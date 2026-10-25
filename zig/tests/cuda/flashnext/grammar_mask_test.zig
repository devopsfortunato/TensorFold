//! Structured output's mask kernel (fn_ops.cu tf_fn_grammar_mask_kernel) through Torch.grammarMask, as the engine
//! launches it, against a host reference: `tf-flashnext-grammar-mask`, PASS/FAIL lines, exit 1 on failure. Rows of
//! bf16 logits `ld` apart, each with its own random mask: a disallowed logit must read -inf (0xff80) and every other
//! byte (allowed logits, the padding past `cols`) must keep its bits; a mask with fewer words than its row disallows
//! the tokens past them. Then the shapes the wrapper refuses.

const std = @import("std");
const cuda = @import("cuda");
const flashnext = @import("flashnext");
const torch_ops = flashnext.torch_ops;

const Case = struct {
    what: []const u8,
    rows: usize,
    cols: usize,
    ld: usize,
    words: usize,
    percent: u32, // each token's chance to be allowed
};

const vocab = 248_320; // Qwen3.8 Flash Next's logits width
const full = (vocab + 31) / 32;

const cases = [_]Case{
    .{ .what = "one row, half allowed", .rows = 1, .cols = vocab, .ld = vocab, .words = full, .percent = 50 },
    .{ .what = "a drafted window's 16 rows, padded rows, few allowed", .rows = 16, .cols = vocab, .ld = vocab + 64, .words = full, .percent = 3 },
    .{ .what = "a mask shorter than its rows", .rows = 3, .cols = 1000, .ld = 1024, .words = 20, .percent = 50 },
    .{ .what = "every token allowed", .rows = 2, .cols = 33, .ld = 40, .words = 2, .percent = 100 },
    .{ .what = "no token allowed", .rows = 2, .cols = 300, .ld = 300, .words = 10, .percent = 0 },
};

pub fn main(init: std.process.Init) !u8 {
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var ops = try torch_ops.Ops.load(&ctx);
    defer ops.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var prng = std.Random.DefaultPrng.init(0x6d61736b);
    for (cases) |c| run(init.gpa, &driver, &ops, stream, c, prng.random()) catch |e| {
        std.debug.print("FAIL grammar mask, {s}: {t}\n", .{ c.what, e });
        return 1;
    };
    refusals(&ops, stream) catch |e| {
        std.debug.print("FAIL grammar mask refusals: {t}\n", .{e});
        return 1;
    };
    std.debug.print("PASS grammar mask: {d} shapes against the host reference, and the refused shapes\n", .{cases.len});
    return 0;
}

fn run(gpa: std.mem.Allocator, d: *const cuda.Driver, ops: *const torch_ops.Ops, stream: cuda.Stream, c: Case, r: std.Random) !void {
    const logits = try gpa.alloc(u16, c.rows * c.ld);
    defer gpa.free(logits);
    for (logits) |*x| x.* = r.int(u16); // any bits, NaNs and infinities included: allowed ones must stay as they are
    const masks = try gpa.alloc(u32, c.rows * c.words);
    defer gpa.free(masks);
    for (masks) |*m| {
        var w: u32 = 0;
        for (0..32) |bit| {
            if (r.uintLessThan(u32, 100) < c.percent) w |= @as(u32, 1) << @intCast(bit);
        }
        m.* = w;
    }
    var dl = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(logits));
    defer dl.free();
    var dm = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(masks));
    defer dm.free();
    try ops.on(stream).grammarMask(dl.ptr, c.ld, c.rows, c.cols, dm.ptr, c.words);
    try stream.synchronize();
    const got = try gpa.alloc(u16, logits.len);
    defer gpa.free(got);
    try dl.download(0, std.mem.sliceAsBytes(got));
    var masked: usize = 0;
    for (0..c.rows) |row| for (0..c.ld) |col| {
        const i = row * c.ld + col;
        const w = col / 32;
        const allowed = col >= c.cols or (w < c.words and (masks[row * c.words + w] >> @intCast(col % 32)) & 1 == 1);
        const want: u16 = if (allowed) logits[i] else 0xff80;
        if (got[i] != want) {
            std.debug.print("FAIL grammar mask, {s}: row {d} column {d} reads 0x{x:0>4}, expected 0x{x:0>4}\n", .{ c.what, row, col, got[i], want });
            return error.TestFailed;
        }
        masked += @intFromBool(!allowed);
    };
    std.debug.print("PASS grammar mask, {s}: {d} rows of {d}, {d} logits masked\n", .{ c.what, c.rows, c.cols, masked });
}

fn refusals(ops: *const torch_ops.Ops, stream: cuda.Stream) !void {
    const t = ops.on(stream);
    if (t.grammarMask(1, 8, 1, 8, 1, 0)) |_| return error.TestFailed else |e| if (e != error.BadGrammarMask) return e; // no words
    if (t.grammarMask(1, 7, 1, 8, 1, 1)) |_| return error.TestFailed else |e| if (e != error.BadGrammarMask) return e; // ld < cols
    if (t.grammarMask(1, 8, 65536, 8, 1, 1)) |_| return error.TestFailed else |e| if (e != error.BadGrammarMask) return e; // rows past a grid's y
    try t.grammarMask(0, 8, 0, 8, 0, 1); // no rows: nothing launched
}
