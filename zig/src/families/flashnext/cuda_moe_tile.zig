//! The routed NVFP4 experts of a prompt chunk in shared-memory form (zig/kernels/cuda/fn_experts_tile.cu): a CTA's four
//! warps share an item's rows through shared memory (cp.async, S - 1 K32 groups ahead), each warp stages its weight
//! blocks there too, and a decoded weight fragment serves up to 4 row tiles of 16 pairs on a plan whose items hold 64
//! pairs. The bits are nvfp4_expert_kernel's (`check` and `replay` compare bytes). TF_FLASHNEXT_PROMPT_EXPERT_TILE=1
//! takes it for gate/up of chunks of at least `gu_rows`, =2 for down as well (default 2 at one rank; 0 off).
const std = @import("std");
const cuda = @import("cuda");
const kern = @import("cuda_kernels.zig");
const moep = @import("cuda_moe_prompt.zig");

pub const env = "TF_FLASHNEXT_PROMPT_EXPERT_TILE";

const threads: usize = 128;
const xbytes: usize = 4 * 1024;

pub const Tile4 = struct {
    mod: cuda.Module,
    gu: [5]Fn, // rt4_s3, rt4_s4, rt2_s3, rt2_s4, rt3_s3
    f32: [3]Fn, // rt4_s3, rt4_s4, rt2_s3
    b16: [3]Fn,
    sms: usize = 0,
    /// the plan's item size (pairs), and the function indices of gate/up (1: 4 row tiles, 4 stages) and down (2: 2 row tiles, 3 stages) (TF_FLASHNEXT_PROMPT_EXPERT_TILE_CFG=tile,gu,down)
    tile: usize = 64,
    i_gu: usize = 1,
    i_dn: usize = 2,
    /// faces taken: gate/up always, down unless TF_FLASHNEXT_PROMPT_EXPERT_TILE=1
    use_down: bool = false,
    gu_rows: usize = 1024,
    down_rows: usize = 1024,

    pub const Fn = struct { f: cuda.Function, per_sm: usize, shared: u32 };

    pub fn init(ctx: *const cuda.Context) !Tile4 {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var p: Tile4 = .{ .mod = undefined, .gu = undefined, .f32 = undefined, .b16 = undefined };
        p.mod = try cuda.Module.load(ctx.d, cuda.kernels.fn_experts_tile);
        errdefer p.mod.unload();
        p.gu = .{ try p.fnOf("fn_tile4_gu_rt4_s3", 2, 3), try p.fnOf("fn_tile4_gu_rt4_s4", 2, 4), try p.fnOf("fn_tile4_gu_rt2_s3", 2, 3), try p.fnOf("fn_tile4_gu_rt2_s4", 2, 4), try p.fnOf("fn_tile4_gu_rt3_s3", 2, 3) };
        p.f32 = .{ try p.fnOf("fn_tile4_f32_rt4_s3", 1, 3), try p.fnOf("fn_tile4_f32_rt4_s4", 1, 4), try p.fnOf("fn_tile4_f32_rt2_s3", 1, 3) };
        p.b16 = .{ try p.fnOf("fn_tile4_b16_rt4_s3", 1, 3), try p.fnOf("fn_tile4_b16_rt4_s4", 1, 4), try p.fnOf("fn_tile4_b16_rt2_s3", 1, 3) };
        p.sms = @intCast(try ctx.attribute(.multiprocessor_count));
        p.use_down = if (std.c.getenv(env)) |v| (v[0] == '2') else true;
        if (std.c.getenv(env ++ "_CFG")) |v| {
            var it = std.mem.tokenizeScalar(u8, std.mem.span(v), ',');
            if (it.next()) |x| p.tile = std.fmt.parseInt(usize, x, 10) catch 64;
            if (it.next()) |x| p.i_gu = std.fmt.parseInt(usize, x, 10) catch 0;
            if (it.next()) |x| p.i_dn = std.fmt.parseInt(usize, x, 10) catch 0;
        }
        if (std.c.getenv(env ++ "_ROWS")) |v| {
            var it = std.mem.tokenizeScalar(u8, std.mem.span(v), ',');
            if (it.next()) |x| p.gu_rows = std.fmt.parseInt(usize, x, 10) catch 1024;
            if (it.next()) |x| p.down_rows = std.fmt.parseInt(usize, x, 10) catch 1024;
        }
        if (p.tile < 16 or p.tile > 1024 or p.tile % 16 != 0 or p.i_gu >= p.gu.len or p.i_dn >= p.f32.len) return error.BadPromptTile;
        return p;
    }

    fn fnOf(p: *const Tile4, name: [:0]const u8, m: usize, s: usize) !Fn {
        const f = try p.mod.function(name);
        const shared: u32 = @intCast(s * (xbytes + 4 * m * 576));
        return .{ .f = f, .per_sm = @max(1, @as(usize, @intCast(try f.occupancy(threads, shared)))), .shared = shared };
    }

    pub fn deinit(p: *Tile4) void {
        p.mod.unload();
    }

    pub fn plan(q: *const Tile4, o: kern.Ops, picks: u64, pairs: usize, experts: usize, p: kern.Plan) !void {
        return moep.planTile(o, picks, pairs, experts, q.tile, p);
    }

    fn run(p: *const Tile4, s: cuda.Stream, f: Fn, x: u64, x_stride: usize, slots: usize, w: u64, scale: u64, kg: usize, nb: usize, pl: kern.Plan, out: u64, n: usize, limit: f32, skip: c_int, max_items: usize) !void {
        const units = max_items * ((nb + 3) / 4);
        const grid = @min(units, f.per_sm * p.sms);
        if (grid < 1) return;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(@as(c_int, @intCast(x_stride)));
        a.add(@as(c_int, @intCast(slots)));
        a.add(w);
        a.add(scale);
        a.add(@as(c_int, @intCast(kg)));
        a.add(@as(c_int, @intCast(nb)));
        for ([_]u64{ pl.items, pl.counts, pl.members, out }) |v| a.add(v);
        a.add(@as(c_int, @intCast(n)));
        a.add(limit);
        a.add(skip);
        try cuda.launch.launch(f.f, .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = threads }, .shared = f.shared }, s, &a);
    }

    /// nvfp4/experts.gate_up on this plan (items of `tile` pairs): x [R, D] -> out [R * slots, NI] bf16 SwiGLU of each routed pair.
    pub fn gateUp(p: *const Tile4, s: cuda.Stream, x: u64, x_stride: usize, ex: kern.Experts4, pl: kern.Plan, plan_slots: usize, plan_experts: usize, out: u64, rows: usize, skip: c_int) !void {
        try p.run(s, p.gu[p.i_gu], x, x_stride, plan_slots, ex.up, ex.up_scale, ex.dims / 32, ex.width / 32, pl, out, ex.width, ex.limit, skip, kern.maxItems(rows * plan_slots, plan_experts, p.tile));
    }

    pub fn down(p: *const Tile4, s: cuda.Stream, act: u64, act_stride: usize, ex: kern.Experts4, pl: kern.Plan, plan_slots: usize, plan_experts: usize, out: u64, f32_out: bool, rows: usize, skip: c_int) !void {
        const f = if (f32_out) p.f32[p.i_dn] else p.b16[p.i_dn];
        try p.run(s, f, act, act_stride, 0, ex.down, ex.down_scale, ex.width / 32, ex.dims / 32, pl, out, ex.dims, 0, skip, kern.maxItems(rows * plan_slots, plan_experts, p.tile));
    }
};

// -- the check and the replay: the old kernels against this one, bytes compared -------------------------------

extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn read(fd: c_int, buf: [*]u8, n: usize) isize;
extern "c" fn close(fd: c_int) c_int;

fn bf16Bits(x: f32) u16 {
    const b: u32 = @bitCast(x);
    const r = b + 0x7FFF + ((b >> 16) & 1);
    return @intCast(r >> 16);
}

fn upload(d: *const cuda.Driver, bytes: []const u8) !cuda.DeviceBuffer {
    var b = try cuda.DeviceBuffer.alloc(d, @max(bytes.len, 256));
    errdefer b.free();
    try b.upload(0, bytes);
    return b;
}

fn countDiffer(a: []const u8, b: []const u8, es: usize) !usize {
    if (std.mem.eql(u8, a, b)) return 0;
    var n: usize = 0;
    var i: usize = 0;
    while (i < a.len) : (i += es) n += @intFromBool(!std.mem.eql(u8, a[i .. i + es], b[i .. i + es]));
    return n;
}

fn readPicks(gpa: std.mem.Allocator, path: []const u8) ![]i32 {
    const z = try gpa.alloc(u8, path.len + 1);
    defer gpa.free(z);
    @memcpy(z[0..path.len], path);
    z[path.len] = 0;
    const fd = open(@ptrCast(z.ptr), 0);
    if (fd < 0) return error.NoPlanFile;
    defer _ = close(fd);
    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(gpa);
    var buf: [1 << 16]u8 = undefined;
    while (true) {
        const n = read(fd, &buf, buf.len);
        if (n <= 0) break;
        try list.appendSlice(gpa, buf[0..@intCast(n)]);
    }
    const out = try gpa.alloc(i32, list.items.len / 4);
    @memcpy(std.mem.sliceAsBytes(out), list.items[0 .. out.len * 4]);
    return out;
}

/// Random packed experts (width `ni`), then each of `rows` chunks (synthetic routing, `skew`: a few hot experts) or each
/// plan file of `files` (the engine's recorded router picks, TF_FLASHNEXT_DUMP_PICKS): gate/up and both down faces of the
/// old register kernel (16-pair items), the prompt kernel (cuda_moe_prompt, as it runs today) and this one, bytes
/// compared against the old kernel, then timed with CUDA events (planning included). True when all equal.
pub fn check(gpa: std.mem.Allocator, ctx: *const cuda.Context, ops: kern.Ops, p: *const moep.Prompt4, tx: *const Tile4, ni: usize, rows: []const usize, files: []const []const u8, skew: bool, special: bool) !bool {
    const d = ctx.d;
    const E: usize = 512;
    const D: usize = 2560;
    const top: usize = 10;
    const slots = top + 1;
    var prng = std.Random.DefaultPrng.init(0xe4e4_0001 + ni);
    const r = prng.random();
    const blk_words = 144;
    const gu_n = E * (ni / 32) * (D / 32) * 2 * blk_words;
    const dn_n = E * (D / 32) * (ni / 32) * blk_words;
    const host_w = try gpa.alloc(u32, @max(gu_n, dn_n));
    defer gpa.free(host_w);
    const fillBlocks = struct {
        fn f(rr: std.Random, w: []u32, any: bool) void {
            for (w, 0..) |*v, i| {
                if (i % blk_words < 128 or any) {
                    v.* = rr.int(u32);
                } else {
                    var x: u32 = 0;
                    for (0..4) |q| x |= (@as(u32, rr.intRangeAtMost(u8, 0x18, 0x5F)) | (if (rr.uintLessThan(u8, 8) == 0) @as(u32, 0x80) else 0)) << @intCast(8 * q);
                    v.* = x;
                }
            }
        }
    }.f;
    fillBlocks(r, host_w[0..gu_n], special);
    var w_gu = try upload(d, std.mem.sliceAsBytes(host_w[0..gu_n]));
    defer w_gu.free();
    fillBlocks(r, host_w[0..dn_n], special);
    var w_dn = try upload(d, std.mem.sliceAsBytes(host_w[0..dn_n]));
    defer w_dn.free();
    var sc: [2 * 512]f32 = undefined;
    for (&sc) |*v| v.* = std.math.ldexp(1.0 + r.float(f32), r.intRangeAtMost(i32, -10, -4));
    if (special) for (&sc, 0..) |*v, i| switch (i % 16) {
        0 => v.* = -0.0,
        1 => v.* = @bitCast(@as(u32, 0x00012345)),
        2 => v.* = std.math.ldexp(@as(f32, 1.0), 100),
        3 => v.* = std.math.nan(f32),
        else => {},
    };
    var s_gu = try upload(d, std.mem.sliceAsBytes(&sc));
    defer s_gu.free();
    var s_dn = try upload(d, std.mem.sliceAsBytes(sc[0..E]));
    defer s_dn.free();
    const ex: kern.Experts4 = .{ .up = w_gu.ptr, .down = w_dn.ptr, .up_scale = s_gu.ptr, .down_scale = s_dn.ptr, .width = ni, .dims = D, .limit = if (special) 7.0 else 0 };
    // plans: recorded picks, or synthetic ones of each chunk size
    var plans: std.ArrayList([]i32) = .empty;
    defer {
        for (plans.items) |q| gpa.free(q);
        plans.deinit(gpa);
    }
    var names: std.ArrayList([]const u8) = .empty;
    defer names.deinit(gpa);
    for (files) |fp| {
        try plans.append(gpa, try readPicks(gpa, fp));
        try names.append(gpa, fp);
    }
    for (rows) |R| {
        const hp = try gpa.alloc(i32, R * slots);
        for (0..R) |row| {
            var k: usize = 0;
            while (k < top) {
                const hot = if (skew) r.uintLessThan(u32, 4) == 0 else r.uintLessThan(u32, 10) == 0;
                const e: i32 = if (hot) @intCast(r.uintLessThan(u32, if (skew) 8 else 32)) else @intCast(r.uintLessThan(u32, E));
                const dup = for (hp[row * slots .. row * slots + k]) |q| {
                    if (q == e) break true;
                } else false;
                if (dup) continue;
                hp[row * slots + k] = e;
                k += 1;
            }
            hp[row * slots + top] = @intCast(E);
        }
        try plans.append(gpa, hp);
        try names.append(gpa, if (skew) "synthetic skewed" else "synthetic");
    }
    var pairs_max: usize = 0;
    for (plans.items) |q| pairs_max = @max(pairs_max, q.len);
    const max_r = pairs_max / slots;
    const hx = try gpa.alloc(u16, max_r * D);
    defer gpa.free(hx);
    for (hx) |*v| v.* = bf16Bits(r.floatNorm(f32));
    var x = try upload(d, std.mem.sliceAsBytes(hx));
    defer x.free();
    const hact = try gpa.alloc(u16, pairs_max * ni);
    defer gpa.free(hact);
    for (hact) |*v| v.* = bf16Bits(r.floatNorm(f32) * 0.5);
    var act_in = try upload(d, std.mem.sliceAsBytes(hact));
    defer act_in.free();
    var picks = try cuda.DeviceBuffer.alloc(d, pairs_max * 4);
    defer picks.free();
    const nblk = (pairs_max + 1023) / 1024;
    const max_items = kern.maxItems(pairs_max, E + 1, kern.plan_tile);
    var members = try cuda.DeviceBuffer.alloc(d, pairs_max * 4);
    defer members.free();
    var items = try cuda.DeviceBuffer.alloc(d, max_items * 12);
    defer items.free();
    var counts = try cuda.DeviceBuffer.alloc(d, 256);
    defer counts.free();
    var rank = try cuda.DeviceBuffer.alloc(d, pairs_max * 4);
    defer rank.free();
    var hist = try cuda.DeviceBuffer.alloc(d, nblk * (E + 1) * 4 + 256);
    defer hist.free();
    const pl: kern.Plan = .{ .members = members.ptr, .items = items.ptr, .counts = counts.ptr, .rank = rank.ptr, .hist = hist.ptr };
    var bufs: [3]cuda.DeviceBuffer = undefined;
    const osz = @max(pairs_max * ni * 2, pairs_max * D * 4);
    for (&bufs) |*b| b.* = try cuda.DeviceBuffer.alloc(d, osz);
    defer for (&bufs) |*b| b.free();
    const h1 = try gpa.alloc(u8, osz);
    defer gpa.free(h1);
    const h2 = try gpa.alloc(u8, osz);
    defer gpa.free(h2);
    const skip: c_int = @intCast(E);
    var all = true;
    var tot = [3][3]f64{ .{ 0, 0, 0 }, .{ 0, 0, 0 }, .{ 0, 0, 0 } }; // [face][old, prompt, tile] summed ms
    for (plans.items, names.items) |hp, name| {
        const pairs = hp.len;
        const R = pairs / slots;
        try picks.upload(0, std.mem.sliceAsBytes(hp));
        const faces = [_]struct { name: []const u8, f32_out: bool }{ .{ .name = "gate/up", .f32_out = false }, .{ .name = "down fp32", .f32_out = true }, .{ .name = "down bf16", .f32_out = false } };
        for (faces, 0..) |face, fi| {
            const n = if (fi == 0) ni else D;
            const es: usize = if (face.f32_out) 4 else 2;
            const bytes = pairs * n * es;
            var ms: [3]f32 = .{ 0, 0, 0 };
            for (0..3) |which| {
                const b: cuda.DeviceBuffer = .{ .d = d, .ptr = bufs[which].ptr, .len = bytes };
                try b.fill8(0x11, ops.s.handle);
                var e0 = try cuda.Event.init(d, true);
                defer e0.deinit();
                var e1 = try cuda.Event.init(d, true);
                defer e1.deinit();
                const reps: usize = 3;
                for (0..reps + 1) |rep| {
                    if (rep == 1) try e0.record(ops.s);
                    const o = bufs[which].ptr;
                    switch (which) {
                        0 => {
                            try ops.plan(picks.ptr, pairs, E + 1, kern.plan_tile, pl);
                            if (fi == 0) try ops.nvfp4GateUp(x.ptr, D, ex, pl, slots, E + 1, o, R, skip) else try ops.nvfp4Down(act_in.ptr, ni, ex, pl, slots, E + 1, o, face.f32_out, R, skip);
                        },
                        1 => {
                            try p.plan(ops, picks.ptr, pairs, E + 1, pl);
                            if (fi == 0) try p.gateUp(ops.s, x.ptr, D, ex, pl, slots, E + 1, o, R, skip) else try p.down(ops.s, act_in.ptr, ni, ex, pl, slots, E + 1, o, face.f32_out, R, skip);
                        },
                        else => {
                            try tx.plan(ops, picks.ptr, pairs, E + 1, pl);
                            if (fi == 0) try tx.gateUp(ops.s, x.ptr, D, ex, pl, slots, E + 1, o, R, skip) else try tx.down(ops.s, act_in.ptr, ni, ex, pl, slots, E + 1, o, face.f32_out, R, skip);
                        },
                    }
                }
                try e1.record(ops.s);
                try e1.synchronize();
                ms[which] = try cuda.Event.elapsedMs(e0, e1) / @as(f32, @floatFromInt(reps));
            }
            const b0: cuda.DeviceBuffer = .{ .d = d, .ptr = bufs[0].ptr, .len = bytes };
            const b1: cuda.DeviceBuffer = .{ .d = d, .ptr = bufs[1].ptr, .len = bytes };
            const b2: cuda.DeviceBuffer = .{ .d = d, .ptr = bufs[2].ptr, .len = bytes };
            try b0.download(0, h1[0..bytes]);
            try b2.download(0, h2[0..bytes]);
            const differ = try countDiffer(h1[0..bytes], h2[0..bytes], es);
            try b1.download(0, h2[0..bytes]);
            const differ_p = try countDiffer(h1[0..bytes], h2[0..bytes], es);
            if (differ != 0 or differ_p != 0) all = false;
            for (0..3) |w| tot[fi][w] += ms[w];
            std.debug.print("{s} {s} NI {d} {s}{s} rows {d}: tile {d} vs old {d} values differ (prompt kernel {d}); old {d:.3} ms, prompt {d:.3} ms, tile {d:.3} ms ({d:.2}x old, {d:.2}x prompt)\n", .{ if (differ == 0) "EQUAL" else "DIFFER", face.name, ni, name, if (special) " special" else "", R, tx.tile, differ, differ_p, ms[0], ms[1], ms[2], ms[0] / @max(ms[2], 1e-6), ms[1] / @max(ms[2], 1e-6) });
        }
    }
    if (plans.items.len > 0) for (0..3) |fi| std.debug.print("SUM face {d} over {d} plans: old {d:.3} ms, prompt {d:.3} ms, tile {d:.3} ms ({d:.3}x old, {d:.3}x prompt)\n", .{ fi, plans.items.len, tot[fi][0], tot[fi][1], tot[fi][2], tot[fi][0] / @max(tot[fi][2], 1e-9), tot[fi][1] / @max(tot[fi][2], 1e-9) });
    return all;
}
