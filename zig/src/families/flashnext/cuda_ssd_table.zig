//! The NVFP4 n-gram table on the host (NVIDIA FP4 checkpoint): per shard `weight` U8 [rows, 80] (two e2m1 codes a
//! byte, low nibble first) and `weight_scale` F8_E4M3 [rows, 10] (one scale a group of 16), one fp32 tensor scale for
//! the layer. A row is 160 bf16 values: bf16_rne((e2m1 * e4m3) * scale_2), the arithmetic of our Python
//! `nvfp4_bf16` (ssd_table.py), one rounding.
//!
//! Rows are read either from the shards' memory maps (page faults, MADV_RANDOM) or with batched positional reads
//! (`pread`) on a pool of short-lived threads; both dedupe the ids, dequantize each distinct row once and repeat it,
//! like SSDNVFP4Table. `warm` asks the kernel to start reading rows (posix_fadvise WILLNEED) before a gather needs
//! them, the prompt-wide read-ahead of our Python build (ple_read.cpp `will_need`).
//!
//! Written for TensorFold by TensorFold Contributors.
const std = @import("std");

extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn pread(fd: c_int, buf: [*]u8, n: usize, off: i64) isize;
extern "c" fn posix_fadvise(fd: c_int, off: i64, len: i64, advice: c_int) c_int;
extern "c" fn madvise(addr: ?*anyopaque, len: usize, advice: c_int) c_int;

const O_CLOEXEC: c_int = 0o2000000;
const FADV_RANDOM: c_int = 1;
const FADV_WILLNEED: c_int = 3;
const MADV_RANDOM: c_int = 1;

/// Reader threads a big gather uses (measured on a Spark: 128 beat 64 and 32 cold).
pub const default_threads: u32 = 128;

pub fn openFd(path: [:0]const u8) !c_int {
    const fd = open(path.ptr, O_CLOEXEC);
    if (fd < 0) return error.NgramOpen;
    return fd;
}

pub fn closeFd(fd: c_int) void {
    _ = close(fd);
}

pub const row_codes = 80;
pub const row_scales = 10;
pub const width = 160;

/// One shard: its two tensors mapped (and as file ranges for `pread`).
pub const Shard = struct {
    codes: []const u8,
    scales: []const u8,
    fd: c_int = -1,
    /// close `fd` in deinit (shards of one file share it: one owner)
    owns_fd: bool = false,
    codes_at: u64 = 0,
    scales_at: u64 = 0,
    rows: u64,
};

pub const ReadMode = enum { mmap, pread };

pub fn modeFromEnv() ReadMode {
    const v = std.c.getenv("TF_NGRAM_READ") orelse return .pread;
    return if (std.mem.eql(u8, std.mem.span(v), "mmap")) .mmap else .pread;
}

pub fn threadsFromEnv(default: u32) u32 {
    const v = std.c.getenv("TF_NGRAM_THREADS") orelse return default;
    return std.fmt.parseInt(u32, std.mem.span(v), 10) catch default;
}

const Ent = struct {
    id: i64,
    pos: u32,

    fn less(_: void, a: Ent, b: Ent) bool {
        return a.id < b.id;
    }
};

pub const Table = struct {
    gpa: std.mem.Allocator,
    shards: []Shard,
    starts: []u64,
    rows: u64,
    scale: f32,
    mode: ReadMode,
    threads: u32,
    pair: [256][2]f32,
    e4: [256]f32,
    /// ids sorted for the last gather (reused)
    ents: std.ArrayList(Ent) = .empty,
    heads: std.ArrayList(u32) = .empty,

    /// Takes ownership of `shards` (a slice allocated with `gpa`); `fd`s are closed by deinit.
    pub fn init(gpa: std.mem.Allocator, shards: []Shard, scale: f32, mode: ReadMode, threads: u32) !Table {
        var t: Table = .{ .gpa = gpa, .shards = shards, .starts = try gpa.alloc(u64, shards.len + 1), .rows = 0, .scale = scale, .mode = mode, .threads = @max(1, threads), .pair = undefined, .e4 = undefined };
        t.starts[0] = 0;
        for (shards, 0..) |s, i| {
            if (s.codes.len != s.rows * row_codes or s.scales.len != s.rows * row_scales) return error.NgramShardShape;
            t.starts[i + 1] = t.starts[i] + s.rows;
        }
        t.rows = t.starts[shards.len];
        const e2m1 = [8]f32{ 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 };
        for (&t.pair, 0..) |*p, b| {
            const lo: usize = b & 0xF;
            const hi: usize = b >> 4;
            p.* = .{ if (lo & 8 != 0) -e2m1[lo & 7] else e2m1[lo], if (hi & 8 != 0) -e2m1[hi & 7] else e2m1[hi] };
        }
        for (&t.e4, 0..) |*v, b| v.* = e4m3(@intCast(b));
        return t;
    }

    pub fn deinit(t: *Table) void {
        for (t.shards) |s| if (s.fd >= 0 and s.owns_fd) {
            _ = close(s.fd);
        };
        t.gpa.free(t.shards);
        t.gpa.free(t.starts);
        t.ents.deinit(t.gpa);
        t.heads.deinit(t.gpa);
        t.* = undefined;
    }

    /// Random access for the maps and the files: the kernel's readahead only wastes memory on 90-byte rows.
    pub fn adviseRandom(t: *Table) void {
        for (t.shards) |s| {
            _ = madvise(@constCast(@ptrCast(s.codes.ptr)), s.codes.len, MADV_RANDOM);
            _ = madvise(@constCast(@ptrCast(s.scales.ptr)), s.scales.len, MADV_RANDOM);
            if (s.fd >= 0) _ = posix_fadvise(s.fd, 0, 0, FADV_RANDOM);
        }
    }

    fn shardOf(t: *const Table, id: u64) usize {
        var lo: usize = 0;
        var hi: usize = t.shards.len;
        while (hi - lo > 1) {
            const mid = (lo + hi) / 2;
            if (t.starts[mid] <= id) lo = mid else hi = mid;
        }
        return lo;
    }

    /// One row's 160 bf16 bits from its 80 code bytes and 10 scale bytes.
    pub fn dequant(t: *const Table, codes: *const [row_codes]u8, scales: *const [row_scales]u8, out: *[width]u16) void {
        for (0..row_scales) |g| {
            const s = t.e4[scales[g]];
            for (0..8) |k| {
                const p = t.pair[codes[g * 8 + k]];
                out[g * 16 + 2 * k] = rne((p[0] * s) * t.scale);
                out[g * 16 + 2 * k + 1] = rne((p[1] * s) * t.scale);
            }
        }
    }

    fn readAll(fd: c_int, dst: []u8, at: u64) !void {
        var done: usize = 0;
        while (done < dst.len) {
            const got = pread(fd, dst.ptr + done, dst.len - done, @intCast(at + done));
            if (got <= 0) return error.NgramShortRead;
            done += @intCast(got);
        }
    }

    fn rowInto(t: *const Table, id: u64, out: *[width]u16) !void {
        const s = &t.shards[t.shardOf(id)];
        const local = id - t.starts[t.shardOf(id)];
        if (t.mode == .mmap or s.fd < 0) {
            t.dequant(s.codes[local * row_codes ..][0..row_codes], s.scales[local * row_scales ..][0..row_scales], out);
        } else {
            var c: [row_codes]u8 = undefined;
            var sc: [row_scales]u8 = undefined;
            try readAll(s.fd, &c, s.codes_at + local * row_codes);
            try readAll(s.fd, &sc, s.scales_at + local * row_scales);
            t.dequant(&c, &sc, out);
        }
    }

    const Job = struct {
        t: *const Table,
        ents: []const Ent,
        heads: []const u32,
        out: []u16,
        failed: *std.atomic.Value(bool),

        fn run(j: Job) void {
            for (j.heads) |h| {
                const e = j.ents[h];
                j.t.rowInto(@intCast(e.id), j.out[@as(usize, e.pos) * width ..][0..width]) catch {
                    j.failed.store(true, .release);
                    return;
                };
            }
        }
    };

    /// Rows `ids` (global) -> `out[ids.len][160]` bf16 bits.
    pub fn gather(t: *Table, ids: []const i64, out: []u16) !void {
        if (out.len != ids.len * width) return error.NgramOutLength;
        const n = ids.len;
        if (n == 0) return;
        try t.ents.resize(t.gpa, n);
        for (ids, t.ents.items, 0..) |id, *e, i| {
            if (id < 0 or @as(u64, @intCast(id)) >= t.rows) return error.NgramIdOutOfRange;
            e.* = .{ .id = id, .pos = @intCast(i) };
        }
        std.sort.pdq(Ent, t.ents.items, {}, Ent.less);
        t.heads.clearRetainingCapacity();
        for (t.ents.items, 0..) |e, i| if (i == 0 or t.ents.items[i - 1].id != e.id) try t.heads.append(t.gpa, @intCast(i));
        var failed: std.atomic.Value(bool) = .init(false);
        const nh = t.heads.items.len;
        const workers: usize = @min(t.threads, (nh + 15) / 16);
        if (workers <= 1) {
            (Job{ .t = t, .ents = t.ents.items, .heads = t.heads.items, .out = out, .failed = &failed }).run();
        } else {
            var th: [128]std.Thread = undefined;
            var spawned: usize = 0;
            const w = @min(workers, th.len);
            const per = (nh + w - 1) / w;
            var lo: usize = 0;
            while (lo + per < nh) {
                const job: Job = .{ .t = t, .ents = t.ents.items, .heads = t.heads.items[lo .. lo + per], .out = out, .failed = &failed };
                lo += per;
                if (std.Thread.spawn(.{}, Job.run, .{job})) |x| {
                    th[spawned] = x;
                    spawned += 1;
                } else |_| job.run();
            }
            // the caller takes the last share
            (Job{ .t = t, .ents = t.ents.items, .heads = t.heads.items[lo..], .out = out, .failed = &failed }).run();
            for (th[0..spawned]) |x| x.join();
        }
        if (failed.load(.acquire)) return error.NgramShortRead;
        // duplicates repeat their first row
        var first: usize = 0;
        for (t.ents.items, 0..) |e, i| {
            if (i > 0 and t.ents.items[i - 1].id == e.id) {
                @memcpy(out[@as(usize, e.pos) * width ..][0..width], out[@as(usize, t.ents.items[first].pos) * width ..][0..width]);
            } else first = i;
        }
    }

    fn warmRange(t: *const Table, ids: []const i64) void {
        for (ids) |id| {
            if (id < 0 or @as(u64, @intCast(id)) >= t.rows) continue;
            const si = t.shardOf(@intCast(id));
            const s = &t.shards[si];
            if (s.fd < 0) continue;
            const local = @as(u64, @intCast(id)) - t.starts[si];
            _ = posix_fadvise(s.fd, @intCast(s.codes_at + local * row_codes), row_codes, FADV_WILLNEED);
            _ = posix_fadvise(s.fd, @intCast(s.scales_at + local * row_scales), row_scales, FADV_WILLNEED);
        }
    }

    /// Ask the kernel to start reading the pages of rows `ids` (a later gather finds them cached); returns once
    /// asked (the reads run on in the kernel). Call it with a prompt's ids ahead of the chunks that gather them.
    pub fn warm(t: *Table, ids: []const i64) void {
        const workers: usize = @min(32, ids.len / 512);
        if (workers <= 1) return t.warmRange(ids);
        var th: [32]std.Thread = undefined;
        var spawned: usize = 0;
        const per = (ids.len + workers - 1) / workers;
        var lo: usize = 0;
        while (lo + per < ids.len) : (lo += per) {
            if (std.Thread.spawn(.{}, warmRange, .{ t, ids[lo .. lo + per] })) |x| {
                th[spawned] = x;
                spawned += 1;
            } else |_| t.warmRange(ids[lo .. lo + per]);
        }
        t.warmRange(ids[lo..]);
        for (th[0..spawned]) |x| x.join();
    }
};

/// float32 -> bf16 bits, round to nearest even by integer arithmetic; NaNs are not special-cased (Python's
/// `(f32 + 0x7FFF + ((f32 >> 16) & 1)) >> 16` in uint64).
pub fn rne(x: f32) u16 {
    const u: u64 = @as(u32, @bitCast(x));
    return @truncate((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

/// e4m3fn byte -> f32 as Python `e4m3()` (exp 0: subnormal m/8 * 2^-6; 0x7f / 0xff NaN).
pub fn e4m3(b: u8) f32 {
    const e: u32 = (b >> 3) & 0xF;
    const m: u32 = b & 7;
    if (b & 0x7F == 0x7F) return std.math.nan(f32);
    const sign: f32 = if (b & 0x80 != 0) -1.0 else 1.0;
    const mag: f32 = if (e == 0)
        @as(f32, @floatFromInt(m)) / 8.0 * (1.0 / 64.0)
    else
        (1.0 + @as(f32, @floatFromInt(m)) / 8.0) * std.math.ldexp(@as(f32, 1.0), @as(i32, @intCast(e)) - 7);
    return sign * mag;
}

test "rne ties to even, e4m3 values" {
    try std.testing.expectEqual(@as(u16, 0x3F80), rne(1.0));
    try std.testing.expectEqual(@as(u16, 0x3F80), rne(@bitCast(@as(u32, 0x3F808000)))); // tie, even stays
    try std.testing.expectEqual(@as(u16, 0x3F82), rne(@bitCast(@as(u32, 0x3F818000)))); // tie, odd rounds up
    try std.testing.expectEqual(@as(f32, 448.0), e4m3(0x7E));
    try std.testing.expectEqual(@as(f32, -0.001953125), e4m3(0x81 | 0x00));
}

test "gather: two shards, duplicate and out-of-order ids, negative nibbles" {
    const a = std.testing.allocator;
    var c0: [2 * row_codes]u8 = @splat(0x21); // low nibble 1 = 0.5, high nibble 2 = 1.0
    var c1: [1 * row_codes]u8 = @splat(0xA9); // low 9 = -0.5, high A = -1.0
    var s0: [2 * row_scales]u8 = @splat(0x38); // e4m3 1.0
    var s1: [1 * row_scales]u8 = @splat(0x40); // 2.0
    c0[row_codes] = 0x77; // row 1: 6.0, 6.0
    const shards = try a.dupe(Shard, &.{
        .{ .codes = &c0, .scales = &s0, .rows = 2 },
        .{ .codes = &c1, .scales = &s1, .rows = 1 },
    });
    var t = try Table.init(a, shards, 2.0, .mmap, 4);
    defer t.deinit();
    var out: [4 * width]u16 = undefined;
    try t.gather(&.{ 2, 1, 0, 2 }, &out);
    // row 2: -0.5 * 2 * 2 = -2.0 (0xC000), -1.0 * 2 * 2 = -4.0 (0xC080)
    try std.testing.expectEqual(@as(u16, 0xC000), out[0]);
    try std.testing.expectEqual(@as(u16, 0xC080), out[1]);
    try std.testing.expectEqual(@as(u16, 0xC000), out[3 * width]);
    // row 1: 6 * 1 * 2 = 12.0 (0x4140)
    try std.testing.expectEqual(@as(u16, 0x4140), out[width]);
    // row 0: 0.5 * 2 = 1.0 (0x3F80), 1.0 * 2 = 2.0 (0x4000)
    try std.testing.expectEqual(@as(u16, 0x3F80), out[2 * width]);
    try std.testing.expectEqual(@as(u16, 0x4000), out[2 * width + 1]);
    try std.testing.expectError(error.NgramIdOutOfRange, t.gather(&.{3}, out[0..width]));
}
