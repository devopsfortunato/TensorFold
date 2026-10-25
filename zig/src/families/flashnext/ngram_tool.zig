//! Host tool for the NVFP4 n-gram table (no GPU): `gather` dumps rows' bf16 bits for a comparison with Python,
//! `bench` times random gathers. Not part of the engine build.
//!   ngram_tool <snapshot dir> gather <ids.i64> <out.u16> [mmap|pread]
//!   ngram_tool <snapshot dir> bench <calls> <rows per call> [mmap|pread] [threads] [seed] [warm]
//! Written for TensorFold by TensorFold Contributors.
const std = @import("std");
const tbl = @import("cuda_ssd_table.zig");

extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn pread(fd: c_int, buf: [*]u8, n: usize, off: i64) isize;
extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) ?*anyopaque;
extern "c" fn lseek(fd: c_int, off: i64, whence: c_int) i64;

fn readFileAlloc(gpa: std.mem.Allocator, path: [:0]const u8) ![]u8 {
    const fd = open(path, 0);
    if (fd < 0) return error.OpenFailed;
    const len: usize = @intCast(lseek(fd, 0, 2));
    const buf = try gpa.alloc(u8, len);
    var done: usize = 0;
    while (done < len) {
        const got = pread(fd, buf.ptr + done, len - done, @intCast(done));
        if (got <= 0) return error.ReadFailed;
        done += @intCast(got);
    }
    return buf;
}

const Mapped = struct { fd: c_int, mem: []const u8, data: usize, header: std.json.Parsed(std.json.Value) };

fn mapFile(gpa: std.mem.Allocator, path: [:0]const u8) !Mapped {
    const fd = open(path, 0);
    if (fd < 0) return error.OpenFailed;
    const len: usize = @intCast(lseek(fd, 0, 2));
    const p = mmap(null, len, 1, 1, fd, 0) orelse return error.MapFailed;
    if (@intFromPtr(p) == std.math.maxInt(usize)) return error.MapFailed;
    const mem = @as([*]const u8, @ptrCast(p))[0..len];
    const hl: usize = @intCast(std.mem.readInt(u64, mem[0..8], .little));
    const header = try std.json.parseFromSlice(std.json.Value, gpa, mem[8..][0..hl], .{});
    return .{ .fd = fd, .mem = mem, .data = 8 + hl, .header = header };
}

fn tensorRange(m: Mapped, name: []const u8) !struct { begin: u64, end: u64, rows: u64, cols: u64 } {
    const e = m.header.value.object.get(name) orelse return error.MissingTensor;
    const offs = e.object.get("data_offsets").?.array.items;
    const shape = e.object.get("shape").?.array.items;
    return .{ .begin = @intCast(offs[0].integer), .end = @intCast(offs[1].integer), .rows = @intCast(shape[0].integer), .cols = if (shape.len > 1) @intCast(shape[1].integer) else 1 };
}

pub fn main(init: std.process.Init.Minimal) !void {
    const gpa = std.heap.c_allocator;
    var it = std.process.Args.Iterator.init(init.args);
    _ = it.next();
    var args: std.ArrayList([:0]const u8) = .empty;
    while (it.next()) |a| try args.append(gpa, a);
    const dir = args.items[0];
    const cmd = args.items[1];
    const idx_path = try std.fmt.allocPrintSentinel(gpa, "{s}/model.safetensors.index.json", .{dir}, 0);
    const idx_text = try readFileAlloc(gpa, idx_path);
    const idx = try std.json.parseFromSlice(std.json.Value, gpa, idx_text, .{});
    const wm = idx.value.object.get("weight_map").?.object;
    // the prefix before "ngram_embedding."
    var prefix: []const u8 = "";
    var kit = wm.iterator();
    while (kit.next()) |e| {
        if (std.mem.endsWith(u8, e.key_ptr.*, "ngram_embedding.shard_0.weight")) prefix = e.key_ptr.*[0 .. e.key_ptr.len - "shard_0.weight".len];
    }
    if (prefix.len == 0) return error.NoNgramTable;
    var count: usize = 0;
    while (wm.contains(try std.fmt.allocPrint(gpa, "{s}shard_{d}.weight", .{ prefix, count }))) count += 1;
    const mode_arg: []const u8 = if (std.mem.eql(u8, cmd, "gather")) (if (args.items.len > 4) args.items[4] else "pread") else (if (args.items.len > 4) args.items[4] else "pread");
    const mode: tbl.ReadMode = if (std.mem.eql(u8, mode_arg, "mmap")) .mmap else .pread;
    const shards = try gpa.alloc(tbl.Shard, count);
    var maps = std.StringHashMap(Mapped).init(gpa);
    for (shards, 0..) |*s, i| {
        const key = try std.fmt.allocPrint(gpa, "{s}shard_{d}.weight", .{ prefix, i });
        const file = wm.get(key).?.string;
        const path = try std.fmt.allocPrintSentinel(gpa, "{s}/{s}", .{ dir, file }, 0);
        const gop = try maps.getOrPut(file);
        if (!gop.found_existing) gop.value_ptr.* = try mapFile(gpa, path);
        const m = gop.value_ptr.*;
        const w = try tensorRange(m, key);
        const sc = try tensorRange(m, try std.fmt.allocPrint(gpa, "{s}_scale", .{key}));
        if (w.cols != 80 or sc.cols != 10 or w.rows != sc.rows) return error.UnexpectedShard;
        s.* = .{ .codes = m.mem[m.data + w.begin .. m.data + w.end], .scales = m.mem[m.data + sc.begin .. m.data + sc.end], .fd = m.fd, .codes_at = m.data + w.begin, .scales_at = m.data + sc.begin, .rows = w.rows };
    }
    // the layer's tensor scale
    const skey = try std.fmt.allocPrint(gpa, "{s}weight_scale_2", .{prefix});
    const sfile = wm.get(skey).?.string;
    const gop = try maps.getOrPut(sfile);
    if (!gop.found_existing) gop.value_ptr.* = try mapFile(gpa, try std.fmt.allocPrintSentinel(gpa, "{s}/{s}", .{ dir, sfile }, 0));
    const sm = gop.value_ptr.*;
    const sr = try tensorRange(sm, skey);
    const g: f32 = @bitCast(std.mem.readInt(u32, sm.mem[sm.data + sr.begin ..][0..4], .little));
    // the fds are shared by shards of one file: deinit would close twice, so duplicate none and close nothing here
    var t = try tbl.Table.init(gpa, shards, g, mode, tbl.threadsFromEnv(32));
    t.adviseRandom();
    std.debug.print("{d} shards, {d} rows, scale_2 {e}, mode {s}\n", .{ count, t.rows, g, @tagName(mode) });

    if (std.mem.eql(u8, cmd, "gather")) {
        const ids_bytes = try readFileAlloc(gpa, args.items[2]);
        const ids = std.mem.bytesAsSlice(i64, ids_bytes[0 .. ids_bytes.len / 8 * 8]);
        const out = try gpa.alloc(u16, ids.len * tbl.width);
        try t.gather(@alignCast(ids), out);
        const f = std.c.fopen(args.items[3], "wb").?;
        _ = std.c.fwrite(@ptrCast(out.ptr), 1, out.len * 2, f);
        _ = std.c.fclose(f);
    } else {
        const calls = try std.fmt.parseInt(usize, args.items[2], 10);
        const n = try std.fmt.parseInt(usize, args.items[3], 10);
        const seed: u64 = if (args.items.len > 6) try std.fmt.parseInt(u64, args.items[6], 10) else 1;
        const use_warm = args.items.len > 7;
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        const ids = try gpa.alloc(i64, n);
        const out = try gpa.alloc(u16, n * tbl.width);
        var total_ns: u64 = 0;
        var ts: std.c.timespec = undefined;
        var sink: u64 = 0;
        for (0..calls) |_| {
            for (ids) |*x| x.* = rnd.intRangeLessThan(i64, 0, @intCast(t.rows));
            if (use_warm) {
                _ = std.c.clock_gettime(.MONOTONIC, &ts);
                const w0: u64 = @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
                t.warm(ids);
                _ = std.c.clock_gettime(.MONOTONIC, &ts);
                std.debug.print("warm call: {d:.2} ms\n", .{@as(f64, @floatFromInt(@as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec)) - w0)) / 1e6});
                _ = std.c.nanosleep(&.{ .sec = 0, .nsec = 200_000_000 }, null);
            }
            _ = std.c.clock_gettime(.MONOTONIC, &ts);
            const a: u64 = @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
            try t.gather(ids, out);
            _ = std.c.clock_gettime(.MONOTONIC, &ts);
            total_ns += @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec)) - a;
            sink +%= out[7];
        }
        const per = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(calls)) / 1e6;
        std.debug.print("{d} calls x {d} rows: {d:.2} ms per call, {d:.2} us per row (sink {d})\n", .{ calls, n, per, per * 1000.0 / @as(f64, @floatFromInt(n)), sink });
    }
}
