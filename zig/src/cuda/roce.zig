//! The one-shot RoCE all-gather between two ranks (decode plan D1, work/research/R1-decode.md 3.1): b12x's
//! "RoCEnante" protocol (https://github.com/local-inference-lab/b12x, b12x/comm/roce, by the b12x contributors at
//! Local Inference Lab, Apache License 2.0) in a new implementation: the host proxy in roce_proxy.c (libibverbs opened
//! at run time), the GPU kernel in zig/kernels/cuda/fn_roce.cu. A gather of up to `slot_bytes` a rank costs one kernel
//! launch, with no host work on the receive side; the byte layout is ncclAllGather's (rank 0's shard first), so it is
//! a drop-in for small gathers and graph-capturable (the sequence lives in device memory).
//!
//! Setup: `open` (region, ports, queue pairs), the two ranks swap `blob`s (the caller's control link), `connect`,
//! `start` (the proxy thread). Both ranks must make the same gathers in the same order, as for NCCL.
const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;
const memory = @import("memory.zig");
const module = @import("module.zig");
const launch = @import("launch.zig");
const Stream = @import("stream.zig").Stream;
const kernels = @import("kernels.zig");

const Proxy = opaque {};
extern fn tf_roce_blob_bytes() u64;
extern fn tf_roce_layout(world: c_int, slot_bytes: u64, out: *[5]u64) c_int;
extern fn tf_roce_open(world: c_int, rank: c_int, names: [*]const [*:0]const u8, ports: c_int, gid_index: c_int, region: *anyopaque, region_bytes: u64, slot_bytes: u64, err: [*]u8, err_len: u64) ?*Proxy;
extern fn tf_roce_blob_of(r: *Proxy, out: *anyopaque) c_int;
extern fn tf_roce_connect(r: *Proxy, peer: *const anyopaque) c_int;
extern fn tf_roce_start(r: *Proxy) c_int;
extern fn tf_roce_failed(r: *Proxy) c_int;
extern fn tf_roce_error(r: *Proxy) [*:0]const u8;
extern fn tf_roce_ops(r: *Proxy) u64;
extern fn tf_roce_close(r: *Proxy) void;

pub const max_ports = 2;

/// Read-only bytes to pull into the L2 while a gather waits (decode plan D2): the next kernels' weights.
pub const Prefetch = struct { ptr: abi.DevicePtr = 0, bytes: usize = 0 };

/// L2 prefetch blocks for `bytes`: four 128-byte lines a thread of 512, at most 64 blocks.
fn prefetchBlocks(bytes: usize) usize {
    if (bytes == 0) return 0;
    return std.math.clamp((bytes / 128 + 2047) / 2048, 1, 64);
}

pub const prefetch_symbol: [:0]const u8 = "_ZN10tf_fn_roce8prefetchEPKcy";

/// The prefetch alone, for gathers NCCL makes: one launch on the gather's stream just before it; the lines it asks
/// for keep arriving while NCCL waits on the network.
pub const L2 = struct {
    mod: module.Module,
    f: module.Function,
    d: *const Driver,

    pub fn init(d: *const Driver) !L2 {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try module.Module.load(d, kernels.fn_roce);
        errdefer mod.unload();
        return .{ .mod = mod, .f = try mod.function(prefetch_symbol), .d = d };
    }

    pub fn deinit(l: *L2) void {
        l.mod.unload();
    }

    pub fn run(l: *const L2, p: Prefetch, stream: abi.Stream) !void {
        const blocks = prefetchBlocks(p.bytes);
        if (blocks == 0 or p.ptr == 0) return;
        var a: launch.Args = .{};
        a.add(p.ptr);
        a.add(@as(u64, p.bytes));
        const s: Stream = .{ .d = l.d, .handle = stream };
        try launch.launch(l.f, .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = 512 } }, s, &a);
    }
};

/// The kernel's two forms: 16-byte units (aligned pointers, a multiple of 16 bytes) and 4-byte units.
pub const symbols = [2][:0]const u8{
    "_ZN10tf_fn_roce6gatherI5uint4EEvPKT_PS2_ijS5_S4_PKjPjyPNS_5StateEiiyiPKcy",
    "_ZN10tf_fn_roce6gatherIjEEvPKT_PS1_ijS4_S3_PKjPjyPNS_5StateEiiyiPKcy",
};

pub const Settings = struct {
    /// RDMA device names, one a port (rail), e.g. rocep1s0f1 and roceP2p1s0f1; the peer lists its own in the same order
    ports: []const []const u8,
    gid_index: c_int,
    /// the largest shard a rank, a multiple of 4096 (larger gathers go elsewhere)
    slot_bytes: u64 = 512 << 10,
    /// how long a kernel waits for the peer before it stops the stream (a trap: the peer is gone or its proxy failed)
    timeout_s: u64 = 600,
};

pub const Roce = struct {
    d: *const Driver,
    proxy: *Proxy,
    region: memory.HostBuffer,
    region_dev: abi.DevicePtr,
    state: memory.DeviceBuffer,
    mod: module.Module,
    fns: [2]module.Function,
    layout: [5]u64,
    rank: u32,
    ports: u32,
    slot_bytes: u64,
    timeout_ns: u64,

    pub fn blobBytes() usize {
        return @intCast(tf_roce_blob_bytes());
    }

    /// Two ranks only. `d` must outlive the value; the current CUDA context is the engine's.
    pub fn open(d: *const Driver, rank: u32, s: Settings) !Roce {
        if (rank > 1 or s.ports.len < 1 or s.ports.len > max_ports) return error.Invalid;
        if (!kernels.available) return error.BuiltWithoutKernels;
        var lay: [5]u64 = undefined;
        if (tf_roce_layout(2, s.slot_bytes, &lay) != 0) return error.Invalid;
        var region = try memory.HostBuffer.allocMapped(d, @intCast(lay[4]));
        errdefer region.free();
        @memset(region.bytes, 0);
        const dev = try region.device();
        var names: [max_ports][256:0]u8 = undefined;
        var ptrs: [max_ports][*:0]const u8 = undefined;
        for (s.ports, 0..) |p, i| {
            if (p.len >= 256) return error.Invalid;
            @memcpy(names[i][0..p.len], p);
            names[i][p.len] = 0;
            ptrs[i] = &names[i];
        }
        var err: [512]u8 = @splat(0);
        const proxy = tf_roce_open(2, @intCast(rank), &ptrs, @intCast(s.ports.len), s.gid_index, region.bytes.ptr, lay[4], s.slot_bytes, &err, err.len) orelse {
            std.log.err("RoCE: {s}", .{std.mem.sliceTo(&err, 0)});
            return error.RoceUnavailable;
        };
        errdefer tf_roce_close(proxy);
        var state = try memory.DeviceBuffer.alloc(d, 16);
        errdefer state.free();
        try state.fill32(0, null);
        var mod = try module.Module.load(d, kernels.fn_roce);
        errdefer mod.unload();
        var fns: [2]module.Function = undefined;
        for (symbols, &fns) |name, *f| f.* = try mod.function(name);
        return .{ .d = d, .proxy = proxy, .region = region, .region_dev = dev, .state = state, .mod = mod, .fns = fns, .layout = lay, .rank = rank, .ports = @intCast(s.ports.len), .slot_bytes = s.slot_bytes, .timeout_ns = s.timeout_s * std.time.ns_per_s };
    }

    pub fn deinit(r: *Roce) void {
        tf_roce_close(r.proxy);
        r.mod.unload();
        r.state.free();
        r.region.free();
    }

    /// What the peer needs to connect (blobBytes() bytes into `out`).
    pub fn blob(r: *Roce, out: []u8) !void {
        if (out.len < blobBytes()) return error.Invalid;
        if (tf_roce_blob_of(r.proxy, out.ptr) != 0) return error.Invalid;
    }

    pub fn connect(r: *Roce, peer: []const u8) !void {
        if (peer.len != blobBytes()) return error.BadRoceBlob;
        if (tf_roce_connect(r.proxy, peer.ptr) != 0) {
            std.log.err("RoCE connect: {s}", .{std.mem.span(tf_roce_error(r.proxy))});
            return error.RoceUnavailable;
        }
    }

    pub fn start(r: *Roce) !void {
        if (tf_roce_start(r.proxy) != 0) {
            std.log.err("RoCE start: {s}", .{std.mem.span(tf_roce_error(r.proxy))});
            return error.RoceUnavailable;
        }
    }

    /// The proxy's failure, if any (an RDMA write that did not complete): the kernels then time out and trap.
    pub fn failure(r: *const Roce) ?[]const u8 {
        if (tf_roce_failed(r.proxy) == 0) return null;
        return std.mem.span(tf_roce_error(r.proxy));
    }

    /// A kernel's wait that timed out (it trapped, a fatal error for the context): the sequence and the peer.
    pub fn timedOut(r: *const Roce) ?struct { seq: u32, peer: u32 } {
        const ctrl: *const [32]u32 = @ptrCast(@alignCast(r.region.bytes[@intCast(r.layout[3])..][0..128].ptr));
        const seq = @atomicLoad(u32, &ctrl[2], .acquire);
        if (seq == 0) return null;
        return .{ .seq = seq, .peer = @atomicLoad(u32, &ctrl[3], .acquire) };
    }

    pub fn ops(r: *const Roce) u64 {
        return tf_roce_ops(r.proxy);
    }

    /// Gathers this rank fits: up to slot_bytes a rank, whole 4-byte words.
    pub fn fits(r: *const Roce, bytes: usize) bool {
        return bytes > 0 and bytes <= r.slot_bytes and bytes % 4 == 0;
    }

    /// fits, with both pointers 4-byte aligned and the send buffer either apart from the receive one or exactly this
    /// rank's block of it (NCCL's in-place form: the kernel then rewrites each word of that block with itself, after
    /// staging it, and never reads the peer's block from `send`).
    pub fn takes(r: *const Roce, send: abi.DevicePtr, recv: abi.DevicePtr, bytes: usize) bool {
        if (!r.fits(bytes) or send % 4 != 0 or recv % 4 != 0) return false;
        if (send == recv + r.rank * bytes) return true;
        return send + bytes <= recv or recv + 2 * bytes <= send;
    }

    /// `bytes` from `send` on each rank into `recv` [2][bytes], rank 0's first, on `stream` (capturable); `pf`'s lines
    /// prefetched into the L2 by extra blocks of the same launch. Gathers must run one after another (one stream, or
    /// streams ordered by events): every launch shares the device sequence and counters.
    pub fn allGather(r: *const Roce, send: abi.DevicePtr, recv: abi.DevicePtr, bytes: usize, stream: abi.Stream, pf: Prefetch) !void {
        if (!r.takes(send, recv, bytes)) return error.Invalid;
        const wide = bytes % 16 == 0 and send % 16 == 0 and recv % 16 == 0;
        const unit: usize = if (wide) 16 else 4;
        const n = bytes / unit;
        const threads: usize = 512;
        // the grid stays resident (every block waits for the peer): at most 16 blocks, ~2 units a thread
        const blocks = std.math.clamp((n + threads * 2 - 1) / (threads * 2), 1, 16);
        const padded: u32 = @intCast((bytes + 15) / 16 * 16);
        var a: launch.Args = .{};
        a.add(send);
        a.add(recv);
        a.add(@as(c_int, @intCast(n)));
        a.add(padded);
        a.add(r.region_dev + r.layout[2]); // send slots
        a.add(r.region_dev + r.layout[0]); // receive slots
        a.add(r.region_dev + r.layout[1]); // flags
        a.add(r.region_dev + r.layout[3]); // ctrl
        a.add(@as(u64, r.slot_bytes));
        a.add(r.state.ptr);
        a.add(@as(c_int, @intCast(r.rank)));
        a.add(@as(c_int, @intCast(r.ports)));
        a.add(@as(u64, r.timeout_ns));
        a.add(@as(c_int, @intCast(blocks)));
        const pf_blocks = if (pf.ptr != 0) prefetchBlocks(pf.bytes) else 0;
        a.add(pf.ptr);
        a.add(@as(u64, if (pf_blocks > 0) pf.bytes else 0));
        const s: Stream = .{ .d = r.d, .handle = stream };
        try launch.launch(r.fns[if (wide) 0 else 1], .{ .grid = .{ .x = @intCast(blocks + pf_blocks) }, .block = .{ .x = @intCast(threads) } }, s, &a);
    }
};

/// NCCL_IB_HCA's device names ("rocep1s0f1,roceP2p1s0f1", an optional leading = and :1 suffixes; the proxy uses port
/// 1, so another port is refused), at most `max_ports`; `out` holds slices of `text`.
pub fn parseHcas(text: []const u8, out: *[max_ports][]const u8) ![]const []const u8 {
    var t = text;
    if (t.len > 0 and t[0] == '^') return error.ExcludeListUnsupported;
    if (t.len > 0 and t[0] == '=') t = t[1..];
    var n: usize = 0;
    var it = std.mem.tokenizeScalar(u8, t, ',');
    while (it.next()) |item| {
        var name = item;
        if (std.mem.indexOfScalar(u8, item, ':')) |c| {
            if (!std.mem.eql(u8, item[c + 1 ..], "1")) return error.PortOtherThanOne;
            name = item[0..c];
        }
        if (name.len == 0) continue;
        if (n == max_ports) break;
        out[n] = name;
        n += 1;
    }
    if (n == 0) return error.NoHcas;
    return out[0..n];
}

test "the timeout reader compiles" {
    _ = &Roce.timedOut;
    _ = &Roce.takes;
}

test "prefetch blocks: four lines a thread, at most 64 blocks" {
    try std.testing.expectEqual(@as(usize, 0), prefetchBlocks(0));
    try std.testing.expectEqual(@as(usize, 1), prefetchBlocks(128));
    try std.testing.expectEqual(@as(usize, 16), prefetchBlocks(4 << 20));
    try std.testing.expectEqual(@as(usize, 64), prefetchBlocks(64 << 20));
}

test "NCCL_IB_HCA lists give the RoCE ports" {
    var buf: [max_ports][]const u8 = undefined;
    const two = try parseHcas("rocep1s0f1,roceP2p1s0f1", &buf);
    try std.testing.expectEqual(@as(usize, 2), two.len);
    try std.testing.expectEqualStrings("roceP2p1s0f1", two[1]);
    const one = try parseHcas("=mlx5_0:1", &buf);
    try std.testing.expectEqualStrings("mlx5_0", one[0]);
    try std.testing.expectError(error.PortOtherThanOne, parseHcas("mlx5_0:2", &buf));
    try std.testing.expectError(error.ExcludeListUnsupported, parseHcas("^mlx5_1", &buf));
    try std.testing.expectError(error.NoHcas, parseHcas("", &buf));
}

test "the region layout: receive slots, flags, send slots, control" {
    var lay: [5]u64 = undefined;
    try std.testing.expectEqual(@as(c_int, 0), tf_roce_layout(2, 512 << 10, &lay));
    try std.testing.expectEqual(@as(u64, 0), lay[0]);
    try std.testing.expectEqual(@as(u64, 4 * (512 << 10)), lay[1]);
    try std.testing.expectEqual(lay[1] + 2 * 2 * 2 * 128, lay[2]);
    try std.testing.expectEqual(lay[2] + 2 * (512 << 10), lay[3]);
    try std.testing.expectEqual(lay[3] + 128, lay[4]);
    try std.testing.expect(tf_roce_layout(2, 1000, &lay) != 0);
    try std.testing.expect(tf_roce_layout(3, 4096, &lay) != 0);
}
