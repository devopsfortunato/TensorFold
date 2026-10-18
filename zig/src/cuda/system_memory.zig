//! Linux anonymous backing for explicitly capability-gated ATS caches, registered with CUDA after commit.
//! Reserve once, lock only committed pages, and never change a live pointer. Caller must fence GPU users
//! before deinit. The CUDA VMM rings are intentionally not implemented by this allocator.
const std = @import("std");
const builtin = @import("builtin");

extern "c" fn mmap(?*anyopaque, usize, c_int, c_int, c_int, i64) ?*anyopaque;
extern "c" fn munmap(?*anyopaque, usize) c_int;
extern "c" fn mprotect(?*anyopaque, usize, c_int) c_int;
extern "c" fn madvise(?*anyopaque, usize, c_int) c_int;
extern "c" fn mlock(?*const anyopaque, usize) c_int;
extern "c" fn munlock(?*const anyopaque, usize) c_int;
extern "c" fn getpagesize() c_int;

pub const Region = struct {
    base: usize,
    reserved: usize,
    mapped: usize = 0,

    pub fn reserve(bytes: usize, alignment: usize) !Region {
        if (builtin.os.tag != .linux) return error.SystemCacheUnsupported;
        const page: usize = @intCast(getpagesize());
        if (alignment < page or !std.math.isPowerOfTwo(alignment)) return error.BadSystemAlignment;
        const n = std.math.add(usize, @max(bytes, 1), alignment - 1) catch return error.OutOfMemory;
        const size = n & ~(alignment - 1);
        const total = std.math.add(usize, size, alignment) catch return error.OutOfMemory;
        const p = mmap(null, total, 0, 0x22, -1, 0); // PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS
        if (@intFromPtr(p) == std.math.maxInt(usize)) return error.OutOfMemory;
        const start = @intFromPtr(p);
        const base = std.mem.alignForward(usize, start, alignment);
        if (base > start) must(munmap(p, base - start));
        const end = base + size;
        if (end < start + total) must(munmap(@ptrFromInt(end), start + total - end));
        errdefer must(munmap(@ptrFromInt(base), size));
        // Base-page physical allocation: avoid making this a CPU THP/compaction experiment.
        if (madvise(@ptrFromInt(base), size, 15) != 0) return error.SystemCacheAdvice; // MADV_NOHUGEPAGE
        // A vision helper may fork; it must not turn live GPU pages into copy-on-write mappings.
        if (madvise(@ptrFromInt(base), size, 10) != 0) return error.SystemCacheAdvice; // MADV_DONTFORK
        return .{ .base = base, .reserved = size };
    }

    /// `want` is page-aligned, monotonic, and physically charged only on success.
    pub fn growTo(r: *Region, want: usize) !usize {
        if (want <= r.mapped) return 0;
        if (want > r.reserved or want % @as(usize, @intCast(getpagesize())) != 0) return error.RegionFull;
        const size = want - r.mapped;
        const p: *anyopaque = @ptrFromInt(r.base + r.mapped);
        if (mprotect(p, size, 3) != 0) { // READ | WRITE
            std.log.warn("system cache mprotect failed: {s}", .{@tagName(std.posix.errno(@as(c_int, -1)))});
            must(mprotect(p, size, 0)); // mprotect can have changed a prefix before failing
            return error.SystemCacheProtect;
        }
        if (mlock(p, size) != 0) {
            std.log.warn("system cache mlock failed: {s}", .{@tagName(std.posix.errno(@as(c_int, -1)))});
            // mlock may have populated/locked a prefix. Undo only the new, unused suffix.
            must(munlock(p, size));
            must(mprotect(p, size, 0));
            must(madvise(p, size, 4)); // MADV_DONTNEED, also drops populated pages
            return error.SystemCacheLock;
        }
        r.mapped = want;
        return size;
    }

    /// Undo only an unpublished, unregistered suffix after registration failed. No GPU may use that suffix.
    pub fn rollbackGrowth(r: *Region, previous: usize) void {
        std.debug.assert(previous <= r.mapped);
        const size = r.mapped - previous;
        if (size == 0) return;
        const p: *anyopaque = @ptrFromInt(r.base + previous);
        must(munlock(p, size));
        must(mprotect(p, size, 0));
        must(madvise(p, size, 4));
        r.mapped = previous;
    }

    pub fn alloc(bytes: usize) !Region {
        var r = try reserve(bytes, @intCast(getpagesize()));
        errdefer r.deinit();
        _ = try r.growTo(r.reserved);
        return r;
    }

    pub fn deinit(r: *Region) void {
        must(munmap(@ptrFromInt(r.base), r.reserved)); // unmapping also unlocks the pages
        r.* = undefined;
    }
};

fn must(rc: c_int) void {
    if (rc != 0) @panic("system cache mapping cleanup failed");
}

test "system cache aligned reservation, incremental growth and stable data" {
    var r = try Region.reserve(4 * 1024 * 1024, 2 * 1024 * 1024);
    defer r.deinit();
    try std.testing.expectEqual(@as(usize, 0), r.base % (2 * 1024 * 1024));
    try std.testing.expectEqual(@as(usize, 0), r.mapped);
    const page: usize = @intCast(getpagesize());
    try std.testing.expectEqual(page, try r.growTo(page));
    const p: [*]u8 = @ptrFromInt(r.base);
    p[0] = 123;
    p[page - 1] = 45;
    try std.testing.expectEqual(page, try r.growTo(2 * page));
    try std.testing.expectEqual(@as(u8, 123), p[0]);
    try std.testing.expectEqual(@as(u8, 45), p[page - 1]);
    try std.testing.expectEqual(@as(u8, 0), p[page]);
    try std.testing.expectEqual(@as(usize, 0), try r.growTo(page));
    try std.testing.expectError(error.RegionFull, r.growTo(r.reserved + page));
    try std.testing.expectEqual(2 * page, r.mapped);
}

test "system cache rejects overflow and invalid alignment before mmap" {
    try std.testing.expectError(error.OutOfMemory, Region.reserve(std.math.maxInt(usize), 2 * 1024 * 1024));
    try std.testing.expectError(error.BadSystemAlignment, Region.reserve(1, 3));
}

// Exercises rlimit rejection, not a failure partway through page population.
test "system cache lock failure leaves the old prefix usable and permits retry" {
    const Limits = extern struct { soft: usize, hard: usize };
    const libc = struct {
        extern "c" fn getrlimit(c_int, *Limits) c_int;
        extern "c" fn setrlimit(c_int, *const Limits) c_int;
    };
    var old: Limits = undefined;
    if (libc.getrlimit(8, &old) != 0) return error.SkipZigTest; // Linux RLIMIT_MEMLOCK
    const page: usize = @intCast(getpagesize());
    var r = try Region.reserve(3 * page, page);
    defer r.deinit();
    _ = try r.growTo(page);
    const p: [*]u8 = @ptrFromInt(r.base);
    p[0] = 87;
    if (libc.setrlimit(8, &.{ .soft = 0, .hard = old.hard }) != 0) return error.SkipZigTest;
    defer must(libc.setrlimit(8, &old));
    if (r.growTo(2 * page)) |_| {
        // CAP_IPC_LOCK can bypass the limit; do not pretend this covered a failure.
        return error.SkipZigTest;
    } else |err| try std.testing.expectEqual(error.SystemCacheLock, err);
    try std.testing.expectEqual(page, r.mapped);
    try std.testing.expectEqual(@as(u8, 87), p[0]);
    must(libc.setrlimit(8, &old));
    try std.testing.expectEqual(page, try r.growTo(2 * page));
    try std.testing.expectEqual(@as(u8, 87), p[0]);
    try std.testing.expectEqual(@as(u8, 0), p[page]);
}

/// On driver 580.173.02, plain ATS pointers pass eager API calls but graph memory nodes do not replay
/// correctly. Register committed pages to give CUDA the allocation identity those nodes require.
pub fn register(d: *const @import("driver.zig").Driver, base: usize, size: usize) !void {
    const rc = d.api.cuMemHostRegister_v2(@ptrFromInt(base), size, 3); // PORTABLE | DEVICEMAP
    if (rc != 0) {
        std.log.warn("system cache cuMemHostRegister: CUDA error {d}", .{rc});
        return error.SystemCacheRegister;
    }
    errdefer unregister(d, base);
    var device: u64 = 0;
    try d.check(d.api.cuMemHostGetDevicePointer_v2(&device, @ptrFromInt(base), 0), "cuMemHostGetDevicePointer");
    if (device != base) return error.SystemCacheAlias; // all graphs and cache pointers use this reserved VA
}

pub fn unregister(d: *const @import("driver.zig").Driver, base: usize) void {
    if (d.api.cuMemHostUnregister(@ptrFromInt(base)) != 0) @panic("system cache unregister failed");
}

/// Any allocation crossing a registration boundary must be copied in parts, including different src/dst
/// offsets. Cache registrations are aligned in the same VMM granularity as both reserved base addresses.
pub fn copyChunk(dst: u64, src: u64, remaining: usize, granularity: usize) usize {
    return @min(remaining, granularity - @as(usize, @intCast(dst % granularity)), granularity - @as(usize, @intCast(src % granularity)));
}

test "registered cache copy chunks cannot cross either registration boundary" {
    const g: usize = 2 * 1024 * 1024;
    var src: u64 = 19;
    var dst: u64 = g - 27;
    var remaining: usize = 5 * g + 51;
    var total: usize = 0;
    while (remaining != 0) {
        const n = copyChunk(dst, src, remaining, g);
        try std.testing.expect(n > 0 and n <= remaining);
        try std.testing.expectEqual(src / g, (src + n - 1) / g);
        try std.testing.expectEqual(dst / g, (dst + n - 1) / g);
        src += n;
        dst += n;
        remaining -= n;
        total += n;
    }
    try std.testing.expectEqual(5 * g + 51, total);
}
