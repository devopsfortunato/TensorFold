//! CUDA virtual memory for Flash Next's caches (work/research/R3-concurrency-kv.md's KV items): a sequence reserves
//! address space for its whole window once and maps physical memory into it as it grows, so a cache never moves (no
//! copy, no old-plus-new peak, pointers that captured graphs keep), and its indexer keys live in a ring: one
//! physical chunk mapped again and again over the window's rows, since only the rows from the last incomplete
//! block on are ever read (attention.py `_pool_block` reads rows [p0 - 3, p0 + R) of the keys attn_prep wrote).
//! The driver's VMM entry points are looked up from libcuda here (the shared driver binding does not carry them).
const std = @import("std");
const cuda = @import("cuda");

const Allocator = std.mem.Allocator;
const Ptr = u64;
const Result = c_int;

/// CUmemLocation, CUmemAllocationProp and CUmemAccessDesc (cuda.h).
const Location = extern struct { type: c_int = 1, id: c_int = 0 }; // CU_MEM_LOCATION_TYPE_DEVICE
const AllocFlags = extern struct { compression: u8 = 0, rdma: u8 = 0, usage: u16 = 0, reserved: [4]u8 = @splat(0) };
const Prop = extern struct {
    type: c_int = 1, // CU_MEM_ALLOCATION_TYPE_PINNED
    handle_types: c_int = 0,
    location: Location = .{},
    win32: ?*anyopaque = null,
    flags: AllocFlags = .{},
};
const Access = extern struct { location: Location = .{}, flags: c_int = 3 }; // CU_MEM_ACCESS_FLAGS_PROT_READWRITE

comptime {
    std.debug.assert(@sizeOf(Prop) == 32 and @sizeOf(Access) == 12);
}

pub const Vmm = struct {
    lib: std.DynLib,
    d: *const cuda.Driver,
    system_cache: bool = false,
    device: c_int,
    granularity: usize,
    granularityFn: *const fn (*usize, *const Prop, c_int) callconv(.c) Result,
    create: *const fn (*u64, usize, *const Prop, u64) callconv(.c) Result,
    release: *const fn (u64) callconv(.c) Result,
    reserveFn: *const fn (*Ptr, usize, usize, Ptr, u64) callconv(.c) Result,
    addressFree: *const fn (Ptr, usize) callconv(.c) Result,
    map: *const fn (Ptr, usize, usize, u64, u64) callconv(.c) Result,
    unmap: *const fn (Ptr, usize) callconv(.c) Result,
    setAccess: *const fn (Ptr, usize, *const Access, usize) callconv(.c) Result,

    /// The VMM entry points of the driver that `ctx` runs on, and its allocation granularity (minimum).
    pub fn init(ctx: *const cuda.Context) !Vmm {
        var lib = std.DynLib.open("libcuda.so.1") catch return error.DriverUnavailable;
        errdefer lib.close();
        var v: Vmm = undefined;
        v.lib = lib;
        v.d = ctx.d;
        v.system_cache = try systemCacheRequested();
        if (v.system_cache) try checkSystemSupport(ctx);
        v.device = @intCast(ctx.device);
        inline for (.{ .{ "granularityFn", "cuMemGetAllocationGranularity" }, .{ "create", "cuMemCreate" }, .{ "release", "cuMemRelease" }, .{ "reserveFn", "cuMemAddressReserve" }, .{ "addressFree", "cuMemAddressFree" }, .{ "map", "cuMemMap" }, .{ "unmap", "cuMemUnmap" }, .{ "setAccess", "cuMemSetAccess" } }) |e| {
            @field(v, e[0]) = v.lib.lookup(@TypeOf(@field(v, e[0])), e[1]) orelse return error.MissingSymbol;
        }
        const want: Prop = .{ .location = .{ .id = v.device } };
        try check(v.granularityFn(&v.granularity, &want, 0), "cuMemGetAllocationGranularity");
        if (v.granularity == 0) return error.NoGranularity;
        return v;
    }

    pub fn deinit(v: *Vmm) void {
        v.lib.close();
    }

    fn props(v: *const Vmm) Prop {
        return .{ .location = .{ .id = v.device } };
    }

    pub fn up(v: *const Vmm, bytes: usize) usize {
        return std.mem.alignForward(usize, bytes, v.granularity);
    }
};

fn check(rc: Result, what: []const u8) !void {
    if (rc != 0) {
        if (@import("builtin").is_test) {
            std.log.warn("injected {s}: CUDA error {d}", .{ what, rc });
        } else std.log.err("{s}: CUDA error {d}", .{ what, rc });
        return error.CudaVmm;
    }
}

/// Address space for a cache that grows to `reserved` bytes; physical chunks mapped at its end as it grows.
pub const Region = struct {
    v: *const Vmm,
    base: Ptr,
    reserved: usize,
    mapped: usize = 0,
    handles: std.ArrayList(u64) = .empty,
    /// every mapping (offset, size), unmapped one by one
    maps: std.ArrayList([2]usize) = .empty,
    /// a ring's one chunk mapped over every granule of the region
    ring: bool = false,
    system: ?cuda.system_memory.Region = null,
    /// False only after exclusive ownership moved into the idle ring cache.
    owned: bool = true,
    checked_cleanup: bool = false,

    pub fn reserve(v: *const Vmm, bytes: usize, native_ring: bool) !Region {
        const size = v.up(@max(bytes, 1));
        if (v.system_cache and !native_ring) {
            const r = try cuda.system_memory.Region.reserve(size, v.granularity);
            return .{ .v = v, .base = r.base, .reserved = size, .system = r };
        }
        var base: Ptr = 0;
        try check(v.reserveFn(&base, size, v.granularity, 0, 0), "cuMemAddressReserve");
        return .{ .v = v, .base = base, .reserved = size };
    }

    /// Physical memory under the first `bytes` (rounded up to the granularity); what it added.
    pub fn growTo(r: *Region, gpa: Allocator, bytes: usize) !usize {
        std.debug.assert(!r.ring);
        const want = r.v.up(bytes);
        if (want <= r.mapped) return 0;
        if (want > r.reserved) return error.RegionFull;
        if (r.system) |*sys| {
            try r.maps.ensureUnusedCapacity(gpa, 1);
            const previous = sys.mapped;
            const added = try sys.growTo(want);
            errdefer sys.rollbackGrowth(previous);
            try cuda.system_memory.register(r.v.d, sys.base + previous, added);
            r.maps.appendAssumeCapacity(.{ previous, added });
            r.mapped = want;
            return added;
        }
        const size = want - r.mapped;
        var h: u64 = 0;
        const p = r.v.props();
        try check(r.v.create(&h, size, &p, 0), "cuMemCreate");
        errdefer _ = r.v.release(h);
        try check(r.v.map(r.base + r.mapped, size, 0, h, 0), "cuMemMap");
        errdefer _ = r.v.unmap(r.base + r.mapped, size);
        const a: Access = .{ .location = .{ .id = r.v.device } };
        try check(r.v.setAccess(r.base + r.mapped, size, &a, 1), "cuMemSetAccess");
        try r.maps.append(gpa, .{ r.mapped, size });
        try r.handles.append(gpa, h);
        r.mapped = want;
        return size;
    }

    /// The whole region as a ring over one chunk of `chunk` bytes (a multiple of the granularity): address
    /// base + x reaches physical byte x mod chunk. Returns the physical bytes.
    pub fn mapRing(r: *Region, gpa: Allocator, chunk: usize) !usize {
        if (r.system != null) return error.SystemRingUnsupported;
        const size = r.v.up(chunk);
        if (r.reserved % size != 0) return error.RingDoesNotTile;
        // Once created, the Region owns every handle/map even on partial failure.
        // Reserve metadata first so a successful map can always be recorded.
        try r.handles.ensureUnusedCapacity(gpa, 1);
        try r.maps.ensureUnusedCapacity(gpa, r.reserved / size);
        var h: u64 = 0;
        const p = r.v.props();
        try check(r.v.create(&h, size, &p, 0), "cuMemCreate");
        r.handles.appendAssumeCapacity(h);
        var at: usize = 0;
        while (at < r.reserved) : (at += size) {
            try check(r.v.map(r.base + at, size, 0, h, 0), "cuMemMap");
            r.maps.appendAssumeCapacity(.{ at, size });
        }
        const a: Access = .{ .location = .{ .id = r.v.device } };
        try check(r.v.setAccess(r.base, r.reserved, &a, 1), "cuMemSetAccess");
        r.mapped = r.reserved;
        r.ring = true;
        return size;
    }

    pub fn deinit(r: *Region, gpa: Allocator) void {
        if (!r.owned) return;
        if (r.system) |*sys| {
            if (r.v.d.api.cuCtxSynchronize() != 0) @panic("system cache GPU fence failed");
            for (r.maps.items) |entry| cuda.system_memory.unregister(r.v.d, r.base + entry[0]);
            r.maps.deinit(gpa);
            sys.deinit();
            r.* = undefined;
            return;
        }
        for (r.maps.items) |m| {
            const rc = r.v.unmap(r.base + m[0], m[1]);
            if (r.checked_cleanup and rc != 0) @panic("ring cache unmap failed");
        }
        for (r.handles.items) |h| {
            const rc = r.v.release(h);
            if (r.checked_cleanup and rc != 0) @panic("ring cache release failed");
        }
        r.maps.deinit(gpa);
        r.handles.deinit(gpa);
        const rc = r.v.addressFree(r.base, r.reserved);
        if (r.checked_cleanup and rc != 0) @panic("ring cache address free failed");
        r.* = undefined;
    }
};

test "the VMM structs match cuda.h's layout" {
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(Prop));
    try std.testing.expectEqual(@as(usize, 8), @offsetOf(Prop, "location"));
    try std.testing.expectEqual(@as(usize, 16), @offsetOf(Prop, "win32"));
    try std.testing.expectEqual(@as(usize, 24), @offsetOf(Prop, "flags"));
    try std.testing.expectEqual(@as(usize, 12), @sizeOf(Access));
}

/// Strict opt-in; no implicit fallback during an allocation-policy experiment.
pub fn systemCacheRequested() !bool {
    const value = std.c.getenv("TF_FLASHNEXT_SYSTEM_CACHE");
    return parseSystemCache(if (value) |v| std.mem.span(v) else null);
}

fn parseSystemCache(value: ?[]const u8) !bool {
    const text = value orelse return false;
    if (std.mem.eql(u8, text, "0")) return false;
    if (std.mem.eql(u8, text, "1")) return true;
    return error.BadSystemCacheSetting;
}

fn checkSystemSupport(ctx: *const cuda.Context) !void {
    inline for (.{ .integrated, .unified_addressing, .concurrent_managed_access, .pageable_memory_access, .pageable_memory_access_uses_host_page_tables }) |a| {
        if (try ctx.attribute(a) != 1) {
            std.log.err("system cache requires CUDA attribute {s}=1", .{@tagName(a)});
            return error.SystemCacheUnsupported;
        }
    }
}

test "system cache opt-in is strict and defaults off" {
    try std.testing.expect(!try parseSystemCache(null));
    try std.testing.expect(!try parseSystemCache("0"));
    try std.testing.expect(try parseSystemCache("1"));
    try std.testing.expectError(error.BadSystemCacheSetting, parseSystemCache(""));
    try std.testing.expectError(error.BadSystemCacheSetting, parseSystemCache("true"));
}

test "system VMM region grows without CUDA allocation and refuses a ring" {
    const Mock = struct {
        var fences: usize = 0;
        var fail_register: bool = false;
        var wrong_alias: bool = false;
        var unregisters: usize = 0;
        fn register(_: ?*anyopaque, _: usize, _: c_uint) callconv(.c) c_int {
            return if (fail_register) 2 else 0;
        }
        fn unregister(_: ?*anyopaque) callconv(.c) c_int {
            unregisters += 1;
            return 0;
        }
        fn device(out: *u64, ptr: ?*anyopaque, _: c_uint) callconv(.c) c_int {
            out.* = @intFromPtr(ptr) + @intFromBool(wrong_alias);
            return 0;
        }
        fn sync() callconv(.c) c_int {
            fences += 1;
            return 0;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuCtxSynchronize = Mock.sync;
    d.api.cuMemHostRegister_v2 = Mock.register;
    d.api.cuMemHostUnregister = Mock.unregister;
    d.api.cuMemHostGetDevicePointer_v2 = Mock.device;
    var v: Vmm = undefined;
    v.d = &d;
    v.system_cache = true;
    v.granularity = 2 * 1024 * 1024;
    Mock.fences = 0;
    Mock.fail_register = false;
    Mock.wrong_alias = false;
    Mock.unregisters = 0;
    var r = try Region.reserve(&v, 2 * v.granularity, false);
    try std.testing.expectEqual(v.granularity, try r.growTo(std.testing.allocator, 1));
    try std.testing.expectEqual(@as(usize, 0), r.handles.items.len);
    try std.testing.expectEqual(@as(usize, 1), r.maps.items.len);
    try std.testing.expectEqual(@as(usize, 0), try r.growTo(std.testing.allocator, 1));
    try std.testing.expectError(error.SystemRingUnsupported, r.mapRing(std.testing.allocator, v.granularity));
    try std.testing.expectError(error.RegionFull, r.growTo(std.testing.allocator, 3 * v.granularity));
    const p: [*]u8 = @ptrFromInt(r.base);
    p[0] = 31;
    Mock.fail_register = true;
    try std.testing.expectError(error.SystemCacheRegister, r.growTo(std.testing.allocator, 2 * v.granularity));
    try std.testing.expectEqual(v.granularity, r.mapped);
    try std.testing.expectEqual(v.granularity, r.system.?.mapped);
    try std.testing.expectEqual(@as(u8, 31), p[0]);
    try std.testing.expectEqual(@as(usize, 1), r.maps.items.len);
    Mock.fail_register = false;
    Mock.wrong_alias = true;
    try std.testing.expectError(error.SystemCacheAlias, r.growTo(std.testing.allocator, 2 * v.granularity));
    try std.testing.expectEqual(v.granularity, r.mapped);
    try std.testing.expectEqual(v.granularity, r.system.?.mapped);
    try std.testing.expectEqual(@as(usize, 1), r.maps.items.len);
    try std.testing.expectEqual(@as(usize, 1), Mock.unregisters);
    Mock.wrong_alias = false;
    try std.testing.expectEqual(v.granularity, try r.growTo(std.testing.allocator, 2 * v.granularity));
    r.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 3), Mock.unregisters);
    try std.testing.expectEqual(@as(usize, 1), Mock.fences);
}

/// Lazy, exclusive cache of complete native ring Regions. Only the single-device
/// system-cache engine enables it. A Vmm pointer keys device, granularity and the
/// immutable allocation/access policy; sizes additionally distinguish VA aliases.
pub const RingCache = struct {
    pub const physical_cap = 64 * 1024 * 1024;
    pub const va_cap = 1024 * 1024 * 1024;
    pub const map_cap = 512;
    gpa: Allocator = undefined,
    enabled: bool = false,
    slots: [32]?Region = @splat(null),
    bytes: usize = 0,
    va_bytes: usize = 0,
    mappings: usize = 0,
    hits: usize = 0,
    misses: usize = 0,
    retained: usize = 0,
    trims: usize = 0,

    pub fn requested(system_cache: bool) !bool {
        const value = std.c.getenv("TF_FLASHNEXT_RING_CACHE");
        return parse(if (value) |v| std.mem.span(v) else null, system_cache);
    }

    fn parse(value: ?[]const u8, system_cache: bool) !bool {
        const text = value orelse return false;
        if (std.mem.eql(u8, text, "0")) return false;
        if (!std.mem.eql(u8, text, "1")) return error.BadRingCacheSetting;
        if (!system_cache) return error.RingCacheRequiresSystemCache;
        return true;
    }

    pub fn fence(v: *const Vmm) !void {
        try check(v.d.api.cuCtxSynchronize(), "ring cache retirement fence");
    }

    /// Caller has fenced the context and invalidated all old sequence graphs.
    /// Charge stays in Budget.used; the caller removes it from the retiring Seq.
    pub fn putFenced(c: *RingCache, r: *Region) bool {
        if (!c.enabled or !r.owned or !r.ring or r.system != null) return false;
        const chunk = r.maps.items[0][1];
        if (c.bytes + chunk > physical_cap or c.va_bytes + r.reserved > va_cap or c.mappings + r.maps.items.len > map_cap) return false;
        for (&c.slots) |*slot| if (slot.* == null) {
            r.checked_cleanup = true;
            slot.* = r.*;
            r.owned = false;
            c.bytes += chunk;
            c.va_bytes += r.reserved;
            c.mappings += r.maps.items.len;
            c.retained += 1;
            return true;
        };
        return false;
    }

    /// Moves ownership out, removing the idle charge. The caller immediately
    /// charges the returned backing to its new Seq before any fallible zeroing.
    pub fn take(c: *RingCache, v: *const Vmm, reserved: usize, chunk: usize, used: *usize) ?Region {
        if (!c.enabled) return null;
        for (&c.slots) |*slot| {
            const r = slot.* orelse continue;
            if (r.v != v or r.reserved != v.up(@max(reserved, 1)) or r.maps.items[0][1] != chunk) continue;
            slot.* = null;
            c.bytes -= chunk;
            c.va_bytes -= r.reserved;
            c.mappings -= r.maps.items.len;
            if (used.* < chunk) @panic("ring cache charge underflow on take");
            used.* -= chunk;
            c.hits += 1;
            return r;
        }
        c.misses += 1;
        return null;
    }

    /// All entries were fenced at retirement and have no remaining graph users.
    pub fn trim(c: *RingCache, used: *usize) void {
        for (&c.slots) |*slot| {
            if (slot.*) |*r| r.deinit(c.gpa);
            slot.* = null;
        }
        if (used.* < c.bytes) @panic("ring cache charge underflow on trim");
        used.* -= c.bytes;
        if (c.bytes > 0) c.trims += 1;
        c.bytes = 0;
        c.va_bytes = 0;
        c.mappings = 0;
    }

    pub fn report(c: *const RingCache) void {
        if (c.enabled) std.log.info("ring cache: hits={d} misses={d} retained={d} trims={d} idle_bytes={d} idle_va={d} idle_maps={d}", .{ c.hits, c.misses, c.retained, c.trims, c.bytes, c.va_bytes, c.mappings });
    }
};

test "ring cache opt-in requires system cache and is strict" {
    try std.testing.expect(!try RingCache.parse(null, false));
    try std.testing.expect(!try RingCache.parse("0", false));
    try std.testing.expectError(error.RingCacheRequiresSystemCache, RingCache.parse("1", false));
    try std.testing.expectError(error.BadRingCacheSetting, RingCache.parse("yes", true));
    try std.testing.expect(try RingCache.parse("1", true));
}

const RingMock = struct {
    var creates: usize = 0;
    var releases: usize = 0;
    var maps: usize = 0;
    var unmaps: usize = 0;
    var frees: usize = 0;
    var fences: usize = 0;
    var fail_map: usize = 0;
    var fail_access: bool = false;
    var fail_fence: bool = false;
    var next_base: u64 = 0x100000000;
    fn reset() void {
        creates = 0;
        releases = 0;
        maps = 0;
        unmaps = 0;
        frees = 0;
        fences = 0;
        fail_map = 0;
        fail_access = false;
        fail_fence = false;
    }
    fn create(out: *u64, _: usize, _: *const Prop, _: u64) callconv(.c) c_int {
        creates += 1;
        out.* = creates;
        return 0;
    }
    fn release(_: u64) callconv(.c) c_int {
        releases += 1;
        return 0;
    }
    fn reserve(out: *u64, bytes: usize, _: usize, _: u64, _: u64) callconv(.c) c_int {
        out.* = next_base;
        next_base += bytes;
        return 0;
    }
    fn free(_: u64, _: usize) callconv(.c) c_int {
        frees += 1;
        return 0;
    }
    fn map(_: u64, _: usize, _: usize, _: u64, _: u64) callconv(.c) c_int {
        maps += 1;
        return if (maps == fail_map) 2 else 0;
    }
    fn unmap(_: u64, _: usize) callconv(.c) c_int {
        unmaps += 1;
        return 0;
    }
    fn access(_: u64, _: usize, _: *const Access, _: usize) callconv(.c) c_int {
        return if (fail_access) 2 else 0;
    }
    fn sync() callconv(.c) c_int {
        fences += 1;
        return if (fail_fence) 2 else 0;
    }
    fn vmm(d: *cuda.Driver) Vmm {
        d.api.cuCtxSynchronize = sync;
        return .{ .lib = undefined, .d = d, .system_cache = true, .device = 0, .granularity = 2 << 20, .granularityFn = undefined, .create = create, .release = release, .reserveFn = reserve, .addressFree = free, .map = map, .unmap = unmap, .setAccess = access };
    }
    fn ring(v: *const Vmm, size: usize, chunk: usize) !Region {
        var r = try Region.reserve(v, size, true);
        errdefer r.deinit(std.testing.allocator);
        _ = try r.mapRing(std.testing.allocator, chunk);
        return r;
    }
};

test "ring cache transfers whole Regions and physical charge, rejects mismatches, trims once" {
    RingMock.reset();
    var d: cuda.Driver = undefined;
    const v = RingMock.vmm(&d);
    var other = v;
    other.device = 1;
    var cache: RingCache = .{ .gpa = std.testing.allocator, .enabled = true };
    var used: usize = 2 << 20;
    var r = try RingMock.ring(&v, 16 << 20, 2 << 20);
    const base = r.base;
    try RingCache.fence(&v);
    try std.testing.expect(cache.putFenced(&r));
    try std.testing.expect(!r.owned);
    r.deinit(std.testing.allocator); // moved-from cleanup must not free anything
    try std.testing.expectEqual(@as(usize, 0), RingMock.releases);
    try std.testing.expect(cache.take(&other, 16 << 20, 2 << 20, &used) == null);
    try std.testing.expect(cache.take(&v, 2 << 20, 2 << 20, &used) == null);
    try std.testing.expect(cache.take(&v, 16 << 20, 4 << 20, &used) == null);
    var hit = cache.take(&v, 16 << 20, 2 << 20, &used).?;
    try std.testing.expectEqual(base, hit.base);
    try std.testing.expectEqual(@as(usize, 0), used);
    try std.testing.expectEqual(@as(usize, 1), RingMock.creates);
    try std.testing.expectEqual(@as(usize, 8), RingMock.maps);
    used += 2 << 20; // immediate new-Seq charge, before zeroing (including failed zero)
    try RingCache.fence(&v);
    try std.testing.expect(cache.putFenced(&hit));
    cache.trim(&used);
    cache.trim(&used);
    try std.testing.expectEqual(@as(usize, 0), used);
    try std.testing.expectEqual(@as(usize, 1), RingMock.releases);
    try std.testing.expectEqual(@as(usize, 8), RingMock.unmaps);
    try std.testing.expectEqual(@as(usize, 1), RingMock.frees);
    try std.testing.expectEqual(@as(usize, 2), RingMock.fences);
}

test "ring cache is lazy, bounded and disabled path retains nothing" {
    RingMock.reset();
    var d: cuda.Driver = undefined;
    const v = RingMock.vmm(&d);
    var cache: RingCache = .{ .gpa = std.testing.allocator };
    var used: usize = 0;
    try std.testing.expectEqual(@as(usize, 0), RingMock.creates);
    var r = try RingMock.ring(&v, 2 << 20, 2 << 20);
    try std.testing.expect(!cache.putFenced(&r));
    r.deinit(std.testing.allocator);
    cache.enabled = true;
    for (0..33) |i| {
        var next = try RingMock.ring(&v, 2 << 20, 2 << 20);
        if (i < 32) {
            used += 2 << 20;
            try std.testing.expect(cache.putFenced(&next));
        } else {
            try std.testing.expect(!cache.putFenced(&next));
        }
        next.deinit(std.testing.allocator);
    }
    try std.testing.expectEqual(@as(usize, RingCache.physical_cap), cache.bytes);
    cache.trim(&used);
    try std.testing.expectEqual(RingMock.creates, RingMock.releases);
    try std.testing.expectEqual(@as(usize, 0), used);
    // VA and mapping caps independently prevent unbounded aliases.
    var big = try RingMock.ring(&v, 1024 << 20, 2 << 20);
    used += 2 << 20;
    try std.testing.expect(cache.putFenced(&big));
    var extra = try RingMock.ring(&v, 2 << 20, 2 << 20);
    try std.testing.expect(!cache.putFenced(&extra));
    extra.deinit(std.testing.allocator);
    cache.trim(&used);
    try std.testing.expectEqual(RingMock.creates, RingMock.releases);
    try std.testing.expectEqual(RingMock.maps, RingMock.unmaps);
}

test "partial ring construction owns every successful map and handle exactly once" {
    for ([_]usize{ 1, 3, 0 }) |fail| {
        RingMock.reset();
        RingMock.fail_map = fail;
        RingMock.fail_access = fail == 0;
        var d: cuda.Driver = undefined;
        const v = RingMock.vmm(&d);
        try std.testing.expectError(error.CudaVmm, RingMock.ring(&v, 8 << 20, 2 << 20));
        try std.testing.expectEqual(@as(usize, 1), RingMock.releases);
        try std.testing.expectEqual(@as(usize, if (fail == 0) 4 else fail - 1), RingMock.unmaps);
        try std.testing.expectEqual(@as(usize, 1), RingMock.frees);
    }
}

test "failed retirement fence does not permit publication" {
    RingMock.reset();
    var d: cuda.Driver = undefined;
    const v = RingMock.vmm(&d);
    var r = try RingMock.ring(&v, 2 << 20, 2 << 20);
    defer r.deinit(std.testing.allocator);
    var cache: RingCache = .{ .gpa = std.testing.allocator, .enabled = true };
    RingMock.fail_fence = true;
    if (RingCache.fence(&v)) |_| {
        _ = cache.putFenced(&r);
        return error.ExpectedFenceFailure;
    } else |err| try std.testing.expectEqual(error.CudaVmm, err);
    try std.testing.expect(r.owned);
    try std.testing.expectEqual(@as(usize, 0), cache.bytes);
}

test "ring mapping cap is independent of VA and physical caps" {
    RingMock.reset();
    var d: cuda.Driver = undefined;
    var v = RingMock.vmm(&d);
    v.granularity = 1 << 20;
    var cache: RingCache = .{ .gpa = std.testing.allocator, .enabled = true };
    var used: usize = 1 << 20;
    var r = try RingMock.ring(&v, 512 << 20, 1 << 20);
    try std.testing.expect(cache.putFenced(&r));
    try std.testing.expect(cache.va_bytes < RingCache.va_cap);
    try std.testing.expect(cache.bytes < RingCache.physical_cap);
    var other = try RingMock.ring(&v, 1 << 20, 1 << 20);
    try std.testing.expect(!cache.putFenced(&other));
    other.deinit(std.testing.allocator);
    cache.trim(&used);
    try std.testing.expectEqual(RingMock.creates, RingMock.releases);
    try std.testing.expectEqual(RingMock.maps, RingMock.unmaps);
}
