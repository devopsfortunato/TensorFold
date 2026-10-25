//! Structured output (Python engine/grammar.py): libtfgrammar (xgrammar, zig/src/grammar/tf_grammar.cc) loaded when
//! the first structured request arrives, and each stream's grammar as the round loop keeps it: it follows the
//! committed tokens, keeps the window rows an accepted path can use and fills each kept row's allowed-token bits.

const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;

const Fns = struct {
    open: *const fn ([*]const u8, usize, c_int, [*]const i32, c_int, [*]u8, usize) callconv(.c) ?*anyopaque,
    close: *const fn (*anyopaque) callconv(.c) void,
    words: *const fn (*anyopaque) callconv(.c) c_int,
    compile: *const fn (*anyopaque, c_int, [*]const u8, usize, [*]u8, usize) callconv(.c) ?*anyopaque,
    free: *const fn (*anyopaque) callconv(.c) void,
    matcher: *const fn (*anyopaque) callconv(.c) ?*anyopaque,
    matcher_free: *const fn (*anyopaque) callconv(.c) void,
    accept: *const fn (*anyopaque, i32) callconv(.c) c_int,
    rollback: *const fn (*anyopaque, c_int) callconv(.c) c_int,
    terminated: *const fn (*anyopaque) callconv(.c) c_int,
    fill: *const fn (*anyopaque, [*]i32, c_int) callconv(.c) c_int,
};

/// Where the library is looked for: beside the server (zig-out/native/lib, as `zig build grammar` installs it), then
/// the loader's own search path.
const names: []const [:0]const u8 = if (builtin.os.tag == .macos)
    &.{ "@executable_path/../lib/libtfgrammar.dylib", "libtfgrammar.dylib" }
else
    &.{"libtfgrammar.so"};

var lib: ?Fns = null;
var lib_failed = false;
var lib_mutex: std.atomic.Mutex = .unlocked;

fn load() ?Fns {
    while (!lib_mutex.tryLock()) std.atomic.spinLoopHint();
    defer lib_mutex.unlock();
    if (lib) |l| return l;
    if (lib_failed) return null;
    lib_failed = true;
    var dl = for (names) |name| {
        break std.DynLib.open(name) catch continue;
    } else return null;
    var f: Fns = undefined;
    const info = @typeInfo(Fns).@"struct";
    inline for (info.field_names, info.field_types) |name, T| {
        @field(f, name) = dl.lookup(T, "tfg_" ++ name) orelse return null;
    }
    lib_failed = false;
    lib = f;
    return f;
}

/// The library's file name, for the refusal when it cannot be loaded.
pub const library = if (builtin.os.tag == .macos) "libtfgrammar.dylib" else "libtfgrammar.so";

pub const Kind = enum(u8) { json, json_schema, regex, choice, grammar };

/// Compiles running now on compileWithin's threads, those a request stopped waiting for included (per process: a
/// thread past its request may outlive the server that started it).
pub var running: std.atomic.Value(u32) = .init(0);

/// One tokenizer's grammar compiler; compiled grammars are cached by their text (xgrammar's cache). xgrammar's
/// compiler and its cache are thread-safe, so requests compile on their own threads at once, with no lock here.
pub const Compiler = struct {
    f: Fns,
    h: *anyopaque,
    words: usize,

    /// `tokenizer_json`: tokenizer.json's text; `vocab`: the logits' width; `stops`: the model's eos ids.
    pub fn open(tokenizer_json: []const u8, vocab: u32, stops: []const u32, err: *[512]u8) !Compiler {
        const f = load() orelse return error.NoGrammarLibrary;
        err[0] = 0;
        const h = f.open(tokenizer_json.ptr, tokenizer_json.len, @intCast(vocab), @ptrCast(stops.ptr), @intCast(stops.len), err, err.len) orelse return error.GrammarSetup;
        return .{ .f = f, .h = h, .words = @intCast(f.words(h)) };
    }

    /// Closes the compiler, unless a compile a request gave up on still runs on it (the process is exiting then).
    pub fn close(c: *Compiler) void {
        if (running.load(.acquire) == 0) c.f.close(c.h);
    }

    /// The compiled grammar, or error.Grammar with xgrammar's words in `err`, on the caller's thread.
    pub fn compile(c: *const Compiler, kind: Kind, text: []const u8, err: *[512]u8) !Compiled {
        err[0] = 0;
        const g = c.f.compile(c.h, @intFromEnum(kind), text.ptr, text.len, err, err.len) orelse return error.Grammar;
        return .{ .f = c.f, .h = g, .words = c.words };
    }

    pub const Limits = struct {
        /// the longest a request waits, for a free slot and then for its compile; past it the request is refused
        /// and the compile ends alone
        timeout_ms: u32 = 10_000,
        /// compiles at once, those past their timeout included; a request waits for a slot within its timeout
        running: u32 = 4,
    };

    /// `compile` on a thread of its own, waited for at most `limits.timeout_ms` (xgrammar cannot stop a compile):
    /// error.GrammarTimeout past it, and the thread frees what it compiles; error.GrammarBusy when `limits.running`
    /// compiles run for the whole wait, so grammars that never finish cannot pile up threads.
    pub fn compileWithin(c: *const Compiler, gpa: Allocator, io: std.Io, kind: Kind, text: []const u8, limits: Limits, err: *[512]u8) !Compiled {
        err[0] = 0;
        const began = std.Io.Clock.awake.now(io).toNanoseconds();
        const budget: i96 = @as(i96, limits.timeout_ms) * std.time.ns_per_ms;
        while (running.fetchAdd(1, .acq_rel) >= limits.running) {
            _ = running.fetchSub(1, .acq_rel);
            if (std.Io.Clock.awake.now(io).toNanoseconds() - began >= budget) return error.GrammarBusy;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch return error.GrammarBusy;
        }
        const job = gpa.create(Job) catch {
            _ = running.fetchSub(1, .acq_rel);
            return error.OutOfMemory;
        };
        job.* = .{ .c = c.*, .gpa = gpa, .kind = kind, .text = gpa.dupe(u8, text) catch {
            gpa.destroy(job);
            _ = running.fetchSub(1, .acq_rel);
            return error.OutOfMemory;
        } };
        const thread = std.Thread.spawn(.{}, Job.run, .{ job, io }) catch {
            gpa.free(job.text);
            gpa.destroy(job);
            _ = running.fetchSub(1, .acq_rel);
            return error.GrammarBusy;
        };
        thread.detach();
        const left = @max(budget - (std.Io.Clock.awake.now(io).toNanoseconds() - began), 0);
        job.done.waitTimeout(io, .{ .duration = .{ .raw = .fromNanoseconds(left), .clock = .awake } }) catch {
            job.release(); // the thread frees the job and its grammar when the compile ends
            return error.GrammarTimeout;
        };
        defer job.release();
        err.* = job.err;
        const g = job.result orelse return error.Grammar;
        job.result = null;
        return g;
    }

    /// One compileWithin: held by the waiting request and by the compile's thread, freed by the last to let go. It
    /// keeps its own copy of the compiler's handles, so the thread reads nothing of the server's.
    const Job = struct {
        c: Compiler,
        gpa: Allocator,
        kind: Kind,
        text: []u8,
        err: [512]u8 = @splat(0),
        result: ?Compiled = null,
        done: std.Io.Event = .unset,
        refs: std.atomic.Value(u8) = .init(2),

        fn run(j: *Job, io: std.Io) void {
            j.result = j.c.compile(j.kind, j.text, &j.err) catch null;
            j.done.set(io);
            j.release();
            _ = running.fetchSub(1, .acq_rel); // last: the compiler may close once nothing runs on it
        }

        fn release(j: *Job) void {
            if (j.refs.fetchSub(1, .acq_rel) != 1) return;
            if (j.result) |g| g.free(); // the request stopped waiting: nobody took it
            j.gpa.free(j.text);
            j.gpa.destroy(j);
        }
    };
};

/// A compiled grammar. Its matchers hold their own reference (xgrammar's shared state), so it can be freed while
/// replies that use it run.
pub const Compiled = struct {
    f: Fns,
    h: *anyopaque,
    words: usize,

    pub fn free(g: Compiled) void {
        g.f.free(g.h);
    }

    pub fn matcher(g: Compiled) !Matcher {
        return .{ .f = g.f, .h = g.f.matcher(g.h) orelse return error.Grammar };
    }

    /// A fresh reply's grammar, starting after `after` when thinking is on: a matcher, no compile.
    pub fn constraint(g: Compiled, gpa: Allocator, after: ?u32) !*Constraint {
        const m = try g.matcher();
        errdefer m.free();
        const out = try gpa.create(Constraint);
        out.* = .init(m.rules(), g.words, after);
        return out;
    }
};

/// A reply's place in its xgrammar grammar.
pub const Matcher = struct {
    f: Fns,
    h: *anyopaque,

    pub fn free(m: Matcher) void {
        m.f.matcher_free(m.h);
    }

    pub fn rules(m: Matcher) Rules {
        return .{ .ptr = m.h, .vtable = &.{ .accept = acceptFn, .rollback = rollbackFn, .terminated = terminatedFn, .fill = fillFn, .free = freeFn } };
    }

    fn acceptFn(h: *anyopaque, token: u32) anyerror!bool {
        return switch (lib.?.accept(h, @intCast(token))) {
            1 => true,
            0 => false,
            else => error.Grammar,
        };
    }

    fn rollbackFn(h: *anyopaque, n: usize) anyerror!void {
        if (n > 0 and lib.?.rollback(h, @intCast(n)) != 0) return error.Grammar;
    }

    fn terminatedFn(h: *anyopaque) bool {
        return lib.?.terminated(h) == 1;
    }

    fn fillFn(h: *anyopaque, words: []u32) anyerror!void {
        if (lib.?.fill(h, @ptrCast(words.ptr), @intCast(words.len)) != 0) return error.Grammar;
    }

    fn freeFn(h: *anyopaque) void {
        lib.?.matcher_free(h);
    }
};

/// A grammar matcher as the round loop drives it: xgrammar's, or a test's.
pub const Rules = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Whether the grammar takes `token` next (it then has); a rejected token leaves it where it was.
        accept: *const fn (ptr: *anyopaque, token: u32) anyerror!bool,
        rollback: *const fn (ptr: *anyopaque, n: usize) anyerror!void,
        /// The grammar has taken its stop token.
        terminated: *const fn (ptr: *anyopaque) bool,
        /// The next token's allowed bits (token t: bit t % 32 of word t / 32).
        fill: *const fn (ptr: *anyopaque, words: []u32) anyerror!void,
        free: *const fn (ptr: *anyopaque) void,
    };
};

/// A window as the grammar keeps it: the rows an accepted path can use, in window order, and their bits.
pub const Window = struct {
    rows: []const u32, // the kept rows' indices in the planned window (row 0, the pending token, always)
    parents: []const i32, // each kept row's parent among the kept rows (row 0: -1)
    masks: []const u32, // a row's allowed bits after another's (rows.len rows; all set where unconstrained); empty: none
};

/// One reply's grammar at its committed tokens (Python Constraint).
pub const Constraint = struct {
    rules: Rules,
    words: usize,
    after: ?u32, // with thinking on, the grammar starts at the token after this one
    active: bool,

    pub fn init(rules: Rules, words: usize, after: ?u32) Constraint {
        return .{ .rules = rules, .words = words, .after = after, .active = after == null };
    }

    pub fn deinit(c: *Constraint, gpa: Allocator) void {
        c.rules.vtable.free(c.rules.ptr);
        gpa.destroy(c);
    }

    /// The grammar has taken its stop token: the reply is complete.
    pub fn finished(c: *const Constraint) bool {
        return c.active and c.rules.vtable.terminated(c.rules.ptr);
    }

    /// Follow a committed token; false when the grammar rejects it (each is chosen under its mask, forced ones aside).
    pub fn follow(c: *Constraint, token: u32) !bool {
        if (!c.active) {
            c.active = token == c.after.?;
            return true;
        }
        if (c.rules.vtable.terminated(c.rules.ptr)) return true;
        return c.rules.vtable.accept(c.rules.ptr, token);
    }

    /// The next token's bits on its own (a first token, or a one-row window); null when unconstrained.
    pub fn next(c: *Constraint, a: Allocator) !?[]u32 {
        if (!c.active or c.finished()) return null;
        const bits = try a.alloc(u32, c.words);
        try c.rules.vtable.fill(c.rules.ptr, bits);
        return bits;
    }

    /// The rows of `tokens` (row 0 the pending token, `parents` first) an accepted path can use, each constrained row's
    /// bits (Python Constraint.window). A draft the grammar rejects goes with its rows: the parent's masked row could
    /// never choose it; nothing is verified after the stop token. `forced` keeps every row: forced tokens land as
    /// they are, and rows past one the grammar rejects stay unconstrained.
    pub fn window(c: *Constraint, a: Allocator, tokens: []const u32, parents: []const i32, forced: bool) !Window {
        const n = tokens.len;
        const all = try a.alloc(u32, n);
        for (all, 0..) |*r, i| r.* = @intCast(i);
        if (c.finished() or (!c.active and std.mem.indexOfScalar(u32, tokens[1..], c.after.?) == null))
            return .{ .rows = all, .parents = parents, .masks = &.{} }; // the reply has ended, or no row reaches the grammar
        const bits = try a.alloc(u32, n * c.words);
        @memset(bits, std.math.maxInt(u32));
        const kept = try a.alloc(bool, n);
        @memset(kept, forced);
        kept[0] = true;
        var walk: Walk = .{ .c = c, .tokens = tokens, .parents = parents, .bits = bits, .kept = kept };
        try walk.visit(0, c.active);
        var count: usize = 0;
        for (kept) |k| count += @intFromBool(k);
        const rows = try a.alloc(u32, count);
        const index = try a.alloc(i32, n);
        var at: usize = 0;
        for (kept, 0..) |k, r| {
            index[r] = if (k) @intCast(at) else -1;
            if (!k) continue;
            rows[at] = @intCast(r);
            at += 1;
        }
        const out_parents = try a.alloc(i32, count);
        const masks: []u32 = if (walk.filled) try a.alloc(u32, count * c.words) else &.{};
        for (rows, out_parents, 0..) |r, *p, i| {
            p.* = if (r == 0) -1 else index[@intCast(parents[r])];
            if (walk.filled) @memcpy(masks[i * c.words ..][0..c.words], bits[r * c.words ..][0..c.words]);
        }
        return .{ .rows = rows, .parents = out_parents, .masks = masks };
    }

    const Walk = struct {
        c: *Constraint,
        tokens: []const u32,
        parents: []const i32,
        bits: []u32,
        kept: []bool,
        filled: bool = false,

        /// The matcher has taken row r's path (when `active`): fill its bits, then each child's.
        fn visit(w: *Walk, r: usize, active: bool) !void {
            const c = w.c;
            if (active) {
                try c.rules.vtable.fill(c.rules.ptr, w.bits[r * c.words ..][0..c.words]);
                w.filled = true;
            }
            for (w.parents[1..], 1..) |p, child| {
                if (p != @as(i32, @intCast(r))) continue;
                if (!active) {
                    w.kept[child] = true;
                    try w.visit(child, w.tokens[child] == c.after.?);
                } else if (try c.rules.vtable.accept(c.rules.ptr, w.tokens[child])) {
                    if (!c.rules.vtable.terminated(c.rules.ptr)) {
                        w.kept[child] = true;
                        try w.visit(child, true);
                    }
                    try c.rules.vtable.rollback(c.rules.ptr, 1);
                } // rejected: forced rows stay, unconstrained (kept and all set from the start)
            }
        }
    };
};

/// Whether `token` is allowed by a row's bits.
pub fn allows(bits: []const u32, token: usize) bool {
    const word = token / 32;
    return word < bits.len and (bits[word] >> @intCast(token % 32)) & 1 == 1;
}

/// A test grammar over small ids: the tokens of `want` in order, then `stop`; any token is allowed after the end.
pub const Sequence = struct {
    want: []const u32,
    stop: u32,
    at: usize = 0,
    log: std.ArrayList(usize) = .empty, // `at` before each accepted token, for rollback
    gpa: Allocator,

    pub fn rules(s: *Sequence) Rules {
        return .{ .ptr = s, .vtable = &.{ .accept = acceptFn, .rollback = rollbackFn, .terminated = terminatedFn, .fill = fillFn, .free = freeFn } };
    }

    fn cast(ptr: *anyopaque) *Sequence {
        return @ptrCast(@alignCast(ptr));
    }

    fn expected(s: *const Sequence) ?u32 {
        if (s.at < s.want.len) return s.want[s.at];
        if (s.at == s.want.len) return s.stop;
        return null;
    }

    fn acceptFn(ptr: *anyopaque, token: u32) anyerror!bool {
        const s = cast(ptr);
        if (s.expected() != token) return false;
        try s.log.append(s.gpa, s.at);
        s.at += 1;
        return true;
    }

    fn rollbackFn(ptr: *anyopaque, n: usize) anyerror!void {
        const s = cast(ptr);
        for (0..n) |_| s.at = s.log.pop() orelse return error.Grammar;
    }

    fn terminatedFn(ptr: *anyopaque) bool {
        return cast(ptr).at > cast(ptr).want.len;
    }

    fn fillFn(ptr: *anyopaque, words: []u32) anyerror!void {
        @memset(words, 0);
        const t = cast(ptr).expected() orelse return;
        words[t / 32] |= @as(u32, 1) << @intCast(t % 32);
    }

    fn freeFn(ptr: *anyopaque) void {
        cast(ptr).log.deinit(cast(ptr).gpa);
    }
};

test "a window keeps the drafts the grammar takes, masks each kept row, and stops after the stop token" {
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var seq: Sequence = .{ .want = &.{ 5, 6, 7 }, .stop = 2, .gpa = gpa };
    var c: Constraint = .init(seq.rules(), 1, null);
    defer seq.log.deinit(gpa);
    // a chain: pending 9 (already followed: the grammar is at 5), drafts 5 6 8 (8 rejected) 7
    const w = try c.window(a, &.{ 9, 5, 6, 8, 7 }, &.{ -1, 0, 1, 2, 3 }, false);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2 }, w.rows);
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, 1 }, w.parents);
    try std.testing.expectEqualSlices(u32, &.{ 1 << 5, 1 << 6, 1 << 7 }, w.masks);
    try std.testing.expectEqual(@as(usize, 0), seq.at); // the walk rolled back
    // a tree: two branches from the pending row, one taken; the stop token's row is not verified
    try std.testing.expect(try c.follow(5));
    try std.testing.expect(try c.follow(6));
    const t = try c.window(a, &.{ 9, 4, 7, 2, 3 }, &.{ -1, 0, 0, 2, 3 }, false);
    try std.testing.expectEqualSlices(u32, &.{ 0, 2 }, t.rows);
    try std.testing.expectEqualSlices(u32, &.{ 1 << 7, 1 << 2 }, t.masks);
    // forced rows all stay; the row after a rejected forced token is unconstrained
    const f = try c.window(a, &.{ 9, 4, 3 }, &.{ -1, 0, 1 }, true);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2 }, f.rows);
    try std.testing.expectEqualSlices(u32, &.{ 1 << 7, std.math.maxInt(u32), std.math.maxInt(u32) }, f.masks);
}

test "with thinking on the grammar starts after the think end, and rows before it are unconstrained" {
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var seq: Sequence = .{ .want = &.{5}, .stop = 2, .gpa = gpa };
    defer seq.log.deinit(gpa);
    var c: Constraint = .init(seq.rules(), 1, 30);
    try std.testing.expect((try c.next(a)) == null);
    const none = try c.window(a, &.{ 9, 10, 11 }, &.{ -1, 0, 1 }, false);
    try std.testing.expectEqual(@as(usize, 0), none.masks.len);
    // drafts 30 (the think end), then 5 and 4 (rejected)
    const w = try c.window(a, &.{ 9, 30, 5, 4 }, &.{ -1, 0, 1, 2 }, false);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2 }, w.rows);
    try std.testing.expectEqualSlices(u32, &.{ std.math.maxInt(u32), 1 << 5, 1 << 2 }, w.masks);
    try std.testing.expect(try c.follow(30));
    try std.testing.expect(c.active and !try c.follow(4));
    try std.testing.expect(try c.follow(5) and try c.follow(2) and c.finished());
}
