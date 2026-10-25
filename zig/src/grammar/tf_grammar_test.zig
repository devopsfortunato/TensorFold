//! libtfgrammar's C ABI as the server and the lane core drive it (lanes/grammar.zig's FFI over tf_grammar.cc and
//! xgrammar), over a tiny raw vocabulary. `zig build grammar-test -Dxgrammar=DIR` builds the library and runs these
//! with it on the loader's path; scripts/prepare.sh runs them when it builds the library. Without the library they
//! are skipped.

const std = @import("std");
const grammar = @import("lanes").grammar;
const testing = std.testing;

// ids 0-6: { } " a : 1 and a space; 7 the stop token; the logits are wider than the vocabulary (8-39 have no text);
// a decoder with no byte steps: xgrammar reads the vocabulary raw
const tokenizer_json =
    \\{"model": {"type": "BPE", "vocab": {"{": 0, "}": 1, "\"": 2, "a": 3, ":": 4, "1": 5, " ": 6}, "merges": []},
    \\ "added_tokens": [{"id": 7, "content": "<eos>", "special": true}], "decoder": {"type": "Sequence", "decoders": []}}
;
const vocab = 40;
const stop = 7;

fn open() !grammar.Compiler {
    var err: [512]u8 = undefined;
    return grammar.Compiler.open(tokenizer_json, vocab, &.{stop}, &err) catch |e| {
        if (e == error.NoGrammarLibrary) return error.SkipZigTest;
        std.debug.print("tfg_open: {s}\n", .{std.mem.sliceTo(&err, 0)});
        return e;
    };
}

/// The row's bits allow exactly `want` among the first `vocab` tokens.
fn expectOnly(bits: []const u32, want: []const u32) !void {
    for (0..vocab) |t| {
        const in = std.mem.indexOfScalar(u32, want, @intCast(t)) != null;
        if (grammar.allows(bits, t) != in) {
            std.debug.print("token {d}: allowed {}, expected {}\n", .{ t, grammar.allows(bits, t), in });
            return error.TestExpectedEqual;
        }
    }
}

test "the compiler opens over tokenizer.json, and a mask row holds the logits' width in words" {
    var c = try open();
    defer c.close();
    try testing.expectEqual(@as(usize, (vocab + 31) / 32), c.words);
}

test "json_object: the reply opens with {, takes a key and a value, and ends with the stop token" {
    const gpa = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var c = try open();
    defer c.close();
    var err: [512]u8 = undefined;
    const g = try c.compile(.json, "", &err);
    const k = try g.constraint(gpa, null);
    g.free(); // the matcher keeps its own reference to the compiled grammar
    defer k.deinit(gpa);
    const first = (try k.next(a)).?;
    try testing.expect(grammar.allows(first, 0));
    for ([_]u32{ 1, 3, 4, 5, stop, 8, 39 }) |t| try testing.expect(!grammar.allows(first, t));
    try testing.expect(!try k.follow(3)); // rejected: the matcher stays where it was
    try testing.expect(grammar.allows((try k.next(a)).?, 0));
    for ([_]u32{ 0, 2, 3, 2, 4, 5 }) |t| try testing.expect(try k.follow(t)); // {"a":1
    try testing.expect(grammar.allows((try k.next(a)).?, 1));
    try testing.expect(try k.follow(1));
    try testing.expect(!k.finished());
    const end = (try k.next(a)).?;
    try testing.expect(grammar.allows(end, stop) and !grammar.allows(end, 3));
    try testing.expect(try k.follow(stop));
    try testing.expect(k.finished());
    try testing.expect((try k.next(a)) == null);
}

test "a window over xgrammar keeps the drafts it takes, masks each kept row, and rolls the matcher back" {
    const gpa = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var c = try open();
    defer c.close();
    var err: [512]u8 = undefined;
    const g = try c.compile(.regex, "a1", &err);
    defer g.free();
    const k = try g.constraint(gpa, null);
    defer k.deinit(gpa);
    // pending row 0 (already followed), drafts a 1 <eos> on a chain: a and 1 kept, nothing verified after the stop
    const w = try k.window(a, &.{ 9, 3, 5, stop }, &.{ -1, 0, 1, 2 }, false);
    try testing.expectEqualSlices(u32, &.{ 0, 1, 2 }, w.rows);
    try testing.expectEqualSlices(i32, &.{ -1, 0, 1 }, w.parents);
    try expectOnly(w.masks[0..c.words], &.{3});
    try expectOnly(w.masks[c.words..][0..c.words], &.{5});
    try expectOnly(w.masks[2 * c.words ..][0..c.words], &.{stop});
    // a rejected draft goes with its rows
    const r = try k.window(a, &.{ 9, 5, 3 }, &.{ -1, 0, 1 }, false);
    try testing.expectEqualSlices(u32, &.{0}, r.rows);
    // the walks rolled back: the reply is still at its start
    try expectOnly((try k.next(a)).?, &.{3});
    try testing.expect(try k.follow(3) and try k.follow(5) and try k.follow(stop) and k.finished());
}

test "choice and EBNF compile; thinking starts the grammar after the think end; a bad schema reads xgrammar's words" {
    const gpa = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var c = try open();
    defer c.close();
    var err: [512]u8 = undefined;
    const choice = try c.compile(.choice, "[\"a1\", \"1a\"]", &err);
    defer choice.free();
    const k = try choice.constraint(gpa, 6); // the space stands in for </think>
    defer k.deinit(gpa);
    try testing.expect((try k.next(a)) == null); // thinking: unconstrained
    try testing.expect(try k.follow(0) and try k.follow(6));
    try expectOnly((try k.next(a)).?, &.{ 3, 5 });
    const ebnf = try c.compile(.grammar, "root ::= \"a\" \"1\"", &err);
    ebnf.free();
    try testing.expectError(error.Grammar, c.compile(.json_schema, "{\"type\": \"nope\"}", &err));
    const words = std.mem.sliceTo(&err, 0);
    try testing.expect(words.len > 0 and words[0] != '[' and std.mem.indexOfScalar(u8, words, '\n') == null);
    try testing.expectError(error.Grammar, c.compile(.regex, "(", &err));
    try testing.expectError(error.Grammar, c.compile(.choice, "{}", &err));
}

test "compileWithin: one past the running limit is refused; one past its time is refused and freed by its thread" {
    const gpa = testing.allocator;
    const io = testing.io;
    var c = try open();
    defer c.close();
    defer settle(io); // a refused compile's thread frees its job with the test's allocator
    var err: [512]u8 = undefined;
    try testing.expectError(error.GrammarBusy, c.compileWithin(gpa, io, .json, "", .{ .running = 0 }, &err));
    const g = try c.compileWithin(gpa, io, .regex, "a1", .{}, &err);
    g.free();
    try testing.expectError(error.Grammar, c.compileWithin(gpa, io, .regex, "(", .{}, &err));
    try testing.expect(std.mem.sliceTo(&err, 0).len > 0);
    // no wait at all: refused unless already done; either way the thread frees what it compiled
    var refused: usize = 0;
    for (0..8) |i| {
        const text = try std.fmt.allocPrint(gpa, "a{d}1", .{i});
        defer gpa.free(text);
        if (c.compileWithin(gpa, io, .regex, text, .{ .timeout_ms = 0 }, &err)) |late| late.free() else |e| {
            try testing.expectEqual(error.GrammarTimeout, e);
            refused += 1;
        }
        settle(io);
    }
    // refused compiles still count until their threads end: one more past the limit is refused at once
    const slow = [_][]const u8{ "a1", "1a", "a", "1" };
    for (slow) |t| {
        if (c.compileWithin(gpa, io, .regex, t, .{ .timeout_ms = 0, .running = slow.len }, &err)) |late| late.free() else |e| try testing.expectEqual(error.GrammarTimeout, e);
    }
    if (grammar.running.load(.acquire) == slow.len) try testing.expectError(error.GrammarBusy, c.compileWithin(gpa, io, .regex, "a1", .{ .running = slow.len }, &err));
    std.debug.print("{d} of 8 compiles past a zero wait were refused\n", .{refused});
}

/// Waits until no compile thread runs.
fn settle(io: std.Io) void {
    while (grammar.running.load(.acquire) > 0) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
}
