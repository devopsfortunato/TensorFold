//! The ids the prompt cache matches a media prompt by (``api.Request.match``): the prompt's, with each image's run of
//! placeholder rows holding words of a SHA-256 over that image (top bit set: no text id has it). The placeholders alone
//! are equal for every image; with these words a kept state resumes only under the same images in the same places.
//! Written for the recipe's patch 0030, split out of prompt.zig so that ``zig build test`` runs its tests.
const std = @import("std");
const api = @import("engine_api");
const Sha256 = std.crypto.hash.sha2.Sha256;

/// The fewest placeholder rows an image run may have and still be keyed: a run of n rows carries n of the digest's
/// 31-bit words (cycling through its first 8), so a run under 4 rows (under 124 bits) is left uncached.
pub const min_rows = 4;

/// A length-prefixed field: no two (url, detail) pairs feed the hash the same bytes.
fn field(h: *Sha256, bytes: []const u8) void {
    const n: u64 = bytes.len;
    h.update(std.mem.asBytes(&std.mem.nativeToLittle(u64, n)));
    h.update(bytes);
}

/// ``ids`` with each image's run of placeholder rows replaced by words of a SHA-256 over: a version tag, the image's
/// url (the data URL's bytes) and detail, the run length and its rotary grid (each row's t, h, w from the run's
/// first). An image that is not a data URL (an https URL: the content behind it may change) also hashes its feature
/// rows. Empty (the prompt is neither kept nor resumed) for video, for a run under ``min_rows`` rows, or when the
/// placeholder runs do not pair one to one with the images. ``sources``: a slice of structs with ``kind`` (``.image``
/// or ``.video``), ``url`` and ``detail`` (server/vision.zig's Source).
pub fn key(a: std.mem.Allocator, ids: []const u32, media: *const api.Media, sources: anytype) ![]const u32 {
    const rows = media.rows;
    if (sources.len == 0 or rows.len == 0 or media.positions.len != 3 * ids.len) return &.{};
    for (sources) |s| if (s.kind != .image) return &.{};
    var runs: usize = 1;
    for (rows[1..], rows[0 .. rows.len - 1]) |r, prev| {
        if (r != prev + 1) runs += 1;
    }
    if (runs != sources.len) return &.{};
    const row_bytes = @as(usize, media.width) * 2;
    if (media.features.len != rows.len * row_bytes) return &.{};
    for (rows) |r| if (r >= ids.len) return &.{};
    const out = try a.dupe(u32, ids);
    errdefer a.free(out);
    var i: usize = 0;
    for (sources) |s| {
        var j = i + 1;
        while (j < rows.len and rows[j] == rows[j - 1] + 1) j += 1;
        if (j - i < min_rows) {
            a.free(out);
            return &.{};
        }
        // SHA-256, not a fast seeded hash: a crafted collision would let one image resume another's kept state
        var h = Sha256.init(.{});
        h.update("tensorfold-media-match-v2\x00");
        field(&h, s.url);
        field(&h, s.detail);
        const n: u32 = @intCast(j - i);
        h.update(std.mem.asBytes(&std.mem.nativeToLittle(u32, n)));
        const r0 = rows[i];
        for (rows[i..j]) |r| for (0..3) |d| {
            const off: i32 = media.positions[3 * r + d] - media.positions[3 * r0 + d];
            h.update(std.mem.asBytes(&std.mem.nativeToLittle(i32, off)));
        };
        if (!std.mem.startsWith(u8, s.url, "data:")) h.update(media.features[i * row_bytes .. j * row_bytes]);
        const digest = h.finalResult();
        for (rows[i..j], 0..) |r, t| out[r] = 0x8000_0000 | (std.mem.readInt(u32, digest[4 * (t % 8) ..][0..4], .little) & 0x7fff_ffff);
        i = j;
    }
    return out;
}

const Source = struct { kind: enum { image, video }, url: []const u8, detail: []const u8 = "auto" };

fn positions(comptime n: usize) [3 * n]i32 {
    var pos: [3 * n]i32 = undefined;
    for (0..n) |r| for (0..3) |d| {
        pos[3 * r + d] = @intCast(r);
    };
    return pos;
}

fn mediaOf(rows: []const u32, pos: []const i32, feats: []const u8) api.Media {
    return .{ .rows = rows, .positions = pos, .delta = 0, .features = feats, .width = 2 };
}

// two images: rows 2-5 and 8-12 of a 15-row prompt (placeholder id 9)
const ids2 = [_]u32{ 1, 2, 9, 9, 9, 9, 3, 4, 9, 9, 9, 9, 9, 5, 6 };
const rows2 = [_]u32{ 2, 3, 4, 5, 8, 9, 10, 11, 12 };

test "match ids tell images apart and leave text ids alone" {
    const a = std.testing.allocator;
    const pos = positions(ids2.len);
    const feats: [rows2.len * 2 * 2]u8 = @splat(0);
    const m = mediaOf(&rows2, &pos, &feats);
    const x = [_]Source{ .{ .kind = .image, .url = "data:image/png;base64,AAAA" }, .{ .kind = .image, .url = "data:image/png;base64,BBBB" } };
    const y = [_]Source{ .{ .kind = .image, .url = "data:image/png;base64,AAAA" }, .{ .kind = .image, .url = "data:image/png;base64,CCCC" } };
    const mx = try key(a, &ids2, &m, &x);
    defer a.free(mx);
    const my = try key(a, &ids2, &m, &y);
    defer a.free(my);
    for ([_]usize{ 0, 1, 6, 7, 13, 14 }) |r| try std.testing.expectEqual(ids2[r], mx[r]);
    for (rows2) |r| try std.testing.expect(mx[r] >= 0x8000_0000);
    try std.testing.expect(mx[2] != mx[3] and mx[3] != mx[4] and mx[4] != mx[5]); // the words of one digest differ
    try std.testing.expectEqualSlices(u32, mx[0..8], my[0..8]); // the same first image: equal through it
    try std.testing.expect(std.mem.indexOfDiff(u32, mx, my).? == 8); // the second image differs at its first row
    const again = try key(a, &ids2, &m, &x);
    defer a.free(again);
    try std.testing.expectEqualSlices(u32, mx, again); // deterministic
}

test "detail, the grid and (for a URL that is not a data URL) the features are in the hash" {
    const a = std.testing.allocator;
    const pos = positions(ids2.len);
    const feats: [rows2.len * 2 * 2]u8 = @splat(0);
    const m = mediaOf(&rows2, &pos, &feats);
    const base = [_]Source{ .{ .kind = .image, .url = "data:image/png;base64,AAAA" }, .{ .kind = .image, .url = "data:image/png;base64,BBBB" } };
    const low = [_]Source{ .{ .kind = .image, .url = "data:image/png;base64,AAAA", .detail = "low" }, base[1] };
    const k0 = try key(a, &ids2, &m, &base);
    defer a.free(k0);
    const k1 = try key(a, &ids2, &m, &low);
    defer a.free(k1);
    try std.testing.expect(std.mem.indexOfDiff(u32, k0, k1).? == 2); // another detail: another ids from the first row
    // the same bytes with another rotary grid (e.g. a different resize): other ids
    var pos2 = pos;
    pos2[3 * 4 + 1] += 1; // row 4's h
    const m2 = mediaOf(&rows2, &pos2, &feats);
    const k2 = try key(a, &ids2, &m2, &base);
    defer a.free(k2);
    try std.testing.expect(k0[2] != k2[2]);
    // data URL: the features are not hashed (the bytes decide); an https URL: they are (the content may change)
    var other = feats;
    other[0] = 1;
    const m3 = mediaOf(&rows2, &pos, &other);
    const k3 = try key(a, &ids2, &m3, &base);
    defer a.free(k3);
    try std.testing.expectEqualSlices(u32, k0, k3);
    const web = [_]Source{ .{ .kind = .image, .url = "https://example.com/a.png" }, base[1] };
    const w0 = try key(a, &ids2, &m, &web);
    defer a.free(w0);
    const w1 = try key(a, &ids2, &m3, &web);
    defer a.free(w1);
    try std.testing.expect(w0[2] != w1[2]);
    try std.testing.expectEqualSlices(u32, w0[8..], w1[8..]); // the data URL image after it is unchanged
}

test "url and detail are length-prefixed: a NUL moved between them does not collide" {
    const a = std.testing.allocator;
    const ids = [_]u32{ 1, 9, 9, 9, 9, 2 };
    const rows = [_]u32{ 1, 2, 3, 4 };
    const pos = positions(ids.len);
    const feats: [rows.len * 2 * 2]u8 = @splat(0);
    const m = mediaOf(&rows, &pos, &feats);
    const p = [_]Source{.{ .kind = .image, .url = "data:a\x00b", .detail = "" }};
    const q = [_]Source{.{ .kind = .image, .url = "data:a", .detail = "b\x00" }};
    const kp = try key(a, &ids, &m, &p);
    defer a.free(kp);
    const kq = try key(a, &ids, &m, &q);
    defer a.free(kq);
    try std.testing.expect(kp[1] != kq[1]);
}

test "no key for video, short runs, runs that do not pair with the images, or media that does not fit the ids" {
    const a = std.testing.allocator;
    const pos = positions(ids2.len);
    const feats: [rows2.len * 2 * 2]u8 = @splat(0);
    const m = mediaOf(&rows2, &pos, &feats);
    const x = [_]Source{ .{ .kind = .image, .url = "data:image/png;base64,AAAA" }, .{ .kind = .image, .url = "data:image/png;base64,BBBB" } };
    try std.testing.expectEqual(@as(usize, 0), (try key(a, &ids2, &m, x[0..1])).len); // one source for two runs
    const v = [_]Source{ .{ .kind = .video, .url = "data:video/mp4;base64,AAAA" }, x[1] };
    try std.testing.expectEqual(@as(usize, 0), (try key(a, &ids2, &m, &v)).len);
    const short_rows = [_]u32{ 2, 3, 4 }; // 3 rows: under min_rows
    const short = mediaOf(&short_rows, &pos, feats[0 .. 3 * 4]);
    try std.testing.expectEqual(@as(usize, 0), (try key(a, &ids2, &short, x[0..1])).len);
    var bad_feats = mediaOf(&rows2, &pos, feats[0 .. feats.len - 1]); // the features do not cover the rows
    try std.testing.expectEqual(@as(usize, 0), (try key(a, &ids2, &bad_feats, &x)).len);
    bad_feats = mediaOf(&rows2, pos[0 .. pos.len - 3], &feats); // the positions do not cover the ids
    try std.testing.expectEqual(@as(usize, 0), (try key(a, &ids2, &bad_feats, &x)).len);
    const far = [_]u32{ 2, 3, 4, 99 }; // a row past the ids
    const past = mediaOf(&far, &pos, feats[0 .. 4 * 4]);
    try std.testing.expectEqual(@as(usize, 0), (try key(a, &ids2, &past, x[0..1])).len);
}
