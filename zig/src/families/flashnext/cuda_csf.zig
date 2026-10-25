//! NVIDIA's "NVFP4 CSF" compressed block scales (quantized_layers weight_scale_encoding "csf") decoded to the plain
//! [n, k/16] e4m3 bytes a `weight_scale` tensor holds. The same codec as the Python engine's nvfp4_csf.py.
//!
//! A projection of n weight rows and k inputs (c = k / 16 scale columns, c even, n a multiple of 16) is stored as
//!   `<proj>.weight_scale.nvfp4_csf_fixed`       u8 [n / 16, 16 * (1 + c / 2)]
//!   `<proj>.weight_scale.nvfp4_csf_exceptions`  u32 [m]
//! fixed: per group of 16 rows a record of 16 row bases (u8, at most 240) and, row after row, c / 2 bytes of nibbles
//! (low nibble: column 2j, high nibble: column 2j + 1); scale[row, col] = base[row] + nibble. exceptions: little-endian
//! words, position (row * c + col) in bits 0..23 and the exact replacement byte in bits 24..31.
//! Everything is lossless; positions are relative to the projection.
const std = @import("std");

pub const fixed_suffix = ".nvfp4_csf_fixed";
pub const exceptions_suffix = ".nvfp4_csf_exceptions";
const max_base = 240;
const position_bits = 24;

pub const Error = error{ BadGeometry, BadCsf };

/// Bytes of the plain scale matrix.
pub fn plainLen(n: usize, k: usize) usize {
    return n * (k / 16);
}

/// Bytes of the fixed stream.
pub fn fixedLen(n: usize, k: usize) usize {
    return (n / 16) * 16 * (1 + k / 32);
}

fn geometry(n: usize, k: usize) Error!usize {
    if (n == 0 or n % 16 != 0 or k == 0 or k % 32 != 0) return error.BadGeometry;
    const c = k / 16;
    if (n * c > 1 << position_bits) return error.BadGeometry;
    return c;
}

/// Decode one projection into `out` (plainLen(n, k) bytes). `exceptions` are the stream's raw little-endian bytes.
pub fn decode(fixed: []const u8, exceptions: []const u8, n: usize, k: usize, out: []u8) Error!void {
    const c = try geometry(n, k);
    if (fixed.len != fixedLen(n, k) or out.len != n * c or exceptions.len % 4 != 0) return error.BadCsf;
    const half = c / 2;
    const record = 16 * (1 + half);
    for (0..n / 16) |g| {
        const rec = fixed[g * record ..][0..record];
        for (0..16) |r| {
            const base = rec[r];
            if (base > max_base) return error.BadCsf;
            const nib = rec[16 + r * half ..][0..half];
            const row = out[(g * 16 + r) * c ..][0..c];
            for (nib, 0..) |b, j| {
                row[2 * j] = base + (b & 0x0f);
                row[2 * j + 1] = base + (b >> 4);
            }
        }
    }
    var i: usize = 0;
    while (i < exceptions.len) : (i += 4) {
        const w = std.mem.readInt(u32, exceptions[i..][0..4], .little);
        const pos = w & ((1 << position_bits) - 1);
        if (pos >= out.len) return error.BadCsf;
        out[pos] = @intCast(w >> position_bits);
    }
}

/// The reference encoder (tests, tooling): row minimum (at most 240) as base, entries outside base + 0..15 as
/// exceptions. `fixed` takes fixedLen bytes; returns the number of exception words written to `exceptions`.
pub fn encode(scales: []const u8, n: usize, k: usize, fixed: []u8, exceptions: []u32) Error!usize {
    const c = try geometry(n, k);
    if (scales.len != n * c or fixed.len != fixedLen(n, k)) return error.BadCsf;
    const half = c / 2;
    const record = 16 * (1 + half);
    var m: usize = 0;
    for (0..n / 16) |g| {
        const rec = fixed[g * record ..][0..record];
        for (0..16) |r| {
            const row = scales[(g * 16 + r) * c ..][0..c];
            const base: u8 = @min(std.mem.min(u8, row), max_base);
            rec[r] = base;
            for (0..half) |j| {
                var pair: [2]u8 = undefined;
                for (0..2) |h| {
                    const s = row[2 * j + h];
                    const d = @as(i32, s) - base;
                    if (d >= 0 and d <= 15) {
                        pair[h] = @intCast(d);
                    } else {
                        pair[h] = 0;
                        if (m >= exceptions.len) return error.BadCsf;
                        exceptions[m] = @as(u32, @intCast((g * 16 + r) * c + 2 * j + h)) | (@as(u32, s) << position_bits);
                        m += 1;
                    }
                }
                rec[16 + r * half + j] = pair[0] | (pair[1] << 4);
            }
        }
    }
    return m;
}

test "decode inverts the reference encoder, exceptions and the 240 clamp included" {
    const n = 48;
    const k = 96;
    const c = k / 16;
    var scales: [n * c]u8 = undefined;
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    for (0..n) |r| {
        const lo: u8 = if (r == 5) 250 else rnd.intRangeAtMost(u8, 0, 200);
        for (0..c) |j| scales[r * c + j] = if (rnd.uintLessThan(u8, 16) == 0) rnd.int(u8) else lo +| rnd.uintLessThan(u8, 12);
    }
    var fixed: [fixedLen(n, k)]u8 = undefined;
    var exc: [n * c]u32 = undefined;
    const m = try encode(&scales, n, k, &fixed, &exc);
    try std.testing.expect(m > 0);
    var out: [n * c]u8 = undefined;
    try decode(&fixed, std.mem.sliceAsBytes(exc[0..m]), n, k, &out);
    try std.testing.expectEqualSlices(u8, &scales, &out);
}

test "decode refuses a bad base, a position out of range and bad geometry" {
    var fixed: [fixedLen(16, 32)]u8 = @splat(0);
    var out: [plainLen(16, 32)]u8 = undefined;
    try decode(&fixed, &.{}, 16, 32, &out);
    fixed[3] = 241;
    try std.testing.expectError(error.BadCsf, decode(&fixed, &.{}, 16, 32, &out));
    fixed[3] = 0;
    var w: [4]u8 = undefined;
    std.mem.writeInt(u32, &w, @intCast(plainLen(16, 32)), .little);
    try std.testing.expectError(error.BadCsf, decode(&fixed, &w, 16, 32, &out));
    try std.testing.expectError(error.BadGeometry, decode(&fixed, &.{}, 15, 32, &out));
}
