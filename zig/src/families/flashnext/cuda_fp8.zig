//! 128x128-block FP8 linears (ModelOpt FP8_PB_WO: e4m3 `weight` + fp32 `weight_scale_inv`) on CUDA, as the Python
//! engine runs them (cuda/nvfp4/linear.py Fp8BlockLinear): e4m3 bytes in the FP8 GEMM's fragment order, an fp32
//! scale per (64 inputs, output column), and the lane matmul in mode FP8G (qmmf.cu) for decode and prompt rows
//! alike: each 64 inputs' bf16 MMA products scaled into an fp32 sum in block order, K slices fixed by the shape, a
//! row's bits independent of every other row and of M. zig/kernels/cuda/fn_qmmf.cu is the device code copied by
//! zig/tests/cuda/copies.py (SASS-equal to the Python extension tensorfold_nvfp4_v3); `matmul` is linear._matmul and
//! qmmf_cuda's host logic. The host layout helpers make the loader's buffers byte for byte as from_checkpoint does.
//!
//! Python source (TensorFold, https://github.com/ashhart/TensorFold, cuda/nvfp4/linear.py and qmmf.cu, authored by
//! Ash Hart (ashhart)).
const std = @import("std");
const cuda = @import("cuda");

/// Fp8BlockLinear: w8 uint8 [npad/64][K/64][8][32][2][8] (fragment order), bs fp32 [npad/64][K/64][64].
///
/// `mx8`: MXFP8 (qmmf mode 2, linear.py Mx8Linear): the same fragment-order bytes, `bs` uint8 [npad/64][K/64][64][2] of
/// e8m0 exponents (an exponent per 32 inputs, a tile's together) instead of fp32 scales per 64.
///
/// `fold`: every e8m0 byte is in 3..134 (Mx8Linear.fold: an e4m3 byte times its power of two stays exact in bf16), so prompt
/// rows may use fn_prompt.cu's chain (`promptMatmul`) rather than the lane matmul.
pub const Linear = struct { w8: u64 = 0, bs: u64 = 0, n: u32 = 0, k: u32 = 0, npad: u32 = 0, mx8: bool = false, fold: bool = false };

/// linear.py FUSED_ROWS (TF_QMMF_FUSED_ROWS default): rows from which a block sums its tile's K slices itself.
pub const fused_rows: usize = 256;

/// Rows of output padding Fp8Linear / Fp8BlockLinear take: ceil(n / 128) * 128.
pub fn npadOf(n: usize) usize {
    return (n + 127) / 128 * 128;
}

/// qmm.split_k(n, k, gs=64, target=192): K slices fixed by the weight's shape, never by the row count.
pub fn splitK(n: usize, k: usize) usize {
    const tiles = (n + 63) / 64;
    const groups = k / 64;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < 192 and groups % (sk * 2) == 0 and groups / (sk * 2) >= 8) sk *= 2;
    return sk;
}

/// qmm.bucket: the row tile.
pub fn bucket(m: usize) usize {
    return if (m <= 16) 16 else if (m <= 32) 32 else 64;
}

/// qmmf.cu Tile<FP8G or MXFP8, BM, 64, 1, 4, 4>::SMEM: max(4 stages, the cluster partials); a stage holds 64 columns'
/// scales: 4 bytes each (FP8G), 2 (MXFP8).
pub fn smem(bm: usize) u32 {
    return smemOf(bm, false);
}

pub fn smemOf(bm: usize, mx8: bool) u32 {
    const stage = std.mem.alignForward(usize, bm * 128 + 64 * 64 + 64 * @as(usize, if (mx8) 2 else 4), 128);
    const partials = (bm / 16) * 2 * 4 * 128 * 4;
    return @intCast(@max(4 * stage, partials));
}

/// qmmf.cu launch's L2 band: row tiles whose inputs fill about 12 MB.
pub fn l2Group(rows_t: usize, bm: usize, k: usize) usize {
    return @max(1, @min(rows_t, (12 << 20) / (bm * k * 2)));
}

/// Mangled names of the FP8G instantiations (fn_qmmf.cu's footer): [F32][cluster][bm 16, 32, 64], and the fused [F32].
pub const sym = struct {
    fn name(comptime mode: u32, comptime bm: u32, comptime f32_out: bool, comptime cluster: bool, comptime fuse: bool) [:0]const u8 {
        return std.fmt.comptimePrint("_ZN10tf_fn_qmmf11qmmf_kernelILi{d}ELi{d}ELi64ELi1ELi4ELi4ELb{d}ELb{d}ELb{d}EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .{ mode, bm, @intFromBool(f32_out), @intFromBool(cluster), @intFromBool(fuse) });
    }
    /// [mode: FP8G, MXFP8][F32][cluster][bm 16, 32, 64]
    pub const tiled = blk: {
        var out: [2][2][2][3][:0]const u8 = undefined;
        for (0..2) |m| for (0..2) |f| for (0..2) |c| for (0..3) |b| {
            out[m][f][c][b] = name(3 - m, 16 << b, f == 1, c == 1, false);
        };
        break :blk out;
    };
    pub const fused = [2][2][:0]const u8{
        .{ name(3, 64, false, false, true), name(3, 64, true, false, true) },
        .{ name(2, 64, false, false, true), name(2, 64, true, false, true) },
    };
    /// fn_qmmf_ld.cu's (tools/zig/flashnext_qmmf_ld.py): bf16 out with a row stride, [cluster][bm 16, 32, 64] and fused
    fn ldName(comptime mode: u32, comptime bm: u32, comptime cluster: bool, comptime fuse: bool) [:0]const u8 {
        return std.fmt.comptimePrint("_ZN13tf_fn_qmmf_ld11qmmf_kernelILi{d}ELi{d}ELi64ELi1ELi4ELi4ELb0ELb{d}ELb{d}EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiiii", .{ mode, bm, @intFromBool(cluster), @intFromBool(fuse) });
    }
    pub const ld_tiled = blk: {
        var out: [2][2][3][:0]const u8 = undefined;
        for (0..2) |m| for (0..2) |c| for (0..3) |b| {
            out[m][c][b] = ldName(3 - m, 16 << b, c == 1, false);
        };
        break :blk out;
    };
    pub const ld_fused = [2][:0]const u8{ ldName(3, 64, false, true), ldName(2, 64, false, true) };
};

/// fn_prompt.cu's prompt_kernel<MXFP8, 128, 128, 2, 4, 3, F32> (bf16 out, fp32 out).
pub const prompt_sym = [2][:0]const u8{
    "_ZN12tf_fn_prompt13prompt_kernelILi2ELi128ELi128ELi2ELi4ELi3ELb0EEEvPK13__nv_bfloat16PKhS5_fPviiiiii",
    "_ZN12tf_fn_prompt13prompt_kernelILi2ELi128ELi128ELi2ELi4ELi3ELb1EEEvPK13__nv_bfloat16PKhS5_fPviiiiii",
};

/// prompt.cu Tile<MXFP8, 128, 128, 2, 4, 3>::SMEM: 3 stages of x (128 rows of 128 bytes), weights (128 columns of 64) and
/// scales (2 bytes a column), on 128-byte lines.
pub const prompt_smem: u32 = 3 * (((128 * 128 + 128 * 64 + 128 * 2) + 127) / 128 * 128);

pub const Kernels = struct {
    prompt_module: cuda.Module,
    prompt: [2]cuda.Function,
    module: cuda.Module,
    tiled: [2][2][2][3]cuda.Function, // [FP8G, MXFP8][bf16, fp32][one slice, cluster][16, 32, 64]
    fused: [2][2]cuda.Function,
    /// fn_qmmf_ld: the same kernels writing bf16 rows `ldo` apart
    ld_module: cuda.Module,
    ld_tiled: [2][2][3]cuda.Function,
    ld_fused: [2]cuda.Function,
    major: c_int,

    pub fn load(ctx: *const cuda.Context) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = undefined;
        k.prompt_module = try cuda.Module.load(ctx.d, cuda.kernels.fn_prompt);
        errdefer k.prompt_module.unload();
        for (0..2) |f| {
            k.prompt[f] = try k.prompt_module.function(prompt_sym[f]);
            try k.prompt[f].allowDynamicShared(prompt_smem);
        }
        k.module = try cuda.Module.load(ctx.d, cuda.kernels.fn_qmmf);
        errdefer k.module.unload();
        for (0..2) |m| for (0..2) |f| {
            for (0..2) |c| for (0..3) |b| {
                k.tiled[m][f][c][b] = try k.module.function(sym.tiled[m][f][c][b]);
                try k.tiled[m][f][c][b].allowDynamicShared(smemOf(@as(usize, 16) << @intCast(b), m == 1));
            };
            k.fused[m][f] = try k.module.function(sym.fused[m][f]);
            try k.fused[m][f].allowDynamicShared(smemOf(64, m == 1));
        };
        k.ld_module = try cuda.Module.load(ctx.d, cuda.kernels.fn_qmmf_ld);
        errdefer k.ld_module.unload();
        for (0..2) |m| {
            for (0..2) |c| for (0..3) |b| {
                k.ld_tiled[m][c][b] = try k.ld_module.function(sym.ld_tiled[m][c][b]);
                try k.ld_tiled[m][c][b].allowDynamicShared(smemOf(@as(usize, 16) << @intCast(b), m == 1));
            };
            k.ld_fused[m] = try k.ld_module.function(sym.ld_fused[m]);
            try k.ld_fused[m].allowDynamicShared(smemOf(64, m == 1));
        }
        k.major = try ctx.attribute(.compute_capability_major);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.prompt_module.unload();
        k.ld_module.unload();
        k.module.unload();
    }
};

/// linear.py `_prompt(MXFP8, ...)`: bf16 prompt rows on fn_prompt.cu, every weight exact in bf16 and one fp32 chain over K
/// (a row's bits never depend on m or on K slices). `x` rows must start on 16-byte boundaries.
pub fn promptMatmul(k: *const Kernels, s: cuda.Stream, x: u64, x_stride: usize, l: Linear, out: u64, f32_out: bool, m: usize) !void {
    if (!l.mx8 or m < 1) return error.Invalid;
    const n: usize = l.n;
    const kk: usize = l.k;
    if (kk % 64 != 0 or l.npad % 128 != 0 or l.npad < n or x % 16 != 0 or (m > 1 and x_stride % 8 != 0)) return error.Invalid;
    const rows_t = (m + 127) / 128;
    const group = @max(1, @min(rows_t, (12 << 20) / (128 * kk * 2)));
    var a: cuda.Args = .{};
    a.add(x);
    a.add(l.w8);
    a.add(l.bs);
    a.add(@as(f32, 1.0));
    a.add(out);
    for ([_]usize{ m, n, kk, l.npad, if (m == 1) kk else x_stride, group }) |v| a.add(@as(c_int, @intCast(v)));
    try cuda.launch.launch(k.prompt[@intFromBool(f32_out)], .{
        .grid = .{ .x = @intCast(rows_t * ((n + 127) / 128)), .y = 1, .z = 1 },
        .block = .{ .x = 256 },
        .shared = prompt_smem,
    }, s, &a);
}

/// Mx8Linear.prefill / __call__: prompt rows of a folded MXFP8 face on `promptMatmul`, everything else on the lane matmul.
pub fn matmulFor(k: *const Kernels, s: cuda.Stream, prompt: bool, x: u64, x_stride: usize, l: Linear, out: u64, f32_out: bool, m: usize) !void {
    if (prompt and l.mx8 and l.fold) return promptMatmul(k, s, x, x_stride, l, out, f32_out, m);
    return matmul(k, s, x, x_stride, l, out, f32_out, m);
}

fn bucketIndex(bm: usize) usize {
    return switch (bm) {
        16 => 0,
        32 => 1,
        else => 2,
    };
}

/// linear._matmul(FP8G, ...) and qmmf_cuda: x (m, K) bf16 rows `x_stride` elements apart -> out (m, n) bf16 (fp32
/// with `f32_out`), contiguous. K slices from the shape alone (split_k), added in a cluster (sm_90 on) or, from
/// FUSED_ROWS rows, by each block in slice order: the same bits either way, a row's bits never depend on m.
pub fn matmul(k: *const Kernels, s: cuda.Stream, x: u64, x_stride: usize, l: Linear, out: u64, f32_out: bool, m: usize) !void {
    return matmulAt(k, s, x, x_stride, l, out, f32_out, m, null);
}

/// `matmul`, bf16 out, the rows `ldo` elements apart (fn_qmmf_ld: the same kernels and values, only the store
/// address differs; a projection's block-FP8 columns written straight into its wider rows).
pub fn matmulLd(k: *const Kernels, s: cuda.Stream, x: u64, x_stride: usize, l: Linear, out: u64, ldo: usize, m: usize) !void {
    if (ldo < l.n or ldo % 2 != 0 or out % 4 != 0) return error.Invalid;
    return matmulAt(k, s, x, x_stride, l, out, false, m, ldo);
}

fn matmulAt(k: *const Kernels, s: cuda.Stream, x: u64, x_stride: usize, l: Linear, out: u64, f32_out: bool, m: usize, ldo: ?usize) !void {
    if (m < 1) return error.Invalid;
    const n: usize = l.n;
    const kk: usize = l.k;
    if (kk % 64 != 0 or l.npad < n) return error.Invalid;
    const sk = splitK(n, kk);
    const bm: usize = if (sk > 1 and m >= fused_rows) 0 else bucket(m);
    const fused = bm == 0;
    const cluster = !fused and sk > 1 and sk <= 8 and k.major >= 9;
    // sm_89 and older would add slices in the reduce (part + reduce_kernel): never on GB10, not ported
    if (!fused and sk > 1 and !cluster) return error.SplitWithoutCluster;
    const tile: usize = if (fused) 64 else bm;
    const rows_t = (m + tile - 1) / tile;
    const md: usize = @intFromBool(l.mx8);
    const f = if (ldo != null) (if (fused) k.ld_fused[md] else k.ld_tiled[md][@intFromBool(cluster)][bucketIndex(bm)]) else if (fused) k.fused[md][@intFromBool(f32_out)] else k.tiled[md][@intFromBool(f32_out)][@intFromBool(cluster)][bucketIndex(bm)];
    var a: cuda.Args = .{};
    a.add(x);
    a.add(l.w8);
    a.add(l.bs);
    a.add(@as(f32, 1.0));
    a.add(out);
    a.add(@as(u64, 0));
    for ([_]usize{ m, n, kk, sk, l.npad, if (m == 1) kk else x_stride, l2Group(rows_t, tile, kk) }) |v| a.add(@as(c_int, @intCast(v)));
    if (ldo) |ld| a.add(@as(c_int, @intCast(ld)));
    const z: u32 = if (fused) 1 else @intCast(sk);
    try cuda.launch.launch(f, .{
        .grid = .{ .x = @intCast(rows_t * ((n + 63) / 64)), .y = 1, .z = z },
        .block = .{ .x = 128 },
        .shared = smemOf(tile, l.mx8),
        .cluster = if (cluster) .{ .x = 1, .y = 1, .z = @intCast(sk) } else null,
    }, s, &a);
}

// ---- host layouts (Fp8BlockLinear.from_checkpoint) ------------------------------------------------------------------

/// Fp8BlockLinear.column_scales: fp32 `scale_inv` [ceil(n/128), k/128] -> `out` [n, k/64], each (row, 64 inputs)'s
/// block scale.
pub fn columnScales(scale_inv: []const f32, n: usize, k: usize, out: []f32) !void {
    if (k % 128 != 0 or scale_inv.len != ((n + 127) / 128) * (k / 128) or out.len != n * (k / 64)) return error.BadScaleShape;
    const kb = k / 128;
    for (0..n) |r| for (0..k / 64) |j| {
        out[r * (k / 64) + j] = scale_inv[(r / 128) * kb + j / 2];
    };
}

/// The input offset within a 32-input step of fragment byte `byte` of lane `lane` (fragment_index's `kin`).
pub fn kin(lane: usize, byte: usize) usize {
    return 16 * (byte / 4) + 2 * (lane % 4) + (byte % 2) + 8 * ((byte % 4) / 2);
}

/// _fragment_order: e4m3 bytes [n, k] (rows past n zero up to npad) -> [npad/64][k/64][8][32][2][8].
pub fn fragmentOrder(codes: []const u8, n: usize, k: usize, npad: usize, out: []u8) !void {
    if (k % 64 != 0 or npad % 64 != 0 or npad < n or codes.len != n * k or out.len != npad * k) return error.BadFp8Shape;
    var i: usize = 0;
    for (0..npad / 64) |a| for (0..k / 64) |b| for (0..8) |c| for (0..32) |lane| for (0..2) |h| for (0..8) |byte| {
        const row = a * 64 + c * 8 + lane / 4;
        const col = b * 64 + h * 32 + kin(lane, byte);
        out[i] = if (row < n) codes[row * k + col] else 0;
        i += 1;
    };
}

/// from_rows' bs: fp32 column scales [n, k/64] (ones for rows past n up to npad) -> [npad/64][k/64][64].
pub fn tileScales(cols: []const f32, n: usize, k: usize, npad: usize, out: []f32) !void {
    const kg = k / 64;
    if (k % 64 != 0 or npad % 64 != 0 or npad < n or cols.len != n * kg or out.len != npad * kg) return error.BadFp8Shape;
    for (0..npad / 64) |a| for (0..kg) |g| for (0..64) |r| {
        const row = a * 64 + r;
        out[(a * kg + g) * 64 + r] = if (row < n) cols[row * kg + g] else 1.0;
    };
}

/// Mx8Linear.from_checkpoint's bs: e8m0 bytes [npad, k/32] -> [npad/64][k/64][64][2] (a tile's rows together).
pub fn tileExponents(exps: []const u8, npad: usize, k: usize, out: []u8) void {
    const kg = k / 64;
    for (0..npad / 64) |a| for (0..kg) |g| for (0..64) |r| for (0..2) |h| {
        out[((a * kg + g) * 64 + r) * 2 + h] = exps[(a * 64 + r) * (k / 32) + g * 2 + h];
    };
}

test "MXFP8 scale tiling follows Mx8Linear.from_checkpoint" {
    var e: [128 * 4]u8 = undefined;
    for (&e, 0..) |*v, i| v.* = @intCast(i % 251);
    var out: [128 * 4]u8 = undefined;
    tileExponents(&e, 128, 128, &out);
    // (tile 1, group 1, row 5, half 1) = row 69, scale column 3
    try std.testing.expectEqual(e[69 * 4 + 3], out[((1 * 2 + 1) * 64 + 5) * 2 + 1]);
    try std.testing.expectEqual(e[2], out[((0 * 2 + 1) * 64 + 0) * 2 + 0]);
}

test "split_k, buckets and shared memory follow qmm.py and qmmf.cu" {
    // the INT4-AutoRound checkpoint's block-FP8 shapes, one GPU and a TP=2 rank
    try std.testing.expectEqual(@as(usize, 1), splitK(16384, 2560));
    try std.testing.expectEqual(@as(usize, 1), splitK(13312, 2560));
    try std.testing.expectEqual(@as(usize, 8), splitK(2560, 6144));
    try std.testing.expectEqual(@as(usize, 4), splitK(2560, 2560));
    try std.testing.expectEqual(@as(usize, 2), splitK(2560, 1280));
    try std.testing.expectEqual(@as(usize, 4), splitK(2560, 3072));
    try std.testing.expectEqual(@as(usize, 1), splitK(2560, 640));
    try std.testing.expectEqual(@as(usize, 4), splitK(1280, 2560));
    try std.testing.expectEqual(@as(usize, 25600), smem(16));
    try std.testing.expectEqual(@as(usize, 33792), smem(32));
    try std.testing.expectEqual(@as(usize, 50176), smem(64));
    try std.testing.expectEqual(@as(usize, 13312), npadOf(13312));
    try std.testing.expectEqual(@as(usize, 128), npadOf(96));
    try std.testing.expectEqual(@as(usize, 64), bucket(33));
    try std.testing.expectEqual(@as(usize, 1), l2Group(1, 16, 2560));
    try std.testing.expectEqual(@as(usize, 38), l2Group(64, 64, 2560));
    try std.testing.expectEqualStrings("_ZN10tf_fn_qmmf11qmmf_kernelILi3ELi32ELi64ELi1ELi4ELi4ELb1ELb1ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", sym.tiled[0][1][1][1]);
    // MXFP8 (mode 2): a stage holds 2 scale bytes a column, not 4
    try std.testing.expectEqual(@as(usize, 25088), smemOf(16, true));
    try std.testing.expectEqual(@as(usize, 33280), smemOf(32, true));
    try std.testing.expectEqual(@as(usize, 49664), smemOf(64, true));
    try std.testing.expectEqualStrings("_ZN10tf_fn_qmmf11qmmf_kernelILi2ELi16ELi64ELi1ELi4ELi4ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", sym.tiled[1][0][0][0]);
    try std.testing.expectEqualStrings("_ZN10tf_fn_qmmf11qmmf_kernelILi3ELi64ELi64ELi1ELi4ELi4ELb0ELb0ELb1EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", sym.fused[0][0]);
}

test "fragment order is fragment_index's (k, n) of every byte" {
    // fragment_index: p = 16 (byte / 4) + 4 (lane % 4) + byte % 4; kin = 16 (p / 16) + 2 ((p % 16) / 4) + p % 2 + 8 ((p % 4) / 2)
    for (0..32) |lane| for (0..8) |byte| {
        const p = 16 * (byte / 4) + 4 * (lane % 4) + (byte % 4);
        try std.testing.expectEqual(16 * (p / 16) + 2 * ((p % 16) / 4) + (p % 2) + 8 * ((p % 4) / 2), kin(lane, byte));
    };
    // each 32-input step's lane bytes cover 32 inputs of 8 rows once (a permutation)
    for (0..4) |q| {
        var seen: [32]bool = @splat(false);
        for (0..8) |byte| {
            const v = kin(q, byte);
            try std.testing.expect(!seen[v]);
            seen[v] = true;
        }
        for (0..8) |byte| try std.testing.expectEqual(kin(q, byte), kin(q + 4 * 5, byte));
    }
    // hand cases: lane 0 byte 0 -> 0, byte 1 -> 1, byte 2 -> 8, byte 3 -> 9, byte 4 -> 16; lane 1 byte 0 -> 2
    try std.testing.expectEqual(@as(usize, 0), kin(0, 0));
    try std.testing.expectEqual(@as(usize, 1), kin(0, 1));
    try std.testing.expectEqual(@as(usize, 8), kin(0, 2));
    try std.testing.expectEqual(@as(usize, 9), kin(0, 3));
    try std.testing.expectEqual(@as(usize, 16), kin(0, 4));
    try std.testing.expectEqual(@as(usize, 2), kin(1, 0));
    try std.testing.expectEqual(@as(usize, 31), kin(7, 7)); // quad 3, byte 7: 16 + 6 + 1 + 8
}

test "fragment order and tile scales on a small matrix" {
    const gpa = std.testing.allocator;
    const n = 70;
    const k = 128;
    const npad = npadOf(n);
    const codes = try gpa.alloc(u8, n * k);
    defer gpa.free(codes);
    for (codes, 0..) |*c, i| c.* = @intCast((i * 7 + 3) % 251);
    const out = try gpa.alloc(u8, npad * k);
    defer gpa.free(out);
    try fragmentOrder(codes, n, k, npad, out);
    // byte (a, b, c, lane, h, byte) at ((((a * kg + b) * 8 + c) * 32 + lane) * 2 + h) * 8 + byte
    const at = struct {
        fn f(a: usize, b: usize, c: usize, lane: usize, h: usize, byte: usize) usize {
            return ((((a * (k / 64) + b) * 8 + c) * 32 + lane) * 2 + h) * 8 + byte;
        }
    }.f;
    try std.testing.expectEqual(codes[0], out[at(0, 0, 0, 0, 0, 0)]);
    try std.testing.expectEqual(codes[1 * k + 64 + 32 + 9], out[at(0, 1, 0, 4, 1, 3)]); // row 1, col 64 + 32 + kin(4, 3) = 9
    try std.testing.expectEqual(codes[69 * k + 0], out[at(1, 0, 0, 20, 0, 0)]); // row 64 + 5 = 69, lane 20: quad 0, kin 0
    try std.testing.expectEqual(@as(u8, 0), out[at(1, 0, 0, 24, 0, 0)]); // row 70: padding
    var inv = [_]f32{ 0.5, 2.0 }; // [ceil(70/128) = 1, 128/128 = 1] ... one block row, one block column
    var cols: [n * 2]f32 = undefined;
    try std.testing.expectError(error.BadScaleShape, columnScales(&inv, n, k, &cols));
    try columnScales(inv[0..1], n, k, &cols);
    for (cols) |v| try std.testing.expectEqual(@as(f32, 0.5), v);
    cols[69 * 2 + 1] = 3;
    var bs: [128 * 2]f32 = undefined;
    try tileScales(&cols, n, k, npad, &bs);
    try std.testing.expectEqual(@as(f32, 3), bs[(1 * 2 + 1) * 64 + 5]);
    try std.testing.expectEqual(@as(f32, 1), bs[(1 * 2 + 0) * 64 + 6]); // row 70: padding scales are one
    try std.testing.expectEqual(@as(f32, 0.5), bs[(0 * 2 + 1) * 64 + 63]);
}

test "column scales repeat 128 rows and two 64-input groups" {
    var inv = [_]f32{ 1, 2, 3, 4, 5, 6 }; // [2, 3]: n 200 -> 2 block rows, k 384 -> 3 block columns
    var cols: [200 * 6]f32 = undefined;
    try columnScales(&inv, 200, 384, &cols);
    try std.testing.expectEqual(@as(f32, 1), cols[0]);
    try std.testing.expectEqual(@as(f32, 1), cols[1]);
    try std.testing.expectEqual(@as(f32, 2), cols[2]);
    try std.testing.expectEqual(@as(f32, 3), cols[127 * 6 + 5]);
    try std.testing.expectEqual(@as(f32, 4), cols[128 * 6 + 0]);
    try std.testing.expectEqual(@as(f32, 6), cols[199 * 6 + 4]);
}

// ---- fp8-check: the port against Python's Fp8BlockLinear (tools/zig/check_fp8block.py's cases) ---------------------

fn readFile(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8, suffix: []const u8) ![]u8 {
    const path = try std.fmt.allocPrint(gpa, "{s}/{s}{s}", .{ dir, name, suffix });
    defer gpa.free(path);
    return std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 31));
}

fn hex(bytes: []const u8) [64]u8 {
    var sum: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &sum, .{});
    return std.fmt.bytesToHex(sum, .lower);
}

/// Every case of `dir` (cases.json + each case's codes, scale_inv and input rows): the host layouts' sha256 against
/// Python's, then `matmul` at every row count (bf16, and fp32 where Python wrote it) against Python's outputs, a
/// row's bits against its one-row call, and the time at 1 and 4096 rows. True when every byte agrees.
pub fn check(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8) !bool {
    const d = ctx.d;
    var k = try Kernels.load(ctx);
    defer k.deinit();
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    const text = try readFile(gpa, io, dir, "cases", ".json");
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    var ok = true;
    var e0 = try cuda.Event.init(d, true);
    defer e0.deinit();
    var e1 = try cuda.Event.init(d, true);
    defer e1.deinit();
    for (parsed.value.object.get("cases").?.array.items) |cv| {
        const c = cv.object;
        const name = c.get("name").?.string;
        const n: usize = @intCast(c.get("n").?.integer);
        const kk: usize = @intCast(c.get("k").?.integer);
        const npad = npadOf(n);
        const codes = try readFile(gpa, io, dir, name, "_codes.bin");
        defer gpa.free(codes);
        const inv_b = try readFile(gpa, io, dir, name, "_inv.bin");
        defer gpa.free(inv_b);
        const xb = try readFile(gpa, io, dir, name, "_x.bin");
        defer gpa.free(xb);
        const inv = try gpa.alloc(f32, inv_b.len / 4);
        defer gpa.free(inv);
        @memcpy(std.mem.sliceAsBytes(inv), inv_b);
        // the layouts
        const cols = try gpa.alloc(f32, n * (kk / 64));
        defer gpa.free(cols);
        try columnScales(inv, n, kk, cols);
        const bs = try gpa.alloc(f32, npad * (kk / 64));
        defer gpa.free(bs);
        try tileScales(cols, n, kk, npad, bs);
        const w8 = try gpa.alloc(u8, npad * kk);
        defer gpa.free(w8);
        try fragmentOrder(codes, n, kk, npad, w8);
        const w8_ok = std.mem.eql(u8, &hex(w8), c.get("w8").?.string);
        const bs_ok = std.mem.eql(u8, &hex(std.mem.sliceAsBytes(bs)), c.get("bs").?.string);
        ok = ok and w8_ok and bs_ok;
        var dw = try cuda.DeviceBuffer.fromHost(d, w8);
        defer dw.free();
        var dbs = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(bs));
        defer dbs.free();
        var dx = try cuda.DeviceBuffer.fromHost(d, xb);
        defer dx.free();
        const max_m = xb.len / (2 * kk);
        var dy = try cuda.DeviceBuffer.alloc(d, max_m * n * 4);
        defer dy.free();
        var dy1 = try cuda.DeviceBuffer.alloc(d, n * 4);
        defer dy1.free();
        const host = try gpa.alloc(u8, max_m * n * 4);
        defer gpa.free(host);
        const one = try gpa.alloc(u8, n * 4);
        defer gpa.free(one);
        const l: Linear = .{ .w8 = dw.ptr, .bs = dbs.ptr, .n = @intCast(n), .k = @intCast(kk), .npad = @intCast(npad) };
        var equal: usize = 0;
        var total: usize = 0;
        var inv_ok = true;
        for ([_][]const u8{ "bf16", "fp32" }) |kind| {
            const f32_out = kind[0] == 'f';
            const es: usize = if (f32_out) 4 else 2;
            var it = c.get(kind).?.object.iterator();
            while (it.next()) |kv| {
                const m = try std.fmt.parseInt(usize, kv.key_ptr.*, 10);
                try matmul(&k, stream, dx.ptr, kk, l, dy.ptr, f32_out, m);
                try stream.synchronize();
                try dy.download(0, host[0 .. m * n * es]);
                total += 1;
                if (std.mem.eql(u8, &hex(host[0 .. m * n * es]), kv.value_ptr.string)) {
                    equal += 1;
                } else {
                    ok = false;
                    std.debug.print("  {s} {s} m {d}: DIFFER\n", .{ name, kind, m });
                }
                // row invariance: rows of this call against one-row calls
                if (m == max_m or m == 300) for ([_]usize{ 0, 1, m / 2, m - 1 }) |r| {
                    try matmul(&k, stream, dx.ptr + r * kk * 2, kk, l, dy1.ptr, f32_out, 1);
                    try stream.synchronize();
                    try dy1.download(0, one[0 .. n * es]);
                    if (!std.mem.eql(u8, one[0 .. n * es], host[r * n * es ..][0 .. n * es])) {
                        inv_ok = false;
                        std.debug.print("  {s} {s} m {d} row {d}: differs from its one-row call\n", .{ name, kind, m, r });
                    }
                };
            }
        }
        ok = ok and inv_ok;
        // matmulLd (fn_qmmf_ld): every bf16 row count's rows written `ldo` apart hold the contiguous call's bytes,
        // and the gaps between them stay untouched
        {
            const ldo = n + 72;
            var dl = try cuda.DeviceBuffer.alloc(d, max_m * ldo * 2);
            defer dl.free();
            const strided = try gpa.alloc(u8, max_m * ldo * 2);
            defer gpa.free(strided);
            var ld_ok = true;
            var ld_n: usize = 0;
            var it = c.get("bf16").?.object.iterator();
            while (it.next()) |kv| {
                const m = try std.fmt.parseInt(usize, kv.key_ptr.*, 10);
                @memset(strided, 0xAB);
                try dl.upload(0, strided);
                try matmul(&k, stream, dx.ptr, kk, l, dy.ptr, false, m);
                try matmulLd(&k, stream, dx.ptr, kk, l, dl.ptr, ldo, m);
                try stream.synchronize();
                try dy.download(0, host[0 .. m * n * 2]);
                try dl.download(0, strided[0 .. m * ldo * 2]);
                ld_n += 1;
                for (0..m) |r| {
                    const row = strided[r * ldo * 2 ..][0 .. ldo * 2];
                    if (!std.mem.eql(u8, row[0 .. n * 2], host[r * n * 2 ..][0 .. n * 2]) or !std.mem.allEqual(u8, row[n * 2 ..], 0xAB)) {
                        ld_ok = false;
                        std.debug.print("  {s} bf16 m {d} row {d}: matmulLd differs from matmul (or wrote past its columns)\n", .{ name, m, r });
                        break;
                    }
                }
            }
            std.debug.print("  {s}: matmulLd (rows {d} apart) {s} matmul at {d} row counts\n", .{ name, ldo, if (ld_ok) "byte-equal to" else "DIFFERS from", ld_n });
            ok = ok and ld_ok;
        }
        // time at 1 and max_m rows (bf16 out)
        var us: [2]f64 = undefined;
        for ([_]usize{ 1, max_m }, 0..) |m, i| {
            const reps: usize = if (m == 1) 200 else 20;
            try matmul(&k, stream, dx.ptr, kk, l, dy.ptr, false, m);
            try e0.record(stream);
            for (0..reps) |_| try matmul(&k, stream, dx.ptr, kk, l, dy.ptr, false, m);
            try e1.record(stream);
            try e1.synchronize();
            us[i] = @as(f64, try cuda.Event.elapsedMs(e0, e1)) * 1000.0 / @as(f64, @floatFromInt(reps));
        }
        const wbytes: f64 = @floatFromInt(npad * kk + npad * (kk / 64) * 4);
        const flops: f64 = 2.0 * @as(f64, @floatFromInt(max_m * n * kk));
        std.debug.print("{s} {s}: n {d} k {d} split_k {d}: layouts w8 {s} bs {s}, outputs {d}/{d} equal, rows {s} one-row calls; 1 row {d:.1} us ({d:.0} GB/s weights), {d} rows {d:.0} us ({d:.1} TFLOP/s)\n", .{
            if (w8_ok and bs_ok and equal == total and inv_ok) "PASS" else "FAIL", name, n, kk, splitK(n, kk),
            if (w8_ok) "equal" else "DIFFER", if (bs_ok) "equal" else "DIFFER", equal, total, if (inv_ok) "equal" else "DIFFER",
            us[0], wbytes / us[0] / 1000.0, max_m, us[1], flops / us[1] / 1e6,
        });
    }
    return ok;
}

/// `tensorfold mx8-check`: tools/zig/check_mx8.py's cases in `dir` (real MXFP8 layers of the NVIDIA checkpoint: e4m3 codes,
/// e8m0 scales, bf16 input rows and Python's Mx8Linear outputs at several row counts). The host layouts feed qmmf
/// mode 2 and every output is compared byte for byte with Python's; the outputs are also compared with an fp64 reference
/// (rows 1 and 5), a row's bits with its one-row call, and matmulLd with matmul. True when every byte agrees.
pub fn checkMx8(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8) !bool {
    const d = ctx.d;
    var k = try Kernels.load(ctx);
    defer k.deinit();
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    const text = try readFile(gpa, io, dir, "cases", ".json");
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    var ok = true;
    var e0 = try cuda.Event.init(d, true);
    defer e0.deinit();
    var e1 = try cuda.Event.init(d, true);
    defer e1.deinit();
    for (parsed.value.object.get("cases").?.array.items) |cv| {
        const c = cv.object;
        const name = c.get("name").?.string;
        const n: usize = @intCast(c.get("n").?.integer);
        const kk: usize = @intCast(c.get("k").?.integer);
        const npad = npadOf(n);
        const codes = try readFile(gpa, io, dir, name, "_codes.bin");
        defer gpa.free(codes);
        const sc = try readFile(gpa, io, dir, name, "_scale.bin");
        defer gpa.free(sc);
        const xb = try readFile(gpa, io, dir, name, "_x.bin");
        defer gpa.free(xb);
        if (codes.len != n * kk or sc.len != n * kk / 32) return error.BadCase;
        const w8 = try gpa.alloc(u8, npad * kk);
        defer gpa.free(w8);
        try fragmentOrder(codes, n, kk, npad, w8);
        const exps = try gpa.alloc(u8, npad * (kk / 32));
        defer gpa.free(exps);
        @memset(exps, 127);
        @memcpy(exps[0..sc.len], sc);
        const bs = try gpa.alloc(u8, exps.len);
        defer gpa.free(bs);
        tileExponents(exps, npad, kk, bs);
        var dw = try cuda.DeviceBuffer.fromHost(d, w8);
        defer dw.free();
        var dbs = try cuda.DeviceBuffer.fromHost(d, bs);
        defer dbs.free();
        var dx = try cuda.DeviceBuffer.fromHost(d, xb);
        defer dx.free();
        const max_m = xb.len / (2 * kk);
        var dy = try cuda.DeviceBuffer.alloc(d, max_m * n * 2);
        defer dy.free();
        var dy1 = try cuda.DeviceBuffer.alloc(d, n * 2);
        defer dy1.free();
        const host = try gpa.alloc(u8, max_m * n * 2);
        defer gpa.free(host);
        const one = try gpa.alloc(u8, n * 2);
        defer gpa.free(one);
        const l: Linear = .{ .w8 = dw.ptr, .bs = dbs.ptr, .n = @intCast(n), .k = @intCast(kk), .npad = @intCast(npad), .mx8 = true };
        var equal: usize = 0;
        var total: usize = 0;
        var inv_ok = true;
        var ld_ok = true;
        var pequal: usize = 0;
        var ptotal: usize = 0;
        const ldo = n + 72;
        var dl = try cuda.DeviceBuffer.alloc(d, max_m * ldo * 2);
        defer dl.free();
        const strided = try gpa.alloc(u8, max_m * ldo * 2);
        defer gpa.free(strided);
        for (c.get("rows").?.array.items) |mv| {
            const m: usize = @intCast(mv.integer);
            const want = try readFileFmt(gpa, io, dir, name, m, "_y");
            defer gpa.free(want);
            try matmul(&k, stream, dx.ptr, kk, l, dy.ptr, false, m);
            try stream.synchronize();
            try dy.download(0, host[0 .. m * n * 2]);
            total += 1;
            if (std.mem.eql(u8, host[0 .. m * n * 2], want)) equal += 1 else {
                ok = false;
                std.debug.print("  {s} m {d}: DIFFERS from Python's Mx8Linear\n", .{ name, m });
            }
            if (m == max_m or m == 17) for ([_]usize{ 0, m / 2, m - 1 }) |r| {
                try matmul(&k, stream, dx.ptr + r * kk * 2, kk, l, dy1.ptr, false, 1);
                try stream.synchronize();
                try dy1.download(0, one[0 .. n * 2]);
                if (!std.mem.eql(u8, one[0 .. n * 2], host[r * n * 2 ..][0 .. n * 2])) {
                    inv_ok = false;
                    std.debug.print("  {s} m {d} row {d}: differs from its one-row call\n", .{ name, m, r });
                }
            };
            if (c.get("fold").?.bool) {
                const wantp = try readFileFmt(gpa, io, dir, name, m, "_p");
                defer gpa.free(wantp);
                const pm = try gpa.alloc(u8, m * n * 2);
                defer gpa.free(pm);
                try promptMatmul(&k, stream, dx.ptr, kk, l, dl.ptr, false, m);
                try stream.synchronize();
                try dl.download(0, pm);
                ptotal += 1;
                if (std.mem.eql(u8, pm, wantp)) pequal += 1 else {
                    ok = false;
                    std.debug.print("  {s} m {d}: promptMatmul DIFFERS from Python's Mx8Linear.prefill\n", .{ name, m });
                }
            }
            @memset(strided, 0xAB);
            try dl.upload(0, strided);
            try matmulLd(&k, stream, dx.ptr, kk, l, dl.ptr, ldo, m);
            try stream.synchronize();
            try dl.download(0, strided[0 .. m * ldo * 2]);
            for (0..m) |r| {
                const row = strided[r * ldo * 2 ..][0 .. ldo * 2];
                if (!std.mem.eql(u8, row[0 .. n * 2], host[r * n * 2 ..][0 .. n * 2]) or !std.mem.allEqual(u8, row[n * 2 ..], 0xAB)) {
                    ld_ok = false;
                    break;
                }
            }
        }
        // fp64 reference of 3 rows from the codes and exponents (the weights exact in bf16), relative to the largest value
        var worst: f64 = 0;
        {
            const m: usize = 3;
            try matmul(&k, stream, dx.ptr, kk, l, dy.ptr, false, m);
            try stream.synchronize();
            try dy.download(0, host[0 .. m * n * 2]);
            const lay = @import("cuda_layouts.zig");
            const ref = try gpa.alloc(f64, m * n);
            defer gpa.free(ref);
            var big: f64 = 0;
            for (0..m) |r| for (0..n) |o| {
                var acc: f64 = 0;
                for (0..kk) |j| {
                    const w: f64 = @as(f64, lay.e4m3ToF32(codes[o * kk + j])) * std.math.pow(f64, 2.0, @as(f64, @floatFromInt(@as(i32, sc[o * (kk / 32) + j / 32]) - 127)));
                    const xv: f64 = lay.bf16ToF32(std.mem.readInt(u16, xb[(r * kk + j) * 2 ..][0..2], .little));
                    acc += w * xv;
                }
                ref[r * n + o] = acc;
                big = @max(big, @abs(acc));
            };
            for (0..m * n) |i| {
                const y: f64 = lay.bf16ToF32(std.mem.readInt(u16, host[i * 2 ..][0..2], .little));
                worst = @max(worst, @abs(y - ref[i]) / big);
            }
        }
        const ref_ok = worst < 8e-3; // bf16 output rounding is 2^-9 of a value, 2^-8 at worst of the largest
        ok = ok and inv_ok and ld_ok and ref_ok and pequal == ptotal;
        var us: f64 = undefined;
        {
            try matmul(&k, stream, dx.ptr, kk, l, dy.ptr, false, 1);
            try e0.record(stream);
            for (0..200) |_| try matmul(&k, stream, dx.ptr, kk, l, dy.ptr, false, 1);
            try e1.record(stream);
            try e1.synchronize();
            us = @as(f64, try cuda.Event.elapsedMs(e0, e1)) * 1000.0 / 200.0;
        }
        const wbytes: f64 = @floatFromInt(npad * kk + npad * (kk / 32));
        std.debug.print("{s} {s}: n {d} k {d} split_k {d}: outputs {d}/{d} byte-equal to Python's Mx8Linear, rows {s} one-row calls, matmulLd {s}, prompt GEMM {d}/{d} byte-equal to Mx8Linear.prefill, fp64 reference worst {e:.2} of max; 1 row {d:.1} us ({d:.0} GB/s weights)\n", .{
            if (equal == total and inv_ok and ld_ok and ref_ok and pequal == ptotal) "PASS" else "FAIL", name, n, kk, splitK(n, kk),
            equal, total, if (inv_ok) "equal" else "DIFFER", if (ld_ok) "equal" else "DIFFERS", pequal, ptotal, worst, us, wbytes / us / 1000.0,
        });
    }
    return ok;
}

fn readFileFmt(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8, m: usize, kind: []const u8) ![]u8 {
    var b: [32]u8 = undefined;
    return readFile(gpa, io, dir, name, try std.fmt.bufPrint(&b, "{s}{d}.bin", .{ kind, m }));
}
