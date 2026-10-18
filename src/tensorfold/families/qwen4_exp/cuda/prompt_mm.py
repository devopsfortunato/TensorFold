"""Prompt-path matmuls whose K slices are summed inside each program: bf16.matmul's and nvfp4.matmul's bits with no
fp32 partials and no ``_reduce`` (the Zig CUDA engine's prompt chunks; work/research/R2-prefill.md W-A).

``bf16.matmul`` and ``nvfp4.matmul`` cut K into ``SK`` slices by the weight's shape (``split_k`` / ``split_for``): one
program a (rows, columns, slice) tile writes its slice's fp32 sums to a partial buffer, then ``_reduce`` adds the
slices in slice order, one fp32 add a slice. Here one program a (rows, columns) tile runs every slice itself: slice s
is exactly what ``_b16mm`` (``_fp4mm``) program ``pid_s = s`` computes (the same K blocks in the same order, the same
loads, dots and scale steps, an accumulator from zero), and the slice totals are added as ``_reduce`` adds them
(``total = p0``, then ``total = total + p_s``), then stored as ``_reduce`` stores them (fp32, or one bf16 rounding).
A row's or a column's bits never depend on its tile (each output element is the same MMA instruction sequence), so
the tiles and warps are free; the slice boundaries are not: ``SK`` stays the shape's own.

Nothing in the Python engine calls these; the Zig engine replays them from the kernel set
(tools/zig/flashnext_prompt_spec.py adds their specializations to zig/tests/cuda/flashnext/kernels.json).
"""

from __future__ import annotations

HAS_TRITON = True
try:
    import triton
    import triton.language as tl

    from .nvfp4 import _e2m1_pattern, _e4m3_value

    @triton.jit
    def _b16mm_ks(X, W, OUT, M, x_stride,
                  N: tl.constexpr, K: tl.constexpr, SK: tl.constexpr, BM: tl.constexpr,
                  BLOCK_N: tl.constexpr, BK: tl.constexpr, F32: tl.constexpr, GROUP: tl.constexpr):
        """``bf16._b16mm`` over all SK slices of one tile, the slices summed in order (``_reduce``'s adds).

        One program a (rows, columns) tile on a 1-D grid in bands of GROUP row tiles (all column tiles of a band
        before the next band), so a band's rows stay in L2 while its column tiles pass: the order of the tiles, not
        any tile's arithmetic."""

        tiles_m = tl.cdiv(M, BM)
        tiles_n: tl.constexpr = (N + BLOCK_N - 1) // BLOCK_N
        pid = tl.program_id(0)
        band = pid // (GROUP * tiles_n)
        first_m = band * GROUP
        size_m = tl.minimum(tiles_m - first_m, GROUP)
        pid_m = first_m + (pid % size_m)
        pid_n = (pid % (GROUP * tiles_n)) // size_m
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BK)
        m_ok = rm < M
        n_ok = rn < N
        KS: tl.constexpr = K // SK
        NB: tl.constexpr = KS // BK               # whole BK blocks a slice, as _b16mm
        total = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
        acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
        # block j = s * NB + i: _b16mm's program pid_s = s, step i (k0 = (pid_s * NB + i) * BK)
        for j in range(SK * NB):
            k0 = j * BK
            x = tl.load(X + rm[:, None] * x_stride + (k0 + rk)[None, :], mask=m_ok[:, None], other=0.0)
            w = tl.load(W + rn[:, None] * K + (k0 + rk)[None, :], mask=n_ok[:, None], other=0.0)
            acc = tl.dot(x, tl.trans(w), acc)
            if (j + 1) % NB == 0:                 # slice s done: _reduce's acc = p0, then acc + p_s
                if j < NB:
                    total = acc
                else:
                    total = total + acc
                acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
        out_mask = m_ok[:, None] & n_ok[None, :]
        tl.store(OUT + rm[:, None] * N + rn[None, :], total if F32 else total.to(tl.bfloat16), mask=out_mask)

    @triton.jit
    def _fp4mm_ks(X, W, S, S2, OUT, M, x_stride,
                  N: tl.constexpr, K: tl.constexpr, SK: tl.constexpr, BM: tl.constexpr,
                  SBN: tl.constexpr, BLOCK_N: tl.constexpr, GPI: tl.constexpr, F32: tl.constexpr,
                  PACKED: tl.constexpr):
        """``nvfp4._fp4mm`` over all SK slices of one tile, the slices summed in order (``_reduce``'s adds)."""

        PER: tl.constexpr = (K // 16) // SK             # quantization blocks per slice
        SUB: tl.constexpr = SBN // BLOCK_N              # programs per stored N tile
        STEPS: tl.constexpr = PER // GPI                # _fp4mm's outer steps a slice
        pid_n = tl.program_id(1)
        rm = tl.program_id(0) * BM + tl.arange(0, BM)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        r16 = tl.arange(0, 16)
        m_ok = rm < M
        n_ok = rn < N
        tile = W + (pid_n // SUB) * ((K // 64) * (32 if PACKED else 64) * SBN)
        local = (pid_n % SUB) * BLOCK_N + tl.arange(0, BLOCK_N)
        total = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
        acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
        s2 = tl.load(S2 + rn, mask=n_ok, other=1.0)
        # step t = s * STEPS + i: _fp4mm's program pid_s = s, step i (b = pid_s * PER + i * GPI + j)
        for t in range(SK * STEPS):
            for j in tl.static_range(GPI):
                b = t * GPI + j
                x = tl.load(X + rm[:, None] * x_stride + (b * 16 + r16)[None, :], mask=m_ok[:, None], other=0.0)
                if PACKED:
                    w8 = tl.load(tile + b * (8 * SBN) + (r16 // 2)[:, None] * SBN + local[None, :])
                    code = ((w8 >> ((r16 % 2) * 4)[:, None]) & 0xF).to(tl.int32)
                    wv = _e2m1_pattern(code).to(tl.bfloat16, bitcast=True)
                else:
                    wbits = tl.load(tile + b * (16 * SBN) + r16[:, None] * SBN + local[None, :])
                    wv = wbits.to(tl.bfloat16, bitcast=True)
                p = tl.dot(x, wv)
                if PACKED:
                    s = _e4m3_value(tl.load(S + b * N + rn, mask=n_ok, other=0).to(tl.int32)) * s2
                else:
                    s = tl.load(S + b * N + rn, mask=n_ok, other=0.0)
                acc += p * s[None, :]
            if (t + 1) % STEPS == 0:              # slice s done: _reduce's acc = p0, then acc + p_s
                if t < STEPS:
                    total = acc
                else:
                    total = total + acc
                acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
        out_mask = m_ok[:, None] & n_ok[None, :]
        tl.store(OUT + rm[:, None] * N + rn[None, :], total if F32 else total.to(tl.bfloat16), mask=out_mask)

    @triton.jit
    def _scores_rows(IQ, POOLED, POS0, SC, NB, ROWS, HI: tl.constexpr, DI: tl.constexpr, RATIO: tl.constexpr,
                     TOP: tl.constexpr, BB: tl.constexpr, RT: tl.constexpr):
        """``attention._scores`` for RT rows a program: the BB pooled keys of block tile j are loaded once (for
        the tile's last row, the longest) and each row's scores are ``_scores``' own expression on that same
        [BB, DI] tile: the per-row blocks past a row's end are loaded here but never stored, as there."""

        r0 = tl.program_id(0) * RT
        j = tl.program_id(1)
        pos0 = tl.load(POS0)
        last = tl.minimum(r0 + RT, ROWS) - 1
        reach = (pos0 + last + 1) // RATIO
        if reach > TOP and j * BB < reach:
            b = j * BB + tl.arange(0, BB)
            d = tl.arange(0, DI)
            kv = tl.load(POOLED + b[:, None].to(tl.int64) * DI + d[None, :], mask=(b < reach)[:, None], other=0.0)
            for i in tl.static_range(RT):
                r = r0 + i
                complete = (pos0 + r + 1) // RATIO
                if r < ROWS and complete > TOP and j * BB < complete:
                    ok = b < complete
                    k = tl.where(ok[:, None], kv, 0.0).to(tl.float32)
                    total = tl.zeros((BB,), dtype=tl.float32)
                    for h in tl.static_range(HI):
                        q = tl.load(IQ + (r * HI + h) * DI + d).to(tl.float32)
                        total = total + tl.maximum(tl.sum(k * q[None, :], axis=1), 0.0)
                    tl.store(SC + r * NB + b, total / tl.sqrt(DI * 1.0), mask=ok)

    from .glue import _bsig

    @triton.jit
    def _hc_up_mix(ACT, W, NORMED, MIXED, M, D: tl.constexpr, S: tl.constexpr, K: tl.constexpr,
                   BM: tl.constexpr, BD: tl.constexpr, BK: tl.constexpr):
        """The hyper-connection read-out's up projection and mix in one pass (the up rows never stored): each
        stream's up = bf16(act @ W[s D + d].T) as ``bf16._b16mm`` computes it (one K slice: BK steps in order from
        a zero accumulator, one bf16 rounding), then ``glue._hc_mix``'s expression in stream order:
        mixed = bf16((sum_s bf16(bsig(up_s) * normed_s)) / S). The 32-group sums are not made (no reader then)."""

        rm = tl.program_id(0) * BM + tl.arange(0, BM)
        rd = tl.program_id(1) * BD + tl.arange(0, BD)
        rk = tl.arange(0, BK)
        m_ok = rm < M
        total = tl.zeros((BM, BD), dtype=tl.float32)
        for s in tl.static_range(S):
            acc = tl.zeros((BM, BD), dtype=tl.float32)
            for i in range(K // BK):
                x = tl.load(ACT + rm[:, None] * K + (i * BK + rk)[None, :], mask=m_ok[:, None], other=0.0)
                w = tl.load(W + (s * D + rd)[:, None] * K + (i * BK + rk)[None, :])
                acc = tl.dot(x, tl.trans(w), acc)
            u = acc.to(tl.bfloat16).to(tl.float32)
            n = tl.load(NORMED + rm[:, None] * (S * D) + s * D + rd[None, :], mask=m_ok[:, None], other=0.0).to(tl.float32)
            total += (_bsig(u) * n).to(tl.bfloat16).to(tl.float32)
        m = (total / S).to(tl.bfloat16)
        tl.store(MIXED + rm[:, None] * D + rd[None, :], m, mask=m_ok[:, None])

    @triton.jit
    def _hc_wb_norm(H, HOUT, PSS, BR, INJ, Y, WTS, RS, SCALE, NORMED, eps,
                    D: tl.constexpr, S: tl.constexpr, MODE: tl.constexpr, TOPK: tl.constexpr, SLOTS: tl.constexpr,
                    BLOCK: tl.constexpr, WORLD: tl.constexpr):
        """``glue._hc_writeback`` and ``glue._hc_normed`` for one row a program (the chunk-invariant prompt path).
        Phase 1 is _hc_writeback's body over the row's D // BLOCK chunks (the same BLOCK and warps: its fp32 squared
        sums per chunk and stream are the same bits, stored to PSS as there); after a barrier phase 2 is _hc_normed's:
        rinv_s from the stream's chunk sums in chunk order, normed = bf16(h * rinv_s * scale). The 32-group sums of
        normed are not made (only the 4-bit MTP draft matrices read them)."""

        r = tl.program_id(0)
        NC: tl.constexpr = D // BLOCK
        for c in tl.static_range(NC):
            d = c * BLOCK + tl.arange(0, BLOCK)
            if MODE == 1:
                branch = tl.load(BR + r * D + d).to(tl.float32)
            elif MODE == 3 or MODE == 4:
                acc = tl.load(BR + r * D + d)
                for k in tl.static_range(1, WORLD):
                    acc = acc + tl.load(BR + k * RS + r * D + d)
                branch = acc.to(tl.bfloat16).to(tl.float32)
            elif MODE == 2:
                acc = tl.zeros((BLOCK,), dtype=tl.float32)
                for k in tl.static_range(TOPK + 1):
                    wk = tl.load(WTS + r * SLOTS + k)
                    yk = tl.load(Y + (r * SLOTS + k) * D + d).to(tl.float32)
                    acc = acc + yk * wk
                branch = acc.to(tl.bfloat16).to(tl.float32)
            for s in tl.static_range(S):
                hv = tl.load(H + r * (S * D) + s * D + d).to(tl.float32)
                if MODE != 0:
                    inj = tl.load(INJ + r * S + s).to(tl.float32)
                    hv = (hv + (branch * inj).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
                    tl.store(HOUT + r * (S * D) + s * D + d, hv.to(tl.bfloat16))
                tl.store(PSS + (r * NC + c) * S + s, tl.sum(hv * hv, axis=0))
        tl.debug_barrier()
        for s in tl.static_range(S):
            total = 0.0
            for c in tl.static_range(NC):
                total += tl.load(PSS + (r * NC + c) * S + s)
            rinv = 1.0 / tl.sqrt(total / D + eps)
            for c in tl.static_range(NC):
                d = c * BLOCK + tl.arange(0, BLOCK)
                hv = tl.load(HOUT + r * (S * D) + s * D + d).to(tl.float32)
                w = tl.load(SCALE + s * D + d).to(tl.float32)
                tl.store(NORMED + r * (S * D) + s * D + d, (hv * rinv * w).to(tl.bfloat16))
except ModuleNotFoundError:
    HAS_TRITON = False
