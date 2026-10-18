"""--kv-dtype fp8 in a concurrent step: attn_multi.layer with kv8.py's FP8 rows (Python's multi-stream path).

attn_multi.py's own kernels stay as they are (the Zig engine replays them for bf16 caches); the two that touch keys and
values have FP8 twins here, calling kv8's row functions with each row's caches found by table: ``_prep_multi8``
(kv8.prep_row8) and ``_chunks_multi8`` (kv8.chunk8). The pool and the merge are attn_multi's (bf16 indexer keys; no
rotation, BITS 0). A stream's rows get the bits a one-stream step gives them, so concurrent == solo holds."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from . import attention as attn_mod, kv8
from .attention import CHUNK
from .attn_multi import Step, _caches, _merge_multi, _pool_multi, _ptr, _rows_from


@triton.jit
def _prep_multi8(P, POSR, SID, CP, VP, QW, KW, IW, INV, Q, IQ, eps, N, PW: tl.constexpr, NQ: tl.constexpr,
                 NKV: tl.constexpr, HD: tl.constexpr, NI: tl.constexpr, IHD: tl.constexpr, HALF: tl.constexpr,
                 VISION: tl.constexpr, S1: tl.constexpr = 11, S2: tl.constexpr = 10):
    r = tl.program_id(0)
    s = tl.load(SID + r)
    rope, delta, length = CP, CP, 0
    if VISION:
        rope, delta = _ptr(VP, s, tl.int32), _ptr(VP + N, s, tl.int32)
        length = tl.load(VP + 2 * N + s).to(tl.int32)
    kv8.prep_row8(P, tl.load(POSR + r), r, tl.program_id(1), QW, KW, IW, INV, Q, _ptr(CP, s, tl.float8e4nv),
                  _ptr(CP + N, s, tl.float8e4nv), _ptr(CP, s, tl.float32), IQ, _ptr(CP + 4 * N, s, tl.bfloat16), eps,
                  PW, NQ, NKV, HD, NI, IHD, HALF, ROPE=rope, DELTA=delta, length=length, MODE=2 if VISION else 0,
                  S1=S1, S2=S2)


@triton.jit
def _chunks_multi8(Q, CP, POSR, SID, PO, PM, PL, IDS, NKR, N, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
                   G: tl.constexpr, CH: tl.constexpr, NCH: tl.constexpr, SCALE: tl.constexpr, IDW: tl.constexpr,
                   QSA: tl.constexpr, RATIO: tl.constexpr, TOP: tl.constexpr):
    r = tl.program_id(0)
    s = tl.load(SID + r)
    n = tl.load(POSR + r) + 1
    sparse = False
    if QSA:                                          # a row is sparse when its select would mark it (end past TOP)
        sparse = n // RATIO > TOP
        n = tl.where(sparse, tl.load(NKR + r), n)
    kv8.chunk8(Q, _ptr(CP, s, tl.float8e4nv), _ptr(CP + N, s, tl.float8e4nv), _ptr(CP, s, tl.float32), n, sparse, r,
               tl.program_id(1), tl.program_id(2), PO, PM, PL, IDS, H, HK, D, G, CH, NCH, SCALE, IDW, QSA)


def layer(layer, w, b, step: Step, mtp: bool, scale: float) -> torch.Tensor:
    """attn_multi.layer over FP8 caches: prep, pool, sparse streams' own selects, chunks, merge -> b.attn_o[:R]."""

    c, a, sc = w.cfg, layer.attn, b.attn
    n, rows, cp = step.n, step.rows, step.ptrs[step.index[layer.index]]
    heads = c.heads + c.kv_heads + c.index_heads + 1
    sections = w.cfg.mrope_section
    _prep_multi8[(rows, heads)](b.pa[:rows], step.posr, step.sid, cp, step.vision_ptrs,
                                a.q_scale, a.k_scale, a.iq_scale, w.inv_freq,
                                b.q, b.iq, c.eps, n, PW=b.pa.shape[1], NQ=c.heads, NKV=c.kv_heads, HD=c.head_dim,
                                NI=c.index_heads, IHD=c.index_dim, HALF=w.inv_freq.numel(),
                                VISION=step.vision, S1=sections[1], S2=sections[2], num_warps=2)
    top = sc.budget // sc.ratio
    if sc.qsa:
        _pool_multi[(n, step.most // sc.ratio + 2)](cp, step.vision_ptrs, step.first, step.counts,
                                                    a.ik_scale, w.inv_freq, c.eps, n,
                                                    DI=c.index_dim, HALF=w.inv_freq.numel(), RATIO=sc.ratio,
                                                    VISION=step.vision, S1=sections[1], S2=sections[2],
                                                    num_warps=1)
        for (st, a0, a1), end in zip(step.segs, step.ends):
            if end // sc.ratio > top:                # this stream has sparse rows: its own select
                _, _, pooled, pos, _ = _caches(layer, st, mtp)
                attn_mod.qsa_rows(b.iq[a0:a1], pooled, pos, _rows_from(sc, a0), a1 - a0, context=end)
    keys = max(step.ends)
    if sc.qsa:
        keys = min(keys, (top + 1) * sc.ratio - 1)
    chunks = min(sc.nch, triton.cdiv(keys, CHUNK))
    hk = c.kv_heads
    g = c.heads // hk
    _chunks_multi8[(rows, hk, chunks)](b.q, cp, step.posr, step.sid, sc.po, sc.pm, sc.pl, sc.ids, sc.nk, n,
                                       H=c.heads, HK=hk, D=c.head_dim, G=g, CH=CHUNK, NCH=sc.nch, SCALE=scale,
                                       IDW=sc.idw, QSA=sc.qsa, RATIO=sc.ratio, TOP=top, num_warps=4, num_stages=1)
    _merge_multi[(rows, hk)](sc.po, sc.pm, sc.pl, step.posr, b.attn_o, sc.nk, H=c.heads, HK=hk, D=c.head_dim, G=g,
                             CH=CHUNK, NCH=sc.nch, QSA=sc.qsa, BITS=0, RATIO=sc.ratio, TOP=top, num_warps=4)
    return b.attn_o[:rows]
