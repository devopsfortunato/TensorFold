"""--kv-dtype fp8: Flash Next's attention caches as FP8 e4m3 rows, one power-of-two scale a (position, KV head) row.

The format is the owner's GLM recipe's (MiaAI-Lab, patch 0038-glm-kv-fp8, glm5_next/cuda/kv8.py), adapted to Flash
Next's per-head GQA caches. Lossy (replies differ from bf16's), never inexact: a row's bytes are a function of its
bf16 row alone (one kernel, ``_attn_prep8``, writes prompt chunks' and decode windows' rows), and the reader takes the
codes to bf16 (exact: every e4m3 value is a bf16 value) and folds each row's scale into its fp32 products (exact for a
power of two), so a row's attention depends on its query, its keys and the cache only: drafted == serial, concurrent ==
solo and resumed == fresh hold as with bf16. The values are those of the bf16 kernels on the dequantized rows up to
fp32 summation order (Triton lays converted tiles out for the tensor cores its own way).

Layout, per position t and KV head h (row i = t * HK + h):
  keys   uint8 [capacity, HK, D + 16]: D e4m3 codes, then s_k (fp32), s_v (fp32), 8 zero bytes;
  values uint8 [capacity, HK, D]:      D e4m3 codes (their scale is in the key row's trailer).
The KV head's ``_attn_prep8`` program writes both rows and both scales of a position, so nothing else writes a trailer.
A scale is 2^(ceil(log2 amax) - SHIFT): the row's largest value quantizes into (128, 256], under e4m3's 448, so nothing
saturates; a floating-point format's relative precision does not depend on the scale, so a per-row power of two loses
nothing to a finer scale but the values it pushes below e4m3's normal range, under 2^-14 of the row's largest.

The indexer's raw keys and pooled block keys stay bf16 (attention._pool, _scores and the prompt scorers unchanged).
These kernels live apart from attention.py and glue.py so the bf16 kernels' sources, Triton hashes and cubins stay as
they were."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from . import attention as attn_mod, glue
from .attention import CHUNK, _merge
from .image_rows import rope_axis

PAD = 16             # bytes after a key row's codes: s_k, s_v (fp32), 8 zero bytes
SHIFT = 8            # a row's scale: 2^(ceil(log2 amax) - SHIFT), so its codes stay within +-256


def key_bytes(head_dim: int) -> int:
    return head_dim + PAD


def row_bytes(kv_heads: int, head_dim: int) -> int:
    """Bytes a position of one layer's keys and values take."""
    return kv_heads * (2 * head_dim + PAD)


# -- the format in torch (tests, and the definition the kernels must equal) -------------------------------------------
def scales(amax: torch.Tensor) -> torch.Tensor:
    """fp32 amax -> the fp32 power-of-two scale (``scale_of``, bit for bit)."""
    bits = amax.float().contiguous().view(torch.int32)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return ((e - SHIFT).clamp(1, 254) << 23).view(torch.float32)


def _codes(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 rows (of bf16 values) [..., D] -> e4m3 codes as uint8 [..., D] and fp32 scales [...]."""
    s = scales(x.abs().amax(dim=-1))
    inv = ((254 - (s.view(torch.int32) >> 23)) << 23).view(torch.float32)
    return (x * inv[..., None]).to(torch.float8_e4m3fn).view(torch.uint8), s


def quantize(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 keys and values [N, HK, D] -> the cache's key rows [N, HK, D + PAD] and value rows [N, HK, D] (uint8)."""
    kq, sk = _codes(k.to(torch.bfloat16).float())
    vq, sv = _codes(v.to(torch.bfloat16).float())
    n, hk, d = k.shape
    keys = torch.zeros((n, hk, d + PAD), dtype=torch.uint8, device=k.device)
    keys[..., :d] = kq
    keys[..., d:d + 8] = torch.stack((sk, sv), dim=-1).contiguous().view(torch.uint8).reshape(n, hk, 8)
    return keys, vq.contiguous()


def dequantize(keys: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A cache's key and value rows -> fp32 keys and values [N, HK, D] (exact: e4m3 codes times a power of two)."""
    d = values.shape[-1]
    s = keys[..., d:d + 8].contiguous().view(torch.float32)          # [N, HK, 2]
    k = keys[..., :d].contiguous().view(torch.float8_e4m3fn).float() * s[..., :1]
    v = values.contiguous().view(torch.float8_e4m3fn).float() * s[..., 1:]
    return k, v


# -- the format in kernels ------------------------------------------------------------------------------------------
@triton.jit
def scale_of(amax):
    """fp32 amax -> 2^(ceil(log2 amax) - SHIFT) and its reciprocal, both exact powers of two (biased exponent held
    within 1 .. 254: a zero row gets 2^-126 and zeros)."""
    bits = amax.to(tl.int32, bitcast=True)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    e = tl.minimum(tl.maximum(e - 8, 1), 254)
    return (e << 23).to(tl.float32, bitcast=True), ((254 - e) << 23).to(tl.float32, bitcast=True)


@triton.jit
def store_kv(KC, VC, KS, row, k, v, HD: tl.constexpr):
    """Key and value row ``row`` (int64: position * HK + head) <- k, v fp32 [HD] (bf16 values): codes and scales."""
    d = tl.arange(0, HD)
    sk, ik = scale_of(tl.max(tl.abs(k), 0))
    sv, iv = scale_of(tl.max(tl.abs(v), 0))
    tl.store(KC + row * (HD + 16) + d, (k * ik).to(tl.float8e4nv))
    tl.store(VC + row * HD + d, (v * iv).to(tl.float8e4nv))
    tl.store(KS + row * ((HD + 16) // 4) + HD // 4, sk)
    tl.store(KS + row * ((HD + 16) // 4) + HD // 4 + 1, sv)


@triton.jit
def _attn_prep8(P, POS0, QW, KW, IW, INV, Q, KC, VC, KS, IQ, IKC, ROPE, DELTA, length, eps,
                PW: tl.constexpr, NQ: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr, NI: tl.constexpr,
                IHD: tl.constexpr, HALF: tl.constexpr, MODE: tl.constexpr = 0,
                S1: tl.constexpr = 11, S2: tl.constexpr = 10):
    """glue._attn_prep with FP8 keys and values: queries, indexer queries and raw indexer keys as it writes them."""

    r = tl.program_id(0)
    prep_row8(P, tl.load(POS0) + r, r, tl.program_id(1), QW, KW, IW, INV, Q, KC, VC, KS, IQ, IKC, eps, PW, NQ, NKV,
              HD, NI, IHD, HALF, ROPE, DELTA, length, MODE, S1, S2)


@triton.jit
def prep_row8(P, pos, r, head, QW, KW, IW, INV, Q, KC, VC, KS, IQ, IKC, eps, PW: tl.constexpr, NQ: tl.constexpr,
              NKV: tl.constexpr, HD: tl.constexpr, NI: tl.constexpr, IHD: tl.constexpr, HALF: tl.constexpr,
              ROPE=None, DELTA=None, length=0, MODE: tl.constexpr = 0, S1: tl.constexpr = 11, S2: tl.constexpr = 10):
    """glue._prep_row's head ``head`` of row r at position ``pos`` (its arithmetic as written there), the KV head's
    key (after norm and rope, rounded to bf16) and value (bf16 from the projection) stored by ``store_kv``."""

    d = tl.arange(0, HD)
    if head < NQ + NKV + NI:
        width = tl.where(head >= NQ + NKV, IHD, HD)
        live = d < width
        if head < NQ:
            src = r * PW + head * 2 * HD
        elif head < NQ + NKV:
            src = r * PW + NQ * 2 * HD + (head - NQ) * HD
        else:
            src = r * PW + NQ * 2 * HD + 2 * NKV * HD + (head - NQ - NKV) * IHD
        x = tl.load(P + src + d, mask=live, other=0.0).to(tl.float32)
        rinv = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / width + eps)
        if head < NQ:
            w = tl.load(QW + d).to(tl.float32)
        elif head < NQ + NKV:
            w = tl.load(KW + d).to(tl.float32)
        else:
            w = tl.load(IW + d, mask=live, other=0.0).to(tl.float32)
        xn = (x * rinv * w).to(tl.bfloat16).to(tl.float32)
        partner = tl.where(d < HALF, d + HALF, tl.where(d < 2 * HALF, d - HALF, d))
        xp = tl.load(P + src + partner, mask=live, other=0.0).to(tl.float32)
        if head < NQ:
            wp = tl.load(QW + partner).to(tl.float32)
        elif head < NQ + NKV:
            wp = tl.load(KW + partner).to(tl.float32)
        else:
            wp = tl.load(IW + partner, mask=live, other=0.0).to(tl.float32)
        xpn = (xp * rinv * wp).to(tl.bfloat16).to(tl.float32)
        i = tl.where(d < HALF, d, tl.where(d < 2 * HALF, d - HALF, 0))
        axis = rope_axis(pos, ROPE, DELTA, length, i, MODE, S1, S2)
        ang = axis.to(tl.float32) * tl.load(INV + i)
        cos = tl.cos(ang)
        sin = tl.sin(ang)
        rot = tl.where(d < HALF, xn * cos - xpn * sin, tl.where(d < 2 * HALF, xpn * sin + xn * cos, xn))
        out = rot.to(tl.bfloat16)
        if head < NQ:
            tl.store(Q + (r * NQ + head) * HD + d, out)
        elif head < NQ + NKV:
            v = tl.load(P + r * PW + NQ * 2 * HD + NKV * HD + (head - NQ) * HD + d).to(tl.float32)
            store_kv(KC, VC, KS, pos.to(tl.int64) * NKV + head - NQ, out.to(tl.float32), v, HD)
        else:
            tl.store(IQ + (r * NI + head - NQ - NKV) * IHD + d, out, mask=live)
    else:
        live = d < IHD
        raw = tl.load(P + r * PW + NQ * 2 * HD + 2 * NKV * HD + NI * IHD + d, mask=live, other=0.0)
        tl.store(IKC + pos.to(tl.int64) * IHD + d, raw, mask=live)


@triton.jit
def _tile8(q, k, v, sk, sv, m, l, o, valid, scale: tl.constexpr):
    """attention._tile on FP8 rows: k, v the codes as bf16, each key's s_k on its dots and s_v on its probabilities
    (powers of two: the products are the dequantized rows' exactly)."""
    scores = tl.dot(q, tl.trans(k)).to(tl.float32) * sk[None, :]
    scores = scores * scale
    scores = tl.where(valid[None, :], scores, float("-inf"))
    tile_m = tl.max(scores, 1)
    active = tile_m != float("-inf")
    next_m = tl.where(active, tl.maximum(m, tile_m), m)
    alpha = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
    p = tl.where(valid[None, :] & active[:, None], tl.exp(scores - next_m[:, None]), 0.0)
    o = o * alpha[:, None] + tl.dot((p * sv[None, :]).to(tl.bfloat16), v)
    l = l * alpha + tl.sum(p, 1)
    return next_m, l, o


@triton.jit
def _chunks8(Q, KC, VC, KS, POS0, PO, PM, PL, IDS, NKR, SPR,
             H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, G: tl.constexpr, CH: tl.constexpr,
             NCH: tl.constexpr, SCALE: tl.constexpr, IDW: tl.constexpr, QSA: tl.constexpr):
    r = tl.program_id(0)
    hk = tl.program_id(1)
    c = tl.program_id(2)
    n = tl.load(POS0) + r + 1
    sparse = False
    if QSA:
        sparse = tl.load(SPR + r) != 0
        n = tl.where(sparse, tl.load(NKR + r), n)
    chunk8(Q, KC, VC, KS, n, sparse, r, hk, c, PO, PM, PL, IDS, H, HK, D, G, CH, NCH, SCALE, IDW, QSA)


@triton.jit
def chunk8(Q, KC, VC, KS, n, sparse, r, hk, c, PO, PM, PL, IDS,
           H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, G: tl.constexpr, CH: tl.constexpr,
           NCH: tl.constexpr, SCALE: tl.constexpr, IDW: tl.constexpr, QSA: tl.constexpr):
    """attention._chunk on FP8 rows: row r's keys in chunk c of its ``n`` (a sparse row's through IDS)."""

    start = c * CH
    if start < n:                       # chunks past a row's keys write nothing: the merge never reads them
        gg = tl.arange(0, 16)
        d = tl.arange(0, D)
        q = tl.load(Q + (r * H + hk * G + gg[:, None]) * D + d[None, :], mask=gg[:, None] < G, other=0.0)
        m = tl.full((16,), float("-inf"), tl.float32)
        l = tl.zeros((16,), tl.float32)
        o = tl.zeros((16, D), tl.float32)
        tiles = tl.minimum(n - start, CH)
        for t in range(0, tl.cdiv(tiles, 64)):
            ki = start + t * 64 + tl.arange(0, 64)
            valid = ki < n
            if QSA:
                if sparse:
                    ki = tl.load(IDS + r * IDW + ki, mask=valid, other=0)
            row = ki.to(tl.int64) * HK + hk
            kk = tl.load(KC + row[:, None] * (D + 16) + d[None, :], mask=valid[:, None], other=0.0).to(tl.bfloat16)
            vv = tl.load(VC + row[:, None] * D + d[None, :], mask=valid[:, None], other=0.0).to(tl.bfloat16)
            sk = tl.load(KS + row * ((D + 16) // 4) + D // 4, mask=valid, other=1.0)
            sv = tl.load(KS + row * ((D + 16) // 4) + D // 4 + 1, mask=valid, other=1.0)
            m, l, o = _tile8(q, kk, vv, sk, sv, m, l, o, valid, SCALE)
        base = (r * NCH + c) * H + hk * G + gg
        tl.store(PO + base[:, None] * D + d[None, :], o, mask=gg[:, None] < G)
        tl.store(PM + base, m, mask=gg < G)
        tl.store(PL + base, l, mask=gg < G)


# -- wrappers -------------------------------------------------------------------------------------------------------
def is_fp8(kc: torch.Tensor, vc: torch.Tensor, bits: int = 0) -> bool:
    """Whether key and value tensors are an FP8 cache's (uint8 key rows D + PAD wide beside D-wide value rows)."""
    return (bits == 0 and kc.dtype == torch.uint8 and vc.dtype == torch.uint8
            and kc.shape[-1] == vc.shape[-1] + PAD)


def views(kc: torch.Tensor, vc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """An FP8 cache's (key codes, value codes, key-row fp32 view holding both scales) as the kernels take them."""
    if not is_fp8(kc, vc) or not kc.is_contiguous() or not vc.is_contiguous():
        raise ValueError("an FP8 KV cache: contiguous uint8 key rows [N, HK, D + 16] and value rows [N, HK, D]")
    return kc.view(torch.float8_e4m3fn), vc.view(torch.float8_e4m3fn), kc.view(torch.float32)


def attn_prep(p: torch.Tensor, pos0: torch.Tensor, q_scale, k_scale, i_scale, inv_freq, q, kc, vc, iq, ikc,
              eps: float, *, q_heads: int, kv_heads: int, head_dim: int, index_heads: int, index_dim: int,
              ks: torch.Tensor | None = None, vs: torch.Tensor | None = None, bits: int = 0,
              rope: torch.Tensor | None = None, delta: torch.Tensor | None = None, length: int = 0,
              sections: tuple[int, int, int] = (11, 11, 10)) -> None:
    """glue.attn_prep's call, into an FP8 cache through ``_attn_prep8`` (the same grid, warps and modes), any other
    cache through glue.attn_prep itself."""

    if not is_fp8(kc, vc, bits):
        return glue.attn_prep(p, pos0, q_scale, k_scale, i_scale, inv_freq, q, kc, vc, iq, ikc, eps,
                              q_heads=q_heads, kv_heads=kv_heads, head_dim=head_dim, index_heads=index_heads,
                              index_dim=index_dim, ks=ks, vs=vs, bits=bits, rope=rope, delta=delta, length=length,
                              sections=sections)
    rows, pw = p.shape
    kq, vq, sc = views(kc, vc)
    mode = 2 if rope is not None else 1 if delta is not None else 0
    _attn_prep8[(rows, q_heads + kv_heads + index_heads + 1)](
        p, pos0, q_scale, k_scale, i_scale, inv_freq, q, kq, vq, sc, iq, ikc,
        rope if rope is not None else pos0, delta if delta is not None else pos0, length, eps, PW=pw, NQ=q_heads,
        NKV=kv_heads, HD=head_dim, NI=index_heads, IHD=index_dim, HALF=inv_freq.numel(), MODE=mode,
        S1=sections[1], S2=sections[2], num_warps=2)


def attention(q: torch.Tensor, kc: torch.Tensor, vc: torch.Tensor, pos0: torch.Tensor, scratch, rows: int,
              scale: float, out: torch.Tensor | None = None, *, context: int | None = None,
              ks: torch.Tensor | None = None, vs: torch.Tensor | None = None, bits: int = 0) -> torch.Tensor:
    """attention.attention's call: over an FP8 cache ``_chunks8``, then ``_merge`` as bf16 caches merge (BITS 0, no
    rotation); any other cache through attention.attention itself."""

    if not is_fp8(kc, vc, bits):
        return attn_mod.attention(q, kc, vc, pos0, scratch, rows, scale, out, context=context, ks=ks, vs=vs, bits=bits)
    kq, vq, sc = views(kc, vc)
    _, h, d = q.shape
    hk = kq.shape[1]
    g = h // hk
    if g > 16:
        raise ValueError(f"this attention kernel tiles 16 query heads per KV head (G={g} does not fit)")
    nch = scratch.nch
    keys = nch * CHUNK if context is None else context
    if scratch.qsa:
        keys = min(keys, (scratch.budget // scratch.ratio + 1) * scratch.ratio - 1)
    chunks = min(nch, triton.cdiv(keys, CHUNK))
    out = scratch.out if out is None else out
    _chunks8[(rows, hk, chunks)](q, kq, vq, sc, pos0, scratch.po, scratch.pm, scratch.pl, scratch.ids, scratch.nk,
                                 scratch.sparse, H=h, HK=hk, D=d, G=g, CH=CHUNK, NCH=nch, SCALE=scale,
                                 IDW=scratch.idw, QSA=scratch.qsa, num_warps=4, num_stages=1)
    _merge[(rows, hk)](scratch.po, scratch.pm, scratch.pl, pos0, out, scratch.nk, scratch.sparse, H=h,
                       HK=hk, D=d, G=g, CH=CHUNK, NCH=nch, QSA=scratch.qsa, BITS=0, num_warps=4)
    return out


def multi_layer(layer, w, b, step, mtp: bool, scale: float) -> torch.Tensor:
    """attn_multi.layer's call: a concurrent step over FP8 caches through kv8_multi's kernels, any other through
    attn_multi.layer itself."""

    from . import attn_multi

    st0 = step.segs[0][0]
    if not (st0.mtp_kc if mtp else st0.kc[0]).fp8:
        return attn_multi.layer(layer, w, b, step, mtp, scale)
    from . import kv8_multi

    return kv8_multi.layer(layer, w, b, step, mtp, scale)
