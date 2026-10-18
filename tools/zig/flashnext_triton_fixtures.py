#!/usr/bin/env python3
"""Flash Next's Triton launches as its Python wrappers make them, for the Zig wrappers' host tests.

Each Python wrapper (qwen4_exp/cuda/{glue,bf16,nvfp4,attention,forward,exl3_mm}.py, cuda/moe.py) runs on CPU tensors
of the NVFP4 checkpoint's shapes (TP=1 and the TP=2 rank shapes of work/PLAN.md) with its Triton kernels replaced by
recorders: nothing launches, and each launch's kernel, grid, runtime arguments (name, Triton type, int and float
values) and constexprs are written to zig/src/families/flashnext/fixtures_cuda_triton.json. cuda_triton.zig's tests
make the same calls and compare. Run from the repo root with the source on PYTHONPATH (CPU only):

    PYTHONPATH=src python tools/zig/flashnext_triton_fixtures.py --out zig/src/families/flashnext/fixtures_cuda_triton.json
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import torch

from tensorfold.cuda import moe as moe_mod
from tensorfold.families.qwen4_exp.cuda import attention as attn_mod
from tensorfold.families.qwen4_exp.cuda import bf16, exl3_mm, forward, glue, nvfp4

TYPES = {torch.bfloat16: "*bf16", torch.float32: "*fp32", torch.float16: "*fp16", torch.int32: "*i32",
         torch.int64: "*i64", torch.uint16: "*u16", torch.uint8: "*u8", torch.int8: "*i8"}
LOG: list[dict] = []


def f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


def const(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return {"int": int(v)}
    if isinstance(v, int):
        return {"int": v}
    if isinstance(v, float):
        return {"f32": f32_bits(v)}
    raise TypeError(f"constexpr {v!r}")


class Recorder:
    """Stands in for a JITFunction: ``kernel[grid](*args, **kw)`` appends the launch to LOG."""

    def __init__(self, jit) -> None:
        self.jit = jit

    def __getitem__(self, grid):
        def call(*args, **kw):
            names = self.jit.arg_names
            bound = dict(zip(names, args))
            bound.update({k: v for k, v in kw.items() if k in names})
            options = {k: v for k, v in kw.items() if k not in names}
            runtime, consts = [], {}
            for p in self.jit.params:
                if p.name not in bound:
                    if p.is_constexpr and p.has_default:      # a constexpr left at its default
                        consts[p.name] = const(p.default)
                    continue
                v = bound[p.name]
                if p.is_constexpr or v is None:
                    consts[p.name] = const(v)
                elif isinstance(v, torch.Tensor):
                    runtime.append({"name": p.name, "type": TYPES[v.dtype]})
                elif isinstance(v, bool):
                    raise TypeError(f"{p.name}: a runtime bool")
                elif isinstance(v, int):
                    runtime.append({"name": p.name, "type": "i32", "int": v})
                elif isinstance(v, float):
                    runtime.append({"name": p.name, "type": "fp32", "f32": f32_bits(v)})
                else:
                    raise TypeError(f"{p.name}: {type(v)}")
            g = list(grid) + [1] * (3 - len(grid))
            LOG.append({"fn": self.jit.fn.__name__, "grid": [int(x) for x in g], "args": runtime, "consts": consts,
                        "num_warps": options.get("num_warps"), "num_stages": options.get("num_stages")})

        return call


def patch() -> None:
    for mod, names in ((glue, ("_hc_writeback", "_hc_normed", "_hc_act", "_hc_mix", "_rmsnorm", "_attn_prep",
                               "_attn_gate", "_ple_embed_bf16", "_ple_gate", "_ple_conv", "_add_streams",
                               "_moe_partial")),
                       (bf16, ("_b16mm", "_reduce")), (nvfp4, ("_fp4mm", "_reduce")), (exl3_mm, ("_embed",)),
                       (attn_mod, ("_chunks", "_merge", "_pool", "_scores", "_select", "_select_tiles")),
                       (forward, ("_shift_windows",)), (moe_mod, ("_router", "_topk_rows"))):
        for n in names:
            setattr(mod, n, Recorder(getattr(mod, n)))


def e(shape, dtype=torch.bfloat16):
    """A tensor of ``shape`` without its storage (the wrappers read shapes, strides and dtypes only)."""

    return torch.empty((1,) * len(shape), dtype=dtype).expand(*shape)


def z(shape, dtype=torch.bfloat16):
    return torch.empty(shape, dtype=dtype)


# the checkpoint's dims (config.json text_config) and a TP=2 rank's (work/PLAN.md "TP=2 design")
D, S, LOW, HD, NI, IHD, HALF, E, TOPK, EPS = 2560, 4, 320, 256, 4, 128, 32, 512, 10, 1e-6
W = S * D
# rows: decode windows, the capture fncap1's prompt pieces (18/19, 29/30, 205/206, 973/974) and full chunks
ROWS = (1, 2, 3, 4, 5, 6, 7, 16, 18, 19, 29, 30, 116, 973, 974, 2048)
RANKS = {1: dict(heads=24, kv=2, mw=640, vocab=248320, nk=16, nv=48),
         2: dict(heads=12, kv=1, mw=320, vocab=124160, nk=8, nv=24)}


def case(cases: dict, name: str, fn) -> None:
    LOG.clear()
    fn()
    cases[name] = list(LOG)


def build() -> dict:
    patch()
    cases: dict = {}
    for r in ROWS:
        case(cases, f"embed/r{r}/s4", lambda: exl3_mm.embed(z((r,), torch.int32), e((248320, D)), D, 4, z((r, W))))
        case(cases, f"embed/r{r}/s1", lambda: exl3_mm.embed(z((r,), torch.int32), e((248320, D)), D, 1, z((r, D))))
        h, pss, inj = z((r, W)), z((r, D // 256, S), torch.float32), z((r, S))
        case(cases, f"hc_writeback/r{r}/m0", lambda: glue.hc_writeback(h, h, pss, S, 0))
        case(cases, f"hc_writeback/r{r}/m1", lambda: glue.hc_writeback(h, h, pss, S, 1, branch=z((r, D)), inject=inj))
        for name, ydt in (("bf16", torch.bfloat16), ("f32", torch.float32)):
            case(cases, f"hc_writeback/r{r}/m2/{name}", lambda: glue.hc_writeback(
                h, h, pss, S, 2, inject=inj, y=z((r, TOPK + 1, D), ydt), wts=z((r, TOPK + 1), torch.float32)))
        case(cases, f"hc_writeback/r{r}/m3", lambda: glue.hc_writeback(
            h, h, pss, S, 3, branch=z((2 * r * D,), torch.float32).view(2, r, D), inject=inj))
        case(cases, f"hc_normed/r{r}", lambda: glue.hc_normed(h, pss, z((W,), torch.float32), z((r, W)),
                                                              z((r, W // 32), torch.float32), S, EPS))
        for ndn, has in ((LOW + S, True), (LOW, False)):
            case(cases, f"hc_act/r{r}/n{ndn}", lambda: glue.hc_act(
                z((r, ndn), torch.float32), z((r, LOW)), z((r, LOW // 32), torch.float32), z((r, S)) if has else None,
                S, LOW))
        case(cases, f"hc_mix/r{r}", lambda: glue.hc_mix(z((r, W)), z((r, W)), z((r, D)),
                                                        z((r, D // 32), torch.float32), S))
        case(cases, f"rmsnorm/r{r}/d{D}", lambda: glue.rmsnorm(z((r, D)), z((D,), torch.float32), EPS,
                                                               out=z((r, D)), xs=z((r, D // 32), torch.float32)))
        case(cases, f"rmsnorm/r{r}/d{W}", lambda: glue.rmsnorm(z((r, W)), z((W,), torch.float32), EPS,
                                                               out=z((r, W)), xs=z((r, W // 32), torch.float32)))
        case(cases, f"add_streams/r{r}", lambda: glue.add_streams(z((r, D)), z((r * S, D)), z((r, W)), S))
        case(cases, f"ple_embed_bf16/r{r}", lambda: glue.ple_embed_bf16(
            r, z((r * 16, 160)), 16, 160, z((r, D)), z((r, D // 32), torch.float32), scale=1.0))
        # TP=2: a rank gathers its 8 of the 16 n-gram heads (work/PLAN.md), the [R, 1280] halves all-gathered
        case(cases, f"ple_embed_bf16/r{r}/tp2", lambda: glue.ple_embed_bf16(
            r, z((r * 8, 160)), 8, 160, z((r, D // 2)), z((r, D // 64), torch.float32), scale=1.0))
        case(cases, f"ple_gate/r{r}", lambda: glue.ple_gate(
            z((r, W)), z((r, D)), z((r, W)), z((W,), torch.float32), z((W,), torch.float32), z((r, W)),
            z((r, S), torch.float32), EPS, S))
        case(cases, f"ple_conv/r{r}", lambda: glue.ple_conv(
            z((r, W)), z((r, S), torch.float32), z((W,), torch.float32), z((9, W)), z((W, 4)), z((r, W)), z((r, W)),
            z((r, W)), EPS, S, 3))
        for f32 in (False, True):
            case(cases, f"moe_partial/r{r}/{'f32' if f32 else 'bf16'}", lambda: glue.moe_partial(
                z((r, TOPK + 1, D), torch.float32 if f32 else torch.bfloat16), z((r, TOPK + 1), torch.float32),
                z((r, D), torch.float32), r))
        case(cases, f"moe/r{r}", lambda: (moe_mod.router(z((r, D)), e((E + 1, D)), z((r, E + 1), torch.float32)),
                                          moe_mod.select_rows(z((r, E + 1), torch.float32), Buf(r), TOPK, E)))
        for world, rk in RANKS.items():
            heads, kv = rk["heads"], rk["kv"]
            pw = heads * 2 * HD + 2 * kv * HD + (NI + 1) * IHD
            case(cases, f"attn_prep/r{r}/tp{world}", lambda: glue.attn_prep(
                z((r, pw)), z((1,), torch.int32), z((HD,), torch.float32), z((HD,), torch.float32),
                z((IHD,), torch.float32), z((HALF,), torch.float32), z((r, heads, HD)), e((4096, kv, HD)),
                e((4096, kv, HD)), z((r, NI, IHD)), e((4096, IHD)), EPS, q_heads=heads, kv_heads=kv, head_dim=HD,
                index_heads=NI, index_dim=IHD, ks=z((1,), torch.float16), vs=z((1,), torch.float16)))
            case(cases, f"attn_gate/r{r}/tp{world}", lambda: glue.attn_gate(
                z((r, heads, HD)), z((r, pw)), z((r, heads * HD)), z((r, heads * HD // 32), torch.float32),
                q_heads=heads, head_dim=HD))
            mw = rk["mw"]
            case(cases, f"fp4/gu/r{r}/tp{world}", lambda: nvfp4.matmul(z((r, D)), fp4(2 * mw, D)))
            act = z((r, TOPK + 1, mw))
            for f32 in (False, True):
                case(cases, f"fp4/down/r{r}/tp{world}/{'f32' if f32 else 'bf16'}", lambda: nvfp4.matmul(
                    act[:, TOPK], fp4(D, mw), f32=f32))
            conv = 2 * rk["nk"] * 128 + rk["nv"] * 128
            proj = conv + rk["nv"] * 128 + 2 * rk["nv"]
            mats = {"hc_down": (LOW + S, W, True), "mix_down": (LOW, W, True), "hc_up": (W, LOW, False),
                    "gdn_proj": (proj, D, False), "gdn_out": (D, rk["nv"] * 128, world > 1),
                    "attn_proj": (pw, D, False), "o_proj": (D, heads * HD, world > 1), "ple_key": (W, D, False),
                    "ple_value": (D, D, False), "fc": (D, D, False), "head": (rk["vocab"], D, False)}
            for mat, (n, k, f32) in mats.items():
                if world > 1 and mat in ("hc_down", "mix_down", "hc_up", "ple_key", "ple_value", "fc"):
                    continue                                   # replicated: the TP=1 launches
                if mat == "head" and r > 16:
                    continue                                   # a prompt pass heads its ends only (ENDS = 16)
                case(cases, f"b16/{mat}/r{r}/tp{world}", lambda: bf16.matmul(
                    z((r, k)), bf16.B16(e((n, k)), n, k), out=z((r, n), torch.float32 if f32 else torch.bfloat16),
                    f32=f32))
    for r in (1, 2, 3, 4, 5, 6, 7, 18, 973, 2048):             # fc_hidden: the MTP's streams, n * S rows
        case(cases, f"b16/fc_h/r{r * S}", lambda: bf16.matmul(z((r * S, D)), bf16.B16(e((D, D)), D, D),
                                                              out=z((r * S, D))))
    attention_cases(cases)
    shift_cases(cases)
    return cases


class Buf:
    def __init__(self, rows: int) -> None:
        self.pick = z((rows, TOPK + 1), torch.int32)
        self.wts = z((rows, TOPK + 1), torch.float32)


def fp4(n: int, k: int):
    """A bf16 matrix as the shared expert's FP4 pattern table (``fp4_from_bf16``'s form), storage elided."""

    return nvfp4.FP4(e((n // 64, k // 64, 64, 64), torch.uint16), e((k // 16, n), torch.float32), n, k,
                     scale2=z((n,), torch.float32))


PROMPTS = ((0, 2048), (4096, 19), (139264, 2048), (2048, 974), (2048, 973), (0, 30), (0, 29), (0, 18),
           (2048, 206), (2048, 205))


def attention_cases(cases: dict) -> None:
    for world, rk in RANKS.items():
        heads, kv = rk["heads"], rk["kv"]
        scale = HD ** -0.5
        for capacity in (1024, 262144, 262151, 1048576):
            for rows, pos, ctx in [(r, p, c) for r in (1, 2, 3, 4, 5, 6, 7, 16) for p in (0, 100, 3000, 140000)
                                   for c in (None, 8192, 16384)]:
                    if pos + rows > capacity or (ctx is not None and (ctx < pos + rows or ctx > capacity)):
                        continue
                    sc = attn_mod.AttnScratch(rows, heads, HD, capacity, "cpu")
                    keys = pos + rows if ctx is None else ctx           # a graph's bucket bounds the launches
                    q, kc = z((rows, heads, HD)), e((capacity, kv, HD))
                    pos0, ks = z((1,), torch.int32), z((1,), torch.float16)
                    tag = f"tp{world}/c{capacity}/r{rows}/p{pos}/x{ctx or 0}"

                    def decode():
                        if sc.qsa:
                            attn_mod.qsa_select(z((rows, NI, IHD)), e((capacity, IHD)), e((-(-capacity // 4), IHD)),
                                                pos0, z((IHD,), torch.float32), z((HALF,), torch.float32), EPS, sc,
                                                rows, context=keys)
                        attn_mod.attention(q, kc, kc, pos0, sc, rows, scale, context=keys, ks=ks, vs=ks)

                    case(cases, f"attention/{tag}", decode)
            # a prompt chunk: pool the chunk, then blocks of ATT_ROWS rows with their ends as the context
            for start, n in PROMPTS:
                if start + n > capacity:
                    continue
                sc = attn_mod.AttnScratch(256, heads, HD, capacity, "cpu")
                pos0, ks = z((1,), torch.int32), z((1,), torch.float16)

                def prefill():
                    if sc.qsa:
                        attn_mod.qsa_pool(e((capacity, IHD)), e((-(-capacity // 4), IHD)), pos0,
                                          z((IHD,), torch.float32), z((HALF,), torch.float32), EPS, sc, n)
                    for r0 in range(0, n, 256):
                        m = min(256, n - r0)
                        ends = start + r0 + m
                        if sc.qsa:
                            attn_mod.qsa_rows(z((m, NI, IHD)), e((-(-capacity // 4), IHD)), pos0, sc, m, context=ends)
                        attn_mod.attention(z((m, heads, HD)), e((capacity, kv, HD)), e((capacity, kv, HD)), pos0, sc,
                                           m, scale, out=z((m, heads, HD)), context=ends, ks=ks, vs=ks)

                case(cases, f"attention_prefill/tp{world}/c{capacity}/s{start}/n{n}", prefill)


def shift_cases(cases: dict) -> None:
    for world, rk in RANKS.items():
        conv = 2 * rk["nk"] * 128 + rk["nv"] * 128
        proj = conv + rk["nv"] * 128 + 2 * rk["nv"]
        for rows, at, r, keep in ((8, 0, 1, 1), (8, 0, 7, 3), (16, 0, 16, 16), (16, 0, 5, 1)):
            buf = z((36, rows, proj))
            case(cases, f"shift/conv/tp{world}/b{rows}/r{r}/k{keep}", lambda: forward.shift_windows(
                z((36, 3, conv)), buf[:, at:at + r], keep, conv))
        for n in (2048, 974, 30, 19, 18, 1):
            buf = z((1, 2048, proj))
            case(cases, f"shift/prefill/tp{world}/n{n}", lambda: forward.shift_windows(
                z((36, 3, conv))[5:6], buf[0:1, 0:n], n, conv))
    for rows, r, keep in ((8, 1, 1), (8, 7, 4), (2048, 2048, 2048), (2048, 974, 974), (2048, 30, 30),
                          (2048, 19, 19), (2048, 18, 18)):
        nrow = z((rows, W))
        tail = z((9, W))
        case(cases, f"shift/ple/b{rows}/r{r}/k{keep}", lambda: forward.shift_windows(
            tail[None], nrow[None, 0:r], keep, tail.shape[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    cases = build()
    a.out.write_text(json.dumps({"generator": "tools/zig/flashnext_triton_fixtures.py", "cases": cases},
                                indent=None, separators=(",", ":")) + "\n")
    print(f"{len(cases)} cases, {sum(len(v) for v in cases.values())} launches -> {a.out}")


if __name__ == "__main__":
    main()
