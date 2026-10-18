"""Flash Next's torch ops against the deployed torch kernels by raw bytes, at the NVFP4 checkpoint's shapes.

Compiles zig/kernels/cuda/torch_ops' fn_*.cu (ours for Flash Next) and the Nemotron operators Flash Next reuses
(pointwise, movement, argmax, topk) with ops_build's pinned nvcc and flags, then runs each op beside the torch code
the Python engine runs (TensorFold 0.6.5 qwen4_exp/cuda: nvfp4_moe.MoE4.shared_act and its slot copies,
decode.Engine.sample_draft, decode.sample_mapped, cuda/sampling.sample_rows, forward.candidates) and compares bytes.
Run on the GPU in nvcr.io/nvidia/pytorch:26.07-py3 from tools/zig:

    python check_flashnext_ops.py --source ../../zig/kernels/cuda/torch_ops --out <dir> --config <snapshot>/config.json
"""

import argparse
import ctypes
import json
from pathlib import Path
import time

import torch

from ops_build import compile_operators, digest
from ops_compare import RawCells
from ops_ffi import P, U, bind, pointer

CAND = 32          # state.CAND: a rank's gathered candidates per row
MARGIN = 8         # exact_sampling.MARGIN


def shared_act(g: torch.Tensor, width: int) -> torch.Tensor:
    """nvfp4_moe.MoE4.shared_act's torch ops on the gate|up rows."""

    gate, up = g[:, :width].to(torch.float32), g[:, width:].to(torch.float32)
    return ((gate / (1.0 + torch.exp(-gate))).to(torch.bfloat16).to(torch.float32) * up).to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--lse", action="store_true", help="also fn_logsumexp.cu against torch.logsumexp")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    config = json.loads(a.config.read_text())
    text = config.get("text_config", config)
    hidden, vocab = int(text["hidden_size"]), int(text["vocab_size"])
    width, top_k = int(text["shared_expert_intermediate_size"]), int(text["num_experts_per_tok"])
    names = ["fn_ops.cu", "pointwise.cu", "movement.cu", "argmax.cu", "topk.cu"]
    if a.lse:
        names.append("fn_logsumexp.cu")
    receipt = {"oracle": {"torch": torch.__version__, "git_version": torch.version.git_version,
                          "cuda": torch.version.cuda},
               "config_sha256": digest(a.config), "cells": [],
               "provenance": "TensorFold 0.6.5 qwen4_exp/cuda sources and CUDA SDK APIs; no code copied",
               "model_loaded": False,
               "qualification": "synthetic values at the checkpoint's shapes; real-weight gate pending"}
    lib, compiled = compile_operators(a.source, a.out, names)
    receipt.update(compiled)
    torch.cuda.set_device(0)
    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError("This packet is qualified for sm_121 only")
    check = RawCells(a.out, receipt)
    handle = P(torch.cuda.current_stream().cuda_stream)
    swiglu = bind(lib, "tf_fn_shared_swiglu", [P, P, U, U, U, P])
    fill = bind(lib, "tf_fn_fill_u64", [P, U, U, P])
    pick = bind(lib, "tf_fn_draft_pick", [P, P, P, P, P])
    pack = bind(lib, "tf_fn_draft_pack", [P, P, P, P, U, P])
    cands = bind(lib, "tf_fn_candidates", [P, P, P, ctypes.c_int64, P, P, U, U, P])
    cast_up = bind(lib, "tf_bf16_to_f32", [P, P, U, P])
    strided = bind(lib, "tf_strided_copy", [P, P, U, U, U, U, U, U, U, P])
    argmax = bind(lib, "tf_argmax_rows_i32", [P, P, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int32, P])
    topk_bytes = lib.tf_topk_f32_unsorted_scratch_bytes
    topk_bytes.argtypes, topk_bytes.restype = [U, U], U
    topk = bind(lib, "tf_topk_f32_unsorted", [P, P, P, U, U, U, U, U, P, U, P])
    torch.manual_seed(20261007)

    def run_topk(x: torch.Tensor, k: int):
        rows, columns = x.shape
        values = torch.empty((rows, k), device="cuda", dtype=torch.float32)
        indices = torch.empty((rows, k), device="cuda", dtype=torch.int64)
        scratch = torch.empty(topk_bytes(rows, columns), device="cuda", dtype=torch.uint8)
        status = topk(pointer(x), pointer(values), pointer(indices), rows, columns, k, x.stride(0) * 4, 4,
                      pointer(scratch), scratch.numel(), handle)
        return values, indices, status

    # the shared expert's SwiGLU written into its slot of buf.act [R, top_k + 1, NI] (TP=1 NI 640, a TP=2 rank 320)
    for ni in (width, width // 2):
        for rows in (1, 7, 16, 19, 2048):
            g = (torch.randn((rows, 2 * ni), device="cuda") * 3).to(torch.bfloat16)
            act = torch.randn((rows, top_k + 1, ni), device="cuda").to(torch.bfloat16)
            want = act.clone()
            want[:, top_k] = shared_act(g, ni)
            got = act.clone()
            status = swiglu(pointer(g), pointer(got[:, top_k]), rows, ni, got.stride(0), handle)
            check(f"shared-swiglu/ni{ni}/r{rows}", got, want, status)
    # every bf16 gate encoding (NaN, infinities, subnormals, signed zeros) against random up values
    ni = 2048
    gate = torch.arange(65536, dtype=torch.int32, device="cuda").to(torch.int16).view(torch.bfloat16).view(32, ni)
    g = torch.cat([gate, (torch.randn((32, ni), device="cuda") * 3).to(torch.bfloat16)], dim=1).contiguous()
    got = torch.empty((32, ni), device="cuda", dtype=torch.bfloat16)
    check("shared-swiglu/all-bf16-gates", got, shared_act(g, ni), swiglu(pointer(g), pointer(got), 32, ni, ni, handle))

    # the shared expert's down rows into slot top_k of buf.y: fp32 (the MTP head's buffers) and bf16 (the main ones)
    for dtype in (torch.float32, torch.bfloat16):
        for rows in (1, 7, 16, 2048):
            src = torch.randn((rows, hidden), device="cuda").to(dtype)
            y = torch.randn((rows, top_k + 1, hidden), device="cuda").to(dtype)
            want = y.clone()
            want[:, top_k] = src
            got = y.clone()
            dst = got[:, top_k]
            size = src.element_size()
            # cuda_torch_ops.slotCopy's geometry: one block a row (outer = rows, middle = 1)
            status = strided(pointer(src), pointer(dst), rows, 1, hidden * size, src.stride(0) * size,
                             hidden * size, dst.stride(0) * size, hidden * size, handle)
            check(f"slot-copy/{str(dtype)[6:]}/r{rows}", got, want, status)

    # Tensor.fill_ of the conv-state pointer (int64)
    word = torch.zeros((1,), device="cuda", dtype=torch.int64)
    want = torch.empty_like(word).fill_(0x7F1234567890)
    check("fill-u64", word, want, fill(pointer(word), 0x7F1234567890, 1, handle))

    # logits.float(): the head rows (TP=1 vocab, a TP=2 rank's half), the draft head's row
    heads = [vocab, vocab // 2]
    drafts = [79591, 39796, 39795]
    for columns in heads + drafts:
        for rows in (1, 16):
            x = (torch.randn((rows, columns), device="cuda") * 4).to(torch.bfloat16)
            out = torch.empty((rows, columns), device="cuda", dtype=torch.float32)
            check(f"logits-float/{columns}/r{rows}", out, x.float(), cast_up(pointer(x), pointer(out), x.numel(), handle))

    # greedy: argmax over the bf16 head rows (sample_rows: logits.argmax(dim=-1)), with ties planted
    for columns in heads + drafts:
        for rows in (1, 7, 16):
            x = (torch.randn((rows, columns), device="cuda") * 4).to(torch.bfloat16)
            x[:, columns // 3] = 30.0
            x[:, columns // 2] = 30.0
            if rows > 1:
                x[1:, :] = x[:1, :]
                x[rows - 1, 5] = float("nan")
                x[rows - 1, 7] = float("nan")
            ids = torch.empty((rows,), device="cuda", dtype=torch.int32)
            status = argmax(pointer(x), pointer(ids), rows, columns, x.stride(0), 0, handle)
            check(f"argmax/{columns}/r{rows}", ids, x.argmax(dim=-1).to(torch.int32), status)

    # top-k: sample_rows / sample_mapped / sample_draft (top_k + MARGIN), candidates (CAND), at top_k 20 and 40
    for columns in heads + drafts:
        for rows in (1, 7, 16):
            for k in (20 + MARGIN, CAND, 40 + MARGIN):
                x = ((torch.randn((rows, columns), device="cuda") * 4).to(torch.bfloat16)).float()
                x[:, 100:140] = x[:, 100:101]                      # ties across the cut
                values, indices, status = run_topk(x, k)
                want_v, want_i = torch.topk(x, k, dim=-1, sorted=False)
                check(f"topk/{columns}/r{rows}/k{k}/values", values, want_v, status)
                check(f"topk/{columns}/r{rows}/k{k}/indices", indices, want_i)
                same = all(set(zip(indices[r].tolist(), values[r].tolist())) ==
                           set(zip(want_i[r].tolist(), want_v[r].tolist())) for r in range(rows))
                receipt.setdefault("topk_sets", []).append({"id": f"topk/{columns}/r{rows}/k{k}", "same_set": same})

    # Engine.sample_draft (one rank): greedy cat([max, lse, col.float()]), sampled cat([vals, lse, idx.float()])
    for columns in drafts:
        x = (torch.randn((1, columns), device="cuda") * 4).to(torch.bfloat16)
        x[0, 77] = x[0, 4000] = 25.0
        row = x[:1].float()
        lse = torch.logsumexp(row, dim=-1, keepdim=True)
        top, col = row.max(dim=-1, keepdim=True)
        want = torch.cat([top, lse, col.float()], dim=1)
        ids = torch.empty((1,), device="cuda", dtype=torch.int32)
        status = argmax(pointer(x), pointer(ids), 1, columns, x.stride(0), 0, handle)
        got = torch.empty((1, 3), device="cuda", dtype=torch.float32)
        status = status or pick(pointer(x), pointer(ids), pointer(lse), pointer(got), handle)
        check(f"draft-pick/{columns}", got, want, status)
        for k in (20 + MARGIN, 40 + MARGIN):
            vals, idx = torch.topk(row, k, dim=-1, sorted=False)
            want = torch.cat([vals, lse, idx.float()], dim=1)
            got = torch.empty((1, 2 * k + 1), device="cuda", dtype=torch.float32)
            check(f"draft-pack/{columns}/k{k}", got, want, pack(pointer(vals), pointer(idx), pointer(lse), pointer(got),
                                                               k, handle))

    # forward.candidates (two ranks): the main head's rank half (ids + offset), the draft head's (id_map[idx])
    draft_ids = torch.randperm(vocab, device="cuda")[:drafts[1]].sort().values.to(torch.int64)
    for columns, id_map, offset in ((vocab // 2, None, vocab // 2), (drafts[1], draft_ids, vocab // 2)):
        for rows in (1, 7, 16):
            lf = (torch.randn((rows, columns), device="cuda") * 4).to(torch.bfloat16).float()
            vals, idx = torch.topk(lf, CAND, dim=-1, sorted=False)
            ids = (id_map[idx] if id_map is not None else idx + offset).to(torch.int32)
            want = torch.empty((rows, 2 * CAND + 1), device="cuda", dtype=torch.float32)
            want[:, :CAND] = vals
            want[:, CAND:2 * CAND] = ids.view(torch.float32)
            want[:, 2 * CAND:] = torch.logsumexp(lf, dim=-1, keepdim=True)
            lse = want[:, 2 * CAND].contiguous()
            got = torch.empty_like(want)
            status = cands(pointer(vals), pointer(idx), pointer(id_map), offset, pointer(lse), pointer(got), rows, CAND,
                           handle)
            check(f"candidates/{columns}/{'map' if id_map is not None else 'offset'}/r{rows}", got, want, status)

    if "fn_logsumexp.cu" in names:
        from check_flashnext_lse import check_lse

        check_lse(lib, check)
    check.finish(started)


if __name__ == "__main__":
    main()
