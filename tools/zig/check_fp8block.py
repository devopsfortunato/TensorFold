#!/usr/bin/env python3
"""Python's block-FP8 linear (cuda/nvfp4/linear.py Fp8BlockLinear, qmmf FP8G) on synthetic tensors at the
INT4-AutoRound checkpoint's shapes: writes each case's e4m3 codes, fp32 weight_scale_inv and bf16 input rows, and the
sha256 of Python's layouts (w8, bs) and of its outputs for every row count, for ``tensorfold fp8-check DIR`` (the Zig
engine's port, cuda_fp8.zig) to compare byte for byte.

Run on a GPU in nvcr.io/nvidia/pytorch:26.07-py3 with the tree at /tensorfold (PYTHONPATH=/tensorfold/src):
    python -B tools/zig/check_fp8block.py OUT_DIR
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

ROWS = [1, 2, 3, 7, 8, 16, 17, 32, 33, 64, 65, 100, 255, 256, 300, 1024, 4096]
# (name, n, k, fp32 out too): one GPU, then a TP=2 rank's shares
SHAPES = [
    ("gdn_qkvz", 16384, 2560, False),
    ("attn_qkv", 13312, 2560, False),
    ("out_proj", 2560, 6144, True),
    ("shared_gu", 2560, 2560, False),
    ("shared_down", 2560, 1280, True),
    ("tp2_gdn_qkvz", 8192, 2560, False),
    ("tp2_attn_qkv", 6656, 2560, False),
    ("tp2_out_proj", 2560, 3072, True),
    ("tp2_shared_gu", 1280, 2560, False),
    ("tp2_shared_down", 2560, 640, False),
]


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    from tensorfold.cuda.kernels import qmm
    from tensorfold.cuda.nvfp4 import linear
    from tensorfold.cuda.nvfp4.linear import FP8G, Fp8BlockLinear

    cases = []
    for i, (name, n, k, f32) in enumerate(SHAPES):
        g = torch.Generator().manual_seed(1000 + i)
        codes = torch.randint(0, 254, (n, k), generator=g, dtype=torch.uint8)
        codes[codes >= 0x7F] += 1                       # no NaN codes (0x7F, 0xFF)
        inv = torch.exp(torch.empty((-(-n // 128), k // 128)).uniform_(-9.0, -4.0, generator=g)).float()
        x = torch.randn((max(ROWS), k), generator=g).to(torch.bfloat16)
        (out / f"{name}_codes.bin").write_bytes(codes.numpy().tobytes())
        (out / f"{name}_inv.bin").write_bytes(inv.numpy().tobytes())
        (out / f"{name}_x.bin").write_bytes(x.view(torch.int16).numpy().tobytes())
        lin = Fp8BlockLinear.from_checkpoint(codes.cuda().view(torch.float8_e4m3fn), inv.cuda())
        xd = x.cuda()
        case = {"name": name, "n": n, "k": k, "npad": lin.npad, "w8": sha(lin.w8), "bs": sha(lin.bs), "bf16": {},
                "fp32": {}}
        for m in ROWS:
            case["bf16"][str(m)] = sha(lin(xd[:m]))
            if f32:                                     # _matmul with an fp32 output (the f32 flag of qmmf)
                sk = qmm.split_k(n, k)
                bm = 0 if sk > 1 and m >= linear.FUSED_ROWS else qmm.bucket(m)
                y = torch.empty((m, n), dtype=torch.float32, device="cuda")
                linear._ext().qmmf(xd[:m], lin.w8, lin.bs, 1.0, y, None, FP8G, n, sk, lin.npad, bm, True)
                case["fp32"][str(m)] = sha(y)
        torch.cuda.synchronize()
        cases.append(case)
        print(f"{name}: n {n} k {k} npad {lin.npad} split_k {qmm.split_k(n, k)}", flush=True)
    (out / "cases.json").write_text(json.dumps({"rows": ROWS, "cases": cases}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
