"""Flash Next's top_k-off draws (cuda/sampling.nucleus_rows) against the deployed torch, and fixtures for the Zig port.

1. fn_ops.cu's tf_fn_nucleus_mass against torch's fixed-point mass (floor(exp(f64(f32(logit)) / t - top) * 2**40),
   int64) and its row sums, raw bytes, at the checkpoint's vocabulary, a TP=2 rank's half and the draft head's.
2. Fixtures of Python's own _shares / _draw on one rank and on two (each rank's shard, the global top), with the
   whole-shard fallback when a nucleus runs past the candidates: zig/src/families/flashnext/fixtures_cuda_nucleus.json,
   which cuda_nucleus.zig's test replays on the host.
Run on the GPU in nvcr.io/nvidia/pytorch:26.07-py3 from tools/zig with TensorFold's src on PYTHONPATH:

    python check_flashnext_nucleus.py --source ../../zig/kernels/cuda/torch_ops --out <dir> --fixture <json>
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch

from ops_build import compile_operators
from ops_ffi import P, U, bind, pointer

from tensorfold.cuda.sampling import MASS, _draw, _shares, one_rank
from tensorfold.engine.exact_sampling import Sampling


def bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def torch_mass(logits: torch.Tensor, t: float, top: torch.Tensor):
    scaled = logits.float().double() / max(float(t), 1e-6)
    mass = torch.floor(torch.exp(scaled - top[:, None]) * MASS).to(torch.int64)
    return scaled, mass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fixture", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    lib, _ = compile_operators(a.source, a.out, ["fn_ops.cu"])
    torch.cuda.set_device(0)
    stream = P(torch.cuda.current_stream().cuda_stream)
    mass_fn = bind(lib, "tf_fn_nucleus_mass", [P, U, U, U, __import__("ctypes").c_double, P, P, P, P])
    torch.manual_seed(20261007)
    bad = 0
    cells = 0
    # 1. the mass kernel, raw bytes
    for vocab in (248320, 124160, 79591):
        for rows in (1, 7):
            for t in (0.6, 1.0, 1.3):
                for spread in (1.0, 6.0):
                    logits = (torch.randn((rows, vocab), device="cuda") * spread).to(torch.bfloat16)
                    scaled = logits.float().double() / max(t, 1e-6)
                    top = scaled.max(dim=-1).values.contiguous()
                    _, want = torch_mass(logits, t, top)
                    got = torch.empty_like(want)
                    sums = torch.empty((rows,), dtype=torch.int64, device="cuda")
                    status = mass_fn(pointer(logits), vocab, rows, vocab, max(t, 1e-6), pointer(top), pointer(got),
                                     pointer(sums), stream)
                    torch.cuda.synchronize()
                    ok = status == 0 and torch.equal(got, want) and torch.equal(sums, want.sum(dim=-1))
                    cells += 1
                    bad += not ok
                    print(f"{'EQUAL' if ok else 'DIFFER'} mass v{vocab} r{rows} t{t} s{spread}", flush=True)
    # 2. Python's draws on one rank and two, for the Zig port
    cases = []
    samplings = [Sampling(7, 1.0, 0, 0.95, 0.0), Sampling(11, 0.7, 0, 0.5, 0.0), Sampling(13, 1.0, 0, 1.0, 0.05),
                 Sampling(17, 1.3, 0, 0.9, 0.02), Sampling(19, 2.5, 0, 0.99, 0.0), Sampling(23, 4.0, 0, 1.0, 0.001)]
    # small shapes (48 candidates of 256-token shards) so the fixture stays small; _draw's rules do not depend on
    # NUCLEUS, the count _shares is handed
    vocab, count0 = 512, 48
    for world in (1, 2):
        for spread in (1.0, 8.0):
            for s in samplings:
                rows = 3
                logits = (torch.randn((rows, vocab), device="cuda") * spread).to(torch.bfloat16)
                positions = [101, 102, 205]
                shard = vocab // world
                scaled_all = logits.float().double() / max(float(s.temperature), 1e-6)
                top = scaled_all.max(dim=-1).values

                def shares(count):
                    parts = []
                    for r in range(world):
                        lg = logits[:, r * shard:(r + 1) * shard]
                        scaled, mass = torch_mass(lg, s.temperature, top)
                        parts.append(_shares(one_rank, scaled, mass, count, r * shard, None))
                    return tuple(np.concatenate([p[i] for p in parts], axis=0) for i in range(5))

                def pack(got):
                    vals, ids, mass, sums, widths = got
                    return {"count": int(vals.shape[2]),
                            "vals": [[[bits(v) for v in row] for row in rk] for rk in vals],
                            "ids": ids.astype(np.int64).tolist(), "mass": mass.astype(np.int64).tolist(),
                            "sums": sums.astype(np.int64).tolist(), "widths": widths.astype(np.int64).tolist()}

                got = shares(count0)
                drawn = _draw(got, positions, s)
                case = {"world": world, "rows": rows, "positions": positions,
                        "sampling": {"seed": s.seed, "temperature": s.temperature, "top_k": s.top_k,
                                     "top_p": s.top_p, "min_p": s.min_p},
                        "first": pack(got), "fallback": None}
                if drawn is None:
                    count = int(got[4].max())
                    fb = shares(count)
                    drawn = _draw(fb, positions, s)
                    case["fallback"] = pack(fb)
                case["drawn"] = [[int(tok), bits(share)] for tok, share in drawn]
                cases.append(case)
    a.fixture.write_text(json.dumps({"mass": MASS, "nucleus": count0, "cases": cases}) + "\n")
    falls = sum(c["fallback"] is not None for c in cases)
    print(f"{'PASS' if bad == 0 else 'FAIL'} nucleus mass: {cells - bad} of {cells} byte-equal; "
          f"{len(cases)} draw fixtures ({falls} with the whole-shard fallback) -> {a.fixture}", flush=True)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
