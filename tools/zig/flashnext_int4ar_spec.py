#!/usr/bin/env python3
"""The Triton specializations the INT4-AutoRound checkpoint (azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound)
needs beyond the NVFP4 kernel set (zig/tests/cuda/flashnext/kernels.json, aot/p1-ks1), and the merge of both sets.

  spec:  kernels.json -> kernels_int4ar.json, only the new entries, each cloned from an entry of the same function:
         top-5 routing (``_topk_rows``, ``_hc_writeback`` mode 2, ``_moe_partial`` at TOPK 5 / SLOTS 6) and the bf16
         parts of the block-FP8 projection stacks that stay on ``bf16.matmul`` (GDN in_proj_b|in_proj_a: 96 rows at
         one GPU, 48 a TP=2 rank; the indexer's index_qk_proj: 640 rows, replicated), K 2560, in every row bucket and
         int form the wrappers launch (``_b16mm`` and the prompt path's ``_b16mm_ks``). Their K-slice sums take
         ``_reduce`` SK 8 F32 False, already in the set (nvfp4._reduce: the same source as bf16._reduce, same name).
  merge: <base>/aot.json + cubins and <new>/aot.json + cubins -> <out> (copies; a hash in both is kept once)

Build the new entries with ``flashnext_aot.py build --spec zig/tests/cuda/flashnext/kernels_int4ar.json`` in the
PyTorch image with this tree at /tensorfold (the spec's source_root), as kernels.json is built."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from flashnext_aot import _key, _reorder  # noqa: E402

TOPK, SLOTS = 5, 6
# (N, K, tp, why): the bf16 parts beside block-FP8 rows in one projection stack
B16 = [(96, 2560, [1], "GDN in_proj_b|in_proj_a, one GPU (48 + 48 rows)"),
       (48, 2560, [2], "GDN in_proj_b|in_proj_a, a TP=2 rank (24 + 24 rows)"),
       (640, 2560, [1, 2], "the indexer's index_qk_proj (4 x 128 q + 128 k rows), replicated")]
TEMPLATE = (2560, 2560)            # a split (SK > 1) bf16 K=2560 family with every bucket and int form
# fast-fp8 (opt-in): the hyper-connections' down projection on block FP8, its 4 bf16 block_inject rows apart, fp32 out
B16_F32 = [(4, 10240, [1, 2], "the HC read-out's 4 block_inject rows beside its block-FP8 down projection (fp32)")]
TEMPLATE_F32 = (324, 10240)        # the bf16 HC down projection's family (SK 32, F32 true)


def b16_split_k(n: int, k: int, target: int = 160, bn: int = 64, bk: int = 64) -> int:
    """bf16.split_k (cuda_triton.zig b16SplitK)."""

    tiles, blocks, sk = -(-n // bn), k // bk, 1
    while sk < 32 and tiles * sk < target and blocks % (sk * 2) == 0 and blocks // (sk * 2) >= 1:
        sk *= 2
    return sk


def ints(k: dict) -> dict:
    return {n: (v.get("int") if "int" in v else v.get("bool")) for n, v in k["constexprs"].items()}


def clone(k: dict, consts: dict, tp: list[int], rule: str, signature: dict | None = None) -> dict:
    out = json.loads(json.dumps(k))
    for n, v in consts.items():
        out["constexprs"][n] = {"bool": v} if isinstance(v, bool) else {"int": v}
    if signature:
        out["signature"].update(signature)
    src = out.pop("hash", None) or out.get("tp1_hash") or out.get("expanded_from")
    for drop in ("cubin_sha256", "tp1_hash", "expanded_from"):
        out.pop(drop, None)
    out.update(tp=tp, cloned_from=src, rule=f"int4ar {rule}")
    return _reorder(out)


def spec(base: Path, out: Path) -> int:
    data = json.loads(base.read_text())
    ks = data["kernels"]
    new: list[dict] = []
    for k in ks:
        f, c = k["function"], ints(k)
        if f.endswith("moe._topk_rows"):
            new.append(clone(k, {"TOPK": TOPK, "SLOTS": SLOTS, "SLOTP": 8}, k["tp"], "top-5 routing"))
        elif f.endswith("glue._hc_writeback") and c["MODE"] == 2:
            new.append(clone(k, {"TOPK": TOPK, "SLOTS": SLOTS}, k["tp"], f"mode 2 over 5 routed slots + shared (Y {k['signature']['Y']})"))
        elif f.endswith("glue._moe_partial"):
            new.append(clone(k, {"TOPK": TOPK, "SLOTS": SLOTS}, k["tp"], f"a rank's MoE partial over 6 slots (Y {k['signature']['Y']})"))
        elif f.endswith(("bf16._b16mm", "prompt_mm._b16mm_ks")) and (c["N"], c["K"]) == TEMPLATE and not c["F32"]:
            for n, kk, tp, why in B16:
                sk = b16_split_k(n, kk)
                sig = {"PART": "*fp32"} if sk > 1 and "PART" in k["signature"] else None
                new.append(clone(k, {"N": n, "K": kk, "SK": sk}, tp, f"{k['name']} {why}, SK {sk}", sig))
        elif f.endswith(("bf16._b16mm", "prompt_mm._b16mm_ks")) and (c["N"], c["K"]) == TEMPLATE_F32 and c["F32"]:
            for n, kk, tp, why in B16_F32:
                sk = b16_split_k(n, kk)
                sig = {"PART": "*fp32"} if "PART" in k["signature"] else None
                new.append(clone(k, {"N": n, "K": kk, "SK": sk}, tp, f"{k['name']} {why}, SK {sk}", sig))
    have = {_key(k) for k in ks}
    uniq: dict[str, dict] = {}
    for k in new:
        key = _key(k)
        if key not in have:
            uniq.setdefault(key, k)
    # the K-slice sum the new splits launch: `_reduce` SK s, F32 False, both `total` forms, under the kernel name
    sks = {(b16_split_k(n, kk), False) for n, kk, _, _ in B16} | {(b16_split_k(n, kk), True) for n, kk, _, _ in B16_F32}
    for sk, f32 in sks:
        forms = [k for k in ks if k["name"] == "_reduce" and ints(k)["SK"] == sk and bool(ints(k)["F32"]) == f32]
        if len({bool(k["attrs"].get("total")) for k in forms}) < 2:
            raise SystemExit(f"_reduce SK {sk} F32 {f32} lacks a `total` form in {base}; add it here")
    out_data = {"generator": "tools/zig/flashnext_int4ar_spec.py spec", "target": data["target"],
                "source_root": data["source_root"], "kernels": list(uniq.values())}
    out.write_text(json.dumps(out_data, indent=1) + "\n")
    for k in uniq.values():
        print(f"{k['name']:16s} {json.dumps(ints(k))} M:{k['signature'].get('M', '-')} "
              f"div16:{[n for n, v in k['attrs'].items() if v]} tp {k['tp']}")
    print(f"{len(uniq)} new entries -> {out}")
    return 0


def merge(base: Path, new: Path, out: Path) -> int:
    a = json.loads((base / "aot.json").read_text())
    b = json.loads((new / "aot.json").read_text())
    (out / "cubins").mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    for src, data in ((base, a), (new, b)):
        for r in data["kernels"]:
            if r["hash"] in seen:
                continue
            seen.add(r["hash"])
            shutil.copyfile(src / "cubins" / f"{r['hash']}.cubin", out / "cubins" / f"{r['hash']}.cubin")
            rows.append(r)
    rows.sort(key=lambda x: (x["fn"], x["hash"]))
    merged = {"generator": "tools/zig/flashnext_int4ar_spec.py merge", "target": a["target"],
              "tp": sorted(set(a.get("tp", [])) | set(b.get("tp", []))), "kernels": rows,
              "merged_from": [str(base), str(new)]}
    (out / "aot.json").write_text(json.dumps(merged, indent=1) + "\n")
    print(f"{len(a['kernels'])} + {len(b['kernels'])} -> {len(rows)} kernels in {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("spec")
    s.add_argument("--base", default=str(ROOT / "zig/tests/cuda/flashnext/kernels.json"))
    s.add_argument("--out", default=str(ROOT / "zig/tests/cuda/flashnext/kernels_int4ar.json"))
    m = sub.add_parser("merge")
    m.add_argument("--base", required=True, help="a built kernel set (aot.json + cubins/), e.g. aot/p1-ks1")
    m.add_argument("--new", required=True, help="the int4ar entries built by flashnext_aot.py build")
    m.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "spec":
        return spec(Path(a.base), Path(a.out))
    return merge(Path(a.base), Path(a.new), Path(a.out))


if __name__ == "__main__":
    sys.exit(main())
