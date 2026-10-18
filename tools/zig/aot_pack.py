#!/usr/bin/env python3
"""Any capture's Triton manifest (or flashnext_aot.py's) as a Zig kernel set: aot.json (launch facts and keys) plus the
cubins. The family-neutral form of zig/tests/cuda/nemotron/aot_pack.py: the same aot.json fields the Zig loader reads
(zig/src/cuda/aot.zig), plus the Python function and source of each kernel, constexprs of any type (ints and bools as
integers, floats by fp32 bits, others named), and every problem listed before it fails."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def const(v):
    """A constexpr as Zig compares it: ints and bools as integers, floats by their fp32 bits; others by name only."""

    if isinstance(v, bool):
        return {"int": int(v)}
    if isinstance(v, int):
        return {"int": v}
    if isinstance(v, dict) and "fp32_bits" in v:
        return {"f32": int(v["fp32_bits"], 16)}
    if v is None:
        return {"none": True}
    return {"str": str(v)}


def entry(k: dict, jit: dict, cache: Path, out: Path, problems: list[str]) -> dict | None:
    """One specialization's launch facts; its cubin copied beside them by hash."""

    fn = jit.get(k["function"])
    if fn is None:
        problems.append(f"{k['function']}: not in jit.json")
        fn = {"do_not_specialize": []}
    nospec = set(fn["do_not_specialize"])
    attrs = {n for n, v in k["attrs"].items() if v}
    runtime = [n for n, t in k["signature"].items() if t != "constexpr"]
    ptx = [x["name"] for x in k["abi"]]
    if ptx != runtime + ["global_scratch", "profile_scratch"]:
        problems.append(f"{k['name']} {k['hash']}: PTX parameters {ptx} are not the runtime arguments {runtime}")
    cubin = cache / k["cubin"]
    data = cubin.read_bytes()
    if hashlib.sha256(data).hexdigest() != k["cubin_sha256"]:
        problems.append(f"{cubin} changed since the manifest was written")
        return None
    (out / "cubins" / f"{k['hash']}.cubin").write_bytes(data)
    md = k["metadata"]
    return {
        "fn": k["name"], "hash": k["hash"], "name": md["name"], "num_warps": md["num_warps"],
        "num_ctas": md.get("num_ctas", 1), "shared": md.get("shared", 0),
        "global_scratch": md.get("global_scratch_size", 0), "global_align": md.get("global_scratch_align", 1),
        "profile_scratch": md.get("profile_scratch_size", 0), "pdl": bool(md.get("launch_pdl", False)),
        "params": [{"name": n, "type": k["signature"][n], "div16": n in attrs, "nospec": n in nospec}
                   for n in runtime],
        "consts": {n: const(v) for n, v in k["constexprs"].items()},
        "function": k["function"], "source": k.get("source"), "cubin_sha256": k["cubin_sha256"],
    }


def pack(manifests: list[Path], caches: list[Path], jit: dict, out: Path) -> tuple[list[dict], list[str]]:
    (out / "cubins").mkdir(parents=True, exist_ok=True)
    kernels, seen, problems = [], set(), []
    for manifest, cache in zip(manifests, caches, strict=True):
        for k in json.loads(Path(manifest).read_text())["kernels"]:
            if k["hash"] not in seen:
                seen.add(k["hash"])
                got = entry(k, jit, Path(cache), out, problems)
                if got is not None:
                    kernels.append(got)
    kernels.sort(key=lambda x: (x["fn"], x["hash"]))
    return kernels, problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", action="append", required=True, help="a capture's manifest (repeatable)")
    ap.add_argument("--cache", action="append", required=True, help="that capture's Triton cache, in the same order")
    ap.add_argument("--jit", action="append", required=True, help="jit.json (repeatable: merged)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jit: dict = {}
    for path in a.jit:
        jit.update(json.loads(Path(path).read_text()))
    out = Path(a.out)
    kernels, problems = pack([Path(m) for m in a.manifest], [Path(c) for c in a.cache], jit, out)
    meta = {"generator": "tools/zig/aot_pack.py", "kernels": kernels}
    (out / "aot.json").write_text(json.dumps(meta, indent=1) + "\n")
    for p in problems:
        print("PROBLEM", p)
    print(f"{len(kernels)} kernels -> {out}" + (f"; {len(problems)} problems" if problems else ""))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
