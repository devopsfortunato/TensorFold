#!/usr/bin/env python3
"""The image and video prompts' rotary specializations (``MODE=2``) for Flash Next's kernel set.

An image or video sequence rotates its queries, keys and pooled indexer keys at the prompt's three-axis positions
(``glue._attn_prep`` and ``attention._pool`` with ``ROPE`` / ``DELTA`` / ``length``, image_rows.rope_axis); text
keeps ``MODE=0``. For every ``MODE=0`` entry of the spec this adds the ``MODE=2`` entries Triton's JIT builds for
such a launch, as a capture of Python TensorFold's image prompts records them (work/out/v1r1/launches.json):

  _attn_prep  the same signature (ROPE and DELTA were already pointers, POS0 passed for them), MODE 2, ``length``
              (the prompt's tokens, a runtime i32) a multiple of 16 or not
  _pool       ROPE and DELTA become ``*i32`` arguments (16-byte aligned: device allocations) instead of the None
              constexprs, MODE 2, ``length`` a multiple of 16 or not; every ``R`` form of the MODE 0 entry

Entries this script made before (``"rule"`` starting with ``vision``) are replaced, so it can run again.

    python3 -B tools/zig/flashnext_vision_spec.py --spec zig/tests/cuda/flashnext/kernels.json \\
        --out zig/tests/cuda/flashnext/kernels.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DIV16 = [["tt.divisibility", 16]]


def _reorder(k: dict) -> dict:
    p = k["params"]
    k["signature"] = {n: k["signature"][n] for n in p}
    k["attrs"] = {n: k["attrs"].get(n, []) for n in p}
    k["constexprs"] = {n: k["constexprs"][n] for n in sorted(k["constexprs"], key=lambda x: p.index(x.split(".")[0]))}
    return k


def _key(k: dict) -> str:
    return json.dumps([k["function"], k["signature"], k["constexprs"], k["attrs"], k["options"]], sort_keys=True)


def entries(kernels: list[dict]) -> list[dict]:
    out, seen = [], set()
    for k in kernels:
        if k["name"] not in ("_attn_prep", "_pool") or k.get("rule", "").startswith("vision"):
            continue
        if k["constexprs"].get("MODE", {}).get("int") != 0:
            continue
        for form in ("div16", "plain"):
            e = json.loads(json.dumps(k))
            for drop in ("hash", "cubin_sha256", "tp1_hash", "expanded_from"):
                e.pop(drop, None)
            e["constexprs"]["MODE"] = {"int": 2}
            if e["name"] == "_pool":
                for n in ("ROPE", "DELTA"):
                    e["signature"][n] = "*i32"
                    e["constexprs"].pop(n, None)
                    e["attrs"][n] = DIV16
            e["attrs"]["length"] = DIV16 if form == "div16" else []
            e["rule"] = (f"vision: {k['name']} MODE 2 (an image or video prompt's rotary table), length {form} "
                         f"(from {k.get('hash') or k.get('tp1_hash') or k.get('expanded_from') or '?'})")
            e = _reorder(e)
            key = _key(e)
            if key in seen:
                continue
            seen.add(key)
            out.append(e)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = json.loads(Path(a.spec).read_text())
    keep = [k for k in data["kernels"] if not k.get("rule", "").startswith("vision")]
    new = entries(keep)
    data["kernels"] = keep + new
    Path(a.out).write_text(json.dumps(data, indent=1) + "\n")
    for e in new:
        c = {n: next(iter(v.values())) for n, v in e["constexprs"].items()}
        print(f"{e['name']} NQ {c.get('NQ', '-')} R {e['signature'].get('R')} {c.get('R', '')} "
              f"length {'div16' if e['attrs']['length'] else 'plain'} tp {e.get('tp')}")
    print(f"{len(new)} vision entries, {len(data['kernels'])} in all -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
