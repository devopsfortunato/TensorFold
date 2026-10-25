#!/usr/bin/env python3
"""The FP8 KV cache's specializations (src/tensorfold/families/qwen4_exp/cuda/kv8.py, --kv-dtype fp8) for the kernel set.

For every ``_attn_prep`` and ``_chunks`` entry of the spec (TP=1 and TP=2 shapes, QSA on and off), an entry of
``_attn_prep8`` / ``_chunks8`` with the same shape constexprs, attributes and options: the key and value codes as
``*fp8e4nv``, the key rows' fp32 view (both scales in their trailer) as ``KS``, no int8/int4 scale tensors and no
``BITS``. The merge stays ``_merge`` with BITS 0 (bf16's entries). Entries this script made before (``"rule"``
starting with ``kv8``) are replaced, so it can run again after a ``derive`` / ``cover``.

    python3 -B tools/zig/flashnext_kv8_spec.py --spec zig/tests/cuda/flashnext/kernels.json \\
        --out zig/tests/cuda/flashnext/kernels.json
    # then (PyTorch image, this tree at /tensorfold):
    python -B tools/zig/flashnext_aot.py build --spec zig/tests/cuda/flashnext/kernels.json --out <dir> --only kv8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODULE = "tensorfold.families.qwen4_exp.cuda.kv8"
SOURCE = "tensorfold/families/qwen4_exp/cuda/kv8.py"
FP8 = "*fp8e4nv"
# name -> (new name, parameters dropped, parameters retyped)
RULES = {
    "_attn_prep": ("_attn_prep8", ("VS", "BITS"), {"KC": FP8, "VC": FP8, "KS": "*fp32"}),
    "_chunks": ("_chunks8", ("VSC", "BITS"), {"KC": FP8, "VC": FP8, "KSC": "*fp32"}),
}
RENAME = {"KSC": "KS"}      # _chunks' key-scale argument is the key rows' fp32 view in _chunks8


def _line(name: str) -> int | None:
    path = Path(__file__).resolve().parents[2] / "src" / SOURCE
    for i, text in enumerate(path.read_text().splitlines(), 1):
        if text.lstrip().startswith(f"def {name}("):
            return i - 1            # the decorator's line, as the captured entries record
    return None


def entries(kernels: list[dict]) -> list[dict]:
    out, seen = [], set()
    for k in kernels:
        rule = RULES.get(k["name"])
        if rule is None or k.get("rule", "").startswith("kv8"):
            continue
        new_name, drop, retype = rule
        e = json.loads(json.dumps(k))
        for x in ("hash", "cubin_sha256", "tp1_hash", "expanded_from", "role"):
            e.pop(x, None)
        if e["constexprs"].get("BITS", {}).get("int", 0) != 0:
            continue                                     # an int8/int4 entry: not a template
        e["function"] = f"{MODULE}.{new_name}"
        e["name"] = new_name
        e["source"] = {"file": SOURCE, "line": _line(new_name)}
        params = [RENAME.get(p, p) for p in k["params"] if p not in drop]
        sig = {RENAME.get(n, n): retype.get(n, t) for n, t in k["signature"].items() if n not in drop}
        attrs = {RENAME.get(n, n): a for n, a in k["attrs"].items() if n not in drop}
        e["params"] = params
        e["signature"] = {n: sig[n] for n in params}
        e["attrs"] = {n: attrs[n] for n in params if n in attrs}
        e["constexprs"] = {n: v for n, v in k["constexprs"].items() if n not in drop}
        e["rule"] = f"kv8: {k['name']} on FP8 rows (from {k.get('hash') or k.get('rule') or '?'})"
        key = json.dumps([e["function"], e["signature"], e["constexprs"], e["attrs"], e["options"]], sort_keys=True)
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
    keep = [k for k in data["kernels"] if not k.get("rule", "").startswith("kv8")]
    new = entries(keep)
    data["kernels"] = keep + new
    Path(a.out).write_text(json.dumps(data, indent=1) + "\n")
    for e in new:
        c = {n: next(iter(v.values())) for n, v in e["constexprs"].items()}
        shape = (f"NQ {c['NQ']} NKV {c['NKV']} PW {c['PW']}" if e["name"] == "_attn_prep8"
                 else f"H {c['H']} HK {c['HK']} NCH {c['NCH']} QSA {c['QSA']}")
        print(f"{e['name']} {shape} tp {e.get('tp')}")
    print(f"{len(new)} kv8 entries, {len(data['kernels'])} in all -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
