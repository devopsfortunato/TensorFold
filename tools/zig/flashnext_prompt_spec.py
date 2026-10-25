#!/usr/bin/env python3
"""The prompt-path matmul specializations (src/tensorfold/families/qwen4_exp/cuda/prompt_mm.py) for the kernel set.

For every ``_b16mm`` / ``_fp4mm`` entry of the spec that a prompt chunk launches with split K (the 128-row bucket,
``SK > 1``, ``M`` a runtime int), an entry of ``_b16mm_ks`` / ``_fp4mm_ks`` with the same shape constexprs, the same
attributes and options, and no ``PART`` argument. Entries this script made before (``"rule"`` starting with
``prompt_mm``) are replaced, so it can run again after a ``derive`` / ``cover``.

    python3 -B tools/zig/flashnext_prompt_spec.py --spec zig/tests/cuda/flashnext/kernels.json \
        --out zig/tests/cuda/flashnext/kernels.json
    # then (PyTorch image, this tree at /tensorfold):
    python -B tools/zig/flashnext_aot.py build --spec zig/tests/cuda/flashnext/kernels.json --out <dir> --only _ks
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODULE = "tensorfold.families.qwen4_exp.cuda.prompt_mm"
SOURCE = "tensorfold/families/qwen4_exp/cuda/prompt_mm.py"
FROM = {"_b16mm": "_b16mm_ks", "_fp4mm": "_fp4mm_ks"}
GROUP = 8                 # _b16mm_ks: row tiles a band (the tile order; no bits depend on it)


def _consts(k: dict) -> dict:
    return {n: next(iter(v.values())) for n, v in k["constexprs"].items()}


def _line(name: str) -> int | None:
    path = Path(__file__).resolve().parents[2] / "src" / SOURCE
    for i, text in enumerate(path.read_text().splitlines(), 1):
        if text.lstrip().startswith(f"def {name}("):
            return i - 1            # the decorator's line, as the captured entries record
    return None


def entries(kernels: list[dict], bms: tuple[int, ...] = (128,)) -> list[dict]:
    out, seen = [], set()
    for k in kernels:
        new_name = FROM.get(k["name"])
        if new_name is None or k.get("rule", "").startswith("prompt_mm"):
            continue
        c = _consts(k)
        if c.get("BM") not in bms or c.get("SK", 1) <= 1 or "M" in c:
            continue
        e = json.loads(json.dumps(k))
        for drop in ("hash", "cubin_sha256", "tp1_hash", "expanded_from"):
            e.pop(drop, None)
        e["function"] = f"{MODULE}.{new_name}"
        e["name"] = new_name
        e["source"] = {"file": SOURCE, "line": _line(new_name)}
        e["params"] = [p for p in k["params"] if p != "PART"]
        e["signature"] = {n: t for n, t in k["signature"].items() if n != "PART"}
        e["attrs"] = {n: a for n, a in k["attrs"].items() if n != "PART"}
        if new_name == "_b16mm_ks":
            e["params"].append("GROUP")
            e["signature"]["GROUP"] = "constexpr"
            e["attrs"]["GROUP"] = []
            e["constexprs"]["GROUP"] = {"int": GROUP}
        e["rule"] = f"prompt_mm: {k['name']}'s K slices in one program (from {k.get('hash') or k.get('expanded_from') or k.get('tp1_hash')})"
        key = json.dumps([e["function"], e["signature"], e["constexprs"], e["attrs"], e["options"]], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


RT = (8, 16)               # _scores_rows: rows a program


def score_entries(kernels: list[dict]) -> list[dict]:
    """``_scores_rows`` from each ``_scores`` entry: ROWS (a runtime int: a multiple of 16 or not) and RT added."""

    out = []
    for k in kernels:
        if k["name"] != "_scores" or k.get("rule", "").startswith("prompt_mm"):
            continue
        for rt in RT:
            for form in ("div16", "plain"):
                e = json.loads(json.dumps(k))
                for drop in ("hash", "cubin_sha256", "tp1_hash", "expanded_from"):
                    e.pop(drop, None)
                e["function"] = f"{MODULE}._scores_rows"
                e["name"] = "_scores_rows"
                e["source"] = {"file": SOURCE, "line": _line("_scores_rows")}
                params = [p for p in k["params"]]
                params.insert(params.index("NB") + 1, "ROWS")
                params.append("RT")
                e["params"] = params
                e["signature"]["ROWS"] = "i32"
                e["signature"]["RT"] = "constexpr"
                e["signature"] = {n: e["signature"][n] for n in params}
                e["attrs"]["ROWS"] = [["tt.divisibility", 16]] if form == "div16" else []
                e["attrs"]["RT"] = []
                e["attrs"] = {n: e["attrs"][n] for n in params}
                e["constexprs"]["RT"] = {"int": rt}
                e["rule"] = f"prompt_mm: _scores for {rt} rows a program (ROWS {form})"
                out.append(e)
    return out


GLUE = ((64, 64), (32, 64))     # _hc_up_mix (BM, BD) tiles


def glue_entries(kernels: list[dict]) -> list[dict]:
    """``_hc_up_mix`` for the hyper-connections' (D 2560, S 4, low 320), options from a 128-row ``_b16mm`` entry."""

    base = next(k for k in kernels if k["name"] == "_b16mm" and _consts(k).get("BM") == 128)
    out = []
    params = ["ACT", "W", "NORMED", "MIXED", "M", "D", "S", "K", "BM", "BD", "BK"]
    for bm, bd in GLUE:
        for form in ("div16", "plain"):
            e = {"function": f"{MODULE}._hc_up_mix", "name": "_hc_up_mix",
                 "source": {"file": SOURCE, "line": _line("_hc_up_mix")}, "params": params,
                 "signature": {"ACT": "*bf16", "W": "*bf16", "NORMED": "*bf16", "MIXED": "*bf16", "M": "i32",
                               "D": "constexpr", "S": "constexpr", "K": "constexpr", "BM": "constexpr",
                               "BD": "constexpr", "BK": "constexpr"},
                 "constexprs": {"D": {"int": 2560}, "S": {"int": 4}, "K": {"int": 320}, "BM": {"int": bm},
                                "BD": {"int": bd}, "BK": {"int": 64}},
                 "attrs": {n: ([["tt.divisibility", 16]] if n in ("ACT", "W", "NORMED", "MIXED") or (n == "M" and form == "div16") else [])
                           for n in params},
                 "options": json.loads(json.dumps(base["options"])), "tp": [1, 2],
                 "rule": f"prompt_mm: hc up + mix, BM {bm} BD {bd} (M {form})"}
            out.append(e)
    return out


def wbnorm_entries(kernels: list[dict]) -> list[dict]:
    """``_hc_wb_norm`` (``_hc_writeback`` + ``_hc_normed`` a row a program) for every ``_hc_writeback`` entry: its
    arguments, constexprs, attributes and options, then SCALE, NORMED and eps."""

    out = []
    for k in kernels:
        if k["name"] != "_hc_writeback":
            continue
        e = json.loads(json.dumps(k))
        for key in ("hash", "cubin_sha256", "expanded_from"):
            e.pop(key, None)
        e["function"] = f"{MODULE}._hc_wb_norm"
        e["name"] = "_hc_wb_norm"
        e["source"] = {"file": SOURCE, "line": _line("_hc_wb_norm")}
        head = [n for n in k["params"] if n not in k["constexprs"]]
        tail = [n for n in k["params"] if n in k["constexprs"]]
        e["params"] = head + ["SCALE", "NORMED", "eps"] + tail
        sig = dict(k["signature"])
        sig.update({"SCALE": "*fp32", "NORMED": "*bf16", "eps": "fp32"})
        e["signature"] = {n: sig[n] for n in e["params"]}
        att = dict(k["attrs"])
        att.update({"SCALE": [["tt.divisibility", 16]], "NORMED": [["tt.divisibility", 16]], "eps": []})
        e["attrs"] = {n: att[n] for n in e["params"]}
        e["rule"] = "prompt_mm: hc write-back + norm, one row a program (" + k["rule"] + ")" if k.get("rule") else "prompt_mm: hc write-back + norm, one row a program"
        out.append(e)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = json.loads(Path(a.spec).read_text())
    keep = [k for k in data["kernels"] if not k.get("rule", "").startswith("prompt_mm")]
    new = entries(keep) + score_entries(keep) + glue_entries(keep) + wbnorm_entries(keep)
    data["kernels"] = keep + new
    Path(a.out).write_text(json.dumps(data, indent=1) + "\n")
    for e in new:
        c = _consts(e)
        if e["name"] == "_hc_wb_norm":
            print(f"{e['name']} MODE {c['MODE']} WORLD {c['WORLD']} TOPK {c['TOPK']} tp {e.get('tp')}")
        elif e["name"] == "_hc_up_mix":
            print(f"{e['name']} BM {c['BM']} BD {c['BD']} M {e['attrs']['M']}")
        elif "N" in c:
            print(f"{e['name']} N {c['N']} K {c['K']} SK {c['SK']} F32 {c['F32']} M {e['attrs']['M']} tp {e.get('tp')}")
        else:
            print(f"{e['name']} RT {c['RT']} NB {e['attrs']['NB']} ROWS {e['attrs']['ROWS']} tp {e.get('tp')}")
    print(f"{len(new)} prompt_mm entries, {len(data['kernels'])} in all -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
