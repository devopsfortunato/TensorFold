#!/usr/bin/env python3
"""Flash Next's Triton kernel set without the checkpoint: every specialization in a spec list compiled with
``triton.compile`` for sm_121 (no GPU, no weights), written as ``aot.json`` + ``cubins/`` (the form
``zig/src/cuda/aot.zig`` loads and ``tools/zig/aot_pack.py`` writes from a capture).

  extract: a capture (launches.json + its Triton cache) -> the spec list (zig/tests/cuda/flashnext/kernels.json)
  derive:  the captured TP=1 entries -> the same plus the TP=2 specializations (PLAN.md's TP=2 split, the Python
           wrappers' own K-split and tile rules for the new shapes)
  cover:   every launch in W5's replay fixtures must find a variant; missing ones are added from a template of the
           same kernel (argument types, int forms, constexprs, warps as the fixture gives them)
  build:   the spec list -> <out>/aot.json + <out>/cubins/<hash>.cubin; --check <manifest.json> compares the TP=1
           entries' cubin sha256 (and Triton's kernel hash) with a capture's

A spec entry is exactly what Triton's JIT hands ``compile``: the function (module + qualname), the signature by
parameter name, the constexpr values by name (typed: int, bool, float64 bits, dtype, None, str, a JIT function), the
divisibility attributes by name, and the compile options. TP=2 entries carry ``"tp": [2]`` and their own constexprs;
the TP=1 ones ``"tp": [1]`` (or ``[1, 2]`` when both ranks launch the same specialization)."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import struct
import sys
from pathlib import Path

TARGET = ("cuda", 121, 32)
OPTION_KEYS = ("num_warps", "num_ctas", "num_stages", "warp_size", "maxnreg", "ptx_version", "ptx_options",
               "ir_override", "enable_fp_fusion", "enable_reflect_ftz", "launch_cooperative_grid", "launch_pdl",
               "supported_fp8_dtypes", "deprecated_fp8_dot_operand_dtypes", "default_dot_input_precision",
               "allowed_dot_input_precisions", "max_num_imprecise_acc_default", "extern_libs", "debug",
               "backend_name", "sanitize_overflow", "arch", "instrumentation_mode")
DTYPES = ("int1", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "fp8e4nv", "fp8e5",
          "fp8e4b15", "fp8e4b8", "fp8e5b16", "fp16", "bf16", "fp32", "fp64", "void")


# ---------------------------------------------------------------------------------------------------------- extract

def typed(v):
    """A Recorder constexpr (``_jsonable``) -> the spec's typed form."""

    if isinstance(v, bool):
        return {"bool": v}
    if isinstance(v, int):
        return {"int": v}
    if v is None:
        return {"none": True}
    if isinstance(v, dict) and "fp64_bits" in v:
        return {"float": v["float"], "fp64_bits": v["fp64_bits"]}
    if isinstance(v, str) and v in DTYPES:
        return {"dtype": v}
    if isinstance(v, str) and v.startswith("JITFunction("):
        return {"jit": v[len("JITFunction("):-1]}
    if isinstance(v, list):
        return {"tuple": [typed(x) for x in v]}
    return {"str": str(v)}


def extract(launches: Path, cache: Path, mount: str, out: Path, roles: dict | None, keep: Path | None) -> int:
    rec = json.loads(launches.read_text())
    old = {}
    if keep is not None and keep.exists():
        old = {k["hash"]: k for k in json.loads(keep.read_text())["kernels"] if "hash" in k}
    kernels = []
    for h, info in sorted(rec["kernels"].items(), key=lambda kv: (kv[1]["function"], kv[0])):
        meta_path = next(p for leaf, p in info["files"].items() if leaf.endswith(".json") and not leaf.startswith("__grp__"))
        meta_path = cache / meta_path[len(mount):].lstrip("/") if meta_path.startswith(mount) else Path(meta_path)
        meta = json.loads(meta_path.read_text())
        cubin = meta_path.with_suffix(".cubin")
        options = {k: meta[k] for k in OPTION_KEYS if k in meta}
        entry = {"function": info["function"], "name": info["name"], "hash": h,
                 "source": {"file": info["source"]["file"].split("/src/", 1)[-1], "line": info["source"]["line"]},
                 "signature": info["signature"], "params": info["params"],
                 "constexprs": {k: typed(v) for k, v in info["constexprs"].items()},
                 "attrs": {k: v for k, v in info["attrs"].items()},
                 "options": options, "tp": [1], "cubin_sha256": hashlib.sha256(cubin.read_bytes()).hexdigest()}
        if roles and info["function"] in roles:
            entry["role"] = roles[info["function"]]
        if h in old and "tp" in old[h]:
            entry["tp"] = old[h]["tp"]
        kernels.append(entry)
    derived = [k for k in json.loads(keep.read_text())["kernels"] if "hash" not in k] if keep and keep.exists() else []
    roots = {info["source"]["file"].rsplit("/tensorfold/", 1)[0] for info in rec["kernels"].values()}
    spec = {"generator": "tools/zig/flashnext_aot.py extract", "target": list(TARGET),
            "source_root": sorted(roots)[0] if len(roots) == 1 else sorted(roots),
            "kernels": kernels + derived}
    out.write_text(json.dumps(spec, indent=1) + "\n")
    print(f"{len(kernels)} captured specializations (+{len(derived)} derived) -> {out}")
    return 0


# ----------------------------------------------------------------------------------------------------------- derive

# PLAN.md "TP=2 design": per rank GDN 8 key / 24 value heads, attention 12 q heads / 1 kv head (indexer replicated),
# shared expert at half width (640 -> 320), out_proj / o_proj input columns halved with fp32 partial outputs, the
# head split by vocabulary, n-gram rows by head (16 -> 8 a rank), write-backs of gathered partials (mode 3, WORLD 2)
B16_TP2 = {(16480, 2560): (8240, 2560, None, "GDN in_proj: qkv 2048+3072 | z 3072 | b 24 | a 24 a rank"),
           (13952, 2560): (7296, 2560, None, "attention proj: q|gate 6144, k 256, v 256 a rank; indexer q 512 + key 128 replicated"),
           (2560, 6144): (2560, 3072, True, "GDN out_proj / attention o_proj: K halves, fp32 partials (mode 3)"),
           (248320, 2560): (124160, 2560, None, "lm_head: vocabulary halves")}
FP4_TP2 = {(1280, 2560): (640, 2560, "shared expert gate|up: 320 + 320 rows a rank"),
           (2560, 640): (2560, 320, "shared expert down: K halves (its y slot joins the fp32 MoE partial)")}


def _ints(k: dict) -> dict:
    return {n: (v.get("int") if "int" in v else v.get("bool")) for n, v in k["constexprs"].items()}


def _with(k: dict, rule: str, consts: dict, signature: dict | None = None, options: dict | None = None) -> dict:
    out = json.loads(json.dumps(k))
    for n, v in consts.items():
        out["constexprs"][n] = {"bool": v} if isinstance(v, bool) else {"int": v}
    if signature:
        out["signature"].update(signature)
    if options:
        out["options"].update(options)
    for drop in ("hash", "cubin_sha256"):
        out.pop(drop, None)
    out.update(tp=[2], tp1_hash=k["hash"], rule=rule)
    return out


def _key(k: dict) -> str:
    return json.dumps([k["function"], k["signature"], k["constexprs"], k["attrs"], k["options"]], sort_keys=True)



# runtime ints whose value Triton specializes (1 -> a constexpr, a multiple of 16 -> divisibility), by kernel: a row
# count or a size that follows it; strides of fixed buffers are left as captured
ROW_INTS = {"bf16._b16mm": ("M", True), "nvfp4._fp4mm": ("M", True), "moe._router": ("M", True),
            "bf16._reduce": ("total", False), "nvfp4._reduce": ("total", False), "attention._pool": ("R", True),
            "glue._ple_conv": ("R", True), "forward._shift_windows": ("keep", True),
            "attention._scores": ("NB", False), "attention._select": ("NB", False),
            "attention._select_tiles": ("NB", False)}


def _int_form(k: dict, name: str, form: str) -> dict:
    out = json.loads(json.dumps(k))
    if form == "one":
        out["signature"][name] = "constexpr"
        out["constexprs"][name] = {"int": 1}
        out["attrs"][name] = []
    else:
        out["signature"][name] = "i32"
        out["constexprs"].pop(name, None)
        out["attrs"][name] = [["tt.divisibility", 16]] if form == "div16" else []
    return _reorder(out)


def _reorder(k: dict) -> dict:
    """Signature, attrs and constexprs in parameter order, as the JIT builds them (the attrs' order is hashed)."""

    p = k["params"]
    k["signature"] = {n: k["signature"][n] for n in p}
    k["attrs"] = {n: k["attrs"].get(n, []) for n in p}
    k["constexprs"] = {n: k["constexprs"][n] for n in sorted(k["constexprs"], key=lambda x: p.index(x.split(".")[0]))}
    return k


def _derived(k: dict, why: str) -> dict:
    out = json.loads(json.dumps(k))
    src = out.pop("hash", None) or out.get("tp1_hash") or out.get("expanded_from")
    out.pop("cubin_sha256", None)
    out["expanded_from"] = src
    out["rule"] = (out.get("rule", "") + "; " if out.get("rule") else "") + why
    return out


def expand(kernels: list[dict]) -> list[dict]:
    """Every row bucket and runtime-int form the wrappers can launch, from the captured and TP=2 entries."""

    from tensorfold.families.qwen4_exp.cuda import attention, nvfp4

    out = []
    for k in kernels:
        f = k["function"].split("qwen4_exp.cuda.")[-1].split("tensorfold.cuda.")[-1]
        c = _ints(k)
        variants = [k]
        # row buckets
        if f == "bf16._b16mm":
            variants = [_derived(_with_consts(k, {"BM": bm}), f"rows bucket BM {bm}") for bm in (16, 128)]
        elif f == "nvfp4._fp4mm":
            variants = []
            for bm in (16, 32, 64, 128):
                g0, warps, _ = nvfp4.CONFIG[bm]
                tuned = nvfp4.SHAPES16.get((c["N"], c["K"])) if bm == 16 else None
                if tuned is not None:
                    _, g0, warps, _ = tuned
                g = nvfp4.gpi_for((c["K"] // nvfp4.GS) // c["SK"], g0)
                v = _with_consts(k, {"BM": bm, "GPI": g})
                v["options"]["num_warps"] = warps
                variants.append(_derived(v, f"rows bucket BM {bm}"))
        elif f == "moe._router":
            variants = []
            for bm in (16, 32, 64, 128):
                be, bk, stages = (32, 256, 4) if bm == 16 else (64, 64, 3)
                v = _with_consts(k, {"BM": bm, "BLOCK_E": be, "BK": bk})
                v["options"]["num_stages"] = stages
                variants.append(_derived(v, f"rows bucket BM {bm}"))
        elif f == "attention._select":
            variants = []
            width = 1
            while width <= attention.SELECT_REGS:
                variants.append(_derived(_with_consts(k, {"BLOCK": width}), f"select width {width}"))
                width *= 2
            for tb, warps in ((4096, 8), (8192, 16)):     # past the registers: the tiled select, same lists
                v = json.loads(json.dumps(k))
                v["function"] = k["function"][:-len("_select")] + "_select_tiles"
                v["name"] = "_select_tiles"
                v["params"] = [("TB" if n == "BLOCK" else n) for n in k["params"]]
                v["signature"] = {("TB" if n == "BLOCK" else n): t for n, t in k["signature"].items()}
                v["attrs"] = {("TB" if n == "BLOCK" else n): a for n, a in k["attrs"].items()}
                v["constexprs"] = {("TB" if n == "BLOCK" else n): x for n, x in k["constexprs"].items()}
                v["constexprs"]["TB"] = {"int": tb}
                v["options"]["num_warps"] = warps
                v["source"] = {"file": k["source"]["file"], "line": None}
                variants.append(_derived(v, f"tiled select TB {tb}"))
        # runtime int forms
        name, one = ROW_INTS.get(f, (None, False))
        for v in variants:
            vname = name
            vf = v["function"].split("qwen4_exp.cuda.")[-1]
            if vf == "attention._select_tiles":
                vname = "NB"
            if vname is None or vname not in v["params"]:
                out.append(v)
                continue
            forms = ["div16", "plain"] + (["one"] if one and _ints(v).get("BM", 16) == 16 else [])
            for form in forms:
                out.append(_derived(_int_form(v, vname, form), f"{vname} {form}"))
    return out


def _with_consts(k: dict, consts: dict) -> dict:
    out = json.loads(json.dumps(k))
    for n, v in consts.items():
        out["constexprs"][n] = {"bool": v} if isinstance(v, bool) else {"int": v}
    return out


def derive(spec: Path, out: Path) -> int:
    """TP=1 captured entries -> the same list plus each TP=2 specialization (``tp1_hash``: the entry it came from)."""

    from tensorfold.families.qwen4_exp.cuda import bf16, nvfp4

    data = json.loads(spec.read_text())
    captured = [k for k in data["kernels"] if "tp1_hash" not in k and "expanded_from" not in k]
    derived: list[dict] = []
    reduce_b16 = {_ints(k)["F32"]: k for k in captured if k["function"].endswith("bf16._reduce")}
    reduce_fp4 = {_ints(k)["F32"]: k for k in captured if k["function"].endswith("nvfp4._reduce")}
    for k in captured:
        f, c = k["function"], _ints(k)
        tp = [1, 2]
        if f.endswith("bf16._b16mm") and (c["N"], c["K"]) in B16_TP2:
            n, kk, f32, why = B16_TP2[(c["N"], c["K"])]
            f32 = c["F32"] if f32 is None else f32
            sk = bf16.split_k(n, kk)
            sig = {"OUT": "*fp32" if f32 else "*bf16", "PART": "*fp32" if sk > 1 else ("*fp32" if f32 else "*bf16")}
            derived.append(_with(k, why, {"N": n, "K": kk, "SK": sk, "F32": f32}, sig))
            if sk > 1:
                derived.append(_with(reduce_b16[f32], f"the K-slice sum of {why}", {"SK": sk, "F32": f32}))
            tp = [1]
        elif f.endswith("nvfp4._fp4mm") and (c["N"], c["K"]) in FP4_TP2:
            n, kk, why = FP4_TP2[(c["N"], c["K"])]
            bm = c["BM"]
            c_gpi, c_warps, _ = nvfp4.CONFIG[bm]
            tuned = nvfp4.SHAPES16.get((n, kk)) if bm == 16 else None
            if tuned is not None:
                _, c_gpi, c_warps, _ = tuned
            sk = nvfp4.split_for(n, kk)
            g = nvfp4.gpi_for((kk // nvfp4.GS) // sk, c_gpi)
            sig = {"PART": "*fp32" if sk > 1 else ("*fp32" if c["F32"] else "*bf16")}
            derived.append(_with(k, why, {"N": n, "K": kk, "SK": sk, "GPI": g}, sig, {"num_warps": c_warps}))
            if sk > 1:
                derived.append(_with(reduce_fp4[bool(c["F32"])], f"the K-slice sum of {why}", {"SK": sk}))
            tp = [1]
        elif f.endswith("glue._attn_prep"):
            derived.append(_with(k, "attention prep: 12 q heads, 1 kv head a rank; the indexer replicated",
                                 {"PW": 7296, "NQ": 12, "NKV": 1}))
            tp = [1]
        elif f.endswith("glue._attn_gate"):
            derived.append(_with(k, "attention gate: 12 q heads a rank", {"PW": 7296, "NQ": 12}))
            tp = [1]
        elif f.endswith("attention._chunks") or f.endswith("attention._merge"):
            derived.append(_with(k, "attention: 12 q heads over 1 kv head a rank (G unchanged)", {"H": 12, "HK": 1}))
            tp = [1]
        elif f.endswith("glue._ple_embed_bf16"):
            derived.append(_with(k, "n-gram rows: this rank's 8 of 16 heads", {"HEADS": 8}))
            tp = [1]
        elif f.endswith("forward._shift_windows") and c["T"] == 3:
            derived.append(_with(k, "GDN conv windows: 5120 channels a rank", {"C": 5120}))
            tp = [1]
        elif f.endswith("glue._hc_writeback"):
            if c["MODE"] == 1:
                derived.append(_with(k, "write-back of gathered fp32 partials [2, R, D] (mode 3)",
                                     {"MODE": 3, "WORLD": 2}, {"BR": "*fp32"}))
            tp = [1, 2] if c["MODE"] == 0 else [1]
        k["tp"] = tp
    # the MoE's fp32 share a rank (glue.moe_partial), which one GPU never launches
    base = next(k for k in captured if k["function"].endswith("glue._hc_writeback"))
    mp = {"function": "tensorfold.families.qwen4_exp.cuda.glue._moe_partial", "name": "_moe_partial",
          "source": {"file": "tensorfold/families/qwen4_exp/cuda/glue.py", "line": 125},
          "signature": {"Y": "*bf16", "WTS": "*fp32", "OUT": "*fp32", "D": "constexpr", "TOPK": "constexpr",
                        "SLOTS": "constexpr", "BLOCK": "constexpr"},
          "params": ["Y", "WTS", "OUT", "D", "TOPK", "SLOTS", "BLOCK"],
          "constexprs": {"D": {"int": 2560}, "TOPK": {"int": 10}, "SLOTS": {"int": 11}, "BLOCK": {"int": 256}},
          "attrs": {"Y": [["tt.divisibility", 16]], "WTS": [["tt.divisibility", 16]], "OUT": [["tt.divisibility", 16]],
                    "D": [], "TOPK": [], "SLOTS": [], "BLOCK": []},
          "options": {**base["options"], "num_warps": 2}, "tp": [2], "tp1_hash": None,
          "rule": "the MoE's fp32 partial a rank: sum_k w_k y_k over the 10 routed slots and the shared one"}
    derived.append(mp)
    # every row bucket and int form the wrappers can launch (W7: a 16-row prompt chunk needs M % 16 == 0)
    tp_of = {}
    for k in captured + derived:
        for e in expand([k]):
            tp_of.setdefault(_key(e), set()).update(k["tp"])
            if "expanded_from" in e:
                derived.append(e)
    for k in derived:
        k["tp"] = sorted(tp_of.get(_key(k), set(k["tp"])))
    have = {_key(k) for k in captured}
    uniq: dict[str, dict] = {}
    for k in derived:
        key = _key(k)
        if key in have:                                   # the TP=2 shape equals a captured one: both ranks share it
            for c in captured:
                if _key(c) == key:
                    c["tp"] = sorted(set(c["tp"]) | set(k["tp"]))
            continue
        uniq.setdefault(key, k)
    data["kernels"] = captured + list(uniq.values())
    data["generator"] = "tools/zig/flashnext_aot.py extract + derive"
    out.write_text(json.dumps(data, indent=1) + "\n")
    print(f"{len(captured)} captured ({sum(1 for k in captured if 2 in k['tp'])} on both ranks), "
          f"{len(uniq)} derived (TP=2 shapes, row buckets, int forms; {sum(1 for k in uniq.values() if k['tp'] == [2])} "
          f"TP=2-only) -> {out}")
    return 0



# ------------------------------------------------------------------------------------------------------------ cover

def _fixture_matches(k: dict, launch: dict) -> bool:
    """zig/src/cuda/aot.zig's ``matches`` on a W5 fixture launch (pointers taken as 16-byte aligned)."""

    if k["name"] != launch["fn"]:
        return False
    zc = {n: zig_const(v) for n, v in k["constexprs"].items()}
    for n, v in launch["consts"].items():
        if not isinstance(v, dict) or not ({"int", "f32"} & set(v)):
            continue                                       # a None (or named) constexpr: Zig does not compare it
        got = zc.get(n)
        if got is None or any(got.get(t) != v.get(t) for t in ("int", "f32") if t in v):
            return False
    runtime = 0
    params = {n for n in k["params"] if k["signature"].get(n) != "constexpr"}
    for a in launch["args"]:
        n, ty = a["name"], a["type"]
        div = bool(k["attrs"].get(n))
        if ty.startswith("*"):
            if n not in params or k["signature"][n] != ty or not div:
                return False
        elif "int" in a:
            x = a["int"]
            if n not in params:
                if x != 1 or zc.get(n, {}).get("int") != 1:
                    return False
                continue
            if k["signature"][n] != ty or x == 1 or div != (x % 16 == 0):
                return False
        elif n not in params or k["signature"][n] != ty:
            return False
        runtime += 1
    return runtime == len(params)


def _from_fixture(t: dict, launch: dict, case: str) -> dict:
    """A spec entry for a fixture launch, on a template of the same kernel: its types, int forms, constexprs, warps."""

    out = json.loads(json.dumps(t))
    for drop in ("hash", "cubin_sha256", "tp1_hash"):
        out.pop(drop, None)
    for a in launch["args"]:
        n = a["name"]
        if "int" in a and a["int"] == 1:
            out["signature"][n], out["constexprs"][n], out["attrs"][n] = "constexpr", {"int": 1}, []
            continue
        out["signature"][n] = a["type"]
        out["constexprs"].pop(n, None)
        if a["type"].startswith("*"):
            out["attrs"][n] = [["tt.divisibility", 16]]
        elif "int" in a:
            out["attrs"][n] = [["tt.divisibility", 16]] if a["int"] % 16 == 0 else []
    for n, v in launch["consts"].items():
        old = t["constexprs"].get(n, {})
        if not isinstance(v, dict) or not ({"int", "f32"} & set(v)):
            continue
        if "f32" in v:
            if old and zig_const(old).get("f32") == v["f32"]:
                continue                                  # the template's own value (its fp64 bits)
            x = struct.unpack("<f", struct.pack("<I", v["f32"]))[0]     # an fp32 value, as a Python float
            out["constexprs"][n] = {"float": x, "fp64_bits": "0x%016x" % struct.unpack("<Q", struct.pack("<d", x))[0]}
            continue
        out["constexprs"][n] = {"bool": bool(v["int"])} if "bool" in old else {"int": v["int"]}
    out["options"]["num_warps"] = launch["num_warps"]
    if launch.get("num_stages") is not None:
        out["options"]["num_stages"] = launch["num_stages"]
    out["tp"] = [2] if "/tp2" in case else [1] if "/tp1" in case else [1, 2]
    out["expanded_from"] = t.get("hash") or t.get("expanded_from")
    out["rule"] = f"W5 fixture {case}"
    return _reorder(out)


def cover(spec: Path, fixtures: Path, out: Path) -> int:
    """Every launch of W5's fixtures (zig/src/families/flashnext/fixtures_cuda_triton.json) must find a variant."""

    data = json.loads(spec.read_text())
    kernels = data["kernels"]
    cases = json.loads(fixtures.read_text())["cases"]
    cases = list(cases.items()) if isinstance(cases, dict) else cases
    added, missing = [], 0
    for case, launches in cases:
        for launch in launches:
            if any(_fixture_matches(k, launch) for k in kernels + added):
                continue
            missing += 1
            same = [k for k in kernels if k["name"] == launch["fn"]
                    and {a["name"] for a in launch["args"]} <= set(k["params"])]
            if not same:
                print(f"NO TEMPLATE {case} {launch['fn']}")
                continue
            # the template closest in constexprs
            best = max(same, key=lambda k: (sum(zig_const(k["constexprs"].get(n, {"none": True})) == v
                                                for n, v in launch["consts"].items() if isinstance(v, dict)),
                                            sum(k["signature"].get(a["name"]) == a["type"] for a in launch["args"])))
            new = _from_fixture(best, launch, case)
            if not _fixture_matches(new, launch):
                raise SystemExit(f"{case}: the synthesized entry does not match its launch")
            added.append(new)
            print(f"added {case} {launch['fn']}: " + ", ".join(f"{a['name']} {a['type']}" for a in launch["args"]
                                                              if a["type"] != best["signature"].get(a["name"])))
    data["kernels"] = kernels + added
    out.write_text(json.dumps(data, indent=1) + "\n")
    print(f"{sum(len(l) for _, l in cases)} fixture launches, {missing} without a variant, {len(added)} entries added "
          f"-> {out}")
    return 0

# ------------------------------------------------------------------------------------------------------------ build

def value(v, tl, registry):
    if "bool" in v:
        return v["bool"]
    if "int" in v:
        return v["int"]
    if "none" in v:
        return None
    if "fp64_bits" in v:
        return struct.unpack("<d", struct.pack("<Q", int(v["fp64_bits"], 16)))[0]
    if "dtype" in v:
        return tl.dtype(v["dtype"])
    if "jit" in v:
        return registry(v["jit"])
    if "tuple" in v:
        return tuple(value(x, tl, registry) for x in v["tuple"])
    return v["str"]


def resolve(function: str):
    """``package.module.qualname`` -> the JITFunction object (the longest importable module prefix)."""

    parts = function.split(".")
    for i in range(len(parts) - 1, 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for p in parts[i:]:
            obj = getattr(obj, p)
        return obj
    raise SystemExit(f"cannot import {function}")


def path_of(name: str, params: list[str]) -> tuple:
    head, *rest = name.split(".")
    return (params.index(head), *map(int, rest))


def compile_one(k: dict, cache_dir: Path | None):
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource, compile

    fn = resolve(k["function"])
    params = k["params"]
    registry = lambda key: resolve(key.replace(":", "."))       # noqa: E731
    constexprs = {path_of(n, params): value(v, tl, registry) for n, v in k["constexprs"].items()}
    attrs = {path_of(n, params): v for n, v in k["attrs"].items()}
    options = {n: tuple(v) if isinstance(v, list) else v for n, v in k["options"].items()}
    src = ASTSource(fn, dict(k["signature"]), constexprs, attrs)
    return compile(src, target=GPUTarget(*TARGET), options=options)


class Compiled:
    """What a worker sends back: the kernel hash, its cubin and metadata (picklable)."""

    def __init__(self, ck) -> None:
        self.hash, self.cubin = ck.hash, ck.asm["cubin"]
        self.metadata = {k: v for k, v in ck.metadata._asdict().items() if isinstance(v, (int, str, bool, float))}
        self.n_regs, self.n_spills = getattr(ck, "n_regs", None), getattr(ck, "n_spills", None)


def _compile_job(k: dict):
    try:
        return Compiled(compile_one(k, None))
    except Exception as exc:                                      # noqa: BLE001  (listed, then the run fails)
        return f"{type(exc).__name__}: {str(exc)[:300]}"


def zig_const(v):
    if "bool" in v:
        return {"int": int(v["bool"])}
    if "int" in v:
        return {"int": v["int"]}
    if "fp64_bits" in v:
        return {"f32": struct.unpack("<I", struct.pack("<f", struct.unpack("<d", struct.pack("<Q", int(v["fp64_bits"], 16)))[0]))[0]}
    if "none" in v:
        return {"none": True}
    return {"str": json.dumps(v)}


def build(spec: Path, out: Path, tps: set[int], check: list[Path], jit: Path | None, only: str, jobs: int = 8) -> int:
    data = json.loads(spec.read_text())
    kernels = data["kernels"]
    root = data.get("source_root")
    import tensorfold

    here = str(Path(tensorfold.__file__).parent.parent)
    if root and here != root:     # Triton's line info names the source file: another path, other cubin bytes
        raise SystemExit(f"the spec was captured with the Python source at {root}, this run imports it from {here}: "
                         f"mount the tree so that {root}/tensorfold is the package (the cubins embed the path)")
    nospec = json.loads(jit.read_text()) if jit else {}
    (out / "cubins").mkdir(parents=True, exist_ok=True)
    captured = {}
    for m in check:
        for k in json.loads(m.read_text())["kernels"]:
            captured[k["hash"]] = k["cubin_sha256"]
    rows, problems, seen = [], [], set()
    same = 0
    todo = [k for k in kernels if tps & set(k.get("tp", [1])) and (not only or only in k["function"])]
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=max(1, jobs)) as pool:
        done = list(pool.map(_compile_job, todo, chunksize=1))
    for k, got in zip(todo, done):
        if isinstance(got, str):
            problems.append(f"{k['function']} {k.get('hash', '?')[:12]}: {got}")
            continue
        ck = got
        cubin = ck.cubin
        sha = hashlib.sha256(cubin).hexdigest()
        if k.get("hash"):
            want = k.get("cubin_sha256") or captured.get(k["hash"])
            if ck.hash != k["hash"]:
                problems.append(f"{k['function']}: Triton hash {ck.hash[:12]} != captured {k['hash'][:12]}")
            if want and sha != want:
                problems.append(f"{k['function']} {k['hash'][:12]}: cubin sha256 {sha[:12]} != captured {want[:12]}")
            elif want:
                same += 1
        if ck.hash in seen:
            continue
        seen.add(ck.hash)
        (out / "cubins" / f"{ck.hash}.cubin").write_bytes(cubin)
        md = ck.metadata
        runtime = [n for n in k["params"] if k["signature"].get(n) != "constexpr"]
        div = {n for n, v in k["attrs"].items() if v}
        dns = set(nospec.get(k["function"], {}).get("do_not_specialize", []))
        rows.append({
            "fn": md["name"], "hash": ck.hash, "name": md["name"], "num_warps": md["num_warps"],
            "num_ctas": md.get("num_ctas", 1), "shared": md.get("shared", 0),
            "global_scratch": md.get("global_scratch_size", 0), "global_align": md.get("global_scratch_align", 1),
            "profile_scratch": md.get("profile_scratch_size", 0), "pdl": bool(md.get("launch_pdl", False)),
            "params": [{"name": n, "type": k["signature"][n], "div16": n in div, "nospec": n in dns} for n in runtime],
            "consts": {n: zig_const(v) for n, v in k["constexprs"].items()},
            "function": k["function"], "tp": k.get("tp", [1]), "cubin_sha256": sha,
            "n_regs": ck.n_regs, "n_spills": ck.n_spills,
        })
    rows.sort(key=lambda x: (x["fn"], x["hash"]))
    (out / "aot.json").write_text(json.dumps({"generator": "tools/zig/flashnext_aot.py", "target": list(TARGET),
                                              "tp": sorted(tps), "kernels": rows}, indent=1) + "\n")
    for p in problems:
        print("PROBLEM", p)
    print(f"{len(rows)} kernels -> {out}; {same} cubins equal to the capture's" +
          (f"; {len(problems)} problems" if problems else ""))
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    x = sub.add_parser("extract")
    x.add_argument("--launches", required=True)
    x.add_argument("--cache", required=True, help="the capture's TRITON_CACHE_DIR as seen here")
    x.add_argument("--mount", default="/aot/triton", help="TRITON_CACHE_DIR as the capture saw it")
    x.add_argument("--roles", help="JSON {function: role} (kernels_md.py's roles)")
    x.add_argument("--keep", help="an existing spec: its TP flags and derived (TP=2) entries are kept")
    x.add_argument("--out", required=True)
    d = sub.add_parser("derive")
    d.add_argument("--spec", required=True, help="an extracted spec (its derived entries are made again)")
    d.add_argument("--out", required=True)
    v = sub.add_parser("cover")
    v.add_argument("--spec", required=True)
    v.add_argument("--fixtures", required=True, help="W5's zig/src/families/flashnext/fixtures_cuda_triton.json")
    v.add_argument("--out", required=True)
    b = sub.add_parser("build")
    b.add_argument("--spec", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--tp", default="1,2")
    b.add_argument("--check", action="append", default=[], help="a capture's manifest.json (repeatable)")
    b.add_argument("--jit", help="a capture's jit.json (do-not-specialize lists for aot.json)")
    b.add_argument("--only", default="", help="substring of the function names to build")
    b.add_argument("--jobs", type=int, default=8, help="compiles in parallel")
    a = ap.parse_args()
    if a.cmd == "extract":
        return extract(Path(a.launches), Path(a.cache), a.mount, Path(a.out),
                       json.loads(Path(a.roles).read_text()) if a.roles else None, Path(a.keep) if a.keep else None)
    if a.cmd == "cover":
        return cover(Path(a.spec), Path(a.fixtures), Path(a.out))
    if a.cmd == "derive":
        return derive(Path(a.spec), Path(a.out))
    return build(Path(a.spec), Path(a.out), {int(t) for t in a.tp.split(",")}, [Path(c) for c in a.check],
                 Path(a.jit) if a.jit else None, a.only, a.jobs)


if __name__ == "__main__":
    sys.exit(main())
