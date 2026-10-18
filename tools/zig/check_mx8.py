"""Writes cases for `tensorfold mx8-check`: real MXFP8 layers of the NVIDIA checkpoint run through Python's
Mx8Linear (qmmf MODE 2) at several row counts. Usage: check_mx8.py SNAPSHOT_DIR OUT_DIR [PYTHON_SRC]"""
import json, struct, sys, pathlib
import torch
from safetensors import safe_open

snap, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
from tensorfold.cuda.nvfp4.linear import Mx8Linear

index = json.load(open(snap / "model.safetensors.index.json"))["weight_map"]
base = "model.language_model.layers."
names = [
    ("gdn_qkv", base + "0.linear_attn.in_proj_qkv"), ("gdn_z", base + "0.linear_attn.in_proj_z"),
    ("gdn_b", base + "0.linear_attn.in_proj_b"), ("gdn_out", base + "0.linear_attn.out_proj"),
    ("attn_q", base + "3.self_attn.q_proj"), ("attn_k", base + "3.self_attn.k_proj"),
    ("attn_o", base + "3.self_attn.o_proj"), ("indexer", base + "3.self_attn.indexer.index_qk_proj"),
    ("shared_gate", base + "5.mlp.shared_expert.gate_proj"), ("shared_down", base + "5.mlp.shared_expert.down_proj"),
]
rows = [1, 2, 5, 16, 17, 33, 64, 255, 256, 300]
cases = []
g = torch.Generator(device="cpu").manual_seed(7)
for tag, nm in names:
    w = s = None
    for suffix in (".weight", ".weight_scale"):
        with safe_open(str(snap / index[nm + suffix]), "pt", device="cpu") as f:
            t = f.get_tensor(nm + suffix)
        if suffix == ".weight":
            w = t
        else:
            s = t
    assert w.dtype == torch.float8_e4m3fn and s.dtype == torch.uint8, (nm, w.dtype, s.dtype)
    n, k = w.shape
    wd, sd = w.cuda(), s.cuda()
    lin = Mx8Linear.from_checkpoint(wd, sd)
    fold = bool(((sd >= 3) & (sd <= 134)).all())
    x = (torch.randn(max(rows), k, generator=g) * 2).to(torch.bfloat16).cuda()
    deq = w.to(torch.float32).cuda() * torch.exp2(sd.to(torch.int32).sub(127).float()).repeat_interleave(32, dim=1)
    (out / f"{tag}_codes.bin").write_bytes(w.view(torch.uint8).numpy().tobytes())
    (out / f"{tag}_scale.bin").write_bytes(s.numpy().tobytes())
    (out / f"{tag}_x.bin").write_bytes(x.cpu().view(torch.int16).numpy().tobytes())
    worst = 0.0
    for m in rows:
        y = lin(x[:m].contiguous())
        (out / f"{tag}_y{m}.bin").write_bytes(y.cpu().view(torch.int16).numpy().tobytes())
        yp = lin.prefill(x[:m].contiguous())          # prompt rows: prompt.cu's chain when every exponent is in 3..134
        (out / f"{tag}_p{m}.bin").write_bytes(yp.cpu().view(torch.int16).numpy().tobytes())
        ref = x[:m].double() @ deq.double().t()
        err = ((y.double() - ref).abs() / (ref.abs().max() + 1e-30)).max().item()
        worst = max(worst, err)
    emin, emax = int(sd.min()), int(sd.max())
    print(f"{tag} n={n} k={k} e8m0 {emin}..{emax} fold={fold} worst err vs f64 (rel to max) {worst:.3e}", flush=True)
    cases.append({"name": tag, "n": n, "k": k, "rows": rows, "fold": fold})
json.dump({"cases": cases}, open(out / "cases.json", "w"))
