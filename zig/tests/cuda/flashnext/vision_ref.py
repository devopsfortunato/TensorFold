#!/usr/bin/env python3
"""Python TensorFold's Flash Next image and video path as the Zig engine's oracle (TP=1, one GPU).

Each case is a chat request with image or video parts, run the way ``tensorfold.cuda.server`` runs it:
``prepare_images`` (``split_images``, the chat template with image parts, Pillow / PyAV loads, the processor) and
``QwenCudaVision.encode`` (the tower), then the serial engine's ``prefill(..., vision=...)`` with per-chunk digests
(the chunk's final streams and the last row's logits, as the Zig engine's TF_FLASHNEXT_PREFILL_DIGEST prints them),
greedy serial decoding (``serial_decode``) and MTP-drafted decoding (``mtp_decode``) of ``--tokens`` tokens, eager.

Writes ``<out>/<case>/``: ``request.json`` (the OpenAI messages, media as data URLs), ``bundle.json`` (token ids,
feature rows, rope delta, digests, expected tokens) and ``positions.bin`` (i32 [n, 3]) + ``features.bin`` (bf16
[rows, 2560]) for ``tensorfold-native flashnext gate-media``; ``<out>/media/`` the generated images and video.

Needs transformers (5.17.0 as the recipes pin it) and PyAV on the path beside TensorFold's ``src``.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import sys
import time
from pathlib import Path

CASES = ("img1", "img3", "img50", "video10", "tool", "long", "video30")


def module_text(name: str, chars: int) -> str:
    """Python's own source as filler text (as capture.py's long prompts): ``chars`` characters of module ``name``."""

    import importlib

    text = Path(importlib.import_module(name).__file__).read_text()
    return (text * (1 + chars // max(1, len(text))))[:chars]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------------------------------- the media

def scene(w: int, h: int, seed: int, mode: str = "RGB"):
    """A deterministic picture: gradients, shapes and a caption (no fonts beyond Pillow's default)."""

    from PIL import Image, ImageDraw

    img = Image.new(mode, (w, h), (0, 0, 0, 0) if mode == "RGBA" else (0, 0, 0))
    px = img.load()
    for y in range(h):
        for x in range(w):
            r = (x * 255 // max(1, w - 1) + seed * 37) % 256
            g = (y * 255 // max(1, h - 1) + seed * 91) % 256
            b = ((x + y) * 3 + seed * 13) % 256
            px[x, y] = (r, g, b, 160 + (x * y) % 96) if mode == "RGBA" else (r, g, b)
    d = ImageDraw.Draw(img)
    d.ellipse((w // 8, h // 6, w // 2, h // 2 + h // 6), fill=(240, 220, 30) if mode == "RGB" else (240, 220, 30, 255))
    d.rectangle((w // 2, h // 2, w - w // 10, h - h // 10), fill=(20, 60, 200) if mode == "RGB" else (20, 60, 200, 200))
    d.text((8, 8), f"PICTURE {seed}", fill=(255, 255, 255) if mode == "RGB" else (255, 255, 255, 255))
    return img


def encoded(img, fmt: str, **kw) -> bytes:
    out = io.BytesIO()
    img.save(out, format=fmt, **kw)
    return out.getvalue()


def video(seconds: float, fps: int = 24, w: int = 320, h: int = 240) -> bytes:
    """A small MP4 (mpeg4): a ball moving over a gradient, a frame counter."""

    import av
    import numpy as np
    from PIL import ImageDraw

    out = io.BytesIO()
    with av.open(out, mode="w", format="mp4") as box:
        stream = box.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        for i in range(int(seconds * fps)):
            img = scene(w, h, 7)
            d = ImageDraw.Draw(img)
            x = int((w - 60) * i / max(1, seconds * fps - 1))
            d.ellipse((x, h // 3, x + 60, h // 3 + 60), fill=(230, 40, 40))
            d.text((w - 70, h - 20), f"frame {i}", fill=(255, 255, 255))
            frame = av.VideoFrame.from_ndarray(np.asarray(img), format="rgb24")
            for packet in stream.encode(frame):
                box.mux(packet)
        for packet in stream.encode():
            box.mux(packet)
    return out.getvalue()


def data_url(data: bytes, media: str) -> str:
    return f"data:{media};base64,{base64.b64encode(data).decode()}"


def requests(media_dir: Path) -> dict[str, dict]:
    media_dir.mkdir(parents=True, exist_ok=True)

    def keep(name: str, data: bytes) -> bytes:
        (media_dir / name).write_bytes(data)
        return data

    def image_part(data: bytes, media: str, detail: str | None = None) -> dict:
        url = {"url": data_url(data, media)}
        if detail:
            url["detail"] = detail
        return {"type": "image_url", "image_url": url}

    one = keep("one.jpg", encoded(scene(640, 480, 1), "JPEG", quality=90))
    png = keep("alpha.png", encoded(scene(300, 500, 2, "RGBA"), "PNG"))
    webp = keep("wide.webp", encoded(scene(900, 300, 3), "WEBP", quality=85))
    small = [keep(f"small{i:02d}.png", encoded(scene(48 + i % 5 * 8, 40 + i % 7 * 6, 10 + i), "PNG"))
             for i in range(50)]
    clip = keep("clip10.mp4", video(10.0))
    clip30 = keep("clip30.mp4", video(30.0, fps=30, w=640, h=360))
    big = keep("big.jpg", encoded(scene(1280, 960, 4), "JPEG", quality=92))
    weather = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
                                                 "parameters": {"type": "object", "properties": {
                                                     "city": {"type": "string"}}, "required": ["city"]}}}]
    return {
        "img1": {"messages": [{"role": "user", "content": [image_part(one, "image/jpeg"),
                                                           {"type": "text", "text": "Describe this picture."}]}]},
        "img3": {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "Compare these three pictures."}, image_part(one, "image/jpeg"),
            image_part(png, "image/png", "high"), image_part(webp, "image/webp", "low")]}]},
        "img50": {"messages": [{"role": "user", "content": [image_part(d, "image/png") for d in small]
                                + [{"type": "text", "text": "How many pictures are there? Name the numbers."}]}]},
        "video10": {"messages": [{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": data_url(clip, "video/mp4")}},
            {"type": "text", "text": "What happens in this video?"}]}]},
        # ~5k tokens: the image straddles the 2048-row prompt chunks, the indexer's sparse attention is on
        "long": {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "Read this module, then the picture, then more of the module.\n\n"
                                     + module_text("textwrap", 7000)},
            image_part(big, "image/jpeg"),
            {"type": "text", "text": module_text("inspect", 8000) + "\n\nWhat does the picture show?"}]}]},
        "video30": {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "Describe the video in detail."},
            {"type": "video_url", "video_url": {"url": data_url(clip30, "video/mp4")}}]}]},
        "tool": {"tools": weather, "messages": [
            {"role": "user", "content": "Look at the weather screenshot tool and tell me what it shows."},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {
                "name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": [
                {"type": "text", "text": "screenshot:"}, image_part(one, "image/jpeg")]}]},
    }


# ---------------------------------------------------------------------------------------------------- the runs

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="comma list of cases")
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--context", type=int, default=65536)
    ap.add_argument("--tools", default="", help="folder of triton_aot_manifest.py: record every Triton launch")
    ap.add_argument("--kv-dtype", default="bf16", choices=("bf16", "fp8"), help="the attention caches' format")
    ap.add_argument("--no-engine", action="store_true", help="prepare and encode only (no language model)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = None
    if a.tools:
        sys.path.insert(0, a.tools)
        import triton_aot_manifest as aot

        rec = aot.Recorder().install()
    import numpy as np
    import torch

    from tensorfold.cuda.chat_template import ChatTemplate
    from tensorfold.server.prompts import prepare_images
    from tensorfold.vision.native_helper import Helper

    model = Path(a.model)
    cases = {k: v for k, v in requests(out / "media").items() if not a.only or k in a.only.split(",")}
    eng = None
    if not a.no_engine:
        from tensorfold.families.qwen4_exp import cuda_engine

        eng = cuda_engine(model, context=a.context, context_explicit=True, kv_dtype=a.kv_dtype)
    # the tower and frontend as the helper builds them (Flash Next's limits), on the same GPU
    helper = Helper(str(model), allow_urls=False, max_images=50, image_tokens=16384, workspace=0)
    tpl = ChatTemplate(model)
    info = {"ready": helper.ready(), "cases": {}}
    print(json.dumps(info["ready"]), flush=True)
    for name, req in cases.items():
        d = out / name
        d.mkdir(exist_ok=True)
        (d / "request.json").write_text(json.dumps(req) + "\n")

        def render(messages, tools=req.get("tools")):
            return tpl.render(messages, tools=tools, enable_thinking=False, allow_images=True)

        t0 = time.perf_counter()
        prepared = prepare_images(helper.vision, req["messages"], render, context_limit=a.context,
                                  limits=helper.limits).vision
        t1 = time.perf_counter()
        enc = helper.vision.encode(prepared, prepared.token_ids)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        feats = enc.features.contiguous().view(torch.int16).cpu().numpy().tobytes()
        positions = enc.positions.t().contiguous().cpu().numpy().astype("<i4").tobytes()
        tokens = [int(t) for t in prepared.token_ids]
        (d / "features.bin").write_bytes(feats)
        (d / "positions.bin").write_bytes(positions)
        bundle = {"case": name, "tokens": tokens, "rows": [int(r) for r in enc.rows], "delta": int(enc.rope_delta),
                  "width": int(enc.features.shape[1]), "features_sha256": sha(feats),
                  "positions_sha256": sha(positions), "tokens_sha256": sha(np.asarray(tokens, "<u4").tobytes()),
                  "images": len(prepared.image_spans), "video_groups": len(prepared.video_spans),
                  "rendered": render(__import__("tensorfold.vision.images", fromlist=["split_images"]).split_images(
                      req["messages"], limits=helper.limits, allow_videos=True)[0]),
                  "prepare_s": round(t1 - t0, 3), "encode_s": round(t2 - t1, 3)}
        if eng is not None:
            bundle["expected"] = run_engine(eng, tokens, enc, a.tokens)
        (d / "bundle.json").write_text(json.dumps(bundle) + "\n")
        info["cases"][name] = {k: v for k, v in bundle.items() if k not in ("tokens", "rows", "rendered")}
        print(json.dumps(info["cases"][name]), flush=True)
        del enc
        torch.cuda.empty_cache()
    if rec is not None:
        rec.dump(out / "launches.json")
    (out / "results.json").write_text(json.dumps(info, indent=1) + "\n")
    return 0


def run_engine(eng, prompt: list[int], enc, count: int) -> dict:
    """prefill with the image rows and rotary table, then serial and drafted greedy decoding (eager)."""

    import torch

    from tensorfold.families.qwen4_exp.cuda import decode

    e = eng.e
    graphs, e.graphs = e.graphs, None                       # eager: the same bits, no capture of image launches
    digests: list[dict] = []
    inner = decode.forward

    def traced(w, st, b, tokens, **kw):
        logits = inner(w, st, b, tokens, **kw)
        if b is e.pbuf:
            R = len(tokens)
            torch.cuda.synchronize()
            row = {"rows": R, "streams": hashlib.sha256(b.streams[:R].contiguous().view(torch.int16).cpu().numpy()
                                                        .tobytes()).hexdigest()[:16]}
            if kw.get("logits", True) and logits is not None:
                row["logits"] = hashlib.sha256(logits[-1:].contiguous().view(torch.int16).cpu().numpy()
                                               .tobytes()).hexdigest()[:16]
            digests.append(row)
        return logits

    decode.forward = traced
    try:
        t0 = time.perf_counter()
        first = decode.prefill(e, prompt, None, mtp=False, vision=enc)
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - t0
        decode.forward = inner
        serial = decode.serial_decode(e, first, count, None).tokens
        chunk_digests = list(digests)
        digests.clear()
        decode.forward = traced
        first2 = decode.prefill(e, prompt, None, vision=enc)
        decode.forward = inner
        drafted = decode.mtp_decode(e, first2, count, None, depth=eng.depth, confidence=eng.confidence)
    finally:
        decode.forward = inner
        e.graphs = graphs
        e.reset()
    return {"first": int(first), "serial": [int(t) for t in serial], "drafted": [int(t) for t in drafted.tokens],
            "drafted_eq_serial": list(serial) == list(drafted.tokens), "digests": chunk_digests,
            "repeat_digests_equal": chunk_digests == digests, "prefill_s": round(prefill_s, 3),
            "rounds": drafted.rounds, "accepted": drafted.accepted, "drafted_n": drafted.drafted}


if __name__ == "__main__":
    sys.exit(main())
