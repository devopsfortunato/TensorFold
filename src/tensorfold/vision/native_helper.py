"""The Zig engine's vision helper: Python TensorFold's own image and video path, as a child process of
``tensorfold-native serve --vision``.

The Zig server renders the chat template (image and video parts as ``{"type": "image"}`` / ``{"type": "video"}``)
and sends each media request here; this process runs exactly what ``tensorfold.server.prompts.prepare_images`` runs
after its ``split_images``: the sources' checks, ``load_images`` (Pillow), ``load_videos`` (PyAV), the processor's
``prepare`` (transformers' image processor, the HF tokenizer, the rotary positions) and ``QwenCudaVision.encode``
(the 27-layer tower on the GPU). So the token ids, positions and visual embeddings are Python's bits by
construction; the Zig engine only moves them into the prompt rows.

Frames on stdin / stdout, little endian. Request: u32 length, then JSON
``{"prompt": str, "media": [{"kind": "image"|"video", "url": str, "detail": str}], "max_prompt_tokens": int|null}``.
Reply: u32 header length, the header JSON, u64 payload length, the payload: token ids u32 [n], feature rows u32 [k]
(prompt positions, ascending), rotary positions i32 [n, 3], features bf16 [k, width]. An error reply is a header
``{"ok": false, "status": 400|503, "error": message}`` and an empty payload. The first frame written is the
ready header (``{"ready": true, ...}``) once the tower is loaded and warm. Logs go to stderr.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from dataclasses import replace

# Flash Next's request limits (the single-Spark recipe's patches 0008 and 0009 by MiaAI-Lab): up to 50 images
# sharing 16,384 visual tokens (each at most 4,096), 10 MB an image, 64 MB and 128 Mi pixels in all; up to two
# videos (user messages), 64 MB each, 96 MB in all
MAX_IMAGES, IMAGE_TOKENS = 50, 16384
IMAGES_TOTAL_BYTES, IMAGES_TOTAL_PIXELS = 64 * 1024 * 1024, 128 * 1024 * 1024
VIDEO_BYTES, VIDEOS_TOTAL_BYTES = 64 * 1024 * 1024, 96 * 1024 * 1024


def _log(message: str) -> None:
    print(f"[vision-helper] {message}", file=sys.stderr, flush=True)


def _read_exact(stream, n: int) -> bytes | None:
    data = bytearray()
    while len(data) < n:
        chunk = stream.read(n - len(data))
        if not chunk:
            return None
        data += chunk
    return bytes(data)


def _write(out, header: dict, payload: list[bytes] = ()) -> None:
    head = json.dumps(header, separators=(",", ":")).encode()
    size = sum(len(p) for p in payload)
    out.write(struct.pack("<I", len(head)) + head + struct.pack("<Q", size))
    for p in payload:
        out.write(p)
    out.flush()


class Helper:
    def __init__(self, model: str, *, allow_urls: bool, max_images: int, image_tokens: int, workspace: int, max_videos: int = 4) -> None:
        import torch

        from .images import DEFAULT_LIMITS
        from .qwen_cuda import QwenCudaVision
        from .videos import DEFAULT_VIDEO_LIMITS

        self.torch = torch
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)
        started = time.perf_counter()
        self.vision = QwenCudaVision(model, device, allow_urls=allow_urls)
        self.allow_urls = allow_urls
        self.limits = replace(DEFAULT_LIMITS, max_images=max_images, max_visual_tokens=image_tokens,
                              max_total_encoded_bytes=IMAGES_TOTAL_BYTES, max_total_pixels=IMAGES_TOTAL_PIXELS)
        self.video_limits = replace(DEFAULT_VIDEO_LIMITS, max_videos=max_videos, max_encoded_bytes=VIDEO_BYTES,
                                    max_total_encoded_bytes=VIDEOS_TOTAL_BYTES)
        if workspace > 0:                   # the caching allocator stays inside the tower and the reserved workspace
            total = torch.cuda.get_device_properties(0).total_memory
            cap = self.vision.weight_bytes + workspace + (256 << 20)
            torch.cuda.set_per_process_memory_fraction(min(1.0, cap / total), 0)
        self.vision.warm()
        torch.cuda.empty_cache()
        self.load_s = time.perf_counter() - started

    def ready(self) -> dict:
        import numpy
        import PIL
        import transformers

        try:
            import av
            av_version = av.__version__
        except ImportError:
            av_version = None
        raw = self.vision.frontend.config
        return {"ready": True, "selftest_sha256": self.selftest(), "pid": os.getpid(), "videos": bool(self.vision.videos),
                "tower_bytes": int(self.vision.weight_bytes), "width": int(self.vision.config["out_hidden_size"]),
                "image_token_id": int(raw["image_token_id"]), "video_token_id": int(raw.get("video_token_id", -1)),
                "load_s": round(self.load_s, 3), "reserved": int(self.torch.cuda.memory_reserved()),
                "versions": {"torch": self.torch.__version__, "transformers": transformers.__version__,
                             "pillow": PIL.__version__, "numpy": numpy.__version__, "av": av_version}}

    def selftest(self) -> str:
        """The features sha of a fixed 96x64 PNG through the whole path (Pillow decode, processor, tower): equal
        wherever the stack (Python packages, CUDA libraries, GPU) is the reference's."""

        import base64
        import io

        import numpy as np
        from PIL import Image

        y, x = np.mgrid[0:64, 0:96]
        pixels = np.stack([(x * 255 // 95), (y * 255 // 63), ((x + y) * 3) % 256], axis=-1).astype(np.uint8)
        out = io.BytesIO()
        Image.fromarray(pixels, "RGB").save(out, format="PNG")
        front = self.vision.frontend
        prompt = f"{front.vision_start or ''}{front.image_token}{front.vision_end or ''}What is this?"
        header, _ = self.handle({"prompt": prompt, "media": [
            {"kind": "image", "url": "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()}]})
        return header["features_sha256"]

    def prepare(self, request: dict):
        """``prepare_images`` after ``split_images``: the same checks, loads, processor call and tower call."""

        from .images import ImageInputError, ImageSource, _check_source, load_images
        from .videos import load_videos, video_source

        prompt, media = request.get("prompt"), request.get("media") or []
        if not isinstance(prompt, str) or not isinstance(media, list) or not media:
            raise ImageInputError("a media request needs its rendered prompt and its images or videos")
        sources = []
        for item in media:
            kind = item.get("kind") if isinstance(item, dict) else None
            if kind == "image":
                if sum(isinstance(s, ImageSource) for s in sources) >= self.limits.max_images:
                    raise ImageInputError(
                        f"a request supports at most {self.limits.max_images} images across the full message "
                        "history, including prior turns; remove older image content, start a new conversation, "
                        "or restart the server with --vision-max-images N to raise the count limit (other image "
                        "limits still apply)")
                source = ImageSource(item.get("url") or "", item.get("detail") or "auto")
                _check_source(source, self.limits, self.allow_urls)
                sources.append(source)
            elif kind == "video":
                if not self.vision.videos:
                    raise ImageInputError("this checkpoint's frontend encodes images only")
                if sum(not isinstance(s, ImageSource) for s in sources) >= self.video_limits.max_videos:
                    raise ImageInputError(f"a request supports at most {self.video_limits.max_videos} videos; send fewer, or restart the server with --vision-max-videos N to raise the count limit")
                sources.append(video_source({"url": item.get("url")}, self.video_limits, self.allow_urls))
            else:
                raise ImageInputError("media parts must be images or videos")
        images = load_images([s for s in sources if isinstance(s, ImageSource)], limits=self.limits,
                             allow_urls=self.allow_urls)
        budget = {"max_visual_tokens": self.limits.max_visual_tokens}
        clips = [s for s in sources if not isinstance(s, ImageSource)]
        if clips:
            budget["videos"] = load_videos(clips, self.vision.video_size, limits=self.video_limits,
                                           allow_urls=self.allow_urls)
        limit = request.get("max_prompt_tokens")
        return self.vision.prepare(prompt, images, max_prompt_tokens=int(limit) if limit else None, **budget)

    def handle(self, request: dict) -> tuple[dict, list[bytes]]:
        import numpy as np

        t0 = time.perf_counter()
        prepared = self.prepare(request)
        t1 = time.perf_counter()
        encoded = self.vision.encode(prepared, prepared.token_ids)
        torch = self.torch
        features = encoded.features.contiguous().view(torch.int16).cpu().numpy().tobytes()
        positions = encoded.positions.t().contiguous().cpu().numpy().astype("<i4").tobytes()
        tokens = np.asarray(prepared.token_ids, dtype="<u4").tobytes()
        rows = np.asarray(encoded.rows, dtype="<u4").tobytes()
        t2 = time.perf_counter()
        del encoded
        torch.cuda.empty_cache()
        header = {"ok": True, "tokens": len(prepared.token_ids), "rows": len(rows) // 4,
                  "delta": int(prepared.rope_delta), "width": int(self.vision.config["out_hidden_size"]),
                  "images": len(prepared.image_spans), "video_groups": len(prepared.video_spans),
                  "videos": len(prepared.video_hashes), "visual_tokens": int(prepared.visual_tokens),
                  "features_sha256": hashlib.sha256(features).hexdigest(),
                  "tokens_sha256": hashlib.sha256(tokens).hexdigest(),
                  "positions_sha256": hashlib.sha256(positions).hexdigest(),
                  "prepare_s": round(t1 - t0, 4), "encode_s": round(t2 - t1, 4),
                  "peak": int(torch.cuda.max_memory_reserved())}
        torch.cuda.reset_peak_memory_stats()
        return header, [tokens, rows, positions, features]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--urls", action="store_true", help="also fetch public https:// images and videos")
    ap.add_argument("--max-images", type=int, default=int(os.environ.get("TENSORFOLD_MAX_IMAGES") or MAX_IMAGES))
    ap.add_argument("--max-videos", type=int, default=int(os.environ.get("TENSORFOLD_MAX_VIDEOS") or 4))
    ap.add_argument("--image-tokens", type=int,
                    default=int(os.environ.get("TENSORFOLD_IMAGE_TOKENS") or IMAGE_TOKENS))
    ap.add_argument("--workspace-mib", type=int,
                    default=int(os.environ.get("TENSORFOLD_VISION_WORKSPACE_MIB") or 4096))
    a = ap.parse_args()
    if not 1 <= a.max_videos <= 64 or not 1 <= a.max_images <= 256 or not 1 <= a.image_tokens <= 65536 or not 0 <= a.workspace_mib <= 16384:
        print("--max-images 1..256, --image-tokens 1..65536, --workspace-mib 0..16384", file=sys.stderr)
        return 2
    out = sys.stdout.buffer
    sys.stdout = sys.stderr                  # nothing but frames on the pipe
    try:
        helper = Helper(a.model, allow_urls=a.urls, max_images=a.max_images, max_videos=a.max_videos, image_tokens=a.image_tokens,
                        workspace=a.workspace_mib << 20)
    except Exception as exc:  # noqa: BLE001 - the parent reads why
        _write(out, {"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1
    _write(out, helper.ready())
    _log(f"ready: tower {helper.vision.weight_bytes / 2**30:.2f} GiB, loaded in {helper.load_s:.1f} s")
    from tensorfold.server.errors import CONTEXT_LIMIT

    from .images_http import ImageInputError

    inp = sys.stdin.buffer
    while True:
        raw = _read_exact(inp, 4)
        if raw is None:
            return 0
        body = _read_exact(inp, struct.unpack("<I", raw)[0])
        if body is None:
            return 0
        try:
            header, payload = helper.handle(json.loads(body))
        except (ImageInputError, ValueError, ImportError) as exc:
            message = str(exc)
            code = "context_length_exceeded" if message.startswith(CONTEXT_LIMIT) else None
            header, payload = {"ok": False, "status": 400, "error": message, "code": code}, []
        except Exception as exc:  # noqa: BLE001 - one request fails, the helper serves on
            helper.torch.cuda.empty_cache()
            _log(f"request failed: {type(exc).__name__}: {exc}")
            header, payload = {"ok": False, "status": 503, "error": f"vision processing failed ({type(exc).__name__})"}, []
        _write(out, header, payload)


if __name__ == "__main__":
    sys.exit(main())
