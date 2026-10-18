#!/usr/bin/env python3
"""The served image and video path against Python TensorFold's: each vision_ref.py case (``<dir>/<case>/request.json``,
its ``bundle.json`` expected tokens) sent to a running ``tensorfold-native serve --vision`` as an OpenAI chat request
(temperature 0, thinking off, ``ignore_eos``, as many tokens as the oracle decoded), drafted and with ``"draft":
false``; the reply's ``runtime.token_sha`` must be the sha of Python's tokens. Then every case at once (concurrent ==
solo). ``--text``: capture.py's text prompts as well, against a capture's serial greedy tokens (the text path
byte-identical with vision on). Standard library only.

    python3 vision_serve.py --url http://127.0.0.1:8888 --cases out/v1r1 [--text out/fncap1] [--report out.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TEXTS = {
    "code": "Write a Python function that merges two sorted lists, with a docstring and three tests.",
    "story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "facts": "Explain how a refrigerator moves heat out of its cabinet, step by step.",
}


def token_sha(tokens) -> str:
    return hashlib.sha256(",".join(str(int(t)) for t in tokens).encode()).hexdigest()[:12]


def post(url: str, body: dict, timeout: float = 1800) -> tuple[dict, float]:
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()), time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        return {"http_error": e.code, "body": e.read().decode(errors="replace")[:2000]}, time.perf_counter() - t0


def body_of(req: dict, n: int, draft: bool) -> dict:
    b = {"model": "x", "messages": req["messages"], "max_tokens": n, "temperature": 0, "ignore_eos": True,
         "chat_template_kwargs": {"enable_thinking": False}}
    if req.get("tools"):
        b["tools"] = req["tools"]
    if not draft:
        b["draft"] = False
    return b


def data_url(path: Path, media: str) -> str:
    import base64

    return f"data:{media};base64,{base64.b64encode(path.read_bytes()).decode()}"


def bench(url: str, media: Path, reps: int) -> dict:
    """TTFT (the server's time_to_first_token: helper decode + preprocess + tower + prefill + first draw) of one
    1280x960 image, ten 640x480 images and a 30 s video (vision_ref.py's media), one reply token, ``reps`` times."""

    one = {"type": "image_url", "image_url": {"url": data_url(media / "big.jpg", "image/jpeg")}}
    small = {"type": "image_url", "image_url": {"url": data_url(media / "one.jpg", "image/jpeg")}}
    clip = {"type": "video_url", "video_url": {"url": data_url(media / "clip30.mp4", "video/mp4")}}
    kinds = {"1 image": [one], "10 images": [small] * 10, "30 s video": [clip]}
    out = {}
    for name, parts in kinds.items():
        times = []
        for _ in range(reps):
            body = {"model": "x", "max_tokens": 1, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "user", "content": parts + [{"type": "text", "text": "Describe it."}]}]}
            got, dt = post(url, body)
            rt = got.get("tensorfold") or got.get("runtime") or {}
            times.append({"ttft": rt.get("time_to_first_token"), "prefill_s": rt.get("prefill_seconds"),
                          "prompt_tokens": (got.get("usage") or {}).get("prompt_tokens"), "wall": round(dt, 3),
                          "error": got.get("http_error")})
        out[name] = times
        print(f"bench {name}: " + ", ".join(f"ttft {t['ttft']} prefill {t['prefill_s']} prompt {t['prompt_tokens']} "
                                           f"wall {t['wall']}" for t in times), flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--cases", action="append", default=[])
    ap.add_argument("--only", default="")
    ap.add_argument("--text", default="", help="a capture dir (results.json): its text prompts' serial greedy tokens")
    ap.add_argument("--text-tokens", type=int, default=64)
    ap.add_argument("--report", default="")
    ap.add_argument("--bench", default="", help="vision_ref.py's media dir: TTFT of 1 image, 10 images, a 30 s video")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    if a.bench and not (a.cases or a.text):
        r = bench(a.url, Path(a.bench), a.reps)
        if a.report:
            Path(a.report).write_text(json.dumps(r, indent=1) + "\n")
        return 0
    cases = {}
    for root in a.cases:
        for d in sorted(Path(root).iterdir()):
            if (d / "bundle.json").exists() and (not a.only or d.name in a.only.split(",")):
                b = json.loads((d / "bundle.json").read_text())
                if "expected" in b:
                    cases[d.name] = (json.loads((d / "request.json").read_text()), b)
    results, bad = {}, 0

    def check(name, req, want, draft):
        n = len(want)
        got, dt = post(a.url, body_of(req, n, draft))
        if "http_error" in got:
            return {"ok": False, "error": got, "seconds": dt}
        rt = got.get("tensorfold") or got.get("runtime") or {}
        usage = got.get("usage") or {}
        return {"ok": rt.get("token_sha") == token_sha(want), "sha": rt.get("token_sha"), "want": token_sha(want),
                "completion_tokens": usage.get("completion_tokens"), "prompt_tokens": usage.get("prompt_tokens"),
                "ttft": rt.get("time_to_first_token"), "prefill_s": rt.get("prefill_seconds"), "seconds": dt,
                "tps": rt.get("tokens_per_second")}

    for name, (req, b) in cases.items():
        want = b["expected"]["serial"]
        for draft in (True, False):
            r = check(name, req, want, draft)
            r["prompt_tokens_python"] = len(b["tokens"])
            results[f"{name}/{'drafted' if draft else 'serial'}"] = r
            bad += not r["ok"]
            print(f"{'EQUAL' if r['ok'] else 'DIFFER'} served {name}/{'drafted' if draft else 'serial'}: sha "
                  f"{r.get('sha')} want {r.get('want')} prompt {r.get('prompt_tokens')} (python {len(b['tokens'])}) "
                  f"ttft {r.get('ttft')} s {r.get('seconds', 0):.2f}" + (f" ERROR {r['error']}" if 'error' in r else ""),
                  flush=True)
    text = {}
    if a.text:
        res = json.loads((Path(a.text) / "results.json").read_text())["results"]
        for name, user in TEXTS.items():
            want = res[f"serial-greedy/{name}"]["tokens"][:a.text_tokens]
            text[name] = ({"messages": [{"role": "user", "content": user}]}, want)
            r = check(name, text[name][0], want, True)
            results[f"text-{name}"] = r
            bad += not r["ok"]
            print(f"{'EQUAL' if r['ok'] else 'DIFFER'} served text {name}: sha {r.get('sha')} want {r.get('want')}",
                  flush=True)
    # every case at once (and the text prompts with them): each reply as alone
    jobs = [(n, req, b["expected"]["serial"]) for n, (req, b) in cases.items()] + \
           [(f"text-{n}", req, want) for n, (req, want) in text.items()]
    if len(jobs) > 1:
        with ThreadPoolExecutor(len(jobs)) as pool:
            got = list(pool.map(lambda j: check(j[0], j[1], j[2], True), jobs))
        for (n, _, _), r in zip(jobs, got):
            results[f"together/{n}"] = r
            bad += not r["ok"]
            print(f"{'EQUAL' if r['ok'] else 'DIFFER'} served together {n}: sha {r.get('sha')} want {r.get('want')}",
                  flush=True)
    if a.bench:  # after the checks: the TTFTs of 1 image, 10 images, a 30 s video
        results["bench"] = bench(a.url, Path(a.bench), a.reps)
    if a.report:
        Path(a.report).write_text(json.dumps(results, indent=1) + "\n")
    print(f"{'PASS' if bad == 0 else 'FAIL'} vision serve: {len(results) - bad - ('bench' in results)} of {len(results) - ('bench' in results)} equal", flush=True)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
