"""The server streams exactly the text it returns whole: its incremental decoder, and the tool calls it holds back."""

import random
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

tokenizers = pytest.importorskip("tokenizers")

from tensorfold.cuda.server import App, StreamDecoder

def _tokenizer() -> Path:
    from tensorfold import hub

    found = hub.cached("Vontra/Qwen3.8-27B-MLX-4bit")
    return found / "tokenizer.json" if found is not None else Path("/nonexistent")


TOKENIZER = _tokenizer()


@pytest.mark.skipif(not TOKENIZER.exists(), reason="needs the Qwen3.8-27B tokenizer (tensorfold pull Vontra/Qwen3.8-27B-MLX-4bit)")
def test_stream_text_matches_full_decode():
    tok = tokenizers.Tokenizer.from_file(str(TOKENIZER))
    text = ("def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n"
            "Matrix multiplication 矩阵乘法 uses GPUs 🚀🚀 — naïve café, 数学 and emoji 👩‍💻.\n") * 3
    ids = tok.encode(text, add_special_tokens=False).ids + [248044, 11, 12]
    eos = (248044,)
    rng = random.Random(0)
    for _ in range(20):
        stream = StreamDecoder(tok, eos)
        i = 0
        while i < len(ids):
            step = rng.randint(1, 9)
            shown = stream.add(ids[i:i + step])
            i += step
            kept = [t for t in ids[:i] if t not in eos]
            full = tok.decode(kept, skip_special_tokens=False)
            assert full.startswith(shown)                  # never shows text the full decode lacks
            assert "�" not in shown
        assert stream.final() == tok.decode([t for t in ids if t not in eos], skip_special_tokens=False)


class _Tok:
    """One character per token (id = ord + 1), eos 0."""

    def encode(self, text: str, add_special_tokens: bool = False):
        return SimpleNamespace(ids=[ord(c) + 1 for c in text])

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(chr(t - 1) for t in ids)


class _Template:
    def render(self, messages, *, tools, enable_thinking, extra=None) -> str:
        return "prompt"


class _Engine:
    """The scripted reply in token chunks, cut at ``max_tokens`` like a real decode."""

    eos = (0,)

    def __init__(self, reply: str, chunk: int):
        self.ids, self.chunk = [ord(c) + 1 for c in reply], chunk

    def generate(self, prompt, max_tokens, sampling, on_tokens, **kwargs):
        ids = self.ids[:max_tokens] + ([0] if len(self.ids) < max_tokens else [])
        for i in range(0, len(ids), self.chunk):
            if on_tokens(ids[i:i + self.chunk]):
                break
        return {}


TOOLS = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
CASES = [
    ('Before.<tool_call>{"name":"other","arguments":{}}</tool_call>Tail.',
     'Before.<tool_call>{"name":"other","arguments":{}}</tool_call>Tail.', None, []),
    ('Before.<tool_call>{invalid}</tool_call>Tail.', 'Before.<tool_call>{invalid}</tool_call>Tail.', None, []),
    ('Before.<TOOL_CALL>{"name":"read","arguments":{"path":"a"}}</TOOL_CALL>Tail.', "Before.Tail.", None, ["read"]),
    ('\nWriting it.\n<tool_call>\n<function=read>\n<parameter=path>\nnotes.txt',
     "Writing it.\n<tool_call>\n<function=read>", 40, []),
    ('<tool_call>{"name":"other","arguments":{}}</tool_call>Between.<tool_call>{"name":"read","argu',
     '<tool_call>{"name":"other","arguments":{}}</tool_call>Between.<tool_call>{"name"', 80, []),
    ("\n\nHello.\n", "Hello.", None, []),
    ('Sure.\n<tool_call>{"name":"read","arguments":{"path":"a"}}</tool_call>', "Sure.", None, ["read"]),
]


@pytest.mark.parametrize("chunk", [1, 3])
@pytest.mark.parametrize("reply, whole, limit, names", CASES)
def test_streamed_content_is_the_whole_reply(reply: str, whole: str, limit: int | None, names: list[str],
                                             chunk: int) -> None:
    app = App.__new__(App)
    app.engine, app.served, app.tok, app.template = _Engine(reply, chunk), "fake", _Tok(), _Template()
    app.default_thinking = False
    app.sampling = {"temperature": 0, "top_k": 1, "top_p": 1.0}
    app.max_tokens, app.lock = 4096, threading.Lock()
    body: dict = {"messages": [{"role": "user", "content": "x"}], "tools": TOOLS}
    if limit:
        body["max_tokens"] = limit
    deltas: list[dict] = []
    result = app.run(body, True, lambda delta: deltas.append(delta) or True)
    streamed = "".join(d.get("content", "") for d in deltas) + result["final"].get("content", "")
    assert streamed == result["content"] == whole
    assert any(d.get("content") for d in deltas)      # streamed as it came, not only in the final chunk
    assert [c["function"]["name"] for c in result["calls"] or []] == names
