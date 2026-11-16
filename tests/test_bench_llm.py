"""bench_llm against a tiny fake server that streams like Ollama / vLLM. No GPU, no cost."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from part_assistant import bench_llm

EVENTS = [
    {"choices": [{"delta": {"role": "assistant"}}]},  # no text yet: not the first token
    {"choices": [{"delta": {"content": "hold"}}]},
    {"choices": [{"delta": {"content": "-up"}}]},
    {"choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": 3}},
]


class Fake(BaseHTTPRequestHandler):
    bodies: list[dict] = []

    def do_POST(self):  # noqa: N802 - http.server's name
        Fake.bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for e in EVENTS:
            self.wfile.write(b"data: " + json.dumps(e).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args):
        pass


@pytest.fixture
def url():
    server = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
    server.shutdown()


def test_one_streamed_request_is_measured(url):
    r = bench_llm.one_request(url, "m", [{"role": "user", "content": "q"}], 16)
    assert (r["prompt_tokens"], r["output_tokens"]) == (40, 3)  # from the usage event
    assert 0 < r["ttft_s"] <= r["total_s"]
    body = Fake.bodies[-1]
    assert body["stream"] and body["stream_options"] == {"include_usage": True}
    assert body["chat_template_kwargs"] == {"enable_thinking": False}  # same speed setting
    assert body["max_tokens"] == 16 and body["temperature"] == 0 and body["ignore_eos"]
    assert body["messages"][0]["content"].startswith("[request ")


def test_every_request_starts_differently_so_no_prompt_cache_hides_the_ttft():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "q"}]
    a, b = bench_llm.unique(msgs), bench_llm.unique(msgs)
    assert a[0]["content"] != b[0]["content"] and a[0]["content"].endswith("\nS")
    assert a[1] == msgs[1] and msgs[0]["content"] == "S"  # the input is not changed


def test_concurrent_requests_are_all_counted(url):
    row = bench_llm.run(url, "m", [{"role": "user", "content": "q"}], concurrency=4,
                        rounds=2, max_tokens=16)
    assert row["requests"] == 8 and row["output_tokens"] == 3
    assert row["total_tok_s"] > 0


def test_long_prompt_holds_the_system_prompt_and_five_passages():
    pytest.importorskip("numpy")
    try:
        sets = bench_llm.prompts()
    except FileNotFoundError:
        pytest.skip("chunks.jsonl not built")
    assert sets["long"][0]["role"] == "system"
    assert sets["long"][1]["content"].count("[search result") == 5
    assert len(sets["long"][1]["content"]) > 5 * len(sets["short"][0]["content"])
