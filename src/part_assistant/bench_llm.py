"""Speed of a local model server (Ollama or vLLM, OpenAI chat-completions format).

    uv run python -m part_assistant.bench_llm                       # the server in local_url()
    PART_ASSISTANT_LOCAL_URL=http://127.0.0.1:8000/v1 PART_ASSISTANT_LOCAL_MODEL=Qwen/Qwen3-4B-AWQ \
        uv run python -m part_assistant.bench_llm --concurrency 1 4

Measures, per prompt size and concurrency:
  - TTFT    time to first token: what the user feels as "is it stuck?". Grows with the
            prompt length (the whole prompt is read before the first token).
  - tok/s   decode speed of one request, after the first token.
  - total   all requests' output tokens / wall time: what the GPU serves when several
            people ask at once (vLLM batches them; this is where KV cache size matters).

Prompts: "short" is one question; "long" is the system prompt plus 5 datasheet passages,
about the size of a real fallback request after search_docs. Thinking is off and vLLM
writes exactly max_tokens (ignore_eos), so every model does the same work; Ollama has no
such switch and may stop earlier. Free: no Claude calls.
Quality is measured separately: eval_agent --local.
"""

import argparse
import json
import os
import statistics
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from part_assistant.agent import SYSTEM
from part_assistant.chunk import load_chunks
from part_assistant.local_agent import DEFAULT_MODEL, local_url

RUNS_DIR = Path("runs")
QUESTION = "PSU 的 hold-up time 是多少？請根據資料回答並附出處。"


def prompts() -> dict[str, list[dict]]:
    passages = [c.text for c in load_chunks() if c.access_level == "public"][100:105]
    context = "\n\n".join(f"[search result {i}]\n{p}" for i, p in enumerate(passages, 1))
    return {
        "short": [{"role": "user", "content": QUESTION}],
        "long": [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": f"{context}\n\n{QUESTION}"}],
    }


def unique(messages: list[dict]) -> list[dict]:
    """A different first line per request. Servers cache a prompt prefix they have seen
    (prefix caching), so repeating one prompt would skip reading it and hide the real
    TTFT; in real use every question brings different search results."""
    first = {**messages[0], "content": f"[request {uuid.uuid4().hex}]\n{messages[0]['content']}"}
    return [first, *messages[1:]]


def one_request(url: str, model: str, messages: list[dict], max_tokens: int) -> dict:
    body = {"model": model, "messages": unique(messages), "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},  # vLLM
            "reasoning_effort": "none",  # Ollama
            "ignore_eos": True}  # vLLM: always write max_tokens, so models compare fairly
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    usage: dict = {}
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:  # server-sent events: "data: {...}"
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            event = json.loads(line[5:])
            if event.get("usage"):
                usage = event["usage"]
            for ch in event.get("choices") or []:
                delta = ch.get("delta") or {}
                if first is None and (delta.get("content") or delta.get("reasoning")
                                      or delta.get("reasoning_content")):
                    first = time.perf_counter()
    end = time.perf_counter()
    out = usage.get("completion_tokens", 0)
    first = first or end
    return {"ttft_s": first - t0, "total_s": end - t0, "prompt_tokens": usage.get(
        "prompt_tokens", 0), "output_tokens": out,
        "tok_s": (out - 1) / (end - first) if out > 1 and end > first else 0.0}


def run(url: str, model: str, messages: list[dict], concurrency: int, rounds: int,
        max_tokens: int) -> dict:
    results: list[dict] = []
    walls: list[float] = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(concurrency) as pool:  # `concurrency` users ask at once
            futures = [pool.submit(one_request, url, model, messages, max_tokens)
                       for _ in range(concurrency)]
            results.extend(f.result() for f in futures)
        walls.append(time.perf_counter() - t0)

    def med(key: str) -> float:
        return round(statistics.median(r[key] for r in results), 3)

    return {"concurrency": concurrency, "requests": len(results),
            "prompt_tokens": results[0]["prompt_tokens"],
            "output_tokens": med("output_tokens"), "ttft_s": med("ttft_s"),
            "tok_s": round(med("tok_s"), 1),
            "total_tok_s": round(sum(r["output_tokens"] for r in results) / sum(walls), 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=None, help="default: PART_ASSISTANT_LOCAL_MODEL")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--rounds", type=int, default=3, help="repeats per setting (median)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--label", default="", help="e.g. 'vllm 4B-AWQ', saved with the numbers")
    args = ap.parse_args()

    model = args.model or os.environ.get("PART_ASSISTANT_LOCAL_MODEL", DEFAULT_MODEL)
    url = local_url()
    sets = prompts()
    one_request(url, model, sets["short"], 8)  # warm-up: load the model, compile graphs
    rows = []
    for name, messages in sets.items():
        for c in args.concurrency:
            row = {"prompt": name, **run(url, model, messages, c, args.rounds,
                                         args.max_tokens)}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    RUNS_DIR.mkdir(exist_ok=True)
    out = RUNS_DIR / f"bench-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"model": model, "url": url, "label": args.label,
                               "max_tokens": args.max_tokens, "rows": rows},
                              ensure_ascii=False, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
