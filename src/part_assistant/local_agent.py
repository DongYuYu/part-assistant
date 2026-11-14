"""Fallback when Claude is unavailable: the same kind of loop, on a local Ollama model.

Measured on the golden set (RTX 5070 Laptop, see design/routing.md): qwen3:4b with
thinking scores 29/48 vs Claude 48/48, and once claimed it had opened a ticket without
calling the tool. So in fallback mode the local model gets READ-ONLY tools: create_ticket
is not offered, and refused here even if the model calls it anyway.

Settings (environment variables):
    PART_ASSISTANT_LOCAL_MODEL   default qwen3:4b
    OLLAMA_HOST                  default 127.0.0.1:11434 (same variable the ollama CLI uses)

No extra package: Ollama's HTTP API is called with the standard library.
"""

import json
import os
import re
import time
import urllib.request
import uuid
from collections.abc import Callable

from part_assistant.agent import MAX_STEPS, SYSTEM, AgentResult
from part_assistant.tools import TOOL_SCHEMAS, Tools

DEFAULT_MODEL = "qwen3:4b"
READ_ONLY = ("search_docs", "lookup_part", "check_stock")

SYSTEM_LOCAL = SYSTEM + """
8. Tickets can NOT be created right now (the main model is unavailable). If the user asks
   for a ticket, say it cannot be opened now and ask them to try again later."""

# Anthropic tool format -> Ollama ("function") format; the JSON schema inside is the same.
OLLAMA_TOOLS = [{"type": "function", "function": {
    "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
    for t in TOOL_SCHEMAS if t["name"] in READ_ONLY]


def ollama_url() -> str:
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    if "://" not in host:
        host = "http://" + host
    return host.rstrip("/") + "/api/chat"


def ollama_chat(body: dict) -> dict:
    req = urllib.request.Request(ollama_url(), json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def strip_reasoning(content: str) -> str:
    """Reasoning must never reach the user. With think=true Ollama puts it in
    message.thinking, but a thinking-only model with think=false leaks it into content,
    with the opening <think> eaten by the chat template: keep only what follows </think>."""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[1]
    return content.strip()


class LocalAgent:
    def __init__(self, tools: Tools, model: str | None = None, think: bool = True,
                 num_ctx: int = 16384, max_steps: int = MAX_STEPS,
                 chat: Callable[[dict], dict] = ollama_chat):  # tests pass a fake chat
        self.tools = tools
        self.model = model or os.environ.get("PART_ASSISTANT_LOCAL_MODEL", DEFAULT_MODEL)
        self.think = think  # measured: 29/48 with thinking, 18/48 without
        self.num_ctx = num_ctx  # 5 search passages can be several thousand tokens
        self.max_steps = max_steps
        self.chat = chat

    def _run_tool(self, name: str, args: dict) -> dict:
        if name not in READ_ONLY:  # not offered, but a model can still name it
            return {"error": f"{name} is not available in fallback mode"}
        try:
            if name == "search_docs":
                return self.tools.search_docs(args["query"])
            return getattr(self.tools, name)(args["part_no"])
        except Exception as e:  # a broken tool must not crash the loop: tell the model
            return {"error": f"{type(e).__name__}: {e}"}

    def run(self, question: str, session_id: str | None = None) -> AgentResult:
        messages = [{"role": "system", "content": SYSTEM_LOCAL},
                    {"role": "user", "content": question}]
        result = AgentResult(answer="", stop="max_steps")
        for step in range(1, self.max_steps + 1):
            t0 = time.perf_counter()
            r = self.chat({"model": self.model, "messages": messages, "tools": OLLAMA_TOOLS,
                           "stream": False, "think": self.think,
                           "options": {"num_ctx": self.num_ctx, "temperature": 0}})
            msg = r["message"]
            result.trace.append({
                "type": "llm", "step": step, "model": self.model,
                "ms": round((time.perf_counter() - t0) * 1000),
                "input_tokens": r.get("prompt_eval_count", 0),
                "output_tokens": r.get("eval_count", 0),
                "cost_usd": 0.0, "stop_reason": r.get("done_reason"),
            })
            messages.append(msg)
            calls = msg.get("tool_calls") or []
            if not calls:
                result.answer = strip_reasoning(msg.get("content") or "")
                result.stop = "done" if r.get("done_reason") == "stop" else "truncated"
                return result
            for c in calls:
                name, args = c["function"]["name"], c["function"]["arguments"]
                t1 = time.perf_counter()
                output = self._run_tool(name, args)
                result.trace.append({"type": "tool", "step": step, "name": name,
                                     "input": args, "is_error": "error" in output,
                                     "ms": round((time.perf_counter() - t1) * 1000)})
                result.tool_calls.append({"name": name, "input": args, "output": output})
                messages.append({"role": "tool", "tool_name": name,
                                 "content": json.dumps(output, ensure_ascii=False)})
        result.answer = "（已達步數上限，停止處理）"
        return result


def _demo() -> None:  # pragma: no cover - manual check that Ollama answers
    from part_assistant.db import connect
    print(LocalAgent(Tools(connect())).run("PSU-3200-54V 還有多少庫存？", uuid.uuid4().hex))


if __name__ == "__main__":
    _demo()
