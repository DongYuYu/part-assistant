"""Fallback when Claude is unavailable: the same kind of loop, on a local model.

Measured on the golden set (RTX 5070 Laptop, see design/routing.md and local-llm.md):
qwen3:4b with thinking scored 29/48 vs Claude 48/48, and once claimed it had opened a
ticket without calling the tool. With rule 9 below: Qwen3-4B AWQ on vLLM 33/48.
So in fallback mode the local model gets READ-ONLY tools: create_ticket is not offered,
and refused here even if the model calls it anyway.

Speaks the OpenAI chat-completions format, which both Ollama (/v1) and vLLM serve, so the
same code runs on either server, and through the gateway later.

Settings (environment variables):
    PART_ASSISTANT_LOCAL_MODEL   default qwen3:4b (Ollama's name; vLLM uses e.g. Qwen/Qwen3-4B-AWQ)
    PART_ASSISTANT_LOCAL_URL     base URL of the server, e.g. http://127.0.0.1:8000/v1 (vLLM)
    OLLAMA_HOST                  used when PART_ASSISTANT_LOCAL_URL is not set: <host>/v1
                                 (default 127.0.0.1:11434, same variable the ollama CLI uses)
    PART_ASSISTANT_LOCAL_API_KEY sent as "Authorization: Bearer ...": the gateway key when
                                 PART_ASSISTANT_LOCAL_URL points at the gateway

Ollama's OpenAI endpoint can not set the context length per request and defaults to 4096
tokens, too short for 5 search passages: start Ollama with OLLAMA_CONTEXT_LENGTH=16384.

No extra package: the HTTP API is called with the standard library.
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

# Rule 9: "(file p.N)" in rule 1 is a placeholder. Claude fills it in; small local models
# copied it word for word in 10-15 of 48 golden answers (design/local-llm.md). Each search
# result already has a ready-made "source" field, so the rule says: copy that. The example
# file name is made up on purpose, so it can not leak a golden-set answer.
SYSTEM_LOCAL = SYSTEM + """
8. Tickets can NOT be created right now (the main model is unavailable). If the user asks
   for a ticket, say it cannot be opened now and ask them to try again later.
9. To cite, copy the "source" field of the search result you used, exactly as written,
   e.g. (abc-1234.pdf p.3). Never write the word "file" instead of the real file name."""

# Anthropic tool format -> OpenAI ("function") format; the JSON schema inside is the same.
LOCAL_TOOLS = [{"type": "function", "function": {
    "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
    for t in TOOL_SCHEMAS if t["name"] in READ_ONLY]


def local_url() -> str:
    base = os.environ.get("PART_ASSISTANT_LOCAL_URL")
    if not base:
        host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
        if "://" not in host:
            host = "http://" + host
        base = host.rstrip("/") + "/v1"
    return base.rstrip("/") + "/chat/completions"


def local_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("PART_ASSISTANT_LOCAL_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    return headers


def local_chat(body: dict) -> dict:
    req = urllib.request.Request(local_url(), json.dumps(body).encode(), local_headers())
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def strip_reasoning(content: str) -> str:
    """Reasoning must never reach the user. Ollama and vLLM (with a reasoning parser) put
    it in a separate field, but a thinking-only model asked not to think leaks it into
    content, with the opening <think> eaten by the chat template: keep only what follows
    </think>."""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[1]
    return content.strip()


def parse_arguments(raw) -> dict | None:
    """OpenAI sends tool arguments as a JSON string; a small model can get it wrong."""
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


class LocalAgent:
    def __init__(self, tools: Tools, model: str | None = None, think: bool = True,
                 max_steps: int = MAX_STEPS,
                 chat: Callable[[dict], dict] = local_chat):  # tests pass a fake chat
        self.tools = tools
        self.model = model or os.environ.get("PART_ASSISTANT_LOCAL_MODEL", DEFAULT_MODEL)
        self.think = think  # measured: 29/48 with thinking, 18/48 without
        self.max_steps = max_steps
        self.chat = chat

    def _request(self, messages: list[dict]) -> dict:
        body = {"model": self.model, "messages": messages, "tools": LOCAL_TOOLS,
                "temperature": 0}
        if not self.think:  # Qwen3 thinks by default on both servers
            body["chat_template_kwargs"] = {"enable_thinking": False}  # vLLM
            body["reasoning_effort"] = "none"  # Ollama
        return body

    def _run_tool(self, name: str, args: dict, user_id: str | None = None) -> dict:
        if name not in READ_ONLY:  # not offered, but a model can still name it
            return {"error": f"{name} is not available in fallback mode"}
        try:
            if name == "search_docs":
                return self.tools.search_docs(args["query"], user_id)
            return getattr(self.tools, name)(args["part_no"])
        except Exception as e:  # a broken tool must not crash the loop: tell the model
            return {"error": f"{type(e).__name__}: {e}"}

    def run(self, question: str, session_id: str | None = None,
            user_id: str | None = None) -> AgentResult:
        messages = [{"role": "system", "content": SYSTEM_LOCAL},
                    {"role": "user", "content": question}]
        result = AgentResult(answer="", stop="max_steps")
        for step in range(1, self.max_steps + 1):
            t0 = time.perf_counter()
            r = self.chat(self._request(messages))
            choice = r["choices"][0]
            msg, usage = choice["message"], r.get("usage") or {}
            result.trace.append({
                "type": "llm", "step": step, "model": self.model,
                "ms": round((time.perf_counter() - t0) * 1000),
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "cost_usd": 0.0, "stop_reason": choice.get("finish_reason"),
            })
            calls = msg.get("tool_calls") or []
            # Send back only what the format defines: no reasoning fields.
            messages.append({"role": "assistant", "content": msg.get("content") or "",
                             **({"tool_calls": calls} if calls else {})})
            if not calls:
                result.answer = strip_reasoning(msg.get("content") or "")
                result.stop = "done" if choice.get("finish_reason") == "stop" else "truncated"
                return result
            for c in calls:
                name = c["function"]["name"]
                args = parse_arguments(c["function"].get("arguments"))
                t1 = time.perf_counter()
                output = (self._run_tool(name, args, user_id) if args is not None
                          else {"error": "arguments are not a valid JSON object"})
                result.trace.append({"type": "tool", "step": step, "name": name,
                                     "input": args, "is_error": "error" in output,
                                     "ms": round((time.perf_counter() - t1) * 1000)})
                result.tool_calls.append({"name": name, "input": args, "output": output})
                messages.append({"role": "tool", "tool_call_id": c.get("id", ""),
                                 "content": json.dumps(output, ensure_ascii=False)})
        result.answer = "（已達步數上限，停止處理）"
        return result


def _demo() -> None:  # pragma: no cover - manual check that the local model answers
    from part_assistant.db import connect
    print(LocalAgent(Tools(connect())).run("PSU-3200-54V 還有多少庫存？", uuid.uuid4().hex))


if __name__ == "__main__":
    _demo()
