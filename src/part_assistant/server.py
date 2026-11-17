"""The parts assistant as an HTTP service: the retrieval models load once at start, so
every question after that finds them warm on the GPU.

    uv run --env-file .env python -m part_assistant.server        # http://127.0.0.1:8000
    curl -s 127.0.0.1:8000/ask -H "X-User: alice" -H "Content-Type: application/json" \\
        -d '{"question": "PSU-3200-54V 還有多少庫存？"}'

Routes
    POST /ask      {"question": "...", "session_id": optional}
                   -> answer, stop, degraded, cost_usd, tool calls, session_id, pending_ticket
    GET  /healthz

Measured (RTX 5070 Laptop): loading bge-m3 + the reranker takes ~14 s, a warm search
~0.6 s. The CLI pays the 14 s on every question; this service pays it once.

Who is asking: the X-User header. This is a placeholder for SSO: trust it only behind a
proxy that authenticates the user and sets it, never with the port open to users.

High-severity tickets: /ask returns the draft (stop "needs_confirmation") and does not
create it. Approval needs a person identified by SSO, in a separate step (not built yet).

Model calls go wherever the environment says: through the gateway when ANTHROPIC_BASE_URL
and PART_ASSISTANT_LOCAL_URL point at it (compose does that), or straight to the APIs.
"""

import argparse
import json
import threading
import uuid
from collections.abc import Callable

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from part_assistant.agent import Agent, AgentResult, save_trace

MAX_QUESTION_CHARS = 2000


class LockedRetriever:
    """The retriever is shared by every request (one copy of the models on the GPU), but
    is not known to be safe in several threads at once: one search at a time. A search
    takes ~0.6 s; the seconds spent waiting for Claude run in parallel."""

    def __init__(self, retriever):
        self._retriever = retriever
        self._lock = threading.Lock()

    def search(self, *args, **kwargs):
        with self._lock:
            return self._retriever.search(*args, **kwargs)

    def __getattr__(self, name):  # chunks, model_filter, ...: read-only, no lock needed
        return getattr(self._retriever, name)


def build_make_agent() -> Callable[[], Agent]:
    """Load everything that is shared once; return a function that builds a cheap Agent
    per request (its own database connection: sqlite3 connections stay in their thread)."""
    import anthropic

    from part_assistant.chunk import load_chunks
    from part_assistant.circuit import CircuitBreaker
    from part_assistant.db import connect, init_schema
    from part_assistant.local_agent import LocalAgent
    from part_assistant.retrieve import Embedder, Reranker, Retriever
    from part_assistant.tools import Tools

    init_schema(connect())
    retriever = LockedRetriever(Retriever(load_chunks(), Embedder(), Reranker()))
    client = anthropic.Anthropic()  # thread-safe; reuses its connections
    breaker = CircuitBreaker()  # in memory: this process lives on (the CLI needs a file)

    def make_agent() -> Agent:
        tools = Tools(connect(), retriever)
        return Agent(tools, client=client, fallback=LocalAgent(tools), breaker=breaker,
                     channel="api")

    return make_agent


def summary(result: AgentResult) -> dict:
    return {"answer": result.answer, "stop": result.stop, "degraded": result.degraded,
            "cost_usd": round(result.cost_usd, 6), "session_id": result.session_id,
            "tool_calls": [c["name"] for c in result.tool_calls],
            "pending_ticket": result.pending_ticket}


def build_app(make_agent: Callable[[], Agent], save: bool = True) -> Starlette:
    async def ask(request: Request) -> JSONResponse:
        user = (request.headers.get("x-user") or "").strip()
        if not user:
            return JSONResponse({"error": "X-User header is required"}, status_code=401)
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return JSONResponse({"error": "body is not JSON"}, status_code=400)
        question = body.get("question") if isinstance(body, dict) else None
        if not isinstance(question, str) or not question.strip():
            return JSONResponse({"error": "question is required"}, status_code=400)
        if len(question) > MAX_QUESTION_CHARS:
            return JSONResponse({"error": f"question is over {MAX_QUESTION_CHARS} characters"},
                                status_code=400)
        session_id = body.get("session_id") or uuid.uuid4().hex

        def answer() -> AgentResult:  # blocking (SDK, database, GPU): in a worker thread
            result = make_agent().run(question, session_id=session_id, user_id=user)
            if save:
                save_trace(question, result)
            return result

        return JSONResponse(summary(await run_in_threadpool(answer)))

    async def healthz(request: Request) -> JSONResponse:
        return JSONResponse({"ok": True})  # listening only after the models are loaded

    return Starlette(routes=[Route("/ask", ask, methods=["POST"]),
                             Route("/healthz", healthz, methods=["GET"])])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    make_agent = build_make_agent()  # ~14 s: before listening, so /healthz means ready
    import uvicorn
    uvicorn.run(build_app(make_agent), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
