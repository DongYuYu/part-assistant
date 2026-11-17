"""Inference gateway: every model call of the app goes through here.

    uv run python -m part_assistant.gateway add-key alice          # prints a new key, once
    uv run --env-file .env python -m part_assistant.gateway serve  # http://127.0.0.1:8100
    uv run python -m part_assistant.gateway usage                  # tokens and cost per user

Routes
    POST /v1/messages          Anthropic format: passed to Claude unchanged
    POST /v1/chat/completions  OpenAI format: passed to the local server (vLLM or Ollama)
    GET  /healthz
    GET  /metrics              Prometheus (see design/gateway.md); no user labels on purpose

What it adds to every call, the same for every app and every model:
  - who:      a gateway key per user (only its sha256 is stored). The real
              ANTHROPIC_API_KEY lives only here, never in the apps.
  - limits:   requests per minute, and tokens per day, per key.
  - allowed:  Claude models without a known price are refused (cost control).
  - metering: one row per call in gateway_usage (user, model, tokens, cost, time, status).

No format translation: the app picks the format, the format picks the backend.
The app needs no code change: the Anthropic SDK reads ANTHROPIC_BASE_URL, LocalAgent reads
PART_ASSISTANT_LOCAL_URL / PART_ASSISTANT_LOCAL_API_KEY (see README).

Settings (environment variables of the gateway itself):
    ANTHROPIC_API_KEY                 the real key
    PART_ASSISTANT_GATEWAY_DB         default data/processed/gateway.db
    PART_ASSISTANT_UPSTREAM_ANTHROPIC default https://api.anthropic.com
    PART_ASSISTANT_UPSTREAM_LOCAL     default http://127.0.0.1:11434/v1 (Ollama)

Limits of this version: rate-limit counters are in memory (one process; several replicas
would share them in Redis), and a request is checked against the daily quota before it
runs, so the last request of the day can go over by its own size.
"""

import argparse
import hashlib
import json
import os
import secrets
import sqlite3
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import httpx2
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    disable_created_metrics,
    generate_latest,
)
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from part_assistant.agent import PRICES

DEFAULT_DB = Path(os.environ.get("PART_ASSISTANT_GATEWAY_DB", "data/processed/gateway.db"))
DEFAULT_RPM = 30
DEFAULT_DAILY_TOKENS = 2_000_000
# Headers passed between the app and Claude. Everything else (the app's key above all)
# stays behind.
FORWARD_TO_CLAUDE = ("anthropic-version", "anthropic-beta", "content-type")
FORWARD_BACK = ("content-type", "request-id", "retry-after")

SCHEMA = """
CREATE TABLE IF NOT EXISTS gateway_keys (
    key_hash     TEXT PRIMARY KEY,  -- sha256 of the key; the key is shown once, never stored
    user_id      TEXT NOT NULL,
    rpm          INTEGER NOT NULL,  -- requests per minute
    daily_tokens INTEGER NOT NULL,  -- input + output tokens per UTC day
    created_at   TEXT NOT NULL,
    revoked_at   TEXT
);
CREATE TABLE IF NOT EXISTS gateway_usage (
    id            INTEGER PRIMARY KEY,
    ts            TEXT NOT NULL,     -- UTC, ISO 8601
    user_id       TEXT NOT NULL,
    route         TEXT NOT NULL,     -- claude | local
    model         TEXT,
    status        INTEGER NOT NULL,  -- HTTP status returned to the app
    input_tokens  INTEGER NOT NULL,  -- for Claude: includes cache reads and writes
    output_tokens INTEGER NOT NULL,
    cost_usd      REAL,              -- NULL: price unknown
    ms            INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS gateway_usage_user_ts ON gateway_usage (user_id, ts);
"""


def connect(path: Path = DEFAULT_DB) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    # The app runs on one event loop, but the test client calls it from another thread.
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def add_key(conn: sqlite3.Connection, user_id: str, rpm: int = DEFAULT_RPM,
            daily_tokens: int = DEFAULT_DAILY_TOKENS) -> str:
    key = "pa-" + secrets.token_urlsafe(24)
    with conn:
        conn.execute("INSERT INTO gateway_keys VALUES (?, ?, ?, ?, ?, NULL)",
                     (hash_key(key), user_id, rpm, daily_tokens, _now()))
    return key


def revoke_user(conn: sqlite3.Connection, user_id: str) -> int:
    with conn:
        return conn.execute("UPDATE gateway_keys SET revoked_at = ? WHERE user_id = ? AND "
                            "revoked_at IS NULL", (_now(), user_id)).rowcount


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# --- usage from a response (the format decides where the numbers are)

def claude_usage(usage: dict) -> tuple[int, int]:
    inp = sum(usage.get(k) or 0 for k in ("input_tokens", "cache_read_input_tokens",
                                          "cache_creation_input_tokens"))
    return inp, usage.get("output_tokens") or 0


def claude_cost(model: str, usage: dict) -> float | None:
    p = PRICES.get(model)
    if p is None:
        return None
    return ((usage.get("input_tokens") or 0) * p["input"]
            + (usage.get("output_tokens") or 0) * p["output"]
            + (usage.get("cache_read_input_tokens") or 0) * p["cache_read"]
            + (usage.get("cache_creation_input_tokens") or 0) * p["cache_write"]) / 1_000_000


class SSEUsage:
    """Reads usage out of a server-sent-events stream while the bytes pass through
    unchanged. Claude: input in message_start, output (running total) in message_delta.
    OpenAI format: one last chunk with "usage" (when stream_options.include_usage)."""

    def __init__(self) -> None:
        self.buffer = b""
        self.usage: dict = {}

    def feed(self, chunk: bytes) -> None:
        self.buffer += chunk
        *lines, self.buffer = self.buffer.split(b"\n")
        for line in lines:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            try:
                event = json.loads(line[5:])
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            found = (event.get("message") or {}).get("usage") or event.get("usage")
            if found:
                self.usage.update({k: v for k, v in found.items() if v is not None})


# --- Prometheus metrics

# LLM calls take seconds, not milliseconds: the default buckets (5 ms .. 10 s) would put
# most calls in the last one.
DURATION_BUCKETS = (0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120)
ROUTES = ("claude", "local")
REJECT_REASONS = ("auth", "rate", "quota", "model", "bad_request")
# No extra *_created series (the time each series was created): not used in queries,
# and it doubles the number of series.
disable_created_metrics()


class Metrics:
    """What Prometheus scrapes from /metrics. Labels are kept few and bounded: every label
    combination is a separate time series. Per-user numbers live in gateway_usage
    (SQLite), not here: 10,000 users as a label would be 10,000 times the series."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()  # one per Gateway: tests do not share counts
        r = self.registry
        self.requests = Counter("gateway_requests", "Calls sent upstream, by result",
                                ["route", "model", "status"], registry=r)
        self.duration = Histogram("gateway_request_duration_seconds",
                                  "Time from request to the last byte of the answer",
                                  ["route"], buckets=DURATION_BUCKETS, registry=r)
        self.tokens = Counter("gateway_tokens", "Tokens, from the upstream's usage",
                              ["route", "model", "kind"], registry=r)
        self.cost = Counter("gateway_cost_usd", "Claude list price of the calls",
                            ["model"], registry=r)
        self.rejections = Counter("gateway_rejections", "Refused by the gateway itself, "
                                  "never sent upstream", ["route", "reason"], registry=r)
        self.in_flight = Gauge("gateway_in_flight_requests", "Calls waiting for upstream",
                               ["route"], registry=r)
        # Start every known series at 0. A series that first appears already at 3 has no
        # earlier sample, so increase()/rate() count those first 3 as 0: the first errors,
        # the ones an alert most needs, would be invisible.
        for route in ROUTES:
            self.in_flight.labels(route)
            self.duration.labels(route)
            for reason in REJECT_REASONS:
                self.rejections.labels(route, reason)
        for model in PRICES:  # the only models that can cost money: "$0", not "No data"
            self.cost.labels(model)


# --- the app

class Gateway:
    def __init__(self, conn: sqlite3.Connection, client: httpx2.AsyncClient | None = None,
                 anthropic_key: str | None = None, upstream_anthropic: str | None = None,
                 upstream_local: str | None = None, clock=time.monotonic):
        self.conn = conn
        # Generous read timeout: a long answer from Claude, or a cold local model.
        self.client = client or httpx2.AsyncClient(timeout=httpx2.Timeout(600, connect=10))
        self.anthropic_key = anthropic_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.upstream_anthropic = (upstream_anthropic or os.environ.get(
            "PART_ASSISTANT_UPSTREAM_ANTHROPIC", "https://api.anthropic.com")).rstrip("/")
        self.upstream_local = (upstream_local or os.environ.get(
            "PART_ASSISTANT_UPSTREAM_LOCAL", "http://127.0.0.1:11434/v1")).rstrip("/")
        self.clock = clock
        self.recent: dict[str, deque[float]] = defaultdict(deque)  # user -> request times
        self.metrics = Metrics()

    # --- checks, in order: who, how often, how much

    def _who(self, request: Request) -> sqlite3.Row | None:
        key = request.headers.get("x-api-key") or ""
        auth = request.headers.get("authorization") or ""
        if not key and auth.lower().startswith("bearer "):
            key = auth[7:].strip()
        if not key:
            return None
        return self.conn.execute("SELECT * FROM gateway_keys WHERE key_hash = ? AND "
                                 "revoked_at IS NULL", (hash_key(key),)).fetchone()

    def _over_rate(self, user: sqlite3.Row) -> bool:
        times, now = self.recent[user["user_id"]], self.clock()
        while times and now - times[0] >= 60:  # sliding window of 60 seconds
            times.popleft()
        if len(times) >= user["rpm"]:
            return True
        times.append(now)
        return False

    def tokens_today(self, user_id: str) -> int:
        day = datetime.now(UTC).date().isoformat()
        row = self.conn.execute(
            "SELECT COALESCE(SUM(input_tokens + output_tokens), 0) FROM gateway_usage "
            "WHERE user_id = ? AND ts >= ?", (user_id, day)).fetchone()
        return row[0]

    def _record(self, user_id: str, route: str, model, status: int, tokens: tuple[int, int],
                cost: float | None, t0: float) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO gateway_usage (ts, user_id, route, model, status, input_tokens, "
                "output_tokens, cost_usd, ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (_now(), user_id, route, model, status, tokens[0], tokens[1], cost,
                 round((time.perf_counter() - t0) * 1000)))
        m = self.metrics
        m.in_flight.labels(route).dec()  # every upstream call ends here, exactly once
        # The app picks the local model name: only a name the server accepted becomes a
        # label, or anyone could create new time series by sending random names.
        label = str(model) if status < 400 else "-"
        m.requests.labels(route, label, str(status)).inc()
        m.duration.labels(route).observe(time.perf_counter() - t0)
        m.tokens.labels(route, label, "input").inc(tokens[0])
        m.tokens.labels(route, label, "output").inc(tokens[1])
        if cost:
            m.cost.labels(label).inc(cost)

    def _reject(self, route: str, reason: str, status: int, kind: str, message: str,
                **kw) -> JSONResponse:
        self.metrics.rejections.labels(route, reason).inc()
        return error(route, status, kind, message, **kw)

    # --- routes

    async def messages(self, request: Request) -> Response:
        return await self._proxy(request, "claude")

    async def chat_completions(self, request: Request) -> Response:
        return await self._proxy(request, "local")

    async def healthz(self, request: Request) -> Response:
        return JSONResponse({"ok": True})

    async def metrics_page(self, request: Request) -> Response:
        # No key needed: in production only the monitoring network can reach this port.
        return Response(generate_latest(self.metrics.registry),
                        headers={"content-type": CONTENT_TYPE_LATEST})

    async def _proxy(self, request: Request, route: str) -> Response:
        t0 = time.perf_counter()
        user = self._who(request)
        if user is None:
            return self._reject(route, "auth", 401, "authentication_error",
                                "missing or unknown gateway key")
        if self._over_rate(user):
            return self._reject(route, "rate", 429, "rate_limit_error",
                                f"over {user['rpm']} requests per minute", retry_after=60)
        if self.tokens_today(user["user_id"]) >= user["daily_tokens"]:
            # No point retrying in seconds: tell the SDK not to (it honors x-should-retry).
            return self._reject(route, "quota", 429, "rate_limit_error",
                                f"daily token quota used ({user['daily_tokens']})",
                                retry_after=3600, retry=False)
        try:
            body = json.loads(await request.body())
        except json.JSONDecodeError:
            return self._reject(route, "bad_request", 400, "invalid_request_error",
                                "body is not JSON")
        if not isinstance(body, dict):
            return self._reject(route, "bad_request", 400, "invalid_request_error",
                                "body is not a JSON object")
        model = body.get("model")
        stream = bool(body.get("stream"))

        if route == "claude":
            if model not in PRICES:  # only models someone approved, with a known price
                return self._reject(route, "model", 400, "invalid_request_error",
                                    f"model {model!r} is not allowed by the gateway")
            url = f"{self.upstream_anthropic}/v1/messages"
            headers = {h: request.headers[h] for h in FORWARD_TO_CLAUDE if h in request.headers}
            headers["x-api-key"] = self.anthropic_key
            content = await request.body()  # unchanged, byte for byte
        else:
            url = f"{self.upstream_local}/chat/completions"
            headers = {"content-type": "application/json"}
            if stream:  # ask for the usage chunk, or a streamed call can not be metered
                body.setdefault("stream_options", {})["include_usage"] = True
            content = json.dumps(body).encode()

        upstream = self.client.build_request("POST", url, params=request.url.query,
                                             headers=headers, content=content)
        self.metrics.in_flight.labels(route).inc()  # back down in _record
        try:
            r = await self.client.send(upstream, stream=stream)
        except httpx2.TimeoutException:
            self._record(user["user_id"], route, model, 504, (0, 0), 0.0, t0)
            return error(route, 504, "api_error", "upstream timed out")
        except httpx2.HTTPError as e:  # connection refused, DNS, ...: the app may fall back
            self._record(user["user_id"], route, model, 502, (0, 0), 0.0, t0)
            return error(route, 502, "api_error", f"upstream unreachable: {type(e).__name__}")
        back = {h: r.headers[h] for h in FORWARD_BACK if h in r.headers}

        if not stream:
            data = await r.aread()
            usage = _json_usage(data)
            self._meter(user["user_id"], route, model, r.status_code, usage, t0)
            return Response(data, status_code=r.status_code, headers=back)

        reader = SSEUsage()

        async def relay() -> AsyncIterator[bytes]:
            try:
                async for chunk in r.aiter_raw():
                    reader.feed(chunk)
                    yield chunk
            finally:  # also when the app hangs up: what was generated is still billed
                await r.aclose()
                self._meter(user["user_id"], route, model, r.status_code, reader.usage, t0)

        return StreamingResponse(relay(), status_code=r.status_code, headers=back)

    def _meter(self, user_id: str, route: str, model, status: int, usage: dict,
               t0: float) -> None:
        if route == "claude":
            tokens, cost = claude_usage(usage), claude_cost(model, usage)
        else:  # local: no bill per call; the GPU is a fixed cost (see design/gateway.md)
            tokens = (usage.get("prompt_tokens") or 0, usage.get("completion_tokens") or 0)
            cost = 0.0
        self._record(user_id, route, model, status, tokens, cost, t0)

    def app(self) -> Starlette:
        return Starlette(routes=[
            Route("/v1/messages", self.messages, methods=["POST"]),
            Route("/v1/chat/completions", self.chat_completions, methods=["POST"]),
            Route("/healthz", self.healthz, methods=["GET"]),
            Route("/metrics", self.metrics_page, methods=["GET"]),
        ])


def _json_usage(data: bytes) -> dict:
    try:
        value = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return (value.get("usage") or {}) if isinstance(value, dict) else {}


def error(route: str, status: int, kind: str, message: str,
          retry_after: int | None = None, retry: bool = True) -> JSONResponse:
    """In the format the app expects, so its SDK raises the usual exception: a 429 or 5xx
    here makes the agent fall back, exactly like the same error from Claude."""
    if route == "claude":
        body = {"type": "error", "error": {"type": kind, "message": message}}
    else:
        body = {"error": {"type": kind, "message": message, "code": status}}
    headers = {"retry-after": str(retry_after)} if retry_after else {}
    if not retry:
        headers["x-should-retry"] = "false"
    return JSONResponse(body, status_code=status, headers=headers)


def usage_report(conn: sqlite3.Connection, day: str | None = None) -> list[dict]:
    day = day or datetime.now(UTC).date().isoformat()
    rows = conn.execute(
        "SELECT user_id, route, model, COUNT(*) AS calls, SUM(status >= 400) AS errors, "
        "SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens, "
        "ROUND(SUM(cost_usd), 4) AS cost_usd, CAST(AVG(ms) AS INTEGER) AS avg_ms "
        "FROM gateway_usage WHERE ts >= ? AND ts < date(?, '+1 day') "
        "GROUP BY user_id, route, model ORDER BY user_id, route", (day, day)).fetchall()
    return [dict(r) for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8100)
    k = sub.add_parser("add-key", help="new key for a user; printed once, stored hashed")
    k.add_argument("user")
    k.add_argument("--rpm", type=int, default=DEFAULT_RPM)
    k.add_argument("--daily-tokens", type=int, default=DEFAULT_DAILY_TOKENS)
    r = sub.add_parser("revoke", help="revoke every key of a user")
    r.add_argument("user")
    u = sub.add_parser("usage", help="calls, tokens and cost per user for one UTC day")
    u.add_argument("--day", help="YYYY-MM-DD, default today")
    args = ap.parse_args()

    conn = connect()
    if args.cmd == "add-key":
        print(add_key(conn, args.user, args.rpm, args.daily_tokens))
    elif args.cmd == "revoke":
        print(f"revoked {revoke_user(conn, args.user)} key(s)")
    elif args.cmd == "usage":
        for row in usage_report(conn, args.day):
            print(json.dumps(row, ensure_ascii=False))
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("warning: ANTHROPIC_API_KEY is not set; /v1/messages will get 401 from Claude")
        import uvicorn
        uvicorn.run(Gateway(conn).app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
