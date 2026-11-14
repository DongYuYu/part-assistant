"""Fallback: when Claude is unavailable, a local model answers with READ-ONLY tools.

No network: Claude is a scripted FakeClient that raises, Ollama is a scripted fake chat.
"""

import urllib.error

import anthropic
import httpx2
import pytest
from test_agent import LOW, FakeClient, n_tickets, text, tool_use, usage

from part_assistant.agent import DEGRADED_BANNER, Agent, call_cost
from part_assistant.db import connect, init_schema
from part_assistant.local_agent import READ_ONLY, LocalAgent, ollama_url, strip_reasoning
from part_assistant.seed import seed
from part_assistant.tools import Tools


@pytest.fixture
def tools(tmp_path):
    conn = connect(tmp_path / "t.db")
    init_schema(conn)
    seed(conn)
    return Tools(conn)


REQ =httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def status_error(cls, code):
    return cls(f"HTTP {code}", response=httpx2.Response(code, request=REQ), body=None)


def ol_text(t):
    return {"message": {"role": "assistant", "content": t}, "done_reason": "stop",
            "prompt_eval_count": 100, "eval_count": 20}


def ol_tool(name, args):
    return {"message": {"role": "assistant", "content": "",
                        "tool_calls": [{"function": {"name": name, "arguments": args}}]},
            "done_reason": "stop"}


class FakeOllama:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.bodies = []

    def __call__(self, body):
        self.bodies.append(body)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


STOCK = {"part_no": "PSU-3200-54V"}


def make(tools, *claude, local=None):
    local = local or FakeOllama(ol_tool("check_stock", STOCK), ol_text("可用庫存 120 台"))
    return Agent(tools, FakeClient(*claude), fallback=LocalAgent(tools, chat=local)), local


# --- which errors fall back

@pytest.mark.parametrize("error", [
    anthropic.APIConnectionError(request=REQ),
    anthropic.APITimeoutError(request=REQ),
    status_error(anthropic.InternalServerError, 500),
    status_error(anthropic.InternalServerError, 529),  # overloaded
    status_error(anthropic.RateLimitError, 429),
], ids=["connection", "timeout", "500", "529", "429"])
def test_unavailable_claude_falls_back(tools, error):
    agent, _ = make(tools, error)
    r = agent.run("PSU-3200-54V 還有多少庫存？")
    assert r.degraded and r.stop == "done"
    assert r.answer == DEGRADED_BANNER + "可用庫存 120 台"
    assert [c["name"] for c in r.tool_calls] == ["check_stock"]
    event = next(t for t in r.trace if t["type"] == "fallback")
    assert type(error).__name__ in event["reason"] and event["to"] == "qwen3:4b"


@pytest.mark.parametrize("error", [
    status_error(anthropic.AuthenticationError, 401),  # bad key: our bug, must be seen
    status_error(anthropic.BadRequestError, 400),
], ids=["401", "400"])
def test_our_own_bugs_are_not_hidden_by_the_fallback(tools, error):
    agent, local = make(tools, error)
    with pytest.raises(type(error)):
        agent.run("PSU-3200-54V 還有多少庫存？")
    assert local.bodies == []  # the local model was never asked


def test_without_a_fallback_the_error_is_raised(tools):
    agent = Agent(tools, FakeClient(anthropic.APIConnectionError(request=REQ)))
    with pytest.raises(anthropic.APIConnectionError):
        agent.run("PSU-3200-54V 還有多少庫存？")


def test_claude_working_means_no_fallback(tools):
    agent, local = make(tools, text("可用庫存 120 台"))
    r = agent.run("PSU-3200-54V 還有多少庫存？")
    assert not r.degraded and local.bodies == []


# --- the local model can not write

def test_local_model_is_only_offered_read_only_tools(tools):
    agent, local = make(tools, anthropic.APIConnectionError(request=REQ))
    agent.run("PSU-3200-54V 還有多少庫存？")
    offered = [t["function"]["name"] for t in local.bodies[0]["tools"]]
    assert offered == list(READ_ONLY) and "create_ticket" not in offered
    assert "can NOT be created" in local.bodies[0]["messages"][0]["content"]


def test_local_model_can_not_create_a_ticket_even_if_it_tries(tools):
    local = FakeOllama(ol_tool("create_ticket", LOW), ol_text("工單已開立"))
    agent, _ = make(tools, anthropic.APIConnectionError(request=REQ), local=local)
    before = n_tickets(tools)
    r = agent.run("FAN-4028-HS 有異音，開一張低嚴重度工單")
    assert n_tickets(tools) == before
    assert "not available in fallback mode" in r.tool_calls[0]["output"]["error"]
    assert r.answer.startswith(DEGRADED_BANNER)  # even a false claim carries the warning


# --- what is kept and reported

def test_failure_midway_keeps_what_claude_already_did(tools):
    u = usage(1000, 100)
    agent, _ = make(tools, tool_use(("check_stock", STOCK), u=u),
                    anthropic.APIConnectionError(request=REQ))
    r = agent.run("PSU-3200-54V 還有多少庫存？")
    assert [c["name"] for c in r.tool_calls] == ["check_stock", "check_stock"]  # Claude, local
    assert r.cost_usd == pytest.approx(call_cost("claude-opus-5-5", u))  # local is free
    kinds = [t["type"] for t in r.trace]
    assert kinds.index("fallback") == 2  # Claude's llm + tool, then the fallback event


def test_ollama_down_too(tools):
    local = FakeOllama(urllib.error.URLError("Connection refused"))
    agent, _ = make(tools, anthropic.APIConnectionError(request=REQ), local=local)
    r = agent.run("PSU-3200-54V 還有多少庫存？")
    assert r.degraded and r.stop == "unavailable"
    assert "都暫時無法使用" in r.answer
    assert "URLError" in next(t for t in r.trace if t["type"] == "fallback")["fallback_error"]


def test_leaked_reasoning_is_removed_from_the_answer(tools):
    local = FakeOllama(ol_text("Okay, the user asks...\n</think>\n\n可用庫存 120 台"))
    agent, _ = make(tools, anthropic.APIConnectionError(request=REQ), local=local)
    r = agent.run("PSU-3200-54V 還有多少庫存？")
    assert r.answer == DEGRADED_BANNER + "可用庫存 120 台"


@pytest.mark.parametrize("content,expected", [
    ("Okay...\n</think>\n\n54 VDC", "54 VDC"),  # opening tag eaten by the template
    ("<think>hmm</think>54 VDC", "54 VDC"),
    ("54 VDC", "54 VDC"),
])
def test_strip_reasoning(content, expected):
    assert strip_reasoning(content) == expected


@pytest.mark.parametrize("host,url", [
    (None, "http://127.0.0.1:11434/api/chat"),
    ("0.0.0.0:8080", "http://0.0.0.0:8080/api/chat"),
    ("http://ollama:11434/", "http://ollama:11434/api/chat"),  # e.g. a Docker Compose service
])
def test_ollama_url(monkeypatch, host, url):
    if host is None:
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
    else:
        monkeypatch.setenv("OLLAMA_HOST", host)
    assert ollama_url() == url
