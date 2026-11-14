"""Circuit breaker: after THRESHOLD failures in a row, skip Claude for a while.

No network and no sleeping: Claude is a FakeClient, time is a fake clock.
"""

import anthropic
import httpx2
import pytest
from test_agent import FakeClient, text
from test_fallback import REQ, FakeOllama, ol_text

from part_assistant.agent import DEGRADED_BANNER, Agent
from part_assistant.circuit import CircuitBreaker
from part_assistant.db import connect, init_schema
from part_assistant.local_agent import LocalAgent
from part_assistant.seed import seed
from part_assistant.tools import Tools

DOWN = anthropic.APIConnectionError(request=REQ)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def tools(tmp_path):
    conn = connect(tmp_path / "t.db")
    init_schema(conn)
    seed(conn)
    return Tools(conn)


@pytest.fixture
def clock():
    return Clock()


def breaker(clock, path=None):
    return CircuitBreaker(path, threshold=3, cooldown_s=60, clock=clock)


# --- the breaker alone

def test_opens_after_threshold_failures_in_a_row(clock):
    b = breaker(clock)
    for _ in range(2):
        b.record_failure()
    assert b.state() == "closed"
    b.record_failure()
    assert b.state() == "open" and not b.allow()


def test_a_success_resets_the_count(clock):
    b = breaker(clock)
    b.record_failure()
    b.record_failure()
    b.record_success()
    b.record_failure()
    assert b.state() == "closed"  # not 3 in a row


def test_half_open_after_cooldown_then_closes_on_success(clock):
    b = breaker(clock)
    for _ in range(3):
        b.record_failure()
    clock.now += 59
    assert b.state() == "open" and b.seconds_left() == 1
    clock.now += 1
    assert b.state() == "half_open" and b.allow()
    b.record_success()
    assert b.state() == "closed"


def test_half_open_failure_opens_again_for_a_full_cooldown(clock):
    b = breaker(clock)
    for _ in range(3):
        b.record_failure()
    clock.now += 60
    b.record_failure()  # the one try failed
    assert b.state() == "open" and b.seconds_left() == 60


def test_state_is_shared_through_the_file(tmp_path, clock):
    path = tmp_path / "circuit.json"
    for _ in range(3):
        breaker(clock, path).record_failure()  # a new process each time (CLI runs)
    assert breaker(clock, path).state() == "open"


@pytest.mark.parametrize("content", ["", "{", "[]", '{"failures": "x"}', '{"other": 1}'])
def test_a_broken_file_means_closed(tmp_path, clock, content):
    path = tmp_path / "circuit.json"
    path.write_text(content)
    assert breaker(clock, path).allow()


# --- with the agent

def make(tools, clock, *claude, answers=1):
    local = FakeOllama(*[ol_text("可用庫存 120 台") for _ in range(answers)])
    agent = Agent(tools, FakeClient(*claude),
                  fallback=LocalAgent(tools, chat=local), breaker=breaker(clock))
    return agent, local


def test_open_breaker_skips_claude(tools, clock):
    agent, _ = make(tools, clock, DOWN, DOWN, DOWN, answers=4)
    for _ in range(3):
        agent.run("PSU-3200-54V 還有多少庫存？")
    assert len(agent.client.requests) == 3 and agent.breaker.state() == "open"

    r = agent.run("PSU-3200-54V 還有多少庫存？")
    assert len(agent.client.requests) == 3  # Claude was not called
    assert r.degraded and r.answer == DEGRADED_BANNER + "可用庫存 120 台"
    event = next(e for e in r.trace if e["type"] == "fallback")
    assert event["reason"] == "circuit open (60s left)"


def test_claude_back_closes_the_breaker(tools, clock):
    agent, _ = make(tools, clock, DOWN, DOWN, DOWN, text("可用庫存 120 台"), answers=3)
    for _ in range(3):
        agent.run("q")
    clock.now += 60  # cooldown over: one try
    r = agent.run("q")
    assert not r.degraded and agent.breaker.state() == "closed"


def test_our_own_bugs_do_not_open_the_breaker(tools, clock):
    bad_key = anthropic.AuthenticationError(
        "HTTP 401", response=httpx2.Response(401, request=REQ), body=None)
    agent, _ = make(tools, clock, bad_key, answers=0)
    with pytest.raises(anthropic.AuthenticationError):
        agent.run("q")
    assert agent.breaker.state() == "closed" and agent.breaker._load()["failures"] == 0


def test_no_breaker_means_always_try_claude(tools, clock):
    local = FakeOllama(*[ol_text("ok") for _ in range(5)])
    agent = Agent(tools, FakeClient(*[DOWN] * 5), fallback=LocalAgent(tools, chat=local))
    for _ in range(5):
        agent.run("q")
    assert len(agent.client.requests) == 5
