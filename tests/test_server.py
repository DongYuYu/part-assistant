"""HTTP service tests: a fake agent, no models, no Claude calls."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from starlette.testclient import TestClient

from part_assistant.agent import AgentResult
from part_assistant.server import LockedRetriever, build_app


class FakeAgent:
    def __init__(self, log: list, delay: float = 0.0):
        self.log, self.delay = log, delay

    def run(self, question, session_id=None, user_id=None) -> AgentResult:
        time.sleep(self.delay)
        self.log.append({"question": question, "session_id": session_id, "user": user_id,
                         "thread": threading.get_ident()})
        r = AgentResult(answer=f"answer to {question}", stop="done")
        r.session_id, r.user_id, r.cost_usd = session_id, user_id, 0.0123
        r.tool_calls = [{"name": "check_stock", "input": {}, "output": {}}]
        return r


@pytest.fixture
def server():
    log: list = []
    agents = {"delay": 0.0}
    app = build_app(lambda: FakeAgent(log, agents["delay"]), save=False)
    with TestClient(app) as client:
        yield client, log, agents


def ask(client, question="PSU-3200-54V 還有多少庫存？", user="alice", **extra):
    headers = {"X-User": user} if user else {}
    return client.post("/ask", json={"question": question, **extra}, headers=headers)


def test_ask_answers_and_says_who_asked(server):
    client, log, _ = server
    r = ask(client)
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "answer to PSU-3200-54V 還有多少庫存？"
    assert body["tool_calls"] == ["check_stock"] and body["cost_usd"] == 0.0123
    assert log[0]["user"] == "alice"
    assert body["session_id"] == log[0]["session_id"] and len(body["session_id"]) == 32


def test_session_id_is_kept_when_given(server):
    client, log, _ = server
    assert ask(client, session_id="s-1").json()["session_id"] == "s-1"


def test_user_is_required(server):
    client, log, _ = server
    assert ask(client, user=None).status_code == 401
    assert log == []


@pytest.mark.parametrize("body", [{}, {"question": "  "}, {"question": 3}, ["q"]])
def test_bad_questions_are_400(server, body):
    client, log, _ = server
    assert client.post("/ask", json=body, headers={"X-User": "a"}).status_code == 400
    assert log == []


def test_long_question_is_400(server):
    client, _, _ = server
    assert ask(client, question="x" * 2001).status_code == 400


def test_agent_runs_off_the_event_loop_so_questions_overlap(server):
    """Answering blocks (SDK, database, GPU): it runs in worker threads, so 4 slow
    questions take about as long as one, not four times as long."""
    client, log, agents = server
    agents["delay"] = 0.5
    t0 = time.perf_counter()
    with ThreadPoolExecutor(4) as pool:
        codes = list(pool.map(lambda i: ask(client, question=f"q{i}").status_code, range(4)))
    assert codes == [200] * 4
    assert time.perf_counter() - t0 < 1.5
    assert len({e["thread"] for e in log}) > 1


def test_locked_retriever_runs_one_search_at_a_time():
    class SlowRetriever:
        chunks = ["c"]
        active = peak = 0

        def search(self, query, **kw):
            SlowRetriever.active += 1
            SlowRetriever.peak = max(SlowRetriever.peak, SlowRetriever.active)
            time.sleep(0.05)
            SlowRetriever.active -= 1
            return [query]

    r = LockedRetriever(SlowRetriever())
    with ThreadPoolExecutor(4) as pool:
        assert list(pool.map(lambda i: r.search(f"q{i}", k=5), range(4))) == [
            ["q0"], ["q1"], ["q2"], ["q3"]]
    assert SlowRetriever.peak == 1
    assert r.chunks == ["c"]  # everything else passes through


def test_healthz(server):
    client, _, _ = server
    assert client.get("/healthz").json() == {"ok": True}
