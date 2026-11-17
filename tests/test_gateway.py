"""Gateway tests: a fake upstream (httpx2.MockTransport), no network, no Claude calls."""

import json

import httpx2
import pytest
from starlette.testclient import TestClient

from part_assistant import gateway
from part_assistant.gateway import Gateway, SSEUsage, add_key, connect, revoke_user

CLAUDE_OK = {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
             "content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn",
             "usage": {"input_tokens": 1000, "output_tokens": 200,
                       "cache_read_input_tokens": 500, "cache_creation_input_tokens": 0}}
LOCAL_OK = {"choices": [{"message": {"role": "assistant", "content": "hi"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 300, "completion_tokens": 40}}


class Upstream:
    """Records what the gateway sent; answers with whatever the test sets."""

    def __init__(self):
        self.requests: list[httpx2.Request] = []
        self.reply = lambda request: httpx2.Response(200, json=CLAUDE_OK,
                                                     headers={"request-id": "req_1"})

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.reply(request)


@pytest.fixture
def setup(tmp_path):
    conn = connect(tmp_path / "gateway.db")
    up = Upstream()
    now = [0.0]
    gw = Gateway(conn, client=httpx2.AsyncClient(transport=httpx2.MockTransport(up)),
                 anthropic_key="sk-real", upstream_anthropic="https://claude.test",
                 upstream_local="http://local.test/v1", clock=lambda: now[0])
    key = add_key(conn, "alice")
    return TestClient(gw.app()), gw, up, conn, key, now


def ask_claude(client, key, **extra):
    body = {"model": "claude-opus-5-5", "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}], **extra}
    return client.post("/v1/messages?beta=true", json=body,
                       headers={"x-api-key": key, "anthropic-version": "2023-06-01"})


class Chunks(httpx2.AsyncByteStream):
    """A response body that arrives in pieces, like a real stream from the network."""

    def __init__(self, data: bytes, size: int = 40):
        self.parts = [data[i:i + size] for i in range(0, len(data), size)]

    async def __aiter__(self):
        for part in self.parts:
            yield part


def usage_rows(conn):
    return [dict(r) for r in conn.execute("SELECT * FROM gateway_usage ORDER BY id")]


def test_claude_call_is_passed_through_with_the_real_key(setup):
    client, _, up, conn, key, _ = setup
    r = ask_claude(client, key)
    assert r.status_code == 200 and r.json() == CLAUDE_OK
    assert r.headers["request-id"] == "req_1"
    sent = up.requests[0]
    assert str(sent.url) == "https://claude.test/v1/messages?beta=true"  # beta API kept
    assert sent.headers["x-api-key"] == "sk-real"  # the app's gateway key never leaves
    assert sent.headers["anthropic-version"] == "2023-06-01"
    assert json.loads(sent.content)["model"] == "claude-opus-5-5"
    row = usage_rows(conn)[0]
    assert (row["user_id"], row["route"], row["status"]) == ("alice", "claude", 200)
    assert (row["input_tokens"], row["output_tokens"]) == (1500, 200)  # cache reads count
    assert row["cost_usd"] == pytest.approx((1000 * 4 + 200 * 20 + 500 * 0.2) / 1e6)


def test_the_anthropic_sdk_works_unchanged_through_the_gateway(setup):
    """What the agent does: only base_url and api_key differ."""
    import anthropic
    client, _, up, _, key, _ = setup
    sdk = anthropic.Anthropic(api_key=key, base_url="http://gw.test", http_client=client)
    msg = sdk.beta.messages.create(model="claude-opus-5-5", max_tokens=100,
                                   messages=[{"role": "user", "content": "hi"}])
    assert msg.content[0].text == "hi"
    assert up.requests[0].headers["x-api-key"] == "sk-real"


def test_bearer_key_works_and_local_route_goes_to_the_local_server(setup):
    client, _, up, conn, key, _ = setup
    up.reply = lambda request: httpx2.Response(200, json=LOCAL_OK)
    r = client.post("/v1/chat/completions", json={"model": "Qwen/Qwen3-4B-AWQ",
                                                  "messages": [{"role": "user", "content": "hi"}]},
                    headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 200 and r.json() == LOCAL_OK
    assert str(up.requests[0].url) == "http://local.test/v1/chat/completions"
    assert "authorization" not in up.requests[0].headers
    row = usage_rows(conn)[0]
    assert (row["route"], row["input_tokens"], row["output_tokens"]) == ("local", 300, 40)
    assert row["cost_usd"] == 0.0


@pytest.mark.parametrize("headers", [{}, {"x-api-key": "pa-wrong"},
                                     {"Authorization": "Bearer pa-wrong"}])
def test_unknown_key_is_refused_before_any_upstream_call(setup, headers):
    client, _, up, conn, _, _ = setup
    r = client.post("/v1/messages", json={"model": "claude-opus-5-5"}, headers=headers)
    assert r.status_code == 401
    assert r.json()["error"]["type"] == "authentication_error"  # Anthropic error format
    assert up.requests == [] and usage_rows(conn) == []


def test_revoked_key_is_refused(setup):
    client, _, _, conn, key, _ = setup
    assert revoke_user(conn, "alice") == 1
    assert ask_claude(client, key).status_code == 401


def test_keys_are_stored_hashed(setup):
    _, _, _, conn, key, _ = setup
    stored = conn.execute("SELECT key_hash FROM gateway_keys").fetchone()[0]
    assert key not in stored and stored == gateway.hash_key(key)


def test_rate_limit_per_minute_then_window_moves_on(setup):
    client, _, up, conn, _, now = setup
    key = add_key(conn, "bob", rpm=2)
    assert [ask_claude(client, key).status_code for _ in range(3)] == [200, 200, 429]
    r = ask_claude(client, key)
    assert r.status_code == 429 and r.headers["retry-after"] == "60"
    assert "x-should-retry" not in r.headers  # per minute: retrying later is fine
    assert r.json()["error"]["type"] == "rate_limit_error"
    assert len(up.requests) == 2
    now[0] = 61  # a minute later
    assert ask_claude(client, key).status_code == 200


def test_rate_limit_is_per_user(setup):
    client, _, _, conn, alice, _ = setup
    bob = add_key(conn, "bob", rpm=1)
    assert ask_claude(client, bob).status_code == 200
    assert ask_claude(client, bob).status_code == 429
    assert ask_claude(client, alice).status_code == 200


def test_daily_token_quota(setup):
    client, _, up, conn, _, _ = setup
    key = add_key(conn, "carol", daily_tokens=2000)  # one call uses 1700
    assert ask_claude(client, key).status_code == 200
    assert ask_claude(client, key).status_code == 200  # 1700 < 2000: allowed, goes over
    r = ask_claude(client, key)
    assert r.status_code == 429 and "daily token quota" in r.json()["error"]["message"]
    assert r.headers["x-should-retry"] == "false"  # the SDK would otherwise retry twice
    assert len(up.requests) == 2


def test_unapproved_claude_model_is_refused(setup):
    client, _, up, _, key, _ = setup
    r = ask_claude(client, key, model="claude-some-expensive-model")
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"
    assert up.requests == []


def test_upstream_errors_keep_their_status_so_the_agent_can_fall_back(setup):
    client, _, up, conn, key, _ = setup
    overloaded = {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}
    up.reply = lambda request: httpx2.Response(529, json=overloaded)
    r = ask_claude(client, key)
    assert r.status_code == 529 and r.json() == overloaded
    assert usage_rows(conn)[0]["status"] == 529


def test_unreachable_upstream_is_502(setup):
    client, _, up, conn, key, _ = setup

    def refuse(request):
        raise httpx2.ConnectError("connection refused", request=request)
    up.reply = refuse
    r = ask_claude(client, key)
    assert r.status_code == 502 and r.json()["error"]["type"] == "api_error"
    assert usage_rows(conn)[0]["status"] == 502


def test_local_errors_use_the_openai_format(setup):
    client, _, _, _, _, _ = setup
    r = client.post("/v1/chat/completions", json={}, headers={"Authorization": "Bearer x"})
    assert r.status_code == 401 and r.json()["error"]["code"] == 401


def test_bad_json_is_400(setup):
    client, _, up, _, key, _ = setup
    r = client.post("/v1/messages", content=b"not json", headers={"x-api-key": key})
    assert r.status_code == 400 and up.requests == []


CLAUDE_STREAM = b"".join([
    b'event: message_start\ndata: {"type":"message_start","message":{"model":"claude-opus-5-5",'
    b'"usage":{"input_tokens":800,"output_tokens":1}}}\n\n',
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
    b'"delta":{"type":"text_delta","text":"hi"}}\n\n',
    b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
    b'"usage":{"output_tokens":50}}\n\n',
    b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
])


def test_claude_stream_passes_through_unchanged_and_is_metered(setup):
    client, _, up, conn, key, _ = setup
    up.reply = lambda request: httpx2.Response(
        200, stream=Chunks(CLAUDE_STREAM), headers={"content-type": "text/event-stream"})
    r = ask_claude(client, key, stream=True)
    assert r.status_code == 200 and r.content == CLAUDE_STREAM
    row = usage_rows(conn)[0]
    assert (row["input_tokens"], row["output_tokens"]) == (800, 50)


def test_local_stream_asks_for_usage_and_is_metered(setup):
    client, _, up, conn, key, _ = setup
    stream = (b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
              b'data: {"choices":[],"usage":{"prompt_tokens":70,"completion_tokens":9}}\n\n'
              b"data: [DONE]\n\n")
    up.reply = lambda request: httpx2.Response(200, stream=Chunks(stream))
    r = client.post("/v1/chat/completions", json={"model": "m", "stream": True, "messages": []},
                    headers={"Authorization": f"Bearer {key}"})
    assert r.content == stream
    assert json.loads(up.requests[0].content)["stream_options"] == {"include_usage": True}
    row = usage_rows(conn)[0]
    assert (row["input_tokens"], row["output_tokens"]) == (70, 9)


def test_sse_usage_survives_events_split_across_chunks():
    reader = SSEUsage()
    for i in range(0, len(CLAUDE_STREAM), 7):  # 7-byte pieces: lines are cut anywhere
        reader.feed(CLAUDE_STREAM[i:i + 7])
    assert reader.usage == {"input_tokens": 800, "output_tokens": 50}


def test_usage_report(setup):
    client, _, up, conn, key, _ = setup
    ask_claude(client, key)
    ask_claude(client, key)
    up.reply = lambda request: httpx2.Response(500, json={})
    ask_claude(client, key)
    (row,) = gateway.usage_report(conn)
    assert (row["user_id"], row["calls"], row["errors"]) == ("alice", 3, 1)
    assert row["input_tokens"] == 3000


def test_local_agent_sends_the_gateway_key(monkeypatch):
    from part_assistant.local_agent import local_headers
    monkeypatch.delenv("PART_ASSISTANT_LOCAL_API_KEY", raising=False)
    assert "Authorization" not in local_headers()
    monkeypatch.setenv("PART_ASSISTANT_LOCAL_API_KEY", "pa-abc")
    assert local_headers()["Authorization"] == "Bearer pa-abc"
