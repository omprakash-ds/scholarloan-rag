import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import llm as llm_module
from llm import LLMClient, LLMError


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setattr(llm_module.time, "sleep", lambda s: None)

    class Handler(BaseHTTPRequestHandler):
        responses: list = []
        seen: list = []

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            Handler.seen.append((self.path, self.headers.get("Authorization"), body))
            status, payload = Handler.responses.pop(0)
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield Handler, f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    httpd.shutdown()


OK = (200, {"choices": [{"message": {"content": "  Answer [1]  "}}]})


def test_success_sends_expected_request(server):
    handler, base = server
    handler.responses = [OK]
    client = LLMClient(base + "/", "secret-key", "test-model")
    assert client.complete("sys", "user") == "Answer [1]"
    path, auth, body = handler.seen[0]
    assert path == "/v1/chat/completions" and auth == "Bearer secret-key"
    assert body["model"] == "test-model" and body["temperature"] == 0
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


def test_retries_on_429_then_succeeds(server):
    handler, base = server
    handler.responses = [(429, {}), OK]
    assert LLMClient(base, "k", "m").complete("s", "u") == "Answer [1]"
    assert len(handler.seen) == 2


def test_auth_error_is_not_retried_and_does_not_leak_key(server):
    handler, base = server
    handler.responses = [(401, {"error": "bad key secret-key"})]
    with pytest.raises(LLMError) as err:
        LLMClient(base, "secret-key", "m").complete("s", "u")
    assert "401" in str(err.value) and "secret-key" not in str(err.value)
    assert len(handler.seen) == 1


def test_gives_up_after_max_retries(server):
    handler, base = server
    handler.responses = [(503, {})] * 5
    with pytest.raises(LLMError):
        LLMClient(base, "k", "m", max_retries=2).complete("s", "u")
    assert len(handler.seen) == 3


def test_unexpected_shape_raises(server):
    handler, base = server
    handler.responses = [(200, {"unexpected": True})]
    with pytest.raises(LLMError):
        LLMClient(base, "k", "m").complete("s", "u")


def test_from_env(monkeypatch):
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    assert LLMClient.from_env() is None
    monkeypatch.setenv("LLM_BASE_URL", "http://x/v1")
    monkeypatch.setenv("LLM_API_KEY", "k")
    assert LLMClient.from_env() is None            # model still missing
    monkeypatch.setenv("LLM_MODEL", "m")
    assert LLMClient.from_env().model == "m"
