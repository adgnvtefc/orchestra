"""providers/: reaching the model server. make_client() builds the client, call_model() makes one call."""

import pytest

from harness.providers import call_model, omlx
from tests.fakes import FakeClient, reply


@pytest.fixture
def env(monkeypatch):
    """A controlled environment: the real .env is never read, and OMLX_* vars start unset."""
    monkeypatch.setattr(omlx, "load_dotenv", lambda: None)
    monkeypatch.delenv("OMLX_API_KEY", raising=False)
    monkeypatch.delenv("OMLX_BASE_URL", raising=False)
    return monkeypatch


class TestMakeClient:
    def test_missing_key_exits_with_help(self, env):
        with pytest.raises(SystemExit, match="OMLX_API_KEY is not set"):
            omlx.make_client()

    def test_empty_key_counts_as_missing(self, env):
        env.setenv("OMLX_API_KEY", "")
        with pytest.raises(SystemExit):
            omlx.make_client()

    def test_default_base_url(self, env):
        env.setenv("OMLX_API_KEY", "test-key")
        client = omlx.make_client()
        assert str(client.base_url).rstrip("/") == "http://127.0.0.1:8000/v1"
        assert client.api_key == "test-key"

    def test_base_url_from_env(self, env):
        env.setenv("OMLX_API_KEY", "test-key")
        env.setenv("OMLX_BASE_URL", "http://example.test:9999/v1")
        assert str(omlx.make_client().base_url).rstrip("/") == "http://example.test:9999/v1"

    def test_long_timeout_for_local_models(self, env):
        env.setenv("OMLX_API_KEY", "test-key")
        assert omlx.make_client().timeout == 600


class TestCallModel:
    def test_returns_the_whole_response(self):
        # The whole response, so usage and finish_reason reach the trace.
        client = FakeClient([reply("hi", usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4})])
        response = call_model(client, [{"role": "user", "content": "x"}], "m", 0.5, [])
        assert response.choices[0].message.content == "hi"
        assert response.choices[0].finish_reason == "stop"
        assert response.usage.prompt_tokens == 3
        assert client.requests == [{"model": "m", "messages": [{"role": "user", "content": "x"}], "temperature": 0.5}]

    def test_includes_tools_when_given(self):
        client = FakeClient([reply("hi")])
        call_model(client, [], "m", 0.5, [{"type": "function"}])
        assert client.requests[0]["tools"] == [{"type": "function"}]
