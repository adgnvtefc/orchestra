"""providers/: reaching the model server. make_client() builds a client, client() is the one
every call uses, and send() makes one call and translates the answer into plain dicts
(calls.chat() records it)."""

import pytest

from orchestra.providers import MissingAPIKey, omlx
from tests.fakes import FakeClient, call, reply


@pytest.fixture
def env(monkeypatch):
    """A controlled environment: the real .env is never read, and OMLX_* vars start unset."""
    monkeypatch.delenv("OMLX_API_KEY", raising=False)
    monkeypatch.delenv("OMLX_BASE_URL", raising=False)
    return monkeypatch


class TestMakeClient:
    def test_missing_key_is_an_ordinary_exception(self, env):
        # A program using the harness can catch it; SystemExit would have ended that program.
        with pytest.raises(MissingAPIKey, match="OMLX_API_KEY is not set"):
            omlx.make_client()
        assert issubclass(MissingAPIKey, Exception)

    def test_does_not_read_dotenv_files(self, env, tmp_path, monkeypatch):
        # Loading .env is the interface's job; the provider only reads the environment.
        (tmp_path / ".env").write_text("OMLX_API_KEY=from-a-file\n")
        monkeypatch.chdir(tmp_path)
        with pytest.raises(MissingAPIKey):
            omlx.make_client()

    def test_empty_key_counts_as_missing(self, env):
        env.setenv("OMLX_API_KEY", "")
        with pytest.raises(MissingAPIKey):
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


class TestClient:
    def test_made_once_then_reused(self, env):
        env.setenv("OMLX_API_KEY", "test-key")
        first = omlx.client()
        assert omlx.client() is first  # one client, so calls share its connections

    def test_a_missing_key_is_not_cached(self, env):
        with pytest.raises(MissingAPIKey):
            omlx.client()
        env.setenv("OMLX_API_KEY", "test-key")  # setting the key later works
        assert omlx.client().api_key == "test-key"

    def test_a_missing_key_fails_before_a_session_exists(self, env, tmp_path, monkeypatch, sessions_dir):
        from orchestra import orchestrator
        (tmp_path / "agents").mkdir()
        (tmp_path / "agents" / "base.md").write_text("---\nmodel: m\ntemperature: 0.1\nmax_steps: 3\n---\nPrompt.\n")
        monkeypatch.chdir(tmp_path)
        with pytest.raises(MissingAPIKey):
            orchestrator.new_conversation("base", app="orchestra-test")
        assert not sessions_dir.exists()


class TestSend:
    REQUEST = {"model": "m", "messages": [{"role": "user", "content": "x"}], "temperature": 0.5}

    def send(self, client):
        """send(), with `client` (a scripted FakeClient) as oMLX's client."""
        omlx.client = lambda: client  # the autouse no_model_client fixture puts the real one back afterwards
        return omlx.send(self.REQUEST)

    def test_sends_the_request_exactly_as_given(self):
        client = FakeClient([reply("hi")])
        self.send(client)
        assert client.requests == [self.REQUEST]

    def test_answer_as_a_plain_dict(self):
        usage = {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25, "prompt_tokens_details": {"cached_tokens": 12}}
        answer = self.send(FakeClient([reply("42", reasoning="6*7", usage=usage)]))
        assert answer == {
            "content": "42", "reasoning": "6*7", "tool_calls": [], "finish_reason": "stop", "model": "fake-model",
            "usage": {"prompt": 20, "completion": 5, "cached": 12},
            "server_usage": answer["server_usage"],
        }
        assert answer["server_usage"]["total_tokens"] == 25  # the server's own block, kept whole

    def test_tool_calls_are_plain_dicts(self):
        wanted = call("bash", {"command": "ls"}, id="call_1")
        answer = self.send(FakeClient([reply(tool_calls=[wanted])]))
        assert (answer["tool_calls"], answer["finish_reason"]) == ([wanted], "tool_calls")

    def test_no_usage_or_reasoning_reported(self):
        answer = self.send(FakeClient([reply("hi")]))
        assert (answer["usage"], answer["server_usage"], answer["reasoning"]) == (
            {"prompt": 0, "completion": 0, "cached": 0}, None, None)

    def test_usage_without_cache_details(self):
        usage = {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
        assert self.send(FakeClient([reply("hi", usage=usage)]))["usage"] == {"prompt": 3, "completion": 1, "cached": 0}

    def test_server_errors_are_raised(self):
        with pytest.raises(ConnectionError):
            self.send(FakeClient([ConnectionError("gone")]))

