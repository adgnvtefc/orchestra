"""Live tests against a real model on oMLX. Opt-in and slow: `uv run pytest -m live`.

These are the Phase 1 acceptance tests, automated. Model output varies run to run,
so they check outcomes (files on disk, facts in the answer), never exact wording.
"""

import os
import urllib.request

import pytest
from dotenv import load_dotenv

from harness import orchestrator
from harness.policy import approve_all

pytestmark = pytest.mark.live



@pytest.fixture(scope="module", autouse=True)
def require_server():
    load_dotenv()
    if not os.environ.get("OMLX_API_KEY"):
        pytest.skip("OMLX_API_KEY not set")
    base = os.environ.get("OMLX_BASE_URL", "http://127.0.0.1:8000/v1")
    try:
        urllib.request.urlopen(base + "/models", timeout=3)
    except urllib.error.HTTPError:
        pass  # 401 etc.: the server is up
    except OSError:
        pytest.skip(f"no model server at {base}")


def test_lists_empty_workspace():
    # environment="none": base's python-dev environment would put .venv/ and .harness/ in the
    # workspace, and an agent that runs `ls -la` would rightly say it isn't empty.
    answer = orchestrator.run("What files are in the workspace? Answer briefly.", approver=approve_all, environment="none")
    assert "empty" in answer.lower() or "no files" in answer.lower(), f"model's answer: {answer!r}"


def test_writes_and_runs_a_script(sessions_dir):
    answer = orchestrator.run("Write primes.py that prints the first 10 primes, run it, and tell me the output", approver=approve_all)
    (session,) = [p for p in sessions_dir.iterdir() if p.is_dir()]
    assert (session / "workspace" / "primes.py").is_file()
    assert "29" in answer  # the 10th prime


def test_recovers_from_missing_file():
    answer = orchestrator.run("Read notes.txt and summarize it", approver=approve_all)
    # The model may word it many ways; what matters is it reports the file as missing, not invented.
    missing = ("not found", "doesn't exist", "does not exist", "couldn't find", "could not find",
               "no such", "isn't", "is not", "wasn't", "no file", "not exist", "missing", "empty")
    assert any(phrase in answer.lower() for phrase in missing), f"model's answer: {answer!r}"
