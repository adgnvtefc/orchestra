"""Providers: how the harness reaches a model server.

    omlx.py     the local oMLX server (OpenAI-compatible): make_client(), call_model()

To add one (e.g. Anthropic's native API): a module here with the same two functions.
That's also when the provider-neutral response format gets designed, from two real
examples instead of one.
"""

from harness.providers.omlx import MissingAPIKey, call_model, make_client

__all__ = ["MissingAPIKey", "call_model", "make_client"]
