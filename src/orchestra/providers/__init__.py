"""Providers: how the harness reaches a model server.

    omlx.py     the local oMLX server (OpenAI-compatible): client(), send()

To add one (e.g. Anthropic's native API): a module here with the same functions,
send() returning the same plain dict. (That dict was designed from one example, oMLX;
expect it to change when a second provider gives a real comparison.)
"""

from orchestra.providers.omlx import MissingAPIKey, send

__all__ = ["MissingAPIKey", "send"]
