"""A scripted stand-in for the OpenAI client, so the agent loop can be tested without a model.

Responses are built as the openai SDK's *real* pydantic types (ChatCompletion), so
attribute access and model_dump() behave exactly as they do against oMLX.
"""

import copy
import itertools
import json

from openai.types.chat import ChatCompletion

from orchestra.policy import Policy, deny_all

_ids = itertools.count(1)

# A policy for tests that don't care about approval: "allow" tools run, "ask" tools are refused.
NO_HUMAN = Policy(approver=deny_all)


def call(name: str, args: dict | str | None = None, id: str | None = None) -> dict:
    """One tool call as the model would send it. `args` may be a raw string, to simulate bad JSON."""
    arguments = args if isinstance(args, str) else json.dumps(args or {})
    return {
        "id": id or f"call_{next(_ids)}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def reply(
    content: str | None = None,
    tool_calls: list[dict] | None = None,
    reasoning: str | None = None,
    usage: dict | None = None,
) -> ChatCompletion:
    """One model response. With tool_calls, finish_reason is "tool_calls"; otherwise "stop".

    `usage`, if given, is the token-count block, e.g. {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}.
    """
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if reasoning is not None:
        message["reasoning_content"] = reasoning  # oMLX/Qwen's extra field
    return ChatCompletion.model_validate(
        {
            "id": f"resp_{next(_ids)}",
            "object": "chat.completion",
            "created": 0,
            "model": "fake-model",
            "choices": [{"index": 0, "finish_reason": "tool_calls" if tool_calls else "stop", "message": message}],
            **({"usage": usage} if usage else {}),
        }
    )


class FakeClient:
    """Mimics `client.chat.completions.create(**kwargs)`, returning scripted replies in order.

    A scripted item that's an exception is raised instead of returned.

    `requests` records a deep copy of every request's kwargs. A copy matters: the agent
    keeps appending to the same `messages` list, so without it every recorded request
    would show the final conversation instead of what was actually sent at that step.
    """

    def __init__(self, replies):
        self._replies = iter(replies)
        self.requests: list[dict] = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs) -> ChatCompletion:
        self.requests.append(copy.deepcopy(kwargs))
        try:
            item = next(self._replies)
        except StopIteration:
            raise AssertionError(f"model called more times than scripted ({len(self.requests)} calls)") from None
        if isinstance(item, BaseException):  # script a failure, e.g. the server going away
            raise item
        return item


def always(response_factory):
    """An endless reply stream, e.g. always(lambda: reply(tool_calls=[call("echo", {"text": "x"})]))."""
    while True:
        yield response_factory()
