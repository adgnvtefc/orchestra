"""Model calls for any app: each one is made and recorded in the same function.

    from orchestra import chat
    reply = chat(app="jewels", model="Qwen3.8-27B-MLX-8bit", messages=[...], metadata={"job": job_id})
    reply.content, reply.id

Recording lives inside the call, not next to it, so no call can happen unrecorded. The
record (format in records.py) holds the full request, so each one is a complete example
on its own: exactly what the model was sent, and everything it sent back.

This module knows nothing about the model server: talking to it, and what its responses
look like, is the provider's (providers/omlx.py).
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from orchestra import records
from orchestra.providers import make_client, send

if TYPE_CHECKING:  # only for the `client` type hint; never imported at runtime
    from openai import OpenAI


@dataclass(frozen=True)
class Reply:
    """What a chat call returns: the model's answer, plus the id of its record."""

    id: str  # the record's id: how the caller refers to this call later
    content: str | None
    reasoning: str | None  # the model's thinking, when the server splits it out of the content
    tool_calls: list[dict]  # [{"id", "type", "function": {"name", "arguments"}}], or []
    finish_reason: str | None  # "stop", "tool_calls", or "length" (cut off by max_tokens)
    model: str | None  # the model the server says answered
    usage: dict  # {"prompt", "completion", "cached"} tokens
    seconds: float

    @property
    def message(self) -> dict:
        """The reply as an assistant message, ready to append to the conversation for the next call.

        Reasoning is left out: a smaller context, and earlier messages stay byte-identical,
        which keeps the server's prompt cache hitting. (It's kept in the record.)
        """
        message = {"role": "assistant"}
        if self.content is not None:
            message["content"] = self.content
        if self.tool_calls:  # their ids are what the tool results point back to
            message["tool_calls"] = self.tool_calls
        return message


def chat(*, app: str, model: str, messages: Sequence[dict], temperature: float | None = None,
         max_tokens: int | None = None, tools: list[dict] | None = None, metadata: dict | None = None,
         parent: str | None = None, session: str | None = None, client: "OpenAI | None" = None) -> Reply:
    """Send `messages` to `model`, record the call under `app`, and return the reply.

    `metadata` is yours: any JSON-storable dict, kept with the record and never read here.
    `parent`/`session` link the record to whatever caused it (records.py). `client` is for
    reusing one, or for tests; by default one is made from the environment (make_client).

    A failed call is recorded too, with its error, and then raised.
    """
    record = records.start("chat", app=app, metadata=metadata, parent=parent, session=session)
    # Exactly what's sent, and recorded as such. Unset options are left out, so the server's defaults apply.
    request = {"model": model, "messages": list(messages)}
    if temperature is not None:
        request["temperature"] = temperature
    if max_tokens is not None:
        request["max_tokens"] = max_tokens
    if tools:  # some servers reject an empty tools list
        request["tools"] = tools
    record["request"] = request

    started = time.monotonic()
    try:
        answer = send(client or make_client(), request)
    except BaseException as error:  # Ctrl-C included: an interrupted call is still a call
        record["seconds"] = _since(started)
        record["error"] = f"{type(error).__name__}: {error}"
        records.save(record)
        raise

    reply = Reply(
        id=record["id"],
        content=answer["content"],
        reasoning=answer["reasoning"],
        tool_calls=answer["tool_calls"],
        finish_reason=answer["finish_reason"],
        model=answer["model"],
        usage=answer["usage"],
        seconds=_since(started),
    )
    record["seconds"] = reply.seconds
    record["response"] = {
        "content": reply.content,
        "reasoning": reply.reasoning,
        "tool_calls": reply.tool_calls,
        "finish_reason": reply.finish_reason,
        "model": reply.model,
    }
    record["usage"] = {**reply.usage, "server": answer["server_usage"]}
    records.save(record)
    return reply


def _since(started: float) -> float:
    return round(time.monotonic() - started, 3)
