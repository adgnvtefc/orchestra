"""oMLX: a local MLX model server that speaks the OpenAI chat-completions API.

Everything that knows how to reach and talk to the model server lives here: building
the client (URL, API key, timeout) and making one call. The rest of the harness only
calls these two functions.

(Responses are still the openai SDK's ChatCompletion, which agent.py and tracer.py read
directly, including oMLX's extra `reasoning_content`. A provider-neutral response shape
belongs here too, but it's designed when a second provider exists to compare against.)
"""

import os
from collections.abc import Sequence

from openai import OpenAI
from openai.types.chat import ChatCompletion


class MissingAPIKey(Exception):
    """The model server's API key isn't set. An ordinary exception, so a program using the
    harness can catch it; only an interface (the CLI) decides to exit over it."""


def make_client() -> OpenAI:
    """Return an OpenAI client pointed at oMLX, from OMLX_BASE_URL and OMLX_API_KEY.

    Only *reads* environment variables. Loading a .env file into them is the interface's
    job (the CLI does it): a library shouldn't change its host program's environment.
    """
    # get the api key
    api_key = os.environ.get("OMLX_API_KEY")
    if not api_key:
        raise MissingAPIKey("OMLX_API_KEY is not set: put it in .env (see .env.example) or in the environment")
    return OpenAI(
        base_url=os.environ.get("OMLX_BASE_URL", "http://127.0.0.1:8000/v1"),
        api_key=api_key,
        timeout=600,
    )


def call_model(client: OpenAI, messages: Sequence[dict], model: str, temperature: float, tools: list[dict]) -> ChatCompletion:
    """Send `messages` + `tools` (schemas) to `model`. Return the whole response.

    The whole response, not just the message: `usage` (token counts, cache hits, and
    oMLX's timing extras) and `finish_reason` live outside the message, and the trace wants them.
    """
    # This becomes: POST {base_url}/chat/completions with a JSON body like
    #   {
    #     "model": "Qwen3.8-27B-MLX-8bit",       which model on the server (one server can host many)
    #     "messages": [                           the ENTIRE conversation so far; the model is stateless
    #       {"role": "system", "content": "..."},
    #       {"role": "user", "content": "..."},
    #       {"role": "assistant", "content": None, "tool_calls": [...]},
    #       {"role": "tool", "tool_call_id": "call_1", "content": "..."},
    #     ],
    #     "tools": [                              menu of functions the model MAY ask us to run
    #       {"type": "function", "function": {"name": ..., "description": ..., "parameters": <JSON Schema>}},
    #     ],
    #     "temperature": 0.6                      randomness; lower = more consistent tool calls
    #   }
    kwargs = {
        "model": model,
        "messages": list(messages),  # a session hands us a read-only tuple; the request wants a list
        "temperature": temperature,
    }
    # Some servers reject an empty "tools" list, so only send it when the agent has tools.
    if tools:
        kwargs["tools"] = tools
    response = client.chat.completions.create(**kwargs)

    # The response JSON looks like
    #   {
    #     "id": "...", "model": "...",
    #     "choices": [                            list because you can request n>1 samples; we use 1
    #       {
    #         "index": 0,
    #         "finish_reason": "stop" | "tool_calls" | "length",   why the model stopped generating
    #         "message": {
    #           "role": "assistant",
    #           "content": "text answer" or None,
    #           "tool_calls": [                   present only when the model wants tools run
    #             {"id": "call_1", "type": "function",
    #              "function": {"name": "bash", "arguments": "{\"command\": \"ls\"}"}},   arguments is a JSON STRING
    #           ],
    #         },
    #       }
    #     ],
    #     "usage": {"prompt_tokens": ..., "completion_tokens": ...},
    #   }
    # The SDK parses this into Python objects, so fields are attributes: response.choices[0].message.content
    return response
