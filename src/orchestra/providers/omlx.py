"""oMLX: a local MLX model server that speaks the OpenAI chat-completions API.

Everything that knows how to talk to the model server lives here: building the client
(URL, API key, timeout), sending one request, and reading the response (including
oMLX's extra `reasoning_content`). The rest of orchestra calls these two functions and
gets plain dicts back, never the openai SDK's types. Recording the call is calls.chat()'s job.

(A second provider, e.g. Anthropic's native API, would be a module here with the same two
functions returning the same dict.)
"""

import os

from openai import OpenAI


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


def send(client: OpenAI, request: dict) -> dict:
    """Send one chat request; return the answer as a plain dict:

        content, reasoning, tool_calls ([{"id", "type", "function": {"name", "arguments"}}], or []),
        finish_reason, model (what the server says answered),
        usage ({"prompt", "completion", "cached"} tokens), server_usage (the server's own block)

    `request` is sent exactly as given: {"model", "messages", and any of "temperature",
    "max_tokens", "tools"}.
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
    response = client.chat.completions.create(**request)

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
    #           "reasoning_content": "...",       oMLX/Qwen's thinking, split out of the content
    #           "tool_calls": [                   present only when the model wants tools run
    #             {"id": "call_1", "type": "function",
    #              "function": {"name": "bash", "arguments": "{\"command\": \"ls\"}"}},   arguments is a JSON STRING
    #           ],
    #         },
    #       }
    #     ],
    #     "usage": {"prompt_tokens": ..., "completion_tokens": ..., "prompt_tokens_details": {"cached_tokens": ...}},
    #   }
    # The SDK parses this into Python objects, so fields are attributes: response.choices[0].message.content
    choice = response.choices[0]
    message = choice.message
    usage = {"prompt": 0, "completion": 0, "cached": 0}
    if response.usage:
        details = response.usage.prompt_tokens_details
        usage = {
            "prompt": response.usage.prompt_tokens or 0,
            "completion": response.usage.completion_tokens or 0,
            "cached": (details.cached_tokens or 0) if details else 0,
        }
    return {
        "content": message.content,
        "reasoning": getattr(message, "reasoning_content", None),
        "tool_calls": [call.model_dump() for call in message.tool_calls or []],
        "finish_reason": choice.finish_reason,
        "model": response.model,
        "usage": usage,
        # oMLX's own usage block adds timings (tokens per second, time to first token)
        "server_usage": response.usage.model_dump() if response.usage else None,
    }
