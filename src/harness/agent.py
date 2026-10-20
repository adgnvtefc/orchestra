"""The agent loop: call the model, run any tool calls, feed results back, repeat."""

import time
from collections.abc import Sequence

from openai import OpenAI
from openai.types.chat import ChatCompletion

from harness.session import Session
from harness.tools import Toolset
from harness.tracer import Tracer


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

# PROVIDER-SPECIFIC: oMLX/Qwen puts thinking in `reasoning_content`; dropping it is a cache
# optimization. Anthropic's native API requires thinking blocks to be sent back. Fix later.
def run_agent(task: str, session: Session, config: dict, client: OpenAI, toolset: Toolset, tracer: Tracer) -> str:
    """Run one turn: add `task` to the session, loop until the model answers without
    calling a tool, or max_steps runs out.

    `session` owns the conversation: earlier turns are already in it (its system prompt
    is message 0), and everything this turn adds goes through session.append(), which
    also saves it. `config` comes from config_loader.load_agent_config; `client` from
    make_client; `toolset` holds only the tools this agent's config lists; `tracer`
    records the turn. The agent is handed everything it needs and never goes looking
    for settings itself.

    Return the model's final text answer. Print each tool call and a short
    preview of its result as you go, so you can watch the agent work.
    """
    max_steps = config["max_steps"]
    tools = toolset.schemas()  # same list every turn, which keeps the prompt cache hitting

    # run_start now. run_end comes from tracer.finish() below, or, if this function
    # raises, from the `with Tracer(...)` block of whoever created the tracer.
    tracer.start(task, config, tools, approver=toolset.policy.approver.__name__)

    # The conversation IS the agent's memory. Every turn we send all of it: the system
    # prompt, every earlier turn, and now this task.
    session.append({"role": "user", "content": task})

    for step in range(1, max_steps + 1):
        # calls the client with model and these messages
        started = time.monotonic()
        response = call_model(client, session.messages, config["model"], config["temperature"], tools)
        tracer.model_response(step, response, _since(started))
        msg = response.choices[0].message

        # Append the assistant turn exactly as returned, tool_calls and their ids included,
        # so the tool results we add next have something to point back to.
        # reasoning_content is dropped: smaller context, and earlier messages stay byte-identical,
        # which keeps the server's prompt cache hitting. (Keeping it is a Phase 3 experiment.)
        session.append(msg.model_dump(exclude_none=True, exclude={"reasoning_content"}))

        # the model needs to make tool calls for there to be indicated a final response
        if not msg.tool_calls:
            answer = msg.content or ""
            tracer.finish("answered", answer)
            return answer

        # The model may request several tools in one turn; each needs exactly one reply.
        for call in msg.tool_calls:
            # prints function call name and arguments
            print(f"[step {step}] {call.function.name}({call.function.arguments})")
            # executes the tool; the toolset refuses tools this agent wasn't given
            started = time.monotonic()
            result = toolset.execute(call.function.name, call.function.arguments)
            tracer.tool_call(step, call, result, _since(started))
            # determines preview to print
            preview = result if len(result) <= 200 else result[:200] + "..."
            print(f"  -> {preview}")
            # give tool full context id
            session.append({"role": "tool", "tool_call_id": call.id, "content": result})

    answer = f"Stopped after {max_steps} steps without a final answer."
    tracer.finish("max_steps", answer)
    return answer


def _since(started: float) -> float:
    """Seconds since `started` (a time.monotonic() value), rounded for readable traces."""
    return round(time.monotonic() - started, 3)
