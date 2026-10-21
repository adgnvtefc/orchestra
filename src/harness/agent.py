"""The agent loop: call the model, run any tool calls, feed results back, repeat."""

import time

from openai import OpenAI

from harness.providers import call_model
from harness.session import Session
from harness.tools import Toolset
from harness.tracer import Tracer


# PROVIDER-SPECIFIC: oMLX/Qwen puts thinking in `reasoning_content`; dropping it is a cache
# optimization. Anthropic's native API requires thinking blocks to be sent back. Fix later.
def run_agent(task: str, session: Session, config: dict, client: OpenAI, toolset: Toolset, tracer: Tracer) -> str:
    """Run one turn: add `task` to the session, loop until the model answers without
    calling a tool, or max_steps runs out.

    `session` owns the conversation: earlier turns are already in it (its system prompt
    is message 0), and everything this turn adds goes through session.append(), which
    also saves it. `config` comes from config.load_agent_config; `client` from
    make_client; `toolset` holds only the tools this agent's config lists; `tracer`
    records the turn. The agent is handed everything it needs and never goes looking
    for settings itself.

    Return the model's final text answer. Nothing is printed here: everything that
    happens is reported to the tracer, and interfaces show it by listening to that.
    """
    max_steps = config["max_steps"]
    tools = toolset.schemas()  # same list every turn, which keeps the prompt cache hitting

    # run_start now. run_end comes from tracer.finish() below, or, if this function
    # raises, from the `with Tracer(...)` block of whoever created the tracer.
    tracer.start(task, config, tools, approver=toolset.policy.name)

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
            # executes the tool; the toolset refuses tools this agent wasn't given
            started = time.monotonic()
            result = toolset.execute(call.function.name, call.function.arguments)
            tracer.tool_call(step, call, result, _since(started))
            # give tool full context id
            session.append({"role": "tool", "tool_call_id": call.id, "content": result})

    answer = f"Stopped after {max_steps} steps without a final answer."
    tracer.finish("max_steps", answer)
    return answer


def _since(started: float) -> float:
    """Seconds since `started` (a time.monotonic() value), rounded for readable traces."""
    return round(time.monotonic() - started, 3)
