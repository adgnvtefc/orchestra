"""The agent loop: call the model, run any tool calls, feed results back, repeat."""

import time

from openai import OpenAI

from orchestra import records
from orchestra.calls import chat
from orchestra.session import Session
from orchestra.tools import Toolset
from orchestra.tracer import Tracer


# PROVIDER-SPECIFIC: oMLX/Qwen puts thinking in `reasoning_content`; Reply.message drops it
# (a cache optimization). Anthropic's native API requires thinking blocks to be sent back. Fix later.
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

    # turn_start now. turn_end comes from tracer.finish() below, or, if this function
    # raises, from the `with Tracer(...)` block of whoever created the tracer.
    tracer.start(task, config, approver=toolset.policy.name, sandbox=toolset.context.sandbox.name)

    # The conversation IS the agent's memory. Every turn we send all of it: the system
    # prompt, every earlier turn, and now this task.
    session.append({"role": "user", "content": task})

    for step in range(1, max_steps + 1):
        # One model call, made and recorded by chat(): the full request and response go to
        # the session's app's log. The trace only notes which record it was.
        reply = chat(app=session.app, model=config["model"], messages=session.messages,
                     temperature=config["temperature"], tools=tools, session=session.id, client=client)
        tracer.model_call(step, reply)

        # Append the assistant turn, tool_calls and their ids included, so the tool results
        # we add next have something to point back to. (Reply.message leaves reasoning out.)
        session.append(reply.message)

        # the model needs to make tool calls for there to be indicated a final response
        if not reply.tool_calls:
            answer = reply.content or ""
            tracer.finish("answered")
            return answer

        # The model may request several tools in one turn; each needs exactly one reply.
        for call in reply.tool_calls:
            result, record_id = _run_tool(toolset, call, session, parent=reply.id)
            tracer.tool_run(step, record_id, call, result)
            # give tool full context id
            session.append({"role": "tool", "tool_call_id": call["id"], "content": result})

    answer = f"Stopped after {max_steps} steps without a final answer."
    tracer.finish("max_steps")
    return answer


def _run_tool(toolset: Toolset, call: dict, session: Session, parent: str) -> tuple[str, str]:
    """Run one tool call and record it. Returns (result, the record's id).

    The tool record holds what only running it produced: who approved it, the result, and
    how long it took. Its name and arguments are in the model call that asked for it
    (`parent`), found by `tool_call_id`, so they aren't copied here.
    """
    record = records.start("tool", app=session.app, session=session.id, parent=parent)
    started = time.monotonic()
    # executes the tool; the toolset refuses tools this agent wasn't given (never raises)
    run = toolset.execute(call["function"]["name"], call["function"]["arguments"])
    record.update(seconds=_since(started), tool_call_id=call["id"], approval=run.approval, result=run.result)
    records.save(record)
    return run.result, record["id"]


def _since(started: float) -> float:
    """Seconds since `started` (a time.monotonic() value), rounded for readable traces."""
    return round(time.monotonic() - started, 3)
