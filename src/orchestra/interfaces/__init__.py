"""Interfaces: where a human talks to the harness. They drive the orchestrator; nothing imports them.

    cli.py      the terminal: `orchestra "task"`, `orchestra chat`, `--resume`, and the y/N approver

An interface supplies the approver (how "ask" tools get a human's answer) and decides
how to show what happens. The orchestrator never reads from or prints to anything.
"""
