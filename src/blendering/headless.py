"""Headless renderer: stream orchestrator events to stdout. Useful for CI / scripting."""

from __future__ import annotations

import asyncio
import contextlib
import signal
import sys
from typing import Any

from rich.console import Console

from .config import Settings
from .orchestrator import EventBus, run
from .schemas import RunOutcome
from .tui.events import (
    ActorTextDelta,
    CriticVerdictEvent,
    IterationStart,
    RunFinished,
    StatusMessage,
    ToolCallResult,
    ToolCallStart,
    ViewportUpdate,
)

console = Console()


def _render(event: Any) -> None:
    if isinstance(event, ActorTextDelta):
        console.print(event.text, end="", soft_wrap=True)
        console.file.flush()
        return
    elif isinstance(event, ToolCallStart):
        args = str(event.arguments)
        if len(args) > 200:
            args = args[:199] + "…"
        console.print(f"\n[yellow]→ tool[/] [bold]{event.name}[/] {args}")
    elif isinstance(event, ToolCallResult):
        color = "red" if event.is_error else "green"
        console.print(f"[{color}]← {event.name}[/]: {event.result_preview}")
    elif isinstance(event, CriticVerdictEvent):
        v = event.verdict
        color = {"done": "bold green", "stuck": "bold red", "continue": "bold blue"}.get(
            v.status, "white"
        )
        console.print(
            f"[{color}]critic[/] ({v.confidence:.2f}): {v.reasoning}\n"
            f"  hint: {v.next_step_hint or '(none)'}"
        )
    elif isinstance(event, ViewportUpdate):
        console.print(f"[dim]viewport → {event.path}[/]")
    elif isinstance(event, IterationStart):
        console.print(f"\n[bold cyan]── iteration {event.n}/{event.total} ──[/]")
    elif isinstance(event, StatusMessage):
        color = {"info": "dim", "warn": "yellow", "error": "red"}.get(event.level, "dim")
        console.print(f"[{color}]{event.text}[/]")
    elif isinstance(event, RunFinished):
        o = event.outcome
        colors = {
            "done": "bold green",
            "stuck": "bold red",
            "cancelled": "yellow",
            "max_iterations": "yellow",
            "error": "bold red",
        }
        color = colors.get(o.status, "white")
        extra = f" — {o.error}" if o.error else ""
        console.print(f"\n[{color}]Run {o.status}[/] after {o.iterations} iterations.{extra}")
    console.file.flush()


async def run_headless(
    settings: Settings, user_prompt: str, clear_scene: bool = False
) -> RunOutcome:
    bus = EventBus()
    cancel = asyncio.Event()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, cancel.set)

    run_task = asyncio.create_task(
        run(settings, user_prompt, bus, cancel, clear_scene=clear_scene)
    )
    outcome: RunOutcome | None = None
    while True:
        event = await bus.get()
        _render(event)
        if isinstance(event, RunFinished):
            outcome = event.outcome
            break
    await run_task
    assert outcome is not None
    return outcome


def main(settings: Settings, user_prompt: str, clear_scene: bool = False) -> int:
    outcome = asyncio.run(run_headless(settings, user_prompt, clear_scene=clear_scene))
    return 0 if outcome.status == "done" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(0)
