"""Textual TUI: streams Actor thinking, shows viewport screenshots, supports interrupt."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, RichLog, Static

try:  # textual-image is optional at import time so tests don't need a terminal
    from textual_image.widget import Image as TextualImage  # type: ignore
    _HAS_IMAGE = True
except Exception:  # pragma: no cover
    _HAS_IMAGE = False
    TextualImage = None  # type: ignore

from ..config import Settings
from ..orchestrator import EventBus, run
from .events import (
    ActorTextDelta,
    CriticVerdictEvent,
    IterationStart,
    RunFinished,
    StatusMessage,
    ToolCallResult,
    ToolCallStart,
    ViewportUpdate,
)


class ThinkingLog(RichLog):
    """Streaming log of Actor text, tool calls, and Critic verdicts."""

    DEFAULT_CSS = """
    ThinkingLog { border: round $primary; padding: 0 1; }
    """

    def __init__(self) -> None:
        super().__init__(wrap=True, markup=False, highlight=False, auto_scroll=True)
        self.border_title = "Thinking"


class ViewportPane(Static):
    """Latest viewport screenshot."""

    DEFAULT_CSS = """
    ViewportPane { border: round $accent; padding: 0 1; align: center middle; }
    """

    def __init__(self) -> None:
        super().__init__("(no screenshot yet)")
        self.border_title = "Viewport"

    def show_image(self, path: Path) -> None:
        if _HAS_IMAGE and TextualImage is not None:
            self.update("")
            self.remove_children()
            self.mount(TextualImage(str(path)))
        else:
            self.update(f"[viewport saved to {path}]")


class StatusBar(Static):
    DEFAULT_CSS = """
    StatusBar { dock: bottom; height: 1; background: $boost; padding: 0 1; }
    """

    def set_state(
        self,
        *,
        iteration: int = 0,
        total: int = 0,
        verdict_status: str = "—",
        running: bool = True,
    ) -> None:
        running_marker = "● running" if running else "○ idle"
        self.update(
            Text.from_markup(
                f"[bold]{running_marker}[/]   "
                f"iter [cyan]{iteration}/{total}[/]   "
                f"verdict [magenta]{verdict_status}[/]   "
                f"[dim]i=interrupt  q=quit[/]"
            )
        )


class BlenderingApp(App[None]):
    CSS = """
    Screen { layout: vertical; }
    #panes { height: 1fr; }
    #left { width: 1fr; }
    #right { width: 1fr; }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("i", "interrupt", "Interrupt"),
        Binding("ctrl+c", "interrupt", "Interrupt", show=False),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, settings: Settings, user_prompt: str) -> None:
        super().__init__()
        self.settings = settings
        self.user_prompt = user_prompt
        self.bus = EventBus()
        self.cancel = asyncio.Event()
        self._run_task: asyncio.Task[Any] | None = None
        self._consumer_task: asyncio.Task[Any] | None = None
        self._iter = 0
        self._total = settings.loop.max_iterations
        self._last_verdict_status = "—"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical(id="panes"), Horizontal():
            self.log_widget = ThinkingLog()
            self.viewport = ViewportPane()
            yield self.log_widget
            yield self.viewport
        self.status = StatusBar()
        yield self.status
        yield Footer()

    async def on_mount(self) -> None:
        self.title = "blendering"
        self.sub_title = self.user_prompt[:80]
        self.status.set_state(iteration=0, total=self._total, running=True)
        self.log_widget.write(Text.from_markup(f"[bold]Goal:[/] {self.user_prompt}"))
        self._run_task = asyncio.create_task(
            run(self.settings, self.user_prompt, self.bus, self.cancel)
        )
        self._consumer_task = asyncio.create_task(self._consume_events())

    async def _consume_events(self) -> None:
        while True:
            event = await self.bus.get()
            self._handle_event(event)
            if isinstance(event, RunFinished):
                return

    def _handle_event(self, event: Any) -> None:
        if isinstance(event, ActorTextDelta):
            self.log_widget.write(Text(event.text, style="white"), expand=True)
        elif isinstance(event, ToolCallStart):
            args_preview = str(event.arguments)
            if len(args_preview) > 160:
                args_preview = args_preview[:159] + "…"
            self.log_widget.write(
                Text.from_markup(f"[yellow]→ tool[/] [bold]{event.name}[/] {args_preview}")
            )
        elif isinstance(event, ToolCallResult):
            color = "red" if event.is_error else "green"
            self.log_widget.write(
                Text.from_markup(f"[{color}]← {event.name}[/]: {event.result_preview}")
            )
        elif isinstance(event, CriticVerdictEvent):
            v = event.verdict
            self._last_verdict_status = v.status
            color = {"done": "bold green", "stuck": "bold red", "continue": "bold blue"}.get(
                v.status, "white"
            )
            self.log_widget.write(
                Text.from_markup(
                    f"[{color}]critic[/] ({v.confidence:.2f}): {v.reasoning}\n"
                    f"  hint: {v.next_step_hint or '(none)'}"
                )
            )
            self.status.set_state(
                iteration=self._iter,
                total=self._total,
                verdict_status=self._last_verdict_status,
            )
        elif isinstance(event, ViewportUpdate):
            self.viewport.show_image(Path(event.path))
        elif isinstance(event, IterationStart):
            self._iter = event.n
            self.log_widget.write(
                Text.from_markup(f"\n[bold cyan]── iteration {event.n}/{event.total} ──[/]")
            )
            self.status.set_state(
                iteration=self._iter,
                total=self._total,
                verdict_status=self._last_verdict_status,
            )
        elif isinstance(event, StatusMessage):
            color = {"info": "dim", "warn": "yellow", "error": "red"}.get(event.level, "dim")
            self.log_widget.write(Text.from_markup(f"[{color}]{event.text}[/]"))
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
            self.log_widget.write(
                Text.from_markup(
                    f"\n[{color}]Run {o.status}[/] after {o.iterations} iterations.{extra}"
                )
            )
            self.status.set_state(
                iteration=o.iterations,
                total=self._total,
                verdict_status=o.status,
                running=False,
            )

    def action_interrupt(self) -> None:
        if not self.cancel.is_set():
            self.cancel.set()
            self.log_widget.write(
                Text.from_markup("[bold yellow]⚠ interrupt requested[/] — winding down…")
            )
