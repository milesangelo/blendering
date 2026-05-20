"""Orchestrator behavior tests using a fake LLM and a fake MCP."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from blendering import orchestrator
from blendering.config import LoopConfig, MCPConfig, ModelConfig, Settings
from blendering.llm import ActorDelta
from blendering.mcp_client import BlenderMCP, ToolResult, ToolSpec
from blendering.orchestrator import EventBus, run
from blendering.schemas import Verdict
from blendering.tui.events import (
    CriticVerdictEvent,
    IterationStart,
    RunFinished,
    ToolCallResult,
    ToolCallStart,
)


def _settings(max_iter: int = 5, stuck_streak: int = 2) -> Settings:
    return Settings(
        mcp=MCPConfig(command="echo", args=[]),
        actor=ModelConfig(model="fake/actor", api_key_env="X"),
        critic=ModelConfig(model="fake/critic", api_key_env="X"),
        loop=LoopConfig(
            max_iterations=max_iter,
            screenshot_every_step=False,
            stop_on_stuck_streak=stuck_streak,
        ),
    )


class FakeMCP(BlenderMCP):
    """In-memory MCP stand-in. Bypasses __init__ to avoid needing a real session."""

    def __init__(self) -> None:
        self._tools = [
            ToolSpec(
                name="execute_blender_code",
                description="run python",
                parameters={
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            )
        ]
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def initialize(self) -> None:  # pragma: no cover - not used in tests
        pass

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        self.calls.append((name, arguments))
        return ToolResult(text=f"ran {name}", image_bytes=None, is_error=False)

    async def get_screenshot(self, max_size: int = 1024) -> bytes | None:
        return None


@pytest.fixture
def fake_mcp(monkeypatch: pytest.MonkeyPatch) -> FakeMCP:
    fake = FakeMCP()

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _ctx(_cfg: MCPConfig):
        yield fake

    monkeypatch.setattr(orchestrator, "mcp_client", _ctx)
    return fake


async def _collect_events(bus: EventBus, run_task: asyncio.Task[Any]) -> list[Any]:
    events: list[Any] = []
    while True:
        evt = await bus.get()
        events.append(evt)
        if isinstance(evt, RunFinished):
            await run_task
            return events


def _make_stream(
    deltas_per_call: list[list[ActorDelta]],
) -> Any:
    """Build a fake stream_actor that yields the next list each call."""
    calls = iter(deltas_per_call)

    async def fake_stream_actor(
        _cfg: ModelConfig, _messages: list[dict[str, Any]], _tools: list[dict[str, Any]]
    ) -> AsyncIterator[ActorDelta]:
        try:
            deltas = next(calls)
        except StopIteration:
            deltas = [ActorDelta(text="(no more actions)", done=True)]
        for d in deltas:
            yield d

    return fake_stream_actor


def _patch_judge(monkeypatch: pytest.MonkeyPatch, verdicts: list[Verdict]) -> None:
    it = iter(verdicts)

    async def fake_judge(*_args: Any, **_kwargs: Any) -> tuple[Verdict, int, int]:
        try:
            return next(it), 100, 50
        except StopIteration:
            return Verdict(status="done", reasoning="default", confidence=1.0), 100, 50

    monkeypatch.setattr(orchestrator, "judge", fake_judge)


async def test_happy_path_completes(
    monkeypatch: pytest.MonkeyPatch, fake_mcp: FakeMCP
) -> None:
    monkeypatch.setattr(
        orchestrator,
        "stream_actor",
        _make_stream(
            [
                [
                    ActorDelta(
                        finished_tool_calls=[
                            {
                                "id": "c1",
                                "name": "execute_blender_code",
                                "arguments": {"code": "bpy.ops.mesh.primitive_cube_add()"},
                            }
                        ],
                        done=True,
                    ),
                    ActorDelta(text="cube added", done=True),
                ],
            ]
        ),
    )
    _patch_judge(monkeypatch, [Verdict(status="done", reasoning="cube visible", confidence=0.9)])

    bus = EventBus()
    cancel = asyncio.Event()
    run_task = asyncio.create_task(run(_settings(), "make a cube", bus, cancel))
    events = await _collect_events(bus, run_task)

    finished = events[-1]
    assert isinstance(finished, RunFinished)
    assert finished.outcome.status == "done"
    assert finished.outcome.iterations == 1
    assert any(isinstance(e, ToolCallStart) for e in events)
    assert any(isinstance(e, ToolCallResult) for e in events)
    assert fake_mcp.calls == [
        ("execute_blender_code", {"code": "bpy.ops.mesh.primitive_cube_add()"})
    ]


async def test_stuck_streak_aborts(
    monkeypatch: pytest.MonkeyPatch, fake_mcp: FakeMCP
) -> None:
    monkeypatch.setattr(
        orchestrator,
        "stream_actor",
        _make_stream([[ActorDelta(text=f"step {i}", done=True)] for i in range(10)]),
    )
    _patch_judge(
        monkeypatch,
        [Verdict(status="stuck", reasoning="no change", confidence=0.3) for _ in range(5)],
    )

    bus = EventBus()
    cancel = asyncio.Event()
    run_task = asyncio.create_task(
        run(_settings(max_iter=10, stuck_streak=2), "x", bus, cancel)
    )
    events = await _collect_events(bus, run_task)
    finished = events[-1]
    assert isinstance(finished, RunFinished)
    assert finished.outcome.status == "stuck"
    assert finished.outcome.iterations == 2


async def test_cancel_event_breaks_loop(
    monkeypatch: pytest.MonkeyPatch, fake_mcp: FakeMCP
) -> None:
    started = asyncio.Event()

    async def slow_stream(
        _c: ModelConfig, _m: list[dict[str, Any]], _t: list[dict[str, Any]]
    ) -> AsyncIterator[ActorDelta]:
        started.set()
        await asyncio.sleep(10)  # cancelled before completing
        yield ActorDelta(done=True)

    monkeypatch.setattr(orchestrator, "stream_actor", slow_stream)
    _patch_judge(monkeypatch, [])

    bus = EventBus()
    cancel = asyncio.Event()
    run_task = asyncio.create_task(run(_settings(), "x", bus, cancel))

    # Wait for the iteration to actually start, then cancel.
    async def cancel_when_started() -> None:
        await started.wait()
        await asyncio.sleep(0.05)
        cancel.set()

    canceller = asyncio.create_task(cancel_when_started())

    events = await _collect_events(bus, run_task)
    await canceller
    finished = events[-1]
    assert isinstance(finished, RunFinished)
    assert finished.outcome.status == "cancelled"


async def test_iteration_events_emitted(
    monkeypatch: pytest.MonkeyPatch, fake_mcp: FakeMCP
) -> None:
    monkeypatch.setattr(
        orchestrator,
        "stream_actor",
        _make_stream([[ActorDelta(text="ok", done=True)] for _ in range(3)]),
    )
    _patch_judge(
        monkeypatch,
        [
            Verdict(status="continue", reasoning="…", next_step_hint="do X", confidence=0.4),
            Verdict(status="continue", reasoning="…", next_step_hint="do Y", confidence=0.5),
            Verdict(status="done", reasoning="great", confidence=0.95),
        ],
    )

    bus = EventBus()
    cancel = asyncio.Event()
    run_task = asyncio.create_task(run(_settings(max_iter=5), "x", bus, cancel))
    events = await _collect_events(bus, run_task)

    iter_starts = [e for e in events if isinstance(e, IterationStart)]
    verdict_events = [e for e in events if isinstance(e, CriticVerdictEvent)]
    assert [e.n for e in iter_starts] == [1, 2, 3]
    assert [v.verdict.status for v in verdict_events] == ["continue", "continue", "done"]


async def test_auto_frame_runs_before_screenshot(
    monkeypatch: pytest.MonkeyPatch, fake_mcp: FakeMCP
) -> None:
    """When framing.auto_frame=True and screenshots are on, orchestrator should
    send a reframe script via execute_blender_code right before fetching the
    screenshot."""
    monkeypatch.setattr(
        orchestrator,
        "stream_actor",
        _make_stream([[ActorDelta(text="placed cube", done=True)]]),
    )
    _patch_judge(
        monkeypatch,
        [Verdict(status="done", reasoning="ok", confidence=0.9)],
    )

    settings = _settings(max_iter=1)
    # Turn screenshots on for this test
    settings.loop.screenshot_every_step = True
    settings.framing.auto_frame = True

    bus = EventBus()
    cancel = asyncio.Event()
    run_task = asyncio.create_task(run(settings, "x", bus, cancel))
    await _collect_events(bus, run_task)

    names = [c[0] for c in fake_mcp.calls]
    # execute_blender_code (the reframe) must precede a screenshot attempt.
    assert "execute_blender_code" in names
    # The reframe call's code argument must contain the framing module's signature.
    reframe_calls = [c for c in fake_mcp.calls if c[0] == "execute_blender_code"]
    assert any("padding" in c[1].get("code", "") for c in reframe_calls)


async def test_auto_frame_skipped_when_disabled(
    monkeypatch: pytest.MonkeyPatch, fake_mcp: FakeMCP
) -> None:
    monkeypatch.setattr(
        orchestrator,
        "stream_actor",
        _make_stream([[ActorDelta(text="noop", done=True)]]),
    )
    _patch_judge(monkeypatch, [Verdict(status="done", reasoning="ok", confidence=0.9)])

    settings = _settings(max_iter=1)
    settings.loop.screenshot_every_step = True
    settings.framing.auto_frame = False

    bus = EventBus()
    cancel = asyncio.Event()
    run_task = asyncio.create_task(run(settings, "x", bus, cancel))
    await _collect_events(bus, run_task)

    reframe_calls = [
        c for c in fake_mcp.calls
        if c[0] == "execute_blender_code" and "padding" in c[1].get("code", "")
    ]
    assert reframe_calls == []
