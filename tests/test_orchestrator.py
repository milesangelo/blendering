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
from blendering.schemas import PartSpec, Plan, PositionSpec, Verdict
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
        self.calls.append(("get_screenshot", {"max_size": max_size}))
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


def _minimal_plan() -> Plan:
    """Return a minimal Plan suitable for test stubs."""
    return Plan(
        goal="test",
        parts=[
            PartSpec(
                id="cube",
                description="cube",
                primitive="cube",
                dimensions={"x": 1.0, "y": 1.0, "z": 1.0},
                position=PositionSpec(mode="absolute", xyz=(0.0, 0.0, 0.0)),
            )
        ],
    )


def _patch_plan(monkeypatch: pytest.MonkeyPatch, plan_obj: Plan) -> None:
    """Stub llm.plan to return a fixed Plan."""

    async def fake_plan(_cfg: Any, _goal: str) -> tuple[Plan, int, int]:
        return plan_obj, 50, 25

    monkeypatch.setattr(orchestrator, "plan", fake_plan)


def _patch_empty_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub gather_scene_snapshot to return an empty scene without an MCP call.
    Avoids polluting fake_mcp.calls for tests that inspect them."""

    async def fake_snapshot(_mcp: Any, _plan: Plan) -> dict[str, Any]:
        return {"objects": {}}

    monkeypatch.setattr(orchestrator, "gather_scene_snapshot", fake_snapshot)


async def test_happy_path_completes(
    monkeypatch: pytest.MonkeyPatch, fake_mcp: FakeMCP
) -> None:
    _patch_plan(monkeypatch, _minimal_plan())
    _patch_empty_snapshot(monkeypatch)
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
    _patch_plan(monkeypatch, _minimal_plan())
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
    _patch_plan(monkeypatch, _minimal_plan())
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
    _patch_plan(monkeypatch, _minimal_plan())
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
    _patch_plan(monkeypatch, _minimal_plan())
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
    # The reframe must precede the screenshot fetch — that's the whole point.
    assert "execute_blender_code" in names
    assert "get_screenshot" in names
    assert names.index("execute_blender_code") < names.index("get_screenshot")
    # And the reframe call must carry a non-empty Blender script.
    reframe_calls = [c for c in fake_mcp.calls if c[0] == "execute_blender_code"]
    assert any("bpy" in c[1].get("code", "") for c in reframe_calls)


async def test_auto_frame_skipped_when_disabled(
    monkeypatch: pytest.MonkeyPatch, fake_mcp: FakeMCP
) -> None:
    _patch_plan(monkeypatch, _minimal_plan())
    _patch_empty_snapshot(monkeypatch)
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
        if c[0] == "execute_blender_code" and "bpy" in c[1].get("code", "")
    ]
    assert reframe_calls == []


async def test_planner_runs_once_at_start(
    monkeypatch: pytest.MonkeyPatch, fake_mcp: FakeMCP
) -> None:
    plan_obj = Plan(
        goal="cube on table",
        parts=[
            PartSpec(
                id="cube",
                description="cube",
                primitive="cube",
                dimensions={"x": 1.0, "y": 1.0, "z": 1.0},
                position=PositionSpec(mode="absolute", xyz=(0.0, 0.0, 0.0)),
            )
        ],
    )
    call_count = {"n": 0}

    async def fake_plan(_cfg: Any, _goal: str) -> tuple[Plan, int, int]:
        call_count["n"] += 1
        return plan_obj, 50, 25

    monkeypatch.setattr(orchestrator, "plan", fake_plan)
    monkeypatch.setattr(
        orchestrator,
        "stream_actor",
        _make_stream([[ActorDelta(text="ok", done=True)]]),
    )
    _patch_judge(monkeypatch, [Verdict(status="done", reasoning="ok", confidence=0.9)])

    settings = _settings(max_iter=1)

    bus = EventBus()
    cancel = asyncio.Event()
    run_task = asyncio.create_task(run(settings, "cube on table", bus, cancel))
    await _collect_events(bus, run_task)

    # Planner ran exactly once at the start.
    assert call_count["n"] == 1


async def test_actor_proposals_are_accumulated(
    monkeypatch: pytest.MonkeyPatch, fake_mcp: FakeMCP
) -> None:
    """When the Actor emits PROPOSED_ADDITION blocks in its text, the orchestrator
    must collect them into pending_proposals (visible via the _LAST_PROPOSALS
    module-level test seam)."""
    from blendering.schemas import PartSpec, Plan, PositionSpec

    plan_obj = Plan(
        goal="x", parts=[
            PartSpec(
                id="t", description="t", primitive="cube",
                dimensions={"x": 1.0, "y": 1.0, "z": 1.0},
                position=PositionSpec(mode="absolute", xyz=(0.0, 0.0, 0.0)),
            )
        ],
    )
    _patch_plan(monkeypatch, plan_obj)

    monkeypatch.setattr(
        orchestrator,
        "stream_actor",
        _make_stream(
            [
                [
                    ActorDelta(
                        text="placing cube.\nPROPOSED_ADDITION: a book on the table\n",
                        done=True,
                    )
                ]
            ]
        ),
    )
    _patch_judge(monkeypatch, [Verdict(status="done", reasoning="ok", confidence=0.9)])

    settings = _settings(max_iter=1)
    bus = EventBus()
    cancel = asyncio.Event()
    run_task = asyncio.create_task(run(settings, "x", bus, cancel))
    await _collect_events(bus, run_task)

    proposals = orchestrator._LAST_PROPOSALS
    assert len(proposals) == 1
    assert "book on the table" in proposals[0].description


async def test_verifier_runs_each_step_and_feeds_actor(
    monkeypatch: pytest.MonkeyPatch, fake_mcp: FakeMCP
) -> None:
    """After each Actor turn, orchestrator must call gather_scene_snapshot + verify
    and pass the diff into the Critic's judge() via plan= and diff= kwargs."""
    from blendering.schemas import PartSpec, Plan, PositionSpec

    plan_obj = Plan(
        goal="x",
        parts=[
            PartSpec(
                id="a", description="a", primitive="cube",
                dimensions={"x": 1.0, "y": 1.0, "z": 1.0},
                position=PositionSpec(mode="absolute", xyz=(0.0, 0.0, 0.0)),
            )
        ],
    )
    _patch_plan(monkeypatch, plan_obj)

    # Return a scene with the cube present + correct so the diff is "ok".
    async def fake_snapshot(_mcp: Any, _plan: Plan) -> dict:
        return {
            "objects": {
                "a": {
                    "primitive_guess": "cube",
                    "vert_count": 8,
                    "world_location": [0.0, 0.0, 0.0],
                    "world_bbox_min": [-0.5, -0.5, -0.5],
                    "world_bbox_max": [0.5, 0.5, 0.5],
                    "rotation_euler_deg": [0.0, 0.0, 0.0],
                }
            }
        }
    monkeypatch.setattr(orchestrator, "gather_scene_snapshot", fake_snapshot)

    captured: dict[str, Any] = {}

    async def fake_judge(
        _cfg: Any, _system: str, _goal: str, transcript: str, _shot: Any,
        *, plan: Plan | None = None, diff: Any = None, **_kw: Any
    ) -> tuple[Verdict, int, int]:
        captured["plan_in_judge"] = plan
        captured["diff_in_judge"] = diff
        return Verdict(status="done", reasoning="ok", confidence=0.9), 10, 5
    monkeypatch.setattr(orchestrator, "judge", fake_judge)

    monkeypatch.setattr(
        orchestrator,
        "stream_actor",
        _make_stream([[ActorDelta(text="placed", done=True)]]),
    )

    settings = _settings(max_iter=1)
    bus = EventBus()
    cancel = asyncio.Event()
    run_task = asyncio.create_task(run(settings, "x", bus, cancel))
    await _collect_events(bus, run_task)

    assert captured["plan_in_judge"] is plan_obj
    assert captured["diff_in_judge"] is not None
    assert captured["diff_in_judge"].parts[0].status == "ok"
