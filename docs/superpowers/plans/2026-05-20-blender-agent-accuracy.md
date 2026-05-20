# Blender Agent Accuracy Improvements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate screenshot clipping and component-placement drift by adding pre-screenshot auto-framing, a Planner agent that produces a structured plan, and a deterministic Verifier that emits placement diffs each step.

**Architecture:** Loop becomes `Planner → (Actor → MCP → snapshot → auto-frame → screenshot → Verifier → Critic) → repeat`. Plan is owned by Planner (Actor reads only, may submit `proposed_addition` items). Verifier is pure code, runs every step. Critic gains a `structural_mismatch` verdict that triggers a Planner replan (capped by `max_replans`).

**Tech Stack:** Python 3.12, Pydantic v2, LiteLLM, MCP Python SDK, pytest + pytest-asyncio, uv for env management.

---

## Reference

- Design spec: `docs/superpowers/specs/2026-05-20-blender-agent-accuracy-design.md`
- Existing orchestrator: `src/blendering/orchestrator.py`
- Existing schemas: `src/blendering/schemas.py`
- Existing prompts: `src/blendering/prompts.py`
- Existing LLM clients: `src/blendering/llm.py` — model is `stream_actor` (Actor) and `judge` (Critic). New Planner client follows `judge`'s shape (single completion, JSON-validated, retry once).
- Existing orchestrator tests: `tests/test_orchestrator.py` — `FakeMCP` and `_make_stream` patterns to reuse.
- Test runner: `uv run pytest` (configured in `pyproject.toml`).

## File structure after this plan

**New files:**
- `src/blendering/framing.py` — pure function returning a Blender Python script for camera reframe.
- `src/blendering/verifier.py` — pure function `verify(plan, snapshot) -> VerifierDiff`.
- `tests/test_framing.py`
- `tests/test_verifier.py`
- `tests/test_planner.py`

**Modified files:**
- `src/blendering/schemas.py` — add Plan/PartSpec/PositionSpec/PartProposal/PartDiff/VerifierDiff; extend `Verdict` with `structural_mismatch`.
- `src/blendering/config.py` — add `PlannerConfig`, `FramingConfig`, `VerifierConfig`; extend `LoopConfig` with `max_replans`; add `planner`/`framing`/`verifier` to `Settings`.
- `src/blendering/prompts.py` — add `PLANNER_SYSTEM`, `REPLANNER_SYSTEM`; edit `ACTOR_SYSTEM` and `CRITIC_SYSTEM`.
- `src/blendering/llm.py` — add `plan()` and `replan()` async functions (mirror `judge`).
- `src/blendering/orchestrator.py` — initial Planner call, auto-frame before screenshot, Verifier call, replan branch.
- `config.example.yaml` — show the new sections.
- `tests/test_orchestrator.py` — extend with planner/verifier mocks for new flow.

## Rollout phases

Tasks below follow the spec's four phases. Each phase ends with a working build:

- **Phase 1 (Tasks 1–2):** Auto-framing ships standalone.
- **Phase 2 (Tasks 3–6):** Plan schema + Planner agent; Actor sees plan; no Verifier.
- **Phase 3 (Tasks 7–10):** Verifier runs every step; diff fed to Actor and Critic; `max_replans=0`.
- **Phase 4 (Task 11):** Replan branch enabled.

---

## Task 1: Auto-framing module

**Files:**
- Create: `src/blendering/framing.py`
- Create: `tests/test_framing.py`

- [ ] **Step 1.1: Write the failing test**

Create `tests/test_framing.py`:

```python
"""Tests for the auto-frame Blender script generator."""

from __future__ import annotations

from blendering.framing import reframe_script


def test_reframe_script_includes_padding_and_min_distance() -> None:
    script = reframe_script(padding=0.2, min_distance=3.0, exclude_tags=["_helper"])
    # Generated script must reference the configured numbers literally so they
    # land in Blender's runtime, not just in Python locals.
    assert "padding = 0.2" in script
    assert "min_distance = 3.0" in script
    assert "_helper" in script


def test_reframe_script_handles_empty_scene() -> None:
    script = reframe_script(padding=0.15, min_distance=2.0, exclude_tags=[])
    # The script must early-return when there are no mesh objects to frame.
    assert "if not bboxes" in script or "if not objs" in script


def test_reframe_script_creates_default_camera_when_missing() -> None:
    script = reframe_script(padding=0.15, min_distance=2.0, exclude_tags=[])
    # On first reframe the scene may have no camera; the script must create one.
    assert "bpy.data.cameras.new" in script
    assert "scene.camera" in script


def test_reframe_script_does_not_change_focal_length() -> None:
    script = reframe_script(padding=0.15, min_distance=2.0, exclude_tags=[])
    # We move the camera, not the lens — perspective stays consistent across steps.
    assert "cam.data.lens" not in script
    assert ".lens =" not in script


def test_reframe_script_excludes_lights_and_cameras() -> None:
    script = reframe_script(padding=0.15, min_distance=2.0, exclude_tags=[])
    # AABB must ignore non-content objects.
    assert "MESH" in script  # filtering by object.type
```

- [ ] **Step 1.2: Run test to verify it fails**

Run: `uv run pytest tests/test_framing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'blendering.framing'`.

- [ ] **Step 1.3: Implement `framing.py`**

Create `src/blendering/framing.py`:

```python
"""Generates a Blender Python script that reframes the camera around scene content.

Sent verbatim through MCP's `execute_blender_code` before each screenshot so the
Critic always sees a tightly-framed view. No I/O happens in this module — it
just returns a string.
"""

from __future__ import annotations

import json


def reframe_script(
    padding: float,
    min_distance: float,
    exclude_tags: list[str],
) -> str:
    """Return Python source that reframes Blender's active camera around all mesh
    objects, with `padding` extra room on each side of the AABB and a hard
    `min_distance` floor. Objects whose names contain any string in `exclude_tags`
    are ignored. Adds a default 3/4-angle camera if none exists.
    """
    exclude_literal = json.dumps(exclude_tags)
    return f"""
import bpy
import math
from mathutils import Vector

padding = {float(padding)}
min_distance = {float(min_distance)}
exclude_tags = {exclude_literal}
scene = bpy.context.scene

def _excluded(name):
    return any(tag in name for tag in exclude_tags)

objs = [o for o in scene.objects if o.type == "MESH" and not _excluded(o.name)]
bboxes = []
for o in objs:
    for corner in o.bound_box:
        bboxes.append(o.matrix_world @ Vector(corner))

if not bboxes:
    # Empty scene — leave the camera where it is.
    pass
else:
    mn = Vector((min(b.x for b in bboxes), min(b.y for b in bboxes), min(b.z for b in bboxes)))
    mx = Vector((max(b.x for b in bboxes), max(b.y for b in bboxes), max(b.z for b in bboxes)))
    centroid = (mn + mx) / 2.0
    diag = (mx - mn).length
    radius = (diag / 2.0) * (1.0 + padding)

    cam = scene.camera
    if cam is None:
        cam_data = bpy.data.cameras.new("AutoCam")
        cam = bpy.data.objects.new("AutoCam", cam_data)
        scene.collection.objects.link(cam)
        scene.camera = cam
        # Default 3/4 angle: from +X +Y +Z looking toward origin.
        cam.location = centroid + Vector((radius, -radius, radius))

    # Aim camera at centroid via track-to math.
    direction = (cam.location - centroid).normalized()
    if direction.length == 0.0:
        direction = Vector((1.0, -1.0, 1.0)).normalized()

    # Solve dolly distance so the bounding sphere fits the camera frustum.
    fov = cam.data.angle if cam.data.type != "ORTHO" else math.radians(50.0)
    fit_distance = radius / max(math.sin(fov / 2.0), 1e-4)
    distance = max(fit_distance, min_distance)
    cam.location = centroid + direction * distance

    # Point at centroid using rotation_euler from a tracking vector.
    look = (centroid - cam.location).normalized()
    # Convert look-direction into euler. Blender cameras look down -Z by default.
    up = Vector((0.0, 0.0, 1.0))
    right = look.cross(up)
    if right.length < 1e-4:
        up = Vector((0.0, 1.0, 0.0))
        right = look.cross(up)
    right = right.normalized()
    new_up = right.cross(look).normalized()
    import mathutils
    mat = mathutils.Matrix((
        (right.x, new_up.x, -look.x, cam.location.x),
        (right.y, new_up.y, -look.y, cam.location.y),
        (right.z, new_up.z, -look.z, cam.location.z),
        (0.0, 0.0, 0.0, 1.0),
    ))
    cam.matrix_world = mat
""".lstrip()
```

- [ ] **Step 1.4: Run test to verify it passes**

Run: `uv run pytest tests/test_framing.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 1.5: Commit**

```bash
git add src/blendering/framing.py tests/test_framing.py
git commit -m "feat(framing): add auto-frame script generator

Pure-function module that emits Blender Python to reframe the active
camera around the scene AABB before each screenshot. Falls back to
creating a default 3/4-angle camera when none exists.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Wire auto-framing into the orchestrator

**Files:**
- Modify: `src/blendering/config.py`
- Modify: `src/blendering/orchestrator.py:219-235` (the screenshot block)
- Modify: `tests/test_orchestrator.py` (the FakeMCP to record reframe calls)
- Modify: `config.example.yaml`

- [ ] **Step 2.1: Write the failing test**

Append to `tests/test_orchestrator.py`:

```python
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
```

- [ ] **Step 2.2: Run test to verify it fails**

Run: `uv run pytest tests/test_orchestrator.py::test_auto_frame_runs_before_screenshot -v`
Expected: FAIL with `AttributeError` for `settings.framing` or test failure on missing reframe call.

- [ ] **Step 2.3: Add `FramingConfig` and expose it on `Settings`**

Modify `src/blendering/config.py`. Add this class above `Settings`:

```python
class FramingConfig(BaseModel):
    auto_frame: bool = True
    padding: float = 0.15
    min_distance: float = 2.0
    exclude_tags: list[str] = Field(default_factory=lambda: ["_helper", "_guide"])
```

Then extend `Settings`:

```python
class Settings(BaseModel):
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    actor: ModelConfig
    critic: ModelConfig
    loop: LoopConfig = Field(default_factory=LoopConfig)
    framing: FramingConfig = Field(default_factory=FramingConfig)
    # ... _check_api_keys unchanged
```

- [ ] **Step 2.4: Wire the reframe call into the orchestrator**

In `src/blendering/orchestrator.py`, add `from .framing import reframe_script` near the other imports, then replace the screenshot block (lines ~219–235) with:

```python
                # Screenshot (preceded by auto-frame when enabled)
                screenshot_bytes: bytes | None = None
                if settings.loop.screenshot_every_step:
                    if settings.framing.auto_frame:
                        script = reframe_script(
                            padding=settings.framing.padding,
                            min_distance=settings.framing.min_distance,
                            exclude_tags=settings.framing.exclude_tags,
                        )
                        frame_task = asyncio.create_task(
                            mcp.call_tool("execute_blender_code", {"code": script})
                        )
                        try:
                            await _wait_or_cancel(frame_task, cancel)
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:
                            await bus.post(
                                StatusMessage(text=f"Auto-frame failed: {exc}", level="warn")
                            )

                    shot_task = asyncio.create_task(mcp.get_screenshot())
                    try:
                        screenshot_bytes = await _wait_or_cancel(shot_task, cancel)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        await bus.post(
                            StatusMessage(text=f"Screenshot failed: {exc}", level="warn")
                        )
                    if screenshot_bytes:
                        path = save_screenshot(screenshot_bytes, screenshot_dir)
                        await bus.post(
                            ViewportUpdate(image_bytes=screenshot_bytes, path=str(path))
                        )
```

- [ ] **Step 2.5: Update `config.example.yaml`**

Append to `config.example.yaml`:

```yaml

framing:
  # Pre-screenshot camera reframe so the Critic always sees a fitted view.
  auto_frame: true
  padding: 0.15
  min_distance: 2.0
  exclude_tags: ["_helper", "_guide"]
```

- [ ] **Step 2.6: Run tests to verify they pass**

Run: `uv run pytest tests/test_orchestrator.py tests/test_framing.py -v`
Expected: PASS for all existing tests plus the two new auto-frame tests.

- [ ] **Step 2.7: Commit**

```bash
git add src/blendering/orchestrator.py src/blendering/config.py config.example.yaml tests/test_orchestrator.py
git commit -m "feat(framing): wire auto-frame call into orchestrator

Before each screenshot, send framing.reframe_script through
execute_blender_code so the Critic always sees a tightly-framed view.
Controlled by framing.auto_frame in config (default on).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

**Phase 1 complete: auto-framing ships and is independently useful.**

---

## Task 3: Plan and PartProposal schemas

**Files:**
- Modify: `src/blendering/schemas.py`
- Modify: `tests/test_schemas.py` (existing file — APPEND new tests, do not overwrite)

- [ ] **Step 3.1: Write the failing test**

Append to `tests/test_schemas.py`. The file already has `test_verdict_*` and `test_run_outcome_*` tests — preserve them. Extend the existing import block with the new types (do not duplicate `Verdict`/`RunOutcome`):

```python
# Add to the existing imports near the top of the file:
from blendering.schemas import (
    PartProposal,
    PartSpec,
    Plan,
    PositionSpec,
)
# (pytest and ValidationError are already imported.)


def test_position_spec_absolute() -> None:
    p = PositionSpec(mode="absolute", xyz=(0.0, 0.0, 0.5))
    assert p.mode == "absolute"
    assert p.xyz == (0.0, 0.0, 0.5)
    assert p.anchor_part is None


def test_position_spec_relative() -> None:
    p = PositionSpec(
        mode="relative",
        anchor_part="table",
        anchor_face="top",
        offset=(0.0, 0.0, 0.0),
    )
    assert p.anchor_part == "table"
    assert p.anchor_face == "top"


def test_part_spec_minimum_valid() -> None:
    part = PartSpec(
        id="lamp_base",
        description="cylindrical wooden lamp base",
        primitive="cylinder",
        dimensions={"radius": 0.15, "height": 0.05},
        position=PositionSpec(mode="absolute", xyz=(0.0, 0.0, 0.0)),
    )
    assert part.id == "lamp_base"
    assert part.orientation_deg == (0.0, 0.0, 0.0)


def test_part_spec_rejects_unknown_primitive() -> None:
    with pytest.raises(ValidationError):
        PartSpec(
            id="x",
            description="x",
            primitive="dodecahedron",  # not in the Literal
            dimensions={},
            position=PositionSpec(mode="absolute", xyz=(0.0, 0.0, 0.0)),
        )


def test_plan_round_trip_json() -> None:
    plan = Plan(
        goal="lamp on table",
        parts=[
            PartSpec(
                id="table",
                description="oak table",
                primitive="cube",
                dimensions={"x": 1.0, "y": 0.6, "z": 0.05},
                position=PositionSpec(mode="absolute", xyz=(0.0, 0.0, 0.75)),
            ),
            PartSpec(
                id="lamp_base",
                description="lamp base",
                primitive="cylinder",
                dimensions={"radius": 0.1, "height": 0.05},
                position=PositionSpec(
                    mode="relative",
                    anchor_part="table",
                    anchor_face="top",
                    offset=(0.0, 0.0, 0.0),
                ),
            ),
        ],
    )
    blob = plan.model_dump_json()
    restored = Plan.model_validate_json(blob)
    assert restored == plan
    assert restored.version == 1


def test_part_proposal_minimal() -> None:
    pp = PartProposal(description="a book on the table")
    assert pp.description == "a book on the table"
    assert pp.rationale is None
```

- [ ] **Step 3.2: Run test to verify it fails**

Run: `uv run pytest tests/test_schemas.py -v`
Expected: FAIL with `ImportError` for the new types.

- [ ] **Step 3.3: Add the new schemas**

In `src/blendering/schemas.py`, append:

```python
class PositionSpec(BaseModel):
    """Where a part sits — either an absolute world coordinate or a position
    expressed relative to another part's named face."""

    mode: Literal["absolute", "relative"]
    # absolute mode
    xyz: tuple[float, float, float] | None = None
    # relative mode
    anchor_part: str | None = None
    anchor_face: Literal[
        "top", "bottom", "front", "back", "left", "right", "center"
    ] | None = None
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


class PartSpec(BaseModel):
    """One logical piece of the scene the Planner has committed to building."""

    id: str = Field(description="Stable handle used as the Blender object name.")
    description: str
    primitive: Literal[
        "cube", "cylinder", "sphere", "cone", "plane", "mesh", "imported"
    ]
    dimensions: dict[str, float] = Field(
        description="Dimension keys depend on primitive (e.g. radius/height for cylinder)."
    )
    position: PositionSpec
    orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    material_hint: str | None = None


class Plan(BaseModel):
    """The Planner's declarative build target. Mutated only by the Planner."""

    goal: str
    parts: list[PartSpec]
    scene_notes: str = ""
    version: int = 1


class PartProposal(BaseModel):
    """Actor-surfaced suggestion for a missing part. Not auto-applied — feeds
    the next (re)plan call where the Planner decides whether to incorporate it."""

    description: str
    rationale: str | None = None
```

Also extend `Verdict` (replace the existing class) to add the new status variant — keep the rest:

```python
class Verdict(BaseModel):
    """Critic's structured judgement of progress after a single Actor step."""

    status: Literal["continue", "done", "stuck", "structural_mismatch"]
    reasoning: str = Field(description="One short paragraph explaining the verdict.")
    next_step_hint: str | None = Field(
        default=None,
        description="Concrete next action for the Actor. Required when status='continue'.",
    )
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    replan_reason: str | None = Field(
        default=None,
        description="Required when status='structural_mismatch' — what the Planner should fix.",
    )
```

- [ ] **Step 3.4: Run test to verify it passes**

Run: `uv run pytest tests/test_schemas.py tests/test_orchestrator.py -v`
Expected: PASS for the new schema tests; orchestrator tests still pass because the `Verdict` extension is backward-compatible (existing callers only pass the three original statuses).

- [ ] **Step 3.5: Commit**

```bash
git add src/blendering/schemas.py tests/test_schemas.py
git commit -m "feat(schemas): add Plan, PartSpec, PositionSpec, PartProposal

Also extend Verdict with structural_mismatch status and replan_reason
field. Backward-compatible — existing Verdict consumers untouched.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Planner LLM client and prompts

**Files:**
- Modify: `src/blendering/prompts.py`
- Modify: `src/blendering/llm.py`
- Create: `tests/test_planner.py`

- [ ] **Step 4.1: Write the failing test**

Create `tests/test_planner.py`:

```python
"""Tests for the Planner LLM client. LiteLLM's acompletion is patched so no
network is hit."""

from __future__ import annotations

import json
from typing import Any

import pytest

from blendering import llm
from blendering.config import ModelConfig
from blendering.schemas import Plan, PartProposal, PartSpec, PositionSpec, VerifierDiff


def _planner_cfg() -> ModelConfig:
    return ModelConfig(model="fake/planner", api_key_env="X")


class _FakeResp:
    def __init__(self, payload: dict[str, Any]) -> None:
        class _Msg:
            content = json.dumps(payload)
        class _Choice:
            message = _Msg()
        self.choices = [_Choice()]
        class _Usage:
            prompt_tokens = 10
            completion_tokens = 5
        self.usage = _Usage()


async def test_plan_returns_validated_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "goal": "lamp on table",
        "parts": [
            {
                "id": "table",
                "description": "oak table",
                "primitive": "cube",
                "dimensions": {"x": 1.0, "y": 0.6, "z": 0.05},
                "position": {"mode": "absolute", "xyz": [0.0, 0.0, 0.75]},
            }
        ],
        "scene_notes": "",
        "version": 1,
    }

    async def fake_completion(**_kwargs: Any) -> _FakeResp:
        return _FakeResp(payload)

    monkeypatch.setattr(llm.litellm, "acompletion", fake_completion)
    plan, in_t, out_t = await llm.plan(_planner_cfg(), "lamp on table")
    assert isinstance(plan, Plan)
    assert plan.goal == "lamp on table"
    assert plan.parts[0].id == "table"
    assert (in_t, out_t) == (10, 5)


async def test_plan_retries_on_invalid_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    good = {
        "goal": "x",
        "parts": [],
        "scene_notes": "",
        "version": 1,
    }
    calls = {"n": 0}

    async def fake_completion(**_kwargs: Any) -> _FakeResp:
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResp({"not_a_plan": True})
        return _FakeResp(good)

    monkeypatch.setattr(llm.litellm, "acompletion", fake_completion)
    plan, _, _ = await llm.plan(_planner_cfg(), "x")
    assert plan.goal == "x"
    assert calls["n"] == 2


async def test_replan_passes_prior_plan_and_diff(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "goal": "lamp on table",
        "parts": [],
        "scene_notes": "revised",
        "version": 2,
    }
    captured: dict[str, Any] = {}

    async def fake_completion(**kwargs: Any) -> _FakeResp:
        captured.update(kwargs)
        return _FakeResp(payload)

    monkeypatch.setattr(llm.litellm, "acompletion", fake_completion)
    prior = Plan(
        goal="lamp on table",
        parts=[
            PartSpec(
                id="t",
                description="t",
                primitive="cube",
                dimensions={"x": 1.0, "y": 1.0, "z": 1.0},
                position=PositionSpec(mode="absolute", xyz=(0.0, 0.0, 0.0)),
            )
        ],
    )
    diff = VerifierDiff(
        plan_version=1,
        parts=[],
        extras=[],
        summary="all parts off",
        is_structural=True,
    )
    new_plan, _, _ = await llm.replan(
        _planner_cfg(),
        prior=prior,
        diff=diff,
        recent_actions="resized table",
        proposals=[PartProposal(description="add a chair")],
    )
    assert new_plan.version == 2
    # The user message should reference the prior plan, the diff summary,
    # the recent-actions blob, and any proposals.
    user_msg = captured["messages"][-1]["content"]
    if isinstance(user_msg, list):
        joined = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in user_msg
        )
    else:
        joined = user_msg
    assert "lamp on table" in joined
    assert "all parts off" in joined
    assert "resized table" in joined
    assert "add a chair" in joined
```

- [ ] **Step 4.2: Run test to verify it fails**

Run: `uv run pytest tests/test_planner.py -v`
Expected: FAIL with `AttributeError: module 'blendering.llm' has no attribute 'plan'`.

- [ ] **Step 4.3: Add Planner prompts**

In `src/blendering/prompts.py`, append:

```python
PLANNER_SYSTEM = """\
You are the **Planner** in a three-model team that creates 3D scenes in Blender.

You produce a structured Plan that the Actor will execute. You do NOT write code.
You do NOT call tools. You output JSON only.

The Plan schema:
{
  "goal": str,
  "parts": [PartSpec, ...],
  "scene_notes": str,
  "version": int
}

PartSpec:
{
  "id": str,                            # stable handle, e.g. "lamp_base". Snake_case.
  "description": str,
  "primitive": "cube"|"cylinder"|"sphere"|"cone"|"plane"|"mesh"|"imported",
  "dimensions": {key: float, ...},      # meters. Keys depend on primitive
                                         # (cube: x,y,z; cylinder: radius,height; etc.)
  "position": PositionSpec,
  "orientation_deg": [float,float,float],
  "material_hint": str | null
}

PositionSpec — absolute:
  {"mode": "absolute", "xyz": [x,y,z]}
PositionSpec — relative (preferred for stacked/attached parts):
  {"mode": "relative", "anchor_part": "<other_id>",
   "anchor_face": "top"|"bottom"|"front"|"back"|"left"|"right"|"center",
   "offset": [x,y,z]}

Rules:
- Be explicit about dimensions in meters.
- Prefer relative positions whenever a part is on/in/attached to another.
- One PartSpec per logical piece — don't merge independently-positioned things.
- All IDs are unique; relative anchor_part must reference an earlier id.
- Output ONLY the JSON object. No prose, no markdown fences.
"""

REPLANNER_SYSTEM = """\
You are the **Planner** revising an existing Plan.

You will receive:
1. The prior Plan (with its parts and version).
2. A VerifierDiff describing which parts are ok/off/missing/extra.
3. A short summary of the Actor's recent actions.
4. Optionally, a list of part proposals the Actor wants you to consider.
5. The latest viewport screenshot.

Your job: emit a revised Plan in the same schema.

Rules:
- Bump `version` by exactly 1.
- Keep `id`s stable for parts that are essentially right. Don't rename unless required.
- Only change what the diff or proposals justify. Minimize churn.
- If a proposal is appropriate to the goal, add it as a new PartSpec with a fresh id.
- Output ONLY the JSON object. No prose, no markdown fences.
"""
```

- [ ] **Step 4.4: Implement `plan()` and `replan()` in `llm.py`**

In `src/blendering/llm.py`, add the imports near the top:

```python
from .schemas import PartProposal, Plan, Verdict, VerifierDiff
```

(Adjust the existing `from .schemas import Verdict` line accordingly.)

Then append at the bottom of the file:

```python
_PLAN_INSTRUCTION = (
    "Respond ONLY with a JSON object matching the Plan schema in the system prompt."
)


async def plan(cfg: ModelConfig, user_goal: str) -> tuple[Plan, int, int]:
    """Initial plan — produce a Plan from the user's goal. Validates against the
    Plan schema; retries once on failure with the error attached."""
    from .prompts import PLANNER_SYSTEM

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": f"USER GOAL:\n{user_goal}\n\n{_PLAN_INSTRUCTION}"},
    ]
    return await _plan_completion(cfg, messages)


async def replan(
    cfg: ModelConfig,
    prior: Plan,
    diff: VerifierDiff,
    recent_actions: str,
    proposals: list[PartProposal],
    screenshot_png: bytes | None = None,
) -> tuple[Plan, int, int]:
    """Revise an existing Plan based on the Verifier diff, recent Actor actions,
    and any pending part proposals."""
    from .prompts import REPLANNER_SYSTEM

    proposal_block = (
        "\n".join(f"- {p.description}" + (f"  (rationale: {p.rationale})" if p.rationale else "")
                  for p in proposals)
        if proposals
        else "(none)"
    )
    text_block = (
        f"PRIOR PLAN (version {prior.version}):\n{prior.model_dump_json(indent=2)}\n\n"
        f"VERIFIER DIFF:\n{diff.model_dump_json(indent=2)}\n\n"
        f"RECENT ACTOR ACTIONS:\n{recent_actions}\n\n"
        f"PART PROPOSALS FROM ACTOR:\n{proposal_block}\n\n"
        f"{_PLAN_INSTRUCTION}"
    )

    user_content: Any = text_block
    if screenshot_png is not None:
        small = thumbnail_bytes(screenshot_png)
        user_content = [
            {"type": "text", "text": text_block},
            {"type": "image_url", "image_url": {"url": encode_b64_data_url(small)}},
        ]

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": REPLANNER_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    return await _plan_completion(cfg, messages)


async def _plan_completion(
    cfg: ModelConfig, messages: list[dict[str, Any]]
) -> tuple[Plan, int, int]:
    for attempt in range(2):
        try:
            log.debug("planner → %s attempt=%d", cfg.model, attempt + 1)
            resp = await litellm.acompletion(
                **_model_kwargs(cfg),
                messages=messages,
                response_format={"type": "json_object"},
                stream=False,
            )
            text = resp.choices[0].message.content or "{}"
            usage = getattr(resp, "usage", None)
            in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
            out_tok = getattr(usage, "completion_tokens", 0) if usage else 0
            data = json.loads(_strip_fences(text))
            return Plan.model_validate(data), in_tok, out_tok
        except (json.JSONDecodeError, ValidationError) as exc:
            if attempt == 0:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response failed validation: {exc}. "
                            "Reply ONLY with a valid Plan JSON object."
                        ),
                    }
                )
                continue
            raise RuntimeError(f"Planner returned invalid Plan twice: {exc}") from exc
    raise RuntimeError("_plan_completion exited loop without returning")
```

> Note: `VerifierDiff` is imported but not yet defined. We add it in Task 7. To keep this task self-contained, add a forward stub now so the import works:

Append to `src/blendering/schemas.py` (will be replaced with the full version in Task 7):

```python
class VerifierDiff(BaseModel):
    """Stub — full definition lands in Task 7."""

    plan_version: int = 0
    parts: list[Any] = Field(default_factory=list)
    extras: list[str] = Field(default_factory=list)
    summary: str = ""
    is_structural: bool = False
```

- [ ] **Step 4.5: Run test to verify it passes**

Run: `uv run pytest tests/test_planner.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 4.6: Commit**

```bash
git add src/blendering/prompts.py src/blendering/llm.py src/blendering/schemas.py tests/test_planner.py
git commit -m "feat(planner): add Planner/Replanner LLM clients and prompts

llm.plan() produces a Plan from a user goal; llm.replan() revises a
Plan given the verifier diff, recent actions, and Actor proposals.
Both validate output against the Plan schema and retry once on failure.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: PlannerConfig and orchestrator integration (initial plan only)

**Files:**
- Modify: `src/blendering/config.py`
- Modify: `src/blendering/orchestrator.py`
- Modify: `src/blendering/cost.py` (add planner role accounting)
- Modify: `tests/test_orchestrator.py`
- Modify: `config.example.yaml`

- [ ] **Step 5.1: Write the failing test**

Append to `tests/test_orchestrator.py`:

```python
def _patch_plan(monkeypatch: pytest.MonkeyPatch, plan_obj: Any) -> None:
    """Stub llm.plan to return a fixed Plan."""
    from blendering.schemas import Plan as _Plan

    async def fake_plan(_cfg: Any, _goal: str) -> tuple[_Plan, int, int]:
        return plan_obj, 50, 25

    monkeypatch.setattr(orchestrator, "plan", fake_plan)


async def test_planner_runs_once_at_start(
    monkeypatch: pytest.MonkeyPatch, fake_mcp: FakeMCP
) -> None:
    from blendering.schemas import PartSpec, Plan, PositionSpec

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
    settings.planner = settings.planner  # ensure default exists

    bus = EventBus()
    cancel = asyncio.Event()
    run_task = asyncio.create_task(run(settings, "cube on table", bus, cancel))
    await _collect_events(bus, run_task)

    # Planner ran exactly once at the start.
    assert call_count["n"] == 1
```

- [ ] **Step 5.2: Run test to verify it fails**

Run: `uv run pytest tests/test_orchestrator.py::test_planner_runs_once_at_start -v`
Expected: FAIL with `AttributeError` on `settings.planner` or missing `plan` in orchestrator namespace.

- [ ] **Step 5.3: Add `PlannerConfig`**

In `src/blendering/config.py`, before `Settings`:

```python
class PlannerConfig(ModelConfig):
    """Same shape as ModelConfig — the Planner is just a third LLM role."""
    pass
```

Then in `Settings`:

```python
class Settings(BaseModel):
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    actor: ModelConfig
    critic: ModelConfig
    planner: PlannerConfig | None = None  # falls back to actor config if None
    loop: LoopConfig = Field(default_factory=LoopConfig)
    framing: FramingConfig = Field(default_factory=FramingConfig)
    # ... _check_api_keys unchanged
```

> Rationale: `planner` is optional so existing config files (no `planner:` section) keep working; when absent, the orchestrator reuses the actor config.

Update `_check_api_keys` to include planner when set:

```python
    @model_validator(mode="after")
    def _check_api_keys(self) -> Settings:
        roles: list[tuple[str, ModelConfig]] = [
            ("actor", self.actor),
            ("critic", self.critic),
        ]
        if self.planner is not None:
            roles.append(("planner", self.planner))
        for role, mc in roles:
            if mc.api_key is None:
                os.environ.setdefault(f"_BLENDERING_MISSING_{role.upper()}_KEY", mc.api_key_env)
        return self
```

- [ ] **Step 5.4: Extend `CostMeter` with a planner role**

In `src/blendering/cost.py`, add `planner` to `CostMeter`:

```python
@dataclass
class CostMeter:
    actor: RoleUsage = field(default_factory=RoleUsage)
    critic: RoleUsage = field(default_factory=RoleUsage)
    planner: RoleUsage = field(default_factory=RoleUsage)

    def step_line(self, actor_cfg: ModelConfig, critic_cfg: ModelConfig) -> str:
        ac = self.actor.cost(actor_cfg)
        cc = self.critic.cost(critic_cfg)
        cost_part = ""
        if ac is not None and cc is not None:
            cost_part = f"  step≈${ac + cc:.4f} (cum)"
        return (
            f"tokens — actor: {self.actor.in_tokens} in / {self.actor.out_tokens} out  "
            f"critic: {self.critic.in_tokens} in / {self.critic.out_tokens} out  "
            f"planner: {self.planner.in_tokens} in / {self.planner.out_tokens} out{cost_part}"
        )
```

(Leave the existing `summary()` method as-is; planner usage will be reported in a follow-up if useful.)

- [ ] **Step 5.5: Wire the Planner into the orchestrator**

In `src/blendering/orchestrator.py`:

Add imports:

```python
from .llm import judge, plan, stream_actor
from .schemas import PartProposal, Plan, RunOutcome, Verdict
```

In `run()`, right after the existing `messages = [...]` and `meter = CostMeter()` block (around line 200), insert the initial Planner call:

```python
            # Initial plan
            planner_cfg = settings.planner or settings.actor
            try:
                plan_task = asyncio.create_task(plan(planner_cfg, user_prompt))
                current_plan, p_in, p_out = await _wait_or_cancel(plan_task, cancel)
            except asyncio.CancelledError:
                raise
            meter.planner.add(p_in, p_out)
            await bus.post(
                StatusMessage(
                    text=f"Plan v{current_plan.version}: "
                    f"{len(current_plan.parts)} part(s) — "
                    f"{', '.join(p.id for p in current_plan.parts) or '(none)'}"
                )
            )

            # Initialize Actor with plan in the system prompt
            actor_system = ACTOR_SYSTEM + "\n\nACTIVE PLAN:\n" + current_plan.model_dump_json(indent=2)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": actor_system},
                {"role": "user", "content": user_prompt},
            ]
            pending_proposals: list[PartProposal] = []
```

Replace the original `messages = [...]` declaration so it isn't duplicated.

- [ ] **Step 5.6: Update `config.example.yaml`**

Append:

```yaml

planner:
  # Optional. Omit this section to reuse the actor config for planning.
  # Recommended: point at a stronger reasoning model.
  model: "openai/Qwen/Qwen2.5-72B-Instruct"
  api_base: "https://api.siliconflow.cn/v1"
  api_key_env: "SILICONFLOW_API_KEY"
  temperature: 0.3
  max_tokens: 2048
```

- [ ] **Step 5.7: Run tests to verify they pass**

Run: `uv run pytest tests/ -v`
Expected: all existing tests pass; new `test_planner_runs_once_at_start` passes.

- [ ] **Step 5.8: Commit**

```bash
git add src/blendering/config.py src/blendering/orchestrator.py src/blendering/cost.py config.example.yaml tests/test_orchestrator.py
git commit -m "feat(planner): run Planner once at start and inject plan into Actor

Adds PlannerConfig (optional; falls back to actor config), runs
llm.plan() before the iteration loop, and embeds the resulting Plan
JSON in the Actor's system prompt as ACTIVE PLAN.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Actor prompt updates + proposed_additions parsing

The Actor's structured output gains a `proposed_additions` channel. Since the Actor is tool-calling (not JSON output), we surface this via a small marker convention the Actor emits in its assistant text — easy for the orchestrator to parse, easy for any model to comply with.

**Files:**
- Modify: `src/blendering/prompts.py`
- Modify: `src/blendering/orchestrator.py`
- Modify: `tests/test_orchestrator.py`

- [ ] **Step 6.1: Write the failing test**

Append to `tests/test_orchestrator.py`:

```python
async def test_actor_proposals_are_accumulated(
    monkeypatch: pytest.MonkeyPatch, fake_mcp: FakeMCP
) -> None:
    """When the Actor emits PROPOSED_ADDITION blocks in its text, the orchestrator
    must collect them into pending_proposals (visible via a debug event)."""
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

    async def fake_plan(_cfg: Any, _goal: str) -> tuple[Plan, int, int]:
        return plan_obj, 50, 25
    monkeypatch.setattr(orchestrator, "plan", fake_plan)

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

    captured: dict[str, Any] = {}

    # Spy on the orchestrator's pending_proposals via a hook on RunFinished
    original_run = orchestrator.run

    async def spy_run(*args: Any, **kwargs: Any) -> Any:
        outcome = await original_run(*args, **kwargs)
        captured["proposals"] = getattr(orchestrator, "_LAST_PROPOSALS", [])
        return outcome

    monkeypatch.setattr(orchestrator, "run", spy_run)

    run_task = asyncio.create_task(orchestrator.run(settings, "x", bus, cancel))
    await _collect_events(bus, run_task)

    assert len(captured["proposals"]) == 1
    assert "book on the table" in captured["proposals"][0].description
```

> The spy works because Task 6.3 will write the final list to `orchestrator._LAST_PROPOSALS` at the end of the run. That module-level variable is a deliberate test seam — minimal surface area, zero impact on production behavior.

- [ ] **Step 6.2: Update `ACTOR_SYSTEM`**

Replace `ACTOR_SYSTEM` in `src/blendering/prompts.py`:

```python
ACTOR_SYSTEM = """\
You are the **Actor** in a three-model team that creates 3D scenes in Blender via MCP tools.

You will receive an ACTIVE PLAN (appended below) listing the parts to build. Each part has a
stable id, primitive, dimensions, and position.

Your role:
- Build the plan part by part. ONE part per step when possible.
- When you create or import a part, set its Blender object name to the part's id.
  Example: `obj = bpy.context.active_object; obj.name = "lamp_base"`.
- Prefer calling MCP tools over describing what you would do. Real progress = tool calls.
- For scene mutations, use `execute_blender_code` with concise, idempotent Python (bpy).
- Use `get_scene_info` / `get_object_info` to inspect state when uncertain.
- If a part is flagged `off` by the Verifier, fix only the flagged dimensions/positions.
  Do NOT rebuild correct parts from scratch.
- Do NOT call `get_viewport_screenshot` — the critic handles screenshots.
- Do NOT modify the plan. If you notice a missing part the goal requires, surface a
  PROPOSED_ADDITION block (see below) and continue with the current plan.

Output format:
- A 1-2 sentence plan for THIS step.
- (Optional) Zero or more lines of the form:
    PROPOSED_ADDITION: <one-line description of a part the plan is missing>
  These do NOT gate the current step; they accumulate and are shown to the Planner.
- Then the tool calls needed for this step.
"""
```

- [ ] **Step 6.3: Parse `PROPOSED_ADDITION:` lines in the orchestrator**

In `src/blendering/orchestrator.py`, add a parsing helper near `_truncate`:

```python
def _extract_proposals(text: str) -> list[PartProposal]:
    out: list[PartProposal] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("PROPOSED_ADDITION:"):
            desc = stripped.split(":", 1)[1].strip()
            if desc:
                out.append(PartProposal(description=desc))
    return out
```

In `_run_actor_turn`, after `text` is accumulated each iteration of the inner `while True`, parse and return proposals alongside the transcript. Modify the return type and signature:

```python
async def _run_actor_turn(
    settings: Settings,
    mcp: BlenderMCP,
    messages: list[dict[str, Any]],
    tools_openai: list[dict[str, Any]],
    bus: EventBus,
    cancel: asyncio.Event,
    meter: CostMeter,
) -> tuple[list[dict[str, Any]], str, list[PartProposal]]:
    """Run one Actor turn (text + tool calls). Returns (new_messages, transcript_for_critic, proposals)."""
    accumulated_text = ""
    appended: list[dict[str, Any]] = []
    transcript_lines: list[str] = []
    proposals: list[PartProposal] = []
    # ... rest unchanged until each `text` is observed ...
```

Inside the loop, after `accumulated_text += text`:

```python
        proposals.extend(_extract_proposals(text))
```

Change the two `return` sites from `return appended, ...` to `return appended, "\n".join(transcript_lines), proposals`.

In `run()`, update the call site:

```python
                _, transcript, new_proposals = await _run_actor_turn(
                    settings, mcp, messages, tools_openai, bus, cancel, meter
                )
                pending_proposals.extend(new_proposals)
```

Add a module-level test seam near the top of `orchestrator.py`:

```python
# Test seam: most-recent run's accumulated proposals. Tests assert against this.
_LAST_PROPOSALS: list[PartProposal] = []
```

Then, just before each `return outcome` site in `run()`, write:

```python
                _LAST_PROPOSALS[:] = pending_proposals
```

- [ ] **Step 6.4: Run tests to verify they pass**

Run: `uv run pytest tests/ -v`
Expected: PASS — the proposal test sees one accumulated proposal, all prior tests still pass.

- [ ] **Step 6.5: Commit**

```bash
git add src/blendering/prompts.py src/blendering/orchestrator.py tests/test_orchestrator.py
git commit -m "feat(actor): parse PROPOSED_ADDITION blocks into pending proposals

Actor system prompt now describes the plan-execution contract and the
PROPOSED_ADDITION marker. Orchestrator extracts each marker line into
a PartProposal and accumulates them across iterations for the next
(re)plan call.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

**Phase 2 complete: Plan is generated and visible to the Actor; Actor can surface proposals; no Verifier yet.**

---

## Task 7: VerifierDiff schemas (full)

**Files:**
- Modify: `src/blendering/schemas.py` (replace the stub from Task 4)
- Modify: `tests/test_schemas.py`

- [ ] **Step 7.1: Write the failing test**

Append to `tests/test_schemas.py`:

```python
from blendering.schemas import PartDiff, VerifierDiff


def test_part_diff_status_values() -> None:
    for s in ("ok", "off", "missing", "extra"):
        d = PartDiff(part_id="x", status=s, issues=[], measured={})
        assert d.status == s


def test_verifier_diff_round_trip() -> None:
    d = VerifierDiff(
        plan_version=2,
        parts=[
            PartDiff(part_id="t", status="ok", issues=[], measured={"x": 1.0}),
            PartDiff(part_id="lamp", status="off",
                     issues=["height 0.42 vs plan 0.30 (40% over)"],
                     measured={"height": 0.42}),
        ],
        extras=["StrayCube"],
        summary="lamp off; 1 extra",
        is_structural=False,
    )
    blob = d.model_dump_json()
    restored = VerifierDiff.model_validate_json(blob)
    assert restored == d
```

- [ ] **Step 7.2: Run test to verify it fails**

Run: `uv run pytest tests/test_schemas.py::test_part_diff_status_values -v`
Expected: FAIL — `PartDiff` doesn't exist.

- [ ] **Step 7.3: Replace the `VerifierDiff` stub with the full schemas**

In `src/blendering/schemas.py`, replace the stub from Task 4 with:

```python
class PartDiff(BaseModel):
    """Per-part status from the Verifier."""

    part_id: str
    status: Literal["ok", "off", "missing", "extra"]
    issues: list[str] = Field(default_factory=list)
    measured: dict[str, Any] = Field(default_factory=dict)


class VerifierDiff(BaseModel):
    """Aggregated Verifier output for one step."""

    plan_version: int
    parts: list[PartDiff] = Field(default_factory=list)
    extras: list[str] = Field(default_factory=list)
    summary: str = ""
    is_structural: bool = False
```

- [ ] **Step 7.4: Run test to verify it passes**

Run: `uv run pytest tests/test_schemas.py tests/test_planner.py tests/test_orchestrator.py -v`
Expected: PASS — the planner test uses `VerifierDiff` with the same field names, so it keeps working.

- [ ] **Step 7.5: Commit**

```bash
git add src/blendering/schemas.py tests/test_schemas.py
git commit -m "feat(schemas): replace VerifierDiff stub with full PartDiff/VerifierDiff

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Verifier module

The Verifier is pure: given a plan and a scene snapshot (already gathered by the orchestrator), it produces a `VerifierDiff`. No I/O, no MCP.

The snapshot is a dict shaped:

```python
{
  "objects": {
    "<name>": {
        "primitive_guess": "cube"|"cylinder"|...|"mesh",
        "vert_count": int,
        "world_location": [x,y,z],
        "world_bbox_min": [x,y,z],
        "world_bbox_max": [x,y,z],
        "rotation_euler_deg": [rx, ry, rz],
    },
    ...
  }
}
```

That snapshot will be gathered by the orchestrator in Task 9. The Verifier just consumes it.

**Files:**
- Create: `src/blendering/verifier.py`
- Create: `tests/test_verifier.py`
- Modify: `src/blendering/config.py` (add `VerifierConfig`)
- Modify: `config.example.yaml`

- [ ] **Step 8.1: Write the failing test**

Create `tests/test_verifier.py`:

```python
"""Verifier tests using synthetic scene snapshots."""

from __future__ import annotations

from blendering.config import VerifierConfig
from blendering.schemas import PartSpec, Plan, PositionSpec
from blendering.verifier import verify


def _abs_part(
    pid: str,
    primitive: str,
    dims: dict[str, float],
    xyz: tuple[float, float, float],
) -> PartSpec:
    return PartSpec(
        id=pid,
        description=pid,
        primitive=primitive,  # type: ignore[arg-type]
        dimensions=dims,
        position=PositionSpec(mode="absolute", xyz=xyz),
    )


def _rel_part(
    pid: str,
    primitive: str,
    dims: dict[str, float],
    anchor: str,
    face: str,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> PartSpec:
    return PartSpec(
        id=pid,
        description=pid,
        primitive=primitive,  # type: ignore[arg-type]
        dimensions=dims,
        position=PositionSpec(
            mode="relative",
            anchor_part=anchor,
            anchor_face=face,  # type: ignore[arg-type]
            offset=offset,
        ),
    )


def _cube_snapshot(name: str, center: tuple[float, float, float], size: float) -> dict:
    cx, cy, cz = center
    h = size / 2.0
    return {
        "primitive_guess": "cube",
        "vert_count": 8,
        "world_location": [cx, cy, cz],
        "world_bbox_min": [cx - h, cy - h, cz - h],
        "world_bbox_max": [cx + h, cy + h, cz + h],
        "rotation_euler_deg": [0.0, 0.0, 0.0],
    }


def _cfg() -> VerifierConfig:
    return VerifierConfig()


def test_all_parts_ok() -> None:
    plan = Plan(goal="x", parts=[_abs_part("a", "cube", {"x": 1.0, "y": 1.0, "z": 1.0}, (0.0, 0.0, 0.0))])
    snapshot = {"objects": {"a": _cube_snapshot("a", (0.0, 0.0, 0.0), 1.0)}}
    diff = verify(plan, snapshot, _cfg())
    assert [p.status for p in diff.parts] == ["ok"]
    assert diff.extras == []
    assert diff.is_structural is False


def test_missing_part_is_structural_by_default() -> None:
    plan = Plan(goal="x", parts=[_abs_part("a", "cube", {"x": 1.0, "y": 1.0, "z": 1.0}, (0.0, 0.0, 0.0))])
    snapshot: dict = {"objects": {}}
    diff = verify(plan, snapshot, _cfg())
    assert [p.status for p in diff.parts] == ["missing"]
    assert diff.is_structural is True


def test_dimension_off_outside_tolerance() -> None:
    plan = Plan(goal="x", parts=[_abs_part("a", "cube", {"x": 1.0, "y": 1.0, "z": 1.0}, (0.0, 0.0, 0.0))])
    snapshot = {"objects": {"a": _cube_snapshot("a", (0.0, 0.0, 0.0), 1.5)}}  # 50% larger
    diff = verify(plan, snapshot, _cfg())
    assert diff.parts[0].status == "off"
    assert any("x" in i or "dimension" in i.lower() for i in diff.parts[0].issues)


def test_position_off_outside_tolerance() -> None:
    plan = Plan(goal="x", parts=[_abs_part("a", "cube", {"x": 1.0, "y": 1.0, "z": 1.0}, (0.0, 0.0, 0.0))])
    snapshot = {"objects": {"a": _cube_snapshot("a", (0.5, 0.0, 0.0), 1.0)}}  # 50cm off, tol is 10cm
    diff = verify(plan, snapshot, _cfg())
    assert diff.parts[0].status == "off"
    assert any("position" in i.lower() for i in diff.parts[0].issues)


def test_relative_position_uses_anchor_top_face() -> None:
    plan = Plan(
        goal="lamp on table",
        parts=[
            _abs_part("table", "cube", {"x": 1.0, "y": 1.0, "z": 1.0}, (0.0, 0.0, 0.0)),
            # lamp_base sits on top of the table (z = 0.5 = table top)
            _rel_part(
                "lamp_base", "cube",
                {"x": 0.2, "y": 0.2, "z": 0.1},
                anchor="table", face="top",
            ),
        ],
    )
    # Place lamp_base centered just above the table top: center at (0, 0, 0.55) = table_top + half-height
    snapshot = {
        "objects": {
            "table": _cube_snapshot("table", (0.0, 0.0, 0.0), 1.0),
            "lamp_base": _cube_snapshot("lamp_base", (0.0, 0.0, 0.55), 0.1),
        }
    }
    diff = verify(plan, snapshot, _cfg())
    statuses = {p.part_id: p.status for p in diff.parts}
    assert statuses == {"table": "ok", "lamp_base": "ok"}


def test_extras_listed() -> None:
    plan = Plan(goal="x", parts=[_abs_part("a", "cube", {"x": 1.0, "y": 1.0, "z": 1.0}, (0.0, 0.0, 0.0))])
    snapshot = {
        "objects": {
            "a": _cube_snapshot("a", (0.0, 0.0, 0.0), 1.0),
            "StrayCube": _cube_snapshot("StrayCube", (5.0, 0.0, 0.0), 1.0),
        }
    }
    diff = verify(plan, snapshot, _cfg())
    assert "StrayCube" in diff.extras


def test_structural_when_two_parts_off() -> None:
    plan = Plan(
        goal="x",
        parts=[
            _abs_part("a", "cube", {"x": 1.0, "y": 1.0, "z": 1.0}, (0.0, 0.0, 0.0)),
            _abs_part("b", "cube", {"x": 1.0, "y": 1.0, "z": 1.0}, (2.0, 0.0, 0.0)),
        ],
    )
    snapshot = {
        "objects": {
            "a": _cube_snapshot("a", (1.0, 0.0, 0.0), 1.0),  # off
            "b": _cube_snapshot("b", (5.0, 0.0, 0.0), 1.0),  # off
        }
    }
    diff = verify(plan, snapshot, _cfg())
    assert diff.is_structural is True
```

- [ ] **Step 8.2: Run test to verify it fails**

Run: `uv run pytest tests/test_verifier.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'blendering.verifier'`.

- [ ] **Step 8.3: Add `VerifierConfig`**

In `src/blendering/config.py`, add before `Settings`:

```python
class VerifierConfig(BaseModel):
    dimension_tolerance: float = 0.15           # fractional, applied to each dim
    position_tolerance: float = 0.10            # meters
    orientation_tolerance_deg: float = 10.0
    missing_is_structural: bool = True
    off_threshold_for_structural: int = 2
```

Then in `Settings`:

```python
class Settings(BaseModel):
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    actor: ModelConfig
    critic: ModelConfig
    planner: PlannerConfig | None = None
    loop: LoopConfig = Field(default_factory=LoopConfig)
    framing: FramingConfig = Field(default_factory=FramingConfig)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    # ... _check_api_keys unchanged
```

- [ ] **Step 8.4: Implement `verifier.py`**

Create `src/blendering/verifier.py`:

```python
"""Deterministic checker that compares the current Blender scene to the Plan.

Pure function: no I/O, no MCP. The orchestrator gathers the snapshot dict
and passes it in. Output is a structured VerifierDiff consumed by both
the Actor (next turn) and the Critic (this turn)."""

from __future__ import annotations

from typing import Any

from .config import VerifierConfig
from .schemas import PartDiff, PartSpec, Plan, PositionSpec, VerifierDiff


def verify(plan: Plan, snapshot: dict[str, Any], cfg: VerifierConfig) -> VerifierDiff:
    """Compare the Plan to the scene snapshot. Returns a structured diff."""
    objects: dict[str, dict[str, Any]] = snapshot.get("objects", {})
    part_diffs: list[PartDiff] = []
    seen_names: set[str] = set()
    off_count = 0

    # Index parts by id for relative-position lookups.
    by_id: dict[str, PartSpec] = {p.id: p for p in plan.parts}

    for part in plan.parts:
        obj = objects.get(part.id)
        # Fuzzy fallback by primitive guess if exact-name not found.
        if obj is None:
            for name, candidate in objects.items():
                if name in seen_names:
                    continue
                if _fuzzy_match(part, candidate, cfg):
                    obj = candidate
                    seen_names.add(name)
                    break

        if obj is None:
            part_diffs.append(
                PartDiff(part_id=part.id, status="missing", issues=["object not found"])
            )
            continue
        seen_names.add(part.id)

        issues: list[str] = []
        measured: dict[str, Any] = {
            "location": obj.get("world_location"),
            "vert_count": obj.get("vert_count"),
        }

        # Primitive sanity (lenient — used only as a soft hint).
        primitive_ok = _primitive_plausible(part.primitive, obj.get("vert_count"))
        if not primitive_ok:
            issues.append(
                f"vert_count {obj.get('vert_count')} unusual for primitive {part.primitive}"
            )

        # Dimensions
        dim_issues, dim_measured = _check_dimensions(part, obj, cfg)
        issues.extend(dim_issues)
        measured.update(dim_measured)

        # Position (absolute or relative)
        pos_issues, expected = _check_position(part, obj, by_id, objects, cfg)
        issues.extend(pos_issues)
        measured["expected_location"] = expected

        # Orientation
        rot = obj.get("rotation_euler_deg") or [0.0, 0.0, 0.0]
        for axis_idx, axis in enumerate(("x", "y", "z")):
            delta = abs(rot[axis_idx] - part.orientation_deg[axis_idx])
            # Wrap delta to [0, 180]
            delta = delta % 360.0
            if delta > 180.0:
                delta = 360.0 - delta
            if delta > cfg.orientation_tolerance_deg:
                issues.append(
                    f"rotation_{axis} {rot[axis_idx]:.1f}° vs plan {part.orientation_deg[axis_idx]:.1f}° "
                    f"(delta {delta:.1f}°)"
                )

        status = "off" if issues else "ok"
        if status == "off":
            off_count += 1
        part_diffs.append(
            PartDiff(part_id=part.id, status=status, issues=issues, measured=measured)
        )

    extras = sorted(name for name in objects if name not in seen_names)

    missing_count = sum(1 for d in part_diffs if d.status == "missing")
    is_structural = (
        (cfg.missing_is_structural and missing_count > 0)
        or off_count >= cfg.off_threshold_for_structural
    )

    summary = _summary(part_diffs, extras)

    return VerifierDiff(
        plan_version=plan.version,
        parts=part_diffs,
        extras=extras,
        summary=summary,
        is_structural=is_structural,
    )


def _fuzzy_match(part: PartSpec, candidate: dict[str, Any], cfg: VerifierConfig) -> bool:
    """Used when exact-name lookup fails. Match on primitive guess + rough dims."""
    if candidate.get("primitive_guess") != part.primitive:
        return False
    extents = _extents(candidate)
    if extents is None:
        return False
    expected = _expected_extents(part)
    if expected is None:
        return False
    for axis_idx in range(3):
        if expected[axis_idx] == 0.0:
            continue
        if abs(extents[axis_idx] - expected[axis_idx]) / expected[axis_idx] > cfg.dimension_tolerance:
            return False
    return True


def _primitive_plausible(primitive: str, vert_count: Any) -> bool:
    if not isinstance(vert_count, int):
        return True  # don't penalize unknown
    table = {
        "cube": (8, 8),
        "plane": (4, 4),
        "cone": (3, 200),
        "cylinder": (6, 200),
        "sphere": (10, 5000),
    }
    if primitive not in table:
        return True
    lo, hi = table[primitive]
    return lo <= vert_count <= hi


def _extents(obj: dict[str, Any]) -> tuple[float, float, float] | None:
    mn = obj.get("world_bbox_min")
    mx = obj.get("world_bbox_max")
    if not mn or not mx:
        return None
    return (mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2])


def _expected_extents(part: PartSpec) -> tuple[float, float, float] | None:
    d = part.dimensions
    if part.primitive == "cube" or part.primitive == "plane":
        return (d.get("x", 0.0), d.get("y", 0.0), d.get("z", 0.0))
    if part.primitive in ("cylinder", "cone"):
        r = d.get("radius", 0.0)
        return (2 * r, 2 * r, d.get("height", 0.0))
    if part.primitive == "sphere":
        r = d.get("radius", 0.0)
        return (2 * r, 2 * r, 2 * r)
    return None


def _check_dimensions(
    part: PartSpec, obj: dict[str, Any], cfg: VerifierConfig
) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    extents = _extents(obj)
    expected = _expected_extents(part)
    if extents is None or expected is None:
        return issues, {"extents": extents}
    for axis_idx, axis in enumerate(("x", "y", "z")):
        exp = expected[axis_idx]
        if exp <= 0.0:
            continue
        actual = extents[axis_idx]
        frac = abs(actual - exp) / exp
        if frac > cfg.dimension_tolerance:
            issues.append(
                f"{axis} extent {actual:.3f} vs plan {exp:.3f} "
                f"({frac * 100:.0f}% off; tol {cfg.dimension_tolerance * 100:.0f}%)"
            )
    return issues, {"extents": list(extents)}


def _check_position(
    part: PartSpec,
    obj: dict[str, Any],
    by_id: dict[str, PartSpec],
    objects: dict[str, dict[str, Any]],
    cfg: VerifierConfig,
) -> tuple[list[str], list[float] | None]:
    actual = obj.get("world_location")
    if actual is None:
        return ["world_location missing in snapshot"], None
    expected = _expected_position(part, by_id, objects)
    if expected is None:
        return [], None
    delta = sum((actual[i] - expected[i]) ** 2 for i in range(3)) ** 0.5
    if delta > cfg.position_tolerance:
        return (
            [
                f"position {tuple(round(v, 3) for v in actual)} vs expected "
                f"{tuple(round(v, 3) for v in expected)} (delta {delta:.3f}m; "
                f"tol {cfg.position_tolerance}m)"
            ],
            list(expected),
        )
    return [], list(expected)


def _expected_position(
    part: PartSpec,
    by_id: dict[str, PartSpec],
    objects: dict[str, dict[str, Any]],
) -> tuple[float, float, float] | None:
    pos: PositionSpec = part.position
    if pos.mode == "absolute":
        return pos.xyz
    # Relative: find anchor object in the snapshot, compute face point + offset.
    if pos.anchor_part is None:
        return None
    anchor_obj = objects.get(pos.anchor_part)
    if anchor_obj is None:
        return None
    face_point = _face_point(anchor_obj, pos.anchor_face)
    if face_point is None:
        return None
    # Expected: place this part such that its corresponding face sits on the anchor's face.
    # We approximate by offsetting the face_point by half this part's extent along the face normal,
    # so the part's CENTER lands on (face_point + normal * half_extent + offset).
    own_extents = _expected_extents(part)
    half_along_normal = 0.0
    if own_extents is not None:
        face_axis = _face_axis(pos.anchor_face)
        if face_axis is not None:
            half_along_normal = own_extents[face_axis] / 2.0
    normal = _face_normal(pos.anchor_face)
    ox, oy, oz = pos.offset
    return (
        face_point[0] + normal[0] * half_along_normal + ox,
        face_point[1] + normal[1] * half_along_normal + oy,
        face_point[2] + normal[2] * half_along_normal + oz,
    )


def _face_point(obj: dict[str, Any], face: str | None) -> tuple[float, float, float] | None:
    mn = obj.get("world_bbox_min")
    mx = obj.get("world_bbox_max")
    if not mn or not mx or face is None:
        return None
    cx = (mn[0] + mx[0]) / 2.0
    cy = (mn[1] + mx[1]) / 2.0
    cz = (mn[2] + mx[2]) / 2.0
    if face == "top":
        return (cx, cy, mx[2])
    if face == "bottom":
        return (cx, cy, mn[2])
    if face == "front":
        return (cx, mn[1], cz)
    if face == "back":
        return (cx, mx[1], cz)
    if face == "left":
        return (mn[0], cy, cz)
    if face == "right":
        return (mx[0], cy, cz)
    if face == "center":
        return (cx, cy, cz)
    return None


def _face_normal(face: str | None) -> tuple[float, float, float]:
    return {
        "top": (0.0, 0.0, 1.0),
        "bottom": (0.0, 0.0, -1.0),
        "front": (0.0, -1.0, 0.0),
        "back": (0.0, 1.0, 0.0),
        "left": (-1.0, 0.0, 0.0),
        "right": (1.0, 0.0, 0.0),
        "center": (0.0, 0.0, 0.0),
    }.get(face or "", (0.0, 0.0, 0.0))


def _face_axis(face: str | None) -> int | None:
    return {
        "top": 2, "bottom": 2,
        "front": 1, "back": 1,
        "left": 0, "right": 0,
    }.get(face or "")


def _summary(parts: list[PartDiff], extras: list[str]) -> str:
    ok = sum(1 for p in parts if p.status == "ok")
    off = sum(1 for p in parts if p.status == "off")
    missing = sum(1 for p in parts if p.status == "missing")
    bits = [f"{ok} ok"]
    if off:
        bits.append(f"{off} off")
    if missing:
        bits.append(f"{missing} missing")
    if extras:
        bits.append(f"{len(extras)} extra")
    return ", ".join(bits)
```

- [ ] **Step 8.5: Run test to verify it passes**

Run: `uv run pytest tests/test_verifier.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 8.6: Update `config.example.yaml`**

Append:

```yaml

verifier:
  dimension_tolerance: 0.15
  position_tolerance: 0.10
  orientation_tolerance_deg: 10
  missing_is_structural: true
  off_threshold_for_structural: 2
```

- [ ] **Step 8.7: Commit**

```bash
git add src/blendering/verifier.py src/blendering/config.py config.example.yaml tests/test_verifier.py
git commit -m "feat(verifier): deterministic plan-vs-scene diff

Pure-function verify(plan, snapshot, cfg) -> VerifierDiff that checks
each part's presence, primitive sanity, dimensions, absolute/relative
position, and orientation against configurable tolerances. Marks the
diff is_structural when parts are missing or N+ are off-plan.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Scene snapshot gathering + Verifier wiring in orchestrator

The Verifier runs every step. The orchestrator needs to gather the snapshot via MCP and pass it in. Replans remain disabled (`max_replans=0` default) per the rollout plan.

**Files:**
- Modify: `src/blendering/orchestrator.py`
- Modify: `src/blendering/config.py` (add `LoopConfig.max_replans`)
- Modify: `src/blendering/llm.py` (Critic now sees plan + diff)
- Modify: `src/blendering/prompts.py` (Critic prompt updated)
- Modify: `tests/test_orchestrator.py`

- [ ] **Step 9.1: Write the failing test**

Append to `tests/test_orchestrator.py`:

```python
async def test_verifier_runs_each_step_and_feeds_actor(
    monkeypatch: pytest.MonkeyPatch, fake_mcp: FakeMCP
) -> None:
    """After each Actor turn, orchestrator must call get_scene_info + verify
    and inject the diff summary into the Actor's next-turn message list."""
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

    async def fake_plan(_c: Any, _g: str) -> tuple[Plan, int, int]:
        return plan_obj, 50, 25
    monkeypatch.setattr(orchestrator, "plan", fake_plan)

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
```

- [ ] **Step 9.2: Run test to verify it fails**

Run: `uv run pytest tests/test_orchestrator.py::test_verifier_runs_each_step_and_feeds_actor -v`
Expected: FAIL — `gather_scene_snapshot` doesn't exist; `judge` doesn't accept `plan=`/`diff=`.

- [ ] **Step 9.3: Add `max_replans` to `LoopConfig`**

In `src/blendering/config.py`:

```python
class LoopConfig(BaseModel):
    max_iterations: int = 25
    screenshot_every_step: bool = True
    stop_on_stuck_streak: int = 3
    max_replans: int = 0  # disabled by default; raise to 2 once tolerances are calibrated.
```

- [ ] **Step 9.4: Update Critic prompt and `judge()` signature**

Replace `CRITIC_SYSTEM` in `src/blendering/prompts.py`:

```python
CRITIC_SYSTEM = """\
You are the **Critic** in a three-model team that creates 3D scenes in Blender.

You will receive:
1. The user's original goal.
2. The ACTIVE PLAN (a structured list of parts the Planner has committed to).
3. The latest VerifierDiff (a deterministic per-part check of the current scene
   against the plan).
4. A short transcript of the Actor's recent actions.
5. A framed viewport screenshot of the current scene.

Your job: judge holistic/aesthetic progress and decide one of these statuses:

- "continue" — Actor should keep working. Always set next_step_hint.
- "done"     — Scene plausibly matches the user's goal. Don't be greedy.
- "stuck"    — No visible change, recurring errors, or the Actor is thrashing.
- "structural_mismatch" — Verifier diff is severe enough that the Planner should
  revise the plan, not just nudge the Actor. Use this ONLY when:
    (a) parts are missing that the Actor cannot reasonably build under the
        current plan, OR
    (b) multiple parts are persistently off in ways that suggest the plan
        itself is wrong (e.g., dimensions specified incorrectly).
  When you choose this, populate `replan_reason` with what the Planner should fix.

The Verifier owns geometric truth (positions, sizes). Don't second-guess its diff —
focus on aesthetic/holistic judgment ("does this look like a lamp?", "is the
lighting acceptable?") and on the decision above.

Return STRICT JSON:
- status, reasoning, next_step_hint, confidence, replan_reason (optional).
- Output ONLY the JSON object. No prose, no markdown fences.
"""
```

In `src/blendering/llm.py`, update `judge()` signature to accept plan and diff:

```python
async def judge(
    cfg: ModelConfig,
    system: str,
    user_goal: str,
    transcript: str,
    screenshot_png: bytes | None,
    *,
    plan: Plan | None = None,
    diff: VerifierDiff | None = None,
) -> tuple[Verdict, int, int]:
    """Ask the Critic to evaluate the scene. Returns a validated Verdict."""
    text_parts = [f"USER GOAL:\n{user_goal}"]
    if plan is not None:
        text_parts.append(f"ACTIVE PLAN (v{plan.version}):\n{plan.model_dump_json(indent=2)}")
    if diff is not None:
        text_parts.append(f"VERIFIER DIFF:\n{diff.model_dump_json(indent=2)}")
    text_parts.append(f"RECENT ACTOR TRANSCRIPT:\n{transcript}")
    text_parts.append(_VERDICT_INSTRUCTION)
    text_block = "\n\n".join(text_parts)

    user_content: list[dict[str, Any]] = [{"type": "text", "text": text_block}]
    if screenshot_png is not None:
        small = thumbnail_bytes(screenshot_png)
        user_content.append(
            {"type": "image_url", "image_url": {"url": encode_b64_data_url(small)}}
        )
    else:
        user_content[0]["text"] += "\n\n(No screenshot available this turn.)"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

    # ... existing retry loop and validation unchanged
```

Update `_VERDICT_INSTRUCTION` to reflect the new schema:

```python
_VERDICT_INSTRUCTION = (
    "Respond ONLY with a JSON object matching this schema:\n"
    '{"status": "continue"|"done"|"stuck"|"structural_mismatch", '
    '"reasoning": str, "next_step_hint": str|null, '
    '"confidence": number in [0,1], "replan_reason": str|null}'
)
```

- [ ] **Step 9.5: Add `gather_scene_snapshot()` in the orchestrator**

In `src/blendering/orchestrator.py`, add near the other helpers:

```python
async def gather_scene_snapshot(mcp: BlenderMCP, plan: Plan) -> dict[str, Any]:
    """Snapshot all objects referenced by the plan plus any extras, in the shape
    the Verifier expects. Uses a single execute_blender_code call to inspect the
    scene — cheaper than N round-trip get_object_info calls."""
    script = """
import bpy, json
out = {"objects": {}}
for o in bpy.context.scene.objects:
    if o.type != "MESH":
        continue
    me = o.data
    vert_count = len(me.vertices) if me else 0
    mn = [float("inf")] * 3
    mx = [float("-inf")] * 3
    import mathutils
    for corner in o.bound_box:
        wc = o.matrix_world @ mathutils.Vector(corner)
        for i in range(3):
            mn[i] = min(mn[i], wc[i])
            mx[i] = max(mx[i], wc[i])
    rot_deg = [r * 180.0 / 3.141592653589793 for r in o.rotation_euler]
    # Cheap primitive guess from vert count.
    if vert_count == 8:
        prim = "cube"
    elif vert_count == 4:
        prim = "plane"
    elif vert_count <= 200 and me and any("Cylinder" in m.name for m in [me]):
        prim = "cylinder"
    else:
        prim = "mesh"
    out["objects"][o.name] = {
        "primitive_guess": prim,
        "vert_count": vert_count,
        "world_location": [o.location.x, o.location.y, o.location.z],
        "world_bbox_min": mn,
        "world_bbox_max": mx,
        "rotation_euler_deg": rot_deg,
    }
print(json.dumps(out))
"""
    result = await mcp.call_tool("execute_blender_code", {"code": script})
    text = (result.text or "").strip()
    # Find the last JSON object emitted (Blender may print other lines).
    import json as _json
    start = text.rfind("{")
    if start < 0:
        return {"objects": {}}
    try:
        return _json.loads(text[start:])
    except _json.JSONDecodeError:
        return {"objects": {}}
```

> Implementation note: the primitive guess from vert count is intentionally crude — the Verifier already does `_fuzzy_match` by primitive AND dims, and the Actor sets `obj.name = part.id` per the prompt. If primitive guessing becomes a problem later, switch to per-object `get_object_info` calls.

- [ ] **Step 9.6: Wire Verifier into the orchestrator loop**

In `src/blendering/orchestrator.py`:

Add imports:

```python
from .schemas import PartProposal, Plan, RunOutcome, Verdict, VerifierDiff
from .verifier import verify
```

In `run()`, after the screenshot block and before the Critic call, insert:

```python
                # Verifier — pure code, runs every step.
                snapshot_task = asyncio.create_task(gather_scene_snapshot(mcp, current_plan))
                try:
                    snapshot = await _wait_or_cancel(snapshot_task, cancel)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    await bus.post(
                        StatusMessage(text=f"Snapshot failed: {exc}", level="warn")
                    )
                    snapshot = {"objects": {}}
                last_diff = verify(current_plan, snapshot, settings.verifier)
                await bus.post(StatusMessage(text=f"Verifier: {last_diff.summary}"))
```

Replace the Critic call site:

```python
                judge_task = asyncio.create_task(
                    judge(
                        settings.critic,
                        CRITIC_SYSTEM,
                        user_prompt,
                        transcript,
                        screenshot_bytes,
                        plan=current_plan,
                        diff=last_diff,
                    )
                )
```

Replace the existing "Inject critic feedback for next iteration" block so the Verifier diff is also surfaced to the Actor:

```python
                # Inject Verifier + Critic feedback for next iteration.
                off_lines = "\n".join(
                    f"  - {p.part_id} ({p.status}): {'; '.join(p.issues) if p.issues else 'ok'}"
                    for p in last_diff.parts
                )
                feedback = (
                    f"VERIFIER DIFF (plan v{last_diff.plan_version}): {last_diff.summary}\n"
                    f"{off_lines}\n"
                    f"Critic feedback (status={verdict.status}, "
                    f"confidence={verdict.confidence:.2f}):\n"
                    f"{verdict.reasoning}\n"
                    f"Next step hint: {verdict.next_step_hint or '(none)'}"
                )
                messages.append({"role": "user", "content": feedback})
```

- [ ] **Step 9.7: Run tests to verify they pass**

Run: `uv run pytest tests/ -v`
Expected: all tests pass, including the new `test_verifier_runs_each_step_and_feeds_actor`.

- [ ] **Step 9.8: Commit**

```bash
git add src/blendering/orchestrator.py src/blendering/config.py src/blendering/llm.py src/blendering/prompts.py tests/test_orchestrator.py
git commit -m "feat(verifier): wire scene snapshot + Verifier into orchestrator loop

Each step: gather_scene_snapshot via execute_blender_code → verify() →
feed diff to Critic (now takes plan= and diff= kwargs) and to the next
Actor turn. Critic prompt updated and gains structural_mismatch verdict
(no replan branch yet — max_replans default is 0).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

**Phase 3 complete: Verifier diffs are visible to Actor and Critic; replans still disabled.**

---

## Task 10: Replan branch on structural_mismatch

**Files:**
- Modify: `src/blendering/orchestrator.py`
- Modify: `src/blendering/llm.py` (`replan` import already present from Task 4)
- Modify: `tests/test_orchestrator.py`
- Modify: `config.example.yaml` (default `max_replans: 2`)

- [ ] **Step 10.1: Write the failing test**

Append to `tests/test_orchestrator.py`:

```python
async def test_structural_mismatch_triggers_replan_under_budget(
    monkeypatch: pytest.MonkeyPatch, fake_mcp: FakeMCP
) -> None:
    """When Critic returns structural_mismatch and max_replans > 0, the
    orchestrator must call llm.replan and continue with the new plan."""
    from blendering.schemas import PartSpec, Plan, PositionSpec, VerifierDiff

    plan_v1 = Plan(
        goal="x",
        parts=[
            PartSpec(
                id="a", description="a", primitive="cube",
                dimensions={"x": 1.0, "y": 1.0, "z": 1.0},
                position=PositionSpec(mode="absolute", xyz=(0.0, 0.0, 0.0)),
            )
        ],
        version=1,
    )
    plan_v2 = plan_v1.model_copy(update={"version": 2, "scene_notes": "revised"})

    plan_calls = {"n": 0}

    async def fake_plan(_c: Any, _g: str) -> tuple[Plan, int, int]:
        plan_calls["n"] += 1
        return plan_v1, 50, 25

    replan_calls = {"n": 0}

    async def fake_replan(
        _c: Any, *, prior: Plan, diff: VerifierDiff, recent_actions: str,
        proposals: list[Any], screenshot_png: Any = None,
    ) -> tuple[Plan, int, int]:
        replan_calls["n"] += 1
        return plan_v2, 60, 30

    monkeypatch.setattr(orchestrator, "plan", fake_plan)
    monkeypatch.setattr(orchestrator, "replan", fake_replan)
    monkeypatch.setattr(
        orchestrator,
        "stream_actor",
        _make_stream(
            [
                [ActorDelta(text="step 1", done=True)],
                [ActorDelta(text="step 2", done=True)],
            ]
        ),
    )
    _patch_judge(
        monkeypatch,
        [
            Verdict(status="structural_mismatch", reasoning="bad plan",
                    replan_reason="resize cube", confidence=0.7),
            Verdict(status="done", reasoning="ok", confidence=0.9),
        ],
    )

    async def fake_snapshot(_m: Any, _p: Plan) -> dict:
        return {"objects": {}}
    monkeypatch.setattr(orchestrator, "gather_scene_snapshot", fake_snapshot)

    settings = _settings(max_iter=5)
    settings.loop.max_replans = 1

    bus = EventBus()
    cancel = asyncio.Event()
    run_task = asyncio.create_task(run(settings, "x", bus, cancel))
    events = await _collect_events(bus, run_task)

    assert plan_calls["n"] == 1
    assert replan_calls["n"] == 1
    finished = events[-1]
    assert isinstance(finished, RunFinished)
    assert finished.outcome.status == "done"


async def test_structural_mismatch_with_zero_budget_exits_stuck(
    monkeypatch: pytest.MonkeyPatch, fake_mcp: FakeMCP
) -> None:
    """With max_replans=0, a structural_mismatch verdict counts as stuck and
    is subject to the stuck_streak abort."""
    from blendering.schemas import PartSpec, Plan, PositionSpec

    plan_v1 = Plan(
        goal="x",
        parts=[
            PartSpec(
                id="a", description="a", primitive="cube",
                dimensions={"x": 1.0, "y": 1.0, "z": 1.0},
                position=PositionSpec(mode="absolute", xyz=(0.0, 0.0, 0.0)),
            )
        ],
    )

    async def fake_plan(_c: Any, _g: str) -> tuple[Plan, int, int]:
        return plan_v1, 50, 25
    monkeypatch.setattr(orchestrator, "plan", fake_plan)

    async def fake_snapshot(_m: Any, _p: Plan) -> dict:
        return {"objects": {}}
    monkeypatch.setattr(orchestrator, "gather_scene_snapshot", fake_snapshot)

    monkeypatch.setattr(
        orchestrator,
        "stream_actor",
        _make_stream(
            [[ActorDelta(text=f"s{i}", done=True)] for i in range(5)]
        ),
    )
    _patch_judge(
        monkeypatch,
        [
            Verdict(status="structural_mismatch", reasoning="bad", confidence=0.5),
            Verdict(status="structural_mismatch", reasoning="bad", confidence=0.5),
        ],
    )

    settings = _settings(max_iter=5, stuck_streak=2)
    settings.loop.max_replans = 0

    bus = EventBus()
    cancel = asyncio.Event()
    run_task = asyncio.create_task(run(settings, "x", bus, cancel))
    events = await _collect_events(bus, run_task)
    finished = events[-1]
    assert isinstance(finished, RunFinished)
    assert finished.outcome.status == "stuck"
```

- [ ] **Step 10.2: Run test to verify it fails**

Run: `uv run pytest tests/test_orchestrator.py::test_structural_mismatch_triggers_replan_under_budget -v`
Expected: FAIL — replan branch not implemented.

- [ ] **Step 10.3: Implement the replan branch**

In `src/blendering/orchestrator.py`, add the import:

```python
from .llm import judge, plan, replan, stream_actor
```

Add state initialization right after `pending_proposals: list[PartProposal] = []`:

```python
            replan_count = 0
            recent_actions_log: list[str] = []  # bounded summary fed to replan
```

After each Actor turn, push a short summary into the log:

```python
                recent_actions_log.append(transcript[:200])
                if len(recent_actions_log) > 5:
                    recent_actions_log.pop(0)
```

Replace the verdict-branching section (currently handles `done` and `stuck`) so it also handles `structural_mismatch`. Insert this branch BEFORE the `if verdict.status == "stuck":` check:

```python
                if verdict.status == "structural_mismatch":
                    if replan_count < settings.loop.max_replans:
                        replan_count += 1
                        await bus.post(
                            StatusMessage(
                                text=f"Replanning ({replan_count}/{settings.loop.max_replans}): "
                                f"{verdict.replan_reason or verdict.reasoning}"
                            )
                        )
                        try:
                            replan_task = asyncio.create_task(
                                replan(
                                    settings.planner or settings.actor,
                                    prior=current_plan,
                                    diff=last_diff,
                                    recent_actions="\n".join(recent_actions_log),
                                    proposals=pending_proposals,
                                    screenshot_png=screenshot_bytes,
                                )
                            )
                            new_plan, p_in, p_out = await _wait_or_cancel(replan_task, cancel)
                        except asyncio.CancelledError:
                            raise
                        meter.planner.add(p_in, p_out)
                        # Diff the plans for the Actor.
                        prior_ids = {p.id for p in current_plan.parts}
                        new_ids = {p.id for p in new_plan.parts}
                        added = sorted(new_ids - prior_ids)
                        removed = sorted(prior_ids - new_ids)
                        plan_diff_msg = (
                            f"Plan revised to v{new_plan.version}. "
                            f"Added: {added or '[]'}. Removed: {removed or '[]'}."
                        )
                        await bus.post(StatusMessage(text=plan_diff_msg))

                        current_plan = new_plan
                        pending_proposals = []  # consumed by the replan
                        stuck_streak = 0
                        # Replace the Actor system prompt so it sees the new plan.
                        actor_system = (
                            ACTOR_SYSTEM
                            + "\n\nACTIVE PLAN:\n"
                            + current_plan.model_dump_json(indent=2)
                        )
                        # Drop the prior system message and insert the new one.
                        messages[0] = {"role": "system", "content": actor_system}
                        # Tell the Actor what changed.
                        messages.append({"role": "user", "content": plan_diff_msg})
                        continue
                    else:
                        # Out of replan budget — treat as stuck.
                        verdict = verdict.model_copy(
                            update={"status": "stuck",
                                    "reasoning": f"structural_mismatch with no replan budget: {verdict.reasoning}"}
                        )
                        # Fall through to the stuck handler below.
```

- [ ] **Step 10.4: Update default `max_replans` in `config.example.yaml`**

Add to the `loop:` section:

```yaml
loop:
  max_iterations: 25
  screenshot_every_step: true
  stop_on_stuck_streak: 3
  max_replans: 2
```

- [ ] **Step 10.5: Run tests to verify they pass**

Run: `uv run pytest tests/ -v`
Expected: all tests pass, including both new replan tests.

- [ ] **Step 10.6: Commit**

```bash
git add src/blendering/orchestrator.py config.example.yaml tests/test_orchestrator.py
git commit -m "feat(replan): handle structural_mismatch verdict via Planner.replan

Critic's structural_mismatch now triggers llm.replan when under budget
(loop.max_replans). The new plan replaces the Actor system prompt, the
plan diff is announced to the Actor, and pending proposals are
consumed. Out of budget: verdict is downgraded to stuck.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

**Phase 4 complete: full feature shipped.**

---

## Task 11: README + smoke check

**Files:**
- Modify: `README.md`
- Modify: `scripts/` (existing smoke script, if any)

- [ ] **Step 11.1: Update README**

In `README.md`, add a brief section under the "Why two models?" section:

```markdown
## Why three models?

This branch adds a **Planner** role between the user goal and the Actor. The
Planner runs once at the start of each run to emit a structured `Plan`
(list of parts with dimensions and positions). The Actor builds against
that plan; a deterministic Verifier diffs the live scene against the plan
after every step and feeds the diff to both the Actor and the Critic.

The Critic gains a `structural_mismatch` verdict that triggers a Planner
*replan* — capped by `loop.max_replans` — when the plan itself is the
problem (vs. the Actor just needing a nudge).

Auto-framing also added: before each screenshot, the orchestrator reframes
the active camera around the scene AABB so the Critic never sees clipped
content.

See `docs/superpowers/specs/2026-05-20-blender-agent-accuracy-design.md`
for the full design.
```

- [ ] **Step 11.2: Run the full test suite once more**

Run: `uv run pytest tests/ -v`
Expected: all tests pass.

- [ ] **Step 11.3: Commit**

```bash
git add README.md
git commit -m "docs: explain Planner/Verifier/auto-frame additions in README

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Manual verification (post-implementation)

After all tasks merge, run an end-to-end check against a real Blender:

1. Ensure Blender is running with the MCP add-on enabled and the server connected.
2. `cp config.example.yaml config.yaml` and set API keys via `.env`.
3. `uv run blendering "build a low-poly desk lamp on a wooden table, lit from above-left"`
4. Watch for:
   - StatusMessage `Plan v1: N part(s) — ...` near the start.
   - StatusMessage `Verifier: M ok, K off, ...` after each iteration.
   - Screenshots that no longer clip content.
   - If the build goes sideways, a `Replanning (1/2): ...` line and a `Plan revised to v2` follow-up.

Take screenshots and check `screenshots/` to confirm visible improvement over the pre-change behavior.
