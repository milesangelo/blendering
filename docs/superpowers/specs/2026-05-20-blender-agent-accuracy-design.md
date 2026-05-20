# Blender Agent Accuracy Improvements — Design

**Date:** 2026-05-20
**Status:** Approved design, ready for implementation plan

## Problem

Two observed failure modes in the current Actor + Critic loop:

1. **Framing** — viewport screenshots clip pieces of the scene, so the Critic can't see what it's judging.
2. **Placement** — components end up positioned wrong relative to one another even when code executed successfully.

The Critic's vision-based feedback is too vague to drive convergence on placement issues, and the Actor has no durable spatial plan to execute against — every step is freshly improvised from the goal.

## Goals and non-goals

**Goals**
- Eliminate clipping in the screenshots the Critic sees.
- Give the Actor a structured, dimensioned plan to execute against.
- Detect placement and dimension errors programmatically (not by vision) and feed targeted, numeric feedback back to the Actor.
- Allow a stronger reasoning model to do the planning while keeping a cheaper code model as the Actor.

**Non-goals**
- Multi-view rendering / depth / wireframe overlays. (Considered, deferred.)
- LLM-based verification. Verification is deterministic code.
- Reference-image / target-sketch comparison.
- Hierarchical decomposition into sub-goals.

## Overall shape

```
goal → Planner.plan ─┐
                     ▼
                   PLAN  ◄────────── Planner.replan (only on structural mismatch)
                     │                                ▲
                     ▼                                │
            Actor.step → MCP tools → get_scene_info  │
                              │            │         │
                              ▼            ▼         │
                         auto-frame   Verifier.diff──┤
                              │            │         │
                              ▼            ▼         │
                          screenshot ─→ Critic.judge ┘
                                            │
                                            ▼
                                       done / continue / stuck / structural_mismatch
```

Key shape decisions:

- The plan is a first-class artifact, owned by the Planner. The Actor reads it, never writes it (with one escape hatch — see Actor section).
- The Verifier is pure code. It runs every step and emits a structured diff.
- Auto-framing is a pre-screenshot step. The Critic always sees a well-framed image.
- The Critic gains one new verdict, `structural_mismatch`. That is the only signal that triggers a replan.
- Planning, Acting, and Critiquing can each use a different model, configured independently.

## Components

### 1. Auto-framing (`framing.py`)

Pre-screenshot Blender-side reframe. Before each `get_viewport_screenshot`, the orchestrator sends a small Python script via `execute_blender_code` that:

1. Computes the world-space AABB across all mesh objects (skipping cameras, lights, and objects whose names match `exclude_tags`).
2. Points the active camera at the bbox centroid.
3. Dollies the camera back along its current view vector until the bbox fits the camera frustum with `padding` extra room on each side.
4. Honors `min_distance` so single small objects don't render at meaningless zoom.
5. Creates a default camera at a reasonable 3/4 angle on first reframe if none exists.
6. No-ops cleanly on empty scenes.

Camera intrinsics (focal length, sensor) are left alone — only the camera transform is changed — so perspective stays consistent across steps.

**Module surface:** `reframe_script(padding: float, min_distance: float, exclude_tags: list[str]) -> str` returns the Python to send. The orchestrator calls it; this module does no I/O.

**Config:**

```yaml
framing:
  auto_frame: true
  padding: 0.15
  min_distance: 2.0
  exclude_tags: ["_helper", "_guide"]
```

`auto_frame: false` disables the feature entirely.

### 2. Plan schema (`schemas.py`)

```python
class PositionSpec(BaseModel):
    mode: Literal["absolute", "relative"]
    # absolute mode
    xyz: tuple[float, float, float] | None = None
    # relative mode
    anchor_part: str | None = None
    anchor_face: Literal[
        "top", "bottom", "front", "back", "left", "right", "center"
    ] | None = None
    offset: tuple[float, float, float] = (0, 0, 0)

class PartSpec(BaseModel):
    id: str
    description: str
    primitive: Literal[
        "cube", "cylinder", "sphere", "cone", "plane", "mesh", "imported"
    ]
    dimensions: dict[str, float]
    position: PositionSpec
    orientation_deg: tuple[float, float, float] = (0, 0, 0)
    material_hint: str | None = None

class Plan(BaseModel):
    goal: str
    parts: list[PartSpec]
    scene_notes: str = ""
    version: int = 1
```

Tolerance lives on the Verifier, not the plan — the plan says *what should be true*, the Verifier decides *how close counts*.

### 3. Planner agent

A third LLM role, alongside Actor and Critic, configured independently:

```yaml
planner:
  model: anthropic/claude-sonnet-4-6
  api_base: null
  api_key_env: ANTHROPIC_API_KEY
```

**When it runs:**
- Once at the start of the loop, producing the initial `Plan` from the user goal.
- On every `structural_mismatch` verdict from the Critic, producing a revised `Plan` with `version += 1`.

**Inputs on replan:** prior plan, latest `VerifierDiff`, latest screenshot, short summary of recent Actor actions, and any pending `proposed_addition` items collected from the Actor since the last (re)plan.

**Prompts** (`PLANNER_SYSTEM`, `REPLANNER_SYSTEM`):
- Embed the `Plan` JSON schema.
- Output JSON only, validated against `Plan` via Pydantic. On schema failure, retry once with the error attached.
- Rules: prefer relative positions for stacked/attached parts; be explicit about dimensions in meters; one logical piece per part; on replan, keep `id`s stable for parts that are essentially right.

**Replan budget:** `loop.max_replans` (default 2). Beyond that, `structural_mismatch` is treated as `stuck` and the loop exits.

### 4. Verifier (`verifier.py`)

Pure code. Single function:

```python
def verify(plan: Plan, scene_snapshot: SceneSnapshot) -> VerifierDiff
```

**Scene snapshot** is gathered by the orchestrator each step via one `get_scene_info` call plus a `get_object_info` per object the plan references.

**Mapping plan parts to scene objects.** Plan part `id`s are used as object names. The Actor is instructed to set `obj.name = part.id` for every created or imported object. Verifier first tries exact-name lookup, then falls back to fuzzy match by primitive type and approximate dimensions before declaring a part `missing`.

**Per-part checks:**

| Check | Method |
|---|---|
| Exists | object with `name == part.id`, or fuzzy match |
| Primitive sanity | mesh vert/face count plausible for declared primitive |
| Dimensions | local bbox extents within `dimension_tolerance` (fractional, default 0.15) of plan dims |
| Absolute position | world-space origin within `position_tolerance` meters (default 0.10) of plan `xyz` |
| Relative position | computed: anchor part's `anchor_face` point + `offset` → expected world point; this part's anchor point within `position_tolerance` of it |
| Orientation | within `orientation_tolerance_deg` per axis (default 10) |

**Output:**

```python
class PartDiff(BaseModel):
    part_id: str
    status: Literal["ok", "off", "missing", "extra"]
    issues: list[str]       # e.g. "height 0.42 vs plan 0.30 (40% over)"
    measured: dict[str, Any]

class VerifierDiff(BaseModel):
    plan_version: int
    parts: list[PartDiff]
    extras: list[str]
    summary: str
    is_structural: bool
```

`is_structural` is true when any part is `missing` (and `missing_is_structural` is set) or when the count of `off` parts meets `off_threshold_for_structural`. It is a hint to the Critic, not a forcing function.

**Config:**

```yaml
verifier:
  dimension_tolerance: 0.15
  position_tolerance: 0.10
  orientation_tolerance_deg: 10
  missing_is_structural: true
  off_threshold_for_structural: 2
```

### 5. Actor changes

The Actor's contract becomes:
- Receive the active plan in its system prompt (plus the plan diff vs prior version on a replan turn).
- Receive the latest `VerifierDiff` as structured context each turn.
- Name every created or imported object with its plan `id`.
- Work on one part at a time. If a part is flagged `off`, fix only the flagged dimensions / positions; do not rebuild.

**Escape hatch — `proposed_addition`.** The Actor's structured output gains an optional `proposed_additions: list[PartProposal]` field. A `PartProposal` is a free-form description (no `id` assigned, no dimensions required) of a part the Actor believes is needed but isn't in the plan. Proposals do not gate the current step and the Actor does not build them — they accumulate in orchestrator state and are passed to the Planner on the next (re)plan call, where the Planner decides whether to incorporate them.

Rationale: the Plan stays single-source-of-truth (only the Planner writes it), but the Actor isn't silent about gaps it notices mid-build.

### 6. Critic changes

- Receives plan + `VerifierDiff` alongside the framed screenshot.
- Verdict enum extended to include `structural_mismatch`, with an optional `replan_reason: str`.
- Job shifts: deterministic geometric checks belong to the Verifier; the Critic now focuses on aesthetic/holistic judgment ("does this look like a lamp?", "is the lighting acceptable?", "is the goal met?") and on deciding when a Verifier diff is bad enough to warrant `structural_mismatch`.

## Orchestrator state additions

- `plan: Plan` — current active plan.
- `plan_history: list[Plan]` — bounded to last 5 versions, for logging.
- `replan_count: int` — checked against `loop.max_replans`.
- `last_diff: VerifierDiff | None` — passed to Actor and Critic each turn.
- `pending_proposals: list[PartProposal]` — accumulated Actor proposals awaiting next (re)plan.

## Files touched

- `orchestrator.py` — new state, new loop branches (initial plan, replan, verify step), auto-frame call before screenshot.
- `llm.py` — Planner client mirroring existing Actor/Critic clients.
- `prompts.py` — new `PLANNER_SYSTEM`, `REPLANNER_SYSTEM`; edits to `ACTOR_SYSTEM` and `CRITIC_SYSTEM`.
- `schemas.py` — `Plan`, `PartSpec`, `PositionSpec`, `PartProposal`, `VerifierDiff`, `PartDiff`; extended `CriticVerdict`.
- `config.py` — new `planner`, `framing`, `verifier` sections; new `loop.max_replans`.
- `verifier.py` — new module, pure-function `verify(...)`.
- `framing.py` — new module, pure-function `reframe_script(...)`.
- `tests/` — unit tests for `framing.reframe_script` (snapshot), `verifier.verify` (synthetic scenes), and orchestrator integration with mocked LLMs.

## Rollout / sequencing

Each step is independently shippable and provides observable value on its own.

1. **Auto-framing.** Self-contained, no schema churn, directly addresses the clipping symptom. Ship first.
2. **Plan schema + Planner agent.** Plan is generated and shown to the Actor; Verifier does not exist yet. Validates that planning alone improves Actor behavior.
3. **Verifier + diff-to-Actor with `max_replans=0`.** Verifier runs every step and feeds diffs to the Actor and Critic, but replans are disabled — observation/calibration period for tolerances.
4. **Enable replans.** Raise `max_replans` to 2 and ship.

## Open questions

None. All design points settled in the brainstorming pass.

## Risks and mitigations

- **Object naming discipline.** The whole Verifier story depends on the Actor naming objects as `part.id`. Mitigations: explicit instruction in `ACTOR_SYSTEM`, fuzzy fallback in the Verifier, and an `extras` list in the diff so unnamed/misnamed objects surface in logs immediately.
- **Tolerance miscalibration.** Too-tight tolerances will pump up false `structural_mismatch` rates. Mitigation: rollout step 3 disables replans precisely so we can observe diff distributions before letting them drive replans.
- **Planner cost.** A reasoning model on every replan is the most expensive call in the loop. Mitigation: `max_replans` cap and the `is_structural` threshold prevent runaway replans.
- **Plan rigidity.** A bad initial plan could trap the Actor. Mitigation: the `proposed_addition` escape hatch plus the explicit replan path on `structural_mismatch`.
