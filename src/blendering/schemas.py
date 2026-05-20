"""Pydantic models exchanged between orchestrator components."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


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


class ToolCallRecord(BaseModel):
    """Record of a single tool invocation by the Actor."""

    name: str
    arguments: dict[str, Any]
    result_preview: str = ""
    is_error: bool = False


class StepResult(BaseModel):
    """Result of one Actor turn: any text it spoke + the tool calls it made."""

    text: str = ""
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)


class RunOutcome(BaseModel):
    """Terminal state of an orchestrator run."""

    status: Literal["done", "stuck", "max_iterations", "cancelled", "error"]
    iterations: int
    last_verdict: Verdict | None = None
    error: str | None = None


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
