"""Pydantic models exchanged between orchestrator components."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Verdict(BaseModel):
    """Critic's structured judgement of progress after a single Actor step."""

    status: Literal["continue", "done", "stuck"]
    reasoning: str = Field(description="One short paragraph explaining the verdict.")
    next_step_hint: str | None = Field(
        default=None,
        description="Concrete next action for the Actor. Required when status='continue'.",
    )
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


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
