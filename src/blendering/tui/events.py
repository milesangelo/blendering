"""Typed events flowing from the orchestrator to the TUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..schemas import RunOutcome, Verdict


@dataclass
class ActorTextDelta:
    text: str


@dataclass
class ToolCallStart:
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolCallResult:
    name: str
    result_preview: str
    is_error: bool


@dataclass
class CriticVerdictEvent:
    verdict: Verdict


@dataclass
class ViewportUpdate:
    image_bytes: bytes
    path: str


@dataclass
class IterationStart:
    n: int
    total: int


@dataclass
class StatusMessage:
    text: str
    level: str = "info"  # "info" | "warn" | "error"


@dataclass
class RunFinished:
    outcome: RunOutcome


Event = (
    ActorTextDelta
    | ToolCallStart
    | ToolCallResult
    | CriticVerdictEvent
    | ViewportUpdate
    | IterationStart
    | StatusMessage
    | RunFinished
)
