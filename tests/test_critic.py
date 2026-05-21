"""Tests for the Critic LLM client (judge). LiteLLM's acompletion is patched
so no network is hit. Focus: verify the user message stays compact (no full
Plan/VerifierDiff JSON dumps) so TPM quotas don't blow up on real providers."""

from __future__ import annotations

import json
from typing import Any

import pytest

from blendering import llm
from blendering.config import ModelConfig
from blendering.schemas import (
    PartDiff,
    PartSpec,
    Plan,
    PositionSpec,
    VerifierDiff,
)


def _critic_cfg() -> ModelConfig:
    return ModelConfig(model="fake/critic", api_key_env="X")


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


def _two_part_plan() -> Plan:
    return Plan(
        goal="lamp on table",
        version=3,
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


def _diff_with_one_off() -> VerifierDiff:
    return VerifierDiff(
        plan_version=3,
        parts=[
            PartDiff(part_id="table", status="ok", issues=[]),
            PartDiff(
                part_id="lamp_base",
                status="off",
                issues=["height 0.42 vs plan 0.05 (740% over)"],
            ),
        ],
        extras=["StrayCube"],
        summary="1 ok, 1 off, 1 extra",
        is_structural=False,
    )


def _extract_text(captured_messages: list[dict[str, Any]]) -> str:
    """Pull the first text content chunk from the user message in a captured
    litellm.acompletion call."""
    user_msg = captured_messages[-1]
    content = user_msg["content"]
    if isinstance(content, str):
        return content
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            return item["text"]
    return ""


async def test_judge_does_not_dump_full_plan_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Critic user message must NOT contain the verbose plan JSON dump;
    that was the source of the ~5K-token per-step Critic input on Qwen runs."""
    captured: dict[str, Any] = {}

    async def fake_completion(**kwargs: Any) -> _FakeResp:
        captured["messages"] = kwargs["messages"]
        return _FakeResp(
            {"status": "continue", "reasoning": "ok", "next_step_hint": "carry on", "confidence": 0.5}
        )

    monkeypatch.setattr(llm.litellm, "acompletion", fake_completion)

    plan = _two_part_plan()
    diff = _diff_with_one_off()
    await llm.judge(
        _critic_cfg(),
        "SYS",
        "lamp on table",
        "actor placed table",
        screenshot_png=None,
        plan=plan,
        diff=diff,
    )

    text = _extract_text(captured["messages"])

    # The forbidden pattern: pretty-printed JSON object with a "goal" key.
    # Both Plan.model_dump_json(indent=2) and VerifierDiff.model_dump_json(indent=2)
    # would produce something like `"goal":` and `"plan_version":` on their own lines.
    assert '"goal":' not in text, "Plan JSON must not be dumped to the Critic"
    assert '"plan_version":' not in text, "VerifierDiff JSON must not be dumped to the Critic"
    assert '"primitive":' not in text, "PartSpec JSON must not be dumped to the Critic"


async def test_judge_compact_format_includes_essentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The compact format still has to carry: goal, plan version + part ids,
    diff summary, off/missing part issues, extras, and the transcript."""
    captured: dict[str, Any] = {}

    async def fake_completion(**kwargs: Any) -> _FakeResp:
        captured["messages"] = kwargs["messages"]
        return _FakeResp(
            {"status": "continue", "reasoning": "ok", "next_step_hint": "carry on", "confidence": 0.5}
        )

    monkeypatch.setattr(llm.litellm, "acompletion", fake_completion)

    plan = _two_part_plan()
    diff = _diff_with_one_off()
    await llm.judge(
        _critic_cfg(),
        "SYS",
        "lamp on table",
        "actor placed table",
        screenshot_png=None,
        plan=plan,
        diff=diff,
    )

    text = _extract_text(captured["messages"])

    # Goal preserved.
    assert "lamp on table" in text
    # Plan version + part ids surfaced as a one-liner.
    assert "v3" in text
    assert "table" in text and "lamp_base" in text
    # Diff summary present.
    assert "1 ok, 1 off, 1 extra" in text
    # Off-part with its issue present.
    assert "lamp_base" in text
    assert "740% over" in text
    # Extras listed.
    assert "StrayCube" in text
    # Transcript preserved.
    assert "actor placed table" in text


async def test_judge_compact_payload_is_much_smaller_than_json_dump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity check: the new compact text must be smaller than the prior
    JSON-dump approach would have been. Ensures the trim is real, not cosmetic."""
    captured: dict[str, Any] = {}

    async def fake_completion(**kwargs: Any) -> _FakeResp:
        captured["messages"] = kwargs["messages"]
        return _FakeResp(
            {"status": "continue", "reasoning": "ok", "next_step_hint": "x", "confidence": 0.5}
        )

    monkeypatch.setattr(llm.litellm, "acompletion", fake_completion)

    plan = _two_part_plan()
    diff = _diff_with_one_off()
    await llm.judge(
        _critic_cfg(),
        "SYS",
        "g",
        "t",
        screenshot_png=None,
        plan=plan,
        diff=diff,
    )

    text = _extract_text(captured["messages"])

    # The would-have-been payload if we still dumped JSON for both:
    legacy_len = len(plan.model_dump_json(indent=2)) + len(diff.model_dump_json(indent=2))
    # The new payload should be substantially smaller (at least 50% reduction).
    assert len(text) < legacy_len * 0.5, (
        f"Compact payload should be <50% of legacy JSON dump size. "
        f"Got {len(text)} chars vs legacy ~{legacy_len}."
    )
