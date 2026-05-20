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
