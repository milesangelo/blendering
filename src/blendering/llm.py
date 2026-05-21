"""LiteLLM-backed Actor (streaming + tools) and Critic (vision + JSON verdict)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import litellm
from pydantic import ValidationError

from .config import ModelConfig
from .schemas import PartProposal, Plan, Verdict, VerifierDiff
from .utils.images import encode_b64_data_url, thumbnail_bytes
from .utils.logging import get_logger

log = get_logger("blendering.llm")


@dataclass
class ActorDelta:
    """One chunk from the Actor's streaming completion."""

    text: str = ""
    # Partial tool calls accumulate across deltas; complete ones land in `finished_tool_calls`.
    finished_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False
    # Final-chunk usage stats from the provider (only present once per call).
    in_tokens: int | None = None
    out_tokens: int | None = None


def _model_kwargs(cfg: ModelConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": cfg.model,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }
    if cfg.api_base:
        kwargs["api_base"] = cfg.api_base
    if cfg.api_key:
        kwargs["api_key"] = cfg.api_key
    return kwargs


async def stream_actor(
    cfg: ModelConfig,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> AsyncIterator[ActorDelta]:
    """Stream the Actor's response, yielding text deltas and completed tool calls."""
    log.debug("actor → %s (api_base=%s) tools=%d msgs=%d",
              cfg.model, cfg.api_base, len(tools), len(messages))
    response = await litellm.acompletion(
        **_model_kwargs(cfg),
        messages=messages,
        tools=tools or None,
        tool_choice="auto" if tools else None,
        stream=True,
        stream_options={"include_usage": True},
    )

    # Tool calls arrive in fragments — accumulate by index.
    partials: dict[int, dict[str, Any]] = {}

    async for chunk in response:
        # Final usage-only chunk has no choices but carries `usage`.
        usage = getattr(chunk, "usage", None)
        if not chunk.choices:
            if usage is not None:
                yield ActorDelta(
                    in_tokens=getattr(usage, "prompt_tokens", None),
                    out_tokens=getattr(usage, "completion_tokens", None),
                    done=True,
                )
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        text = getattr(delta, "content", None) or ""
        new_finished: list[dict[str, Any]] = []

        tc_list = getattr(delta, "tool_calls", None) or []
        for tc in tc_list:
            idx = getattr(tc, "index", 0) or 0
            slot = partials.setdefault(
                idx,
                {"id": None, "name": "", "arguments": ""},
            )
            if getattr(tc, "id", None):
                slot["id"] = tc.id
            fn = getattr(tc, "function", None)
            if fn is not None:
                if getattr(fn, "name", None):
                    slot["name"] += fn.name
                if getattr(fn, "arguments", None):
                    slot["arguments"] += fn.arguments

        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "tool_calls":
            for idx in sorted(partials):
                slot = partials.pop(idx)
                try:
                    args = json.loads(slot["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": slot["arguments"]}
                new_finished.append(
                    {"id": slot["id"], "name": slot["name"], "arguments": args}
                )

        done = finish_reason is not None
        if text or new_finished or done:
            yield ActorDelta(text=text, finished_tool_calls=new_finished, done=done)
        if done:
            return


_VERDICT_INSTRUCTION = (
    "Respond ONLY with a JSON object matching this schema:\n"
    '{"status": "continue"|"done"|"stuck"|"structural_mismatch", '
    '"reasoning": str, "next_step_hint": str|null, '
    '"confidence": number in [0,1], "replan_reason": str|null}'
)


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
    """Ask the Critic to evaluate the scene. Returns a validated Verdict.

    The user message stays deliberately compact — no Plan or VerifierDiff JSON
    dumps — so the per-step input fits inside tight provider TPM quotas. The
    Critic gets enough signal (goal + part ids + off/missing parts + extras +
    transcript + screenshot) to make holistic and structural-mismatch calls."""
    text_parts = [f"USER GOAL:\n{user_goal}"]
    if plan is not None:
        ids = ", ".join(p.id for p in plan.parts) or "(no parts)"
        plan_line = f"ACTIVE PLAN: v{plan.version} — {len(plan.parts)} part(s): {ids}"
        if plan.scene_notes:
            plan_line += f"\nScene notes: {plan.scene_notes}"
        text_parts.append(plan_line)
    if diff is not None:
        diff_lines = [f"VERIFIER DIFF (plan v{diff.plan_version}): {diff.summary}"]
        for p in diff.parts:
            if p.status == "ok":
                continue  # summary line already counts these
            issues = "; ".join(p.issues) if p.issues else "(no detail)"
            diff_lines.append(f"  - {p.part_id} [{p.status}]: {issues}")
        if diff.extras:
            diff_lines.append(f"  extras (not in plan): {', '.join(diff.extras)}")
        text_parts.append("\n".join(diff_lines))
    text_parts.append(f"RECENT ACTOR TRANSCRIPT:\n{transcript}")
    text_parts.append(_VERDICT_INSTRUCTION)
    text_block = "\n\n".join(text_parts)

    user_content: list[dict[str, Any]] = [{"type": "text", "text": text_block}]
    if screenshot_png is not None:
        small = thumbnail_bytes(screenshot_png)
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": encode_b64_data_url(small)},
            }
        )
    else:
        user_content[0]["text"] += "\n\n(No screenshot available this turn.)"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

    for attempt in range(2):
        try:
            log.debug("critic → %s (api_base=%s) attempt=%d", cfg.model, cfg.api_base, attempt + 1)
            resp = await litellm.acompletion(
                **_model_kwargs(cfg),
                messages=messages,
                response_format={"type": "json_object"},
                stream=False,
            )
            text = resp.choices[0].message.content or "{}"
            log.debug("critic raw: %s", text[:500])
            usage = getattr(resp, "usage", None)
            in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
            out_tok = getattr(usage, "completion_tokens", 0) if usage else 0
            data = json.loads(_strip_fences(text))
            return Verdict.model_validate(data), in_tok, out_tok
        except (json.JSONDecodeError, ValidationError) as exc:
            if attempt == 0:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response failed validation: {exc}. "
                            "Reply ONLY with the JSON object."
                        ),
                    }
                )
                continue
            return (
                Verdict(
                    status="stuck",
                    reasoning=f"Critic returned invalid JSON twice: {exc}",
                    confidence=0.0,
                ),
                0,
                0,
            )
    # Unreachable; satisfy type checker.
    raise RuntimeError("judge() exited loop without returning")


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        # If language tag present like ```json\n{...}
        if "\n" in t:
            t = t.split("\n", 1)[1]
    return t.strip()


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
        "\n".join(
            f"- {p.description}" + (f"  (rationale: {p.rationale})" if p.rationale else "")
            for p in proposals
        )
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
