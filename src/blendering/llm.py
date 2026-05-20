"""LiteLLM-backed Actor (streaming + tools) and Critic (vision + JSON verdict)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import litellm
from pydantic import ValidationError

from .config import ModelConfig
from .schemas import Verdict
from .utils.images import encode_b64_data_url, thumbnail_bytes


@dataclass
class ActorDelta:
    """One chunk from the Actor's streaming completion."""

    text: str = ""
    # Partial tool calls accumulate across deltas; complete ones land in `finished_tool_calls`.
    finished_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False


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
    response = await litellm.acompletion(
        **_model_kwargs(cfg),
        messages=messages,
        tools=tools or None,
        tool_choice="auto" if tools else None,
        stream=True,
    )

    # Tool calls arrive in fragments — accumulate by index.
    partials: dict[int, dict[str, Any]] = {}

    async for chunk in response:
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
    '{"status": "continue"|"done"|"stuck", "reasoning": str, '
    '"next_step_hint": str|null, "confidence": number in [0,1]}'
)


async def judge(
    cfg: ModelConfig,
    system: str,
    user_goal: str,
    transcript: str,
    screenshot_png: bytes | None,
) -> Verdict:
    """Ask the Critic to evaluate the scene. Returns a validated Verdict."""
    user_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"USER GOAL:\n{user_goal}\n\n"
                f"RECENT ACTOR TRANSCRIPT:\n{transcript}\n\n"
                f"{_VERDICT_INSTRUCTION}"
            ),
        }
    ]
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
            resp = await litellm.acompletion(
                **_model_kwargs(cfg),
                messages=messages,
                response_format={"type": "json_object"},
                stream=False,
            )
            text = resp.choices[0].message.content or "{}"
            data = json.loads(_strip_fences(text))
            return Verdict.model_validate(data)
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
            return Verdict(
                status="stuck",
                reasoning=f"Critic returned invalid JSON twice: {exc}",
                confidence=0.0,
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
