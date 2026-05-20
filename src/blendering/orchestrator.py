"""Actor + Critic loop driving Blender via MCP, with cooperative cancellation."""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any

from .config import Settings
from .cost import CostMeter
from .framing import reframe_script
from .llm import judge, stream_actor
from .mcp_client import BlenderMCP, mcp_client
from .prompts import ACTOR_SYSTEM, CRITIC_SYSTEM
from .schemas import RunOutcome, Verdict
from .tui.events import (
    ActorTextDelta,
    CriticVerdictEvent,
    Event,
    IterationStart,
    RunFinished,
    StatusMessage,
    ToolCallResult,
    ToolCallStart,
    ViewportUpdate,
)
from .utils.images import save_screenshot
from .utils.logging import get_logger

log = get_logger("blendering.orchestrator")


class EventBus:
    """Bounded queue for orchestrator → UI events."""

    def __init__(self, maxsize: int = 256) -> None:
        self._q: asyncio.Queue[Event] = asyncio.Queue(maxsize=maxsize)

    async def post(self, event: Event) -> None:
        await self._q.put(event)

    def post_nowait(self, event: Event) -> None:
        with contextlib.suppress(asyncio.QueueFull):
            self._q.put_nowait(event)

    async def get(self) -> Event:
        return await self._q.get()


async def _wait_or_cancel(coro_task: asyncio.Task[Any], cancel: asyncio.Event) -> Any:
    """Await `coro_task` but bail (cancelling it) the moment `cancel` is set."""
    cancel_task = asyncio.create_task(cancel.wait())
    try:
        done, _pending = await asyncio.wait(
            {coro_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if cancel_task in done and coro_task not in done:
            coro_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await coro_task
            raise asyncio.CancelledError()
        return coro_task.result()
    finally:
        if not cancel_task.done():
            cancel_task.cancel()


def _truncate(text: str, n: int = 400) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


async def _run_actor_turn(
    settings: Settings,
    mcp: BlenderMCP,
    messages: list[dict[str, Any]],
    tools_openai: list[dict[str, Any]],
    bus: EventBus,
    cancel: asyncio.Event,
    meter: CostMeter,
) -> tuple[list[dict[str, Any]], str]:
    """Run one Actor turn (text + tool calls). Returns (new_messages, transcript_for_critic)."""
    accumulated_text = ""
    appended: list[dict[str, Any]] = []
    transcript_lines: list[str] = []

    # The Actor may emit text then tool calls. We loop: stream → execute tool calls →
    # feed results back → stream again, until the model finishes without tool calls.
    while True:
        if cancel.is_set():
            raise asyncio.CancelledError()

        async def _drain() -> tuple[str, list[dict[str, Any]]]:
            text = ""
            finished: list[dict[str, Any]] = []
            log.debug("actor: requesting completion (%d messages)", len(messages))
            async for delta in stream_actor(settings.actor, messages, tools_openai):
                if delta.text:
                    text += delta.text
                    await bus.post(ActorTextDelta(delta.text))
                if delta.finished_tool_calls:
                    finished.extend(delta.finished_tool_calls)
                if delta.in_tokens is not None and delta.out_tokens is not None:
                    meter.actor.add(delta.in_tokens, delta.out_tokens)
            log.debug("actor: stream complete — %d chars text, %d tool calls",
                      len(text), len(finished))
            return text, finished

        stream_task = asyncio.create_task(_drain())
        text, finished_tool_calls = await _wait_or_cancel(stream_task, cancel)

        accumulated_text += text
        if text:
            transcript_lines.append(f"[actor] {_truncate(text)}")

        if not finished_tool_calls:
            # Plain text completion — record and return.
            messages.append({"role": "assistant", "content": text or ""})
            appended.append(messages[-1])
            return appended, "\n".join(transcript_lines)

        # Append the assistant tool-calls message
        tool_calls_payload = [
            {
                "id": tc["id"] or f"call_{i}",
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["arguments"]),
                },
            }
            for i, tc in enumerate(finished_tool_calls)
        ]
        messages.append(
            {"role": "assistant", "content": text or None, "tool_calls": tool_calls_payload}
        )
        appended.append(messages[-1])

        # Execute each tool call against MCP
        for tc, payload in zip(finished_tool_calls, tool_calls_payload, strict=True):
            log.info("tool call: %s args=%s", tc["name"], tc["arguments"])
            await bus.post(ToolCallStart(name=tc["name"], arguments=tc["arguments"]))
            call_task = asyncio.create_task(mcp.call_tool(tc["name"], tc["arguments"]))
            try:
                result = await _wait_or_cancel(call_task, cancel)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                preview = f"ERROR: {exc!r}"
                is_error = True
                result_text = preview
            else:
                result_text = result.text or "(no text)"
                preview = _truncate(result_text)
                is_error = result.is_error

            log.info("tool result: %s%s — %s",
                     tc["name"], " [ERR]" if is_error else "", _truncate(result_text, 200))
            await bus.post(
                ToolCallResult(name=tc["name"], result_preview=preview, is_error=is_error)
            )
            transcript_lines.append(
                f"[tool {tc['name']}] {'ERR ' if is_error else ''}{_truncate(result_text, 200)}"
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": payload["id"],
                    "name": tc["name"],
                    "content": result_text,
                }
            )
            appended.append(messages[-1])


async def run(
    settings: Settings,
    user_prompt: str,
    bus: EventBus,
    cancel: asyncio.Event,
    screenshot_dir: Path | None = None,
) -> RunOutcome:
    """Drive the full Actor+Critic loop."""
    screenshot_dir = screenshot_dir or Path("screenshots")

    try:
        log.info("starting run: %r", user_prompt)
        async with mcp_client(settings.mcp) as mcp:
            tools_openai = [t.to_openai() for t in mcp.tools]
            log.info("MCP connected — %d tools: %s",
                     len(tools_openai), [t.name for t in mcp.tools])
            await bus.post(
                StatusMessage(text=f"MCP connected. {len(tools_openai)} tools available.")
            )

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": ACTOR_SYSTEM},
                {"role": "user", "content": user_prompt},
            ]
            stuck_streak = 0
            last_verdict: Verdict | None = None
            meter = CostMeter()

            for i in range(1, settings.loop.max_iterations + 1):
                if cancel.is_set():
                    outcome = RunOutcome(
                        status="cancelled", iterations=i - 1, last_verdict=last_verdict
                    )
                    await bus.post(RunFinished(outcome))
                    return outcome

                log.info("── iteration %d/%d ──", i, settings.loop.max_iterations)
                await bus.post(IterationStart(n=i, total=settings.loop.max_iterations))

                _, transcript = await _run_actor_turn(
                    settings, mcp, messages, tools_openai, bus, cancel, meter
                )

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

                # Critic
                judge_task = asyncio.create_task(
                    judge(
                        settings.critic,
                        CRITIC_SYSTEM,
                        user_prompt,
                        transcript,
                        screenshot_bytes,
                    )
                )
                verdict, c_in, c_out = await _wait_or_cancel(judge_task, cancel)
                meter.critic.add(c_in, c_out)
                last_verdict = verdict
                log.info("critic verdict: status=%s confidence=%.2f hint=%r",
                         verdict.status, verdict.confidence, verdict.next_step_hint)
                await bus.post(CriticVerdictEvent(verdict))
                step_line = meter.step_line(settings.actor, settings.critic)
                log.info(step_line)
                await bus.post(StatusMessage(text=step_line))

                if verdict.status == "done":
                    summary = meter.summary(settings.actor, settings.critic)
                    log.info("run summary:\n%s", summary)
                    await bus.post(StatusMessage(text=summary))
                    outcome = RunOutcome(
                        status="done", iterations=i, last_verdict=verdict
                    )
                    await bus.post(RunFinished(outcome))
                    return outcome
                if verdict.status == "stuck":
                    stuck_streak += 1
                    if stuck_streak >= settings.loop.stop_on_stuck_streak:
                        summary = meter.summary(settings.actor, settings.critic)
                        log.info("run summary:\n%s", summary)
                        await bus.post(StatusMessage(text=summary))
                        outcome = RunOutcome(
                            status="stuck", iterations=i, last_verdict=verdict
                        )
                        await bus.post(RunFinished(outcome))
                        return outcome
                else:
                    stuck_streak = 0

                # Inject critic feedback for next iteration.
                feedback = (
                    f"Critic feedback (status={verdict.status}, "
                    f"confidence={verdict.confidence:.2f}):\n"
                    f"{verdict.reasoning}\n"
                    f"Next step hint: {verdict.next_step_hint or '(none)'}"
                )
                messages.append({"role": "user", "content": feedback})

            summary = meter.summary(settings.actor, settings.critic)
            log.info("run summary:\n%s", summary)
            await bus.post(StatusMessage(text=summary))
            outcome = RunOutcome(
                status="max_iterations",
                iterations=settings.loop.max_iterations,
                last_verdict=last_verdict,
            )
            await bus.post(RunFinished(outcome))
            return outcome

    except asyncio.CancelledError:
        outcome = RunOutcome(status="cancelled", iterations=0)
        bus.post_nowait(RunFinished(outcome))
        return outcome
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        outcome = RunOutcome(status="error", iterations=0, error=f"{exc!r}\n{tb}")
        bus.post_nowait(RunFinished(outcome))
        return outcome
