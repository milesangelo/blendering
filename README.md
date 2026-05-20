# blendering

A two-model agent for driving Blender via [blender-mcp](https://github.com/ahujasid/blender-mcp).

- **Actor model** — tool-calling LLM that mutates the Blender scene (e.g. `execute_blender_code`, asset imports).
- **Critic model** — vision LLM that looks at the viewport screenshot each step, returns a structured verdict, and tells the Actor what to do next — or that the goal is met.

The loop runs until the Critic returns `done`, hits a stuck streak, hits max iterations, or you press `i` to interrupt.

```
┌── Thinking ─────────────────┬── Viewport ─────────┐
│ → tool execute_blender_code │  [live screenshot]  │
│ ← execute_blender_code: ok  │                     │
│ critic (0.92): looks good   │                     │
└─────────────────────────────┴─────────────────────┘
 ● running   iter 3/25   verdict continue   i=interrupt q=quit
```

## Why two models?

Vision-language models are usually the strongest judges of "does this scene look right" but are often *not* the strongest at long-horizon tool use. Letting a code/tool specialist drive while a separate vision model judges progress means you can pick the best model for each role independently. Model selection is per-role in `config.yaml`.

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

## Install

```bash
git clone https://github.com/milesangelo/blendering
cd blendering
uv sync
```

You also need the Blender MCP server. The default config uses [ahujasid/blender-mcp](https://github.com/ahujasid/blender-mcp) via `uvx blender-mcp`. Install the corresponding Blender add-on, open Blender, and click **Connect to MCP server**.

## Configure

```bash
cp .env.example .env
cp config.example.yaml config.yaml
```

Edit `.env` to set `SILICONFLOW_API_KEY` (or whichever provider key you use), and `config.yaml` to choose your Actor and Critic models. The defaults target SiliconFlow's Qwen lineup:

```yaml
actor:
  model: openai/Qwen/Qwen2.5-Coder-32B-Instruct
  api_base: https://api.siliconflow.cn/v1
  api_key_env: SILICONFLOW_API_KEY
critic:
  model: openai/Qwen/Qwen2-VL-72B-Instruct
  api_base: https://api.siliconflow.cn/v1
  api_key_env: SILICONFLOW_API_KEY
```

Because the LLM layer is [LiteLLM](https://github.com/BerriAI/litellm), you can point either role at any supported provider — OpenAI, Anthropic, local Ollama, etc. — by changing the `model` string and `api_base`.

## Run

```bash
uv run blendering "build a low-poly desk lamp on a wooden table, lit from above-left"
```

You'll get a terminal UI with:

- **Thinking pane (left)** — streaming Actor text, every MCP tool call + result preview, and each Critic verdict with `next_step_hint`.
- **Viewport pane (right)** — the latest screenshot, rendered inline if your terminal supports images (Kitty, WezTerm, iTerm2, Ghostty), otherwise shown as a saved file path.
- **Status bar** — current iteration, last verdict, and the `i` / `q` hotkeys.

Press **`i`** (or **Ctrl+C**) at any time to interrupt. The current step is cancelled cooperatively and the orchestrator winds down within ~1s, leaving Blender in whatever state it was in.

## Config reference

| Section | Key | Default | Notes |
|---|---|---|---|
| `mcp` | `command` | `uvx` | Command to launch the MCP server. |
| `mcp` | `args` | `["blender-mcp"]` | Args to that command. |
| `actor` / `critic` | `model` | — | LiteLLM model string (e.g. `openai/...`, `anthropic/...`, `ollama/...`). |
| `actor` / `critic` | `api_base` | — | Optional override; needed for SiliconFlow. |
| `actor` / `critic` | `api_key_env` | `OPENAI_API_KEY` | Env var to read the API key from. |
| `loop` | `max_iterations` | 25 | Hard cap on Actor turns. |
| `loop` | `stop_on_stuck_streak` | 3 | Abort after this many consecutive `stuck` verdicts. |
| `loop` | `screenshot_every_step` | true | Disable to skip Critic vision calls (text-only loop). |

## Develop

```bash
uv sync
uv run pytest                          # unit tests with fake LLM + MCP
uv run ruff check                       # lint
uv run mypy src                         # type-check
```

## License

MIT
