"""Configuration loaded from YAML + environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class MCPConfig(BaseModel):
    command: str = "uvx"
    args: list[str] = Field(default_factory=lambda: ["blender-mcp"])
    env: dict[str, str] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    model: str
    api_base: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.2
    max_tokens: int = 4096
    # Optional USD pricing per 1M tokens — when set, cost is reported per run.
    price_in_per_1m: float | None = None
    price_out_per_1m: float | None = None

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)


class LoopConfig(BaseModel):
    max_iterations: int = 25
    screenshot_every_step: bool = True
    stop_on_stuck_streak: int = 3


class Settings(BaseModel):
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    actor: ModelConfig
    critic: ModelConfig
    loop: LoopConfig = Field(default_factory=LoopConfig)

    @model_validator(mode="after")
    def _check_api_keys(self) -> Settings:
        # Soft check — surface a clear error if the user forgot to export their key.
        for role, mc in (("actor", self.actor), ("critic", self.critic)):
            if mc.api_key is None:
                # We don't fail here; the LLM call will fail loudly. But warn via env marker.
                os.environ.setdefault(f"_BLENDERING_MISSING_{role.upper()}_KEY", mc.api_key_env)
        return self


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_settings(path: str | Path | None = None) -> Settings:
    """Load settings from a YAML file. Falls back to ./config.yaml or $BLENDERING_CONFIG."""
    if path is None:
        path = os.environ.get("BLENDERING_CONFIG", "config.yaml")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. Copy config.example.yaml to config.yaml."
        )
    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}
    return Settings.model_validate(_expand_env(raw))
