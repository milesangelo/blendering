from __future__ import annotations

from pathlib import Path

import pytest

from blendering.config import load_settings


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(body)
    return p


def test_load_minimal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SILICONFLOW_API_KEY", "sk-test")
    p = _write(
        tmp_path,
        """
actor:
  model: openai/Qwen/Qwen2.5-Coder-32B-Instruct
  api_base: https://api.siliconflow.cn/v1
  api_key_env: SILICONFLOW_API_KEY
critic:
  model: openai/Qwen/Qwen2-VL-72B-Instruct
  api_base: https://api.siliconflow.cn/v1
  api_key_env: SILICONFLOW_API_KEY
""",
    )
    s = load_settings(p)
    assert s.actor.api_key == "sk-test"
    assert s.critic.api_base == "https://api.siliconflow.cn/v1"
    assert s.loop.max_iterations == 25
    assert s.mcp.command == "uvx"


def test_env_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_KEY", "ENV_VALUE")
    p = _write(
        tmp_path,
        """
actor:
  model: openai/x
  api_base: https://${MY_KEY}.example
  api_key_env: SILICONFLOW_API_KEY
critic:
  model: openai/y
  api_base: https://${MY_KEY}.example
  api_key_env: SILICONFLOW_API_KEY
""",
    )
    s = load_settings(p)
    assert s.actor.api_base == "https://ENV_VALUE.example"


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "does-not-exist.yaml")
