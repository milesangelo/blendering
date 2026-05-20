"""Per-role token + dollar accounting for a single run."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import ModelConfig


@dataclass
class RoleUsage:
    in_tokens: int = 0
    out_tokens: int = 0
    calls: int = 0

    def add(self, in_tokens: int, out_tokens: int) -> None:
        self.in_tokens += in_tokens
        self.out_tokens += out_tokens
        self.calls += 1

    def cost(self, cfg: ModelConfig) -> float | None:
        if cfg.price_in_per_1m is None or cfg.price_out_per_1m is None:
            return None
        return (
            self.in_tokens / 1_000_000 * cfg.price_in_per_1m
            + self.out_tokens / 1_000_000 * cfg.price_out_per_1m
        )


@dataclass
class CostMeter:
    actor: RoleUsage = field(default_factory=RoleUsage)
    critic: RoleUsage = field(default_factory=RoleUsage)
    planner: RoleUsage = field(default_factory=RoleUsage)

    def step_line(self, actor_cfg: ModelConfig, critic_cfg: ModelConfig) -> str:
        ac = self.actor.cost(actor_cfg)
        cc = self.critic.cost(critic_cfg)
        cost_part = ""
        if ac is not None and cc is not None:
            cost_part = f"  step≈${ac + cc:.4f} (cum)"
        return (
            f"tokens — actor: {self.actor.in_tokens} in / {self.actor.out_tokens} out  "
            f"critic: {self.critic.in_tokens} in / {self.critic.out_tokens} out  "
            f"planner: {self.planner.in_tokens} in / {self.planner.out_tokens} out{cost_part}"
        )

    def summary(self, actor_cfg: ModelConfig, critic_cfg: ModelConfig) -> str:
        ac = self.actor.cost(actor_cfg)
        cc = self.critic.cost(critic_cfg)
        lines = [
            f"actor:  {self.actor.calls} calls  "
            f"{self.actor.in_tokens} in / {self.actor.out_tokens} out"
            + (f"  ${ac:.4f}" if ac is not None else ""),
            f"critic: {self.critic.calls} calls  "
            f"{self.critic.in_tokens} in / {self.critic.out_tokens} out"
            + (f"  ${cc:.4f}" if cc is not None else ""),
        ]
        if ac is not None and cc is not None:
            lines.append(f"total:  ${ac + cc:.4f}")
        return "\n".join(lines)
