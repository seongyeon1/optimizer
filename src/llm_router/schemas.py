from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class BudgetTier(Enum):
    FAST = "fast"
    BALANCED = "balanced"
    PREMIUM = "premium"

    @classmethod
    def parse(cls, value: str | "BudgetTier") -> "BudgetTier":
        if isinstance(value, BudgetTier):
            return value
        normalized = value.strip().lower()
        for tier in cls:
            if tier.value == normalized:
                return tier
        raise ValueError(f"Unknown budget tier: {value}")


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    prompt: str
    domain: str | None = None
    task_type: str | None = None

    def __post_init__(self) -> None:
        if not self.prompt_id:
            raise ValueError("prompt_id must be non-empty")
        if not self.prompt:
            raise ValueError("prompt must be non-empty")


@dataclass(frozen=True)
class CandidateModel:
    model_id: str
    cost: float
    latency: float | None = None
    family: str | None = None
    quality_prior: float | None = None

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if self.cost < 0:
            raise ValueError("cost must be non-negative")
        if self.latency is not None and self.latency < 0:
            raise ValueError("latency must be non-negative")
        if self.quality_prior is not None and not 0 <= self.quality_prior <= 1:
            raise ValueError("quality_prior must be between 0 and 1")


@dataclass(frozen=True)
class ObservedOutput:
    prompt_id: str
    model_id: str
    output: str
    quality: float | None = None

    def __post_init__(self) -> None:
        if not self.prompt_id:
            raise ValueError("prompt_id must be non-empty")
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if not self.output:
            raise ValueError("output must be non-empty")
        if self.quality is not None and not 0 <= self.quality <= 1:
            raise ValueError("quality must be between 0 and 1")


@dataclass(frozen=True)
class CallHistory:
    observed_outputs: list[ObservedOutput] = field(default_factory=list)

    def outputs_for_prompt(self, prompt_id: str) -> list[ObservedOutput]:
        return [output for output in self.observed_outputs if output.prompt_id == prompt_id]


@dataclass(frozen=True)
class RouteDecision:
    action: str
    model_id: str
    output: str | None = None
    reason: str = ""
    expected_quality: float | None = None
    expected_cost: float | None = None

    def __post_init__(self) -> None:
        if self.action not in {"call", "select"}:
            raise ValueError("action must be 'call' or 'select'")
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if self.action == "select" and not self.output:
            raise ValueError("select decisions require output")
