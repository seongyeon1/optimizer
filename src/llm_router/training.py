from __future__ import annotations

from dataclasses import dataclass

from llm_router.schemas import CandidateModel, ObservedOutput


@dataclass(frozen=True)
class UtilityTable:
    model_quality: dict[str, float]
    global_quality: float = 0.5

    def expected_quality(
        self,
        model_id: str,
        difficulty: float,
        quality_prior: float | None = None,
    ) -> float:
        if model_id in self.model_quality:
            return self.model_quality[model_id]
        if quality_prior is not None:
            return quality_prior
        return _clamp(self.global_quality + 0.15 * (difficulty - 0.5))

    def to_dict(self) -> dict[str, object]:
        return {
            "model_quality": self.model_quality,
            "global_quality": self.global_quality,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "UtilityTable":
        model_quality = payload.get("model_quality", {})
        if not isinstance(model_quality, dict):
            raise ValueError("model_quality must be an object")
        return cls(
            model_quality={str(key): float(value) for key, value in model_quality.items()},
            global_quality=float(payload.get("global_quality", 0.5)),
        )


def train_utility_table(
    models: list[CandidateModel],
    outputs: list[ObservedOutput],
) -> UtilityTable:
    known_model_ids = {model.model_id for model in models}
    values: dict[str, list[float]] = {model_id: [] for model_id in known_model_ids}

    for output in outputs:
        if output.quality is None:
            continue
        values.setdefault(output.model_id, []).append(output.quality)

    model_quality = {
        model_id: sum(scores) / len(scores)
        for model_id, scores in values.items()
        if scores
    }
    all_scores = [score for scores in values.values() for score in scores]
    global_quality = sum(all_scores) / len(all_scores) if all_scores else 0.5
    return UtilityTable(model_quality=model_quality, global_quality=global_quality)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
