from __future__ import annotations

from dataclasses import dataclass

from llm_router.schemas import BudgetTier, CandidateModel
from llm_router.training import UtilityTable


@dataclass(frozen=True)
class TierPolicy:
    cost_penalty: float
    select_threshold: float
    escalation_threshold: float


TIER_POLICIES = {
    BudgetTier.FAST: TierPolicy(cost_penalty=0.55, select_threshold=0.72, escalation_threshold=0.72),
    BudgetTier.BALANCED: TierPolicy(cost_penalty=0.22, select_threshold=0.78, escalation_threshold=0.5),
    BudgetTier.PREMIUM: TierPolicy(cost_penalty=0.08, select_threshold=0.84, escalation_threshold=0.0),
}


def choose_model(
    tier: BudgetTier,
    models: list[CandidateModel],
    utility_table: UtilityTable,
    difficulty: float,
) -> tuple[CandidateModel, float]:
    if not models:
        raise ValueError("at least one candidate model is required")

    policy = TIER_POLICIES[tier]
    cheapest = min(models, key=lambda model: (model.cost, model.model_id))
    max_cost = max(model.cost for model in models) or 1.0
    scored = []

    for model in models:
        expected_quality = utility_table.expected_quality(
            model.model_id,
            difficulty,
            quality_prior=model.quality_prior,
        )
        normalized_cost = model.cost / max_cost
        utility = expected_quality - policy.cost_penalty * normalized_cost
        scored.append((utility, expected_quality, -model.cost, model.model_id, model))

    scored.sort(reverse=True)
    best_utility, best_quality, _, _, best_model = scored[0]

    if tier is BudgetTier.FAST and difficulty < policy.escalation_threshold:
        return cheapest, utility_table.expected_quality(
            cheapest.model_id,
            difficulty,
            quality_prior=cheapest.quality_prior,
        )

    cheapest_quality = utility_table.expected_quality(
        cheapest.model_id,
        difficulty,
        quality_prior=cheapest.quality_prior,
    )
    if best_model.model_id != cheapest.model_id and best_quality - cheapest_quality < 0.12:
        return cheapest, cheapest_quality

    return best_model, best_quality if best_utility > float("-inf") else cheapest_quality
