from __future__ import annotations

from llm_router.difficulty import score_difficulty
from llm_router.features import extract_prompt_features
from llm_router.policy import TIER_POLICIES, choose_model
from llm_router.schemas import BudgetTier, CallHistory, CandidateModel, PromptRecord, RouteDecision
from llm_router.training import UtilityTable


class LLMRouter:
    def __init__(self, utility_table: UtilityTable | None = None) -> None:
        self.utility_table = utility_table or UtilityTable(model_quality={}, global_quality=0.5)

    def route(
        self,
        prompt: PromptRecord,
        tier: BudgetTier | str,
        history: CallHistory,
        candidates: list[CandidateModel],
    ) -> RouteDecision:
        parsed_tier = BudgetTier.parse(tier)
        features = extract_prompt_features(prompt)
        difficulty = score_difficulty(features)
        existing = self._best_existing_output(prompt, parsed_tier, history, difficulty)
        if existing is not None:
            return existing

        model, expected_quality = choose_model(parsed_tier, candidates, self.utility_table, difficulty)
        return RouteDecision(
            action="call",
            model_id=model.model_id,
            reason=f"{parsed_tier.value} utility policy selected model at difficulty {difficulty:.3f}",
            expected_quality=expected_quality,
            expected_cost=model.cost,
        )

    def validate_final_answer(self, output: str, history: CallHistory) -> None:
        if output not in {observed.output for observed in history.observed_outputs}:
            raise ValueError("final answer must come from observed outputs")

    def _best_existing_output(
        self,
        prompt: PromptRecord,
        tier: BudgetTier,
        history: CallHistory,
        difficulty: float,
    ) -> RouteDecision | None:
        threshold = TIER_POLICIES[tier].select_threshold
        best = None
        for output in history.outputs_for_prompt(prompt.prompt_id):
            quality = output.quality
            if quality is None:
                quality = self.utility_table.expected_quality(output.model_id, difficulty)
            if quality >= threshold and (best is None or quality > best.expected_quality):
                best = RouteDecision(
                    action="select",
                    model_id=output.model_id,
                    output=output.output,
                    reason=f"observed output meets {tier.value} threshold",
                    expected_quality=quality,
                    expected_cost=0.0,
                )
        return best
