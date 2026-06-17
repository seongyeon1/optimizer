import pytest

from llm_router.schemas import BudgetTier, CandidateModel, ObservedOutput


def test_budget_tier_parses_case_insensitive_values():
    assert BudgetTier.parse("fast") is BudgetTier.FAST
    assert BudgetTier.parse("Balanced") is BudgetTier.BALANCED
    assert BudgetTier.parse("PREMIUM") is BudgetTier.PREMIUM


def test_budget_tier_rejects_unknown_values():
    with pytest.raises(ValueError, match="Unknown budget tier"):
        BudgetTier.parse("cheap")


def test_candidate_model_rejects_negative_cost():
    with pytest.raises(ValueError, match="cost"):
        CandidateModel(model_id="bad", cost=-1)


def test_observed_output_requires_non_empty_output():
    with pytest.raises(ValueError, match="output"):
        ObservedOutput(prompt_id="p1", model_id="m1", output="")
