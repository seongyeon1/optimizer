import pytest

from llm_router.router import LLMRouter
from llm_router.schemas import BudgetTier, CallHistory, CandidateModel, ObservedOutput, PromptRecord
from llm_router.training import UtilityTable


def test_fast_routes_easy_prompt_to_cheapest_model():
    router = LLMRouter(UtilityTable({"cheap": 0.65, "smart": 0.92}, global_quality=0.75))
    decision = router.route(
        PromptRecord("p1", "안녕?"),
        BudgetTier.FAST,
        CallHistory(),
        [CandidateModel("cheap", cost=1), CandidateModel("smart", cost=10)],
    )

    assert decision.action == "call"
    assert decision.model_id == "cheap"


def test_balanced_escalates_hard_prompt_when_gain_is_meaningful():
    router = LLMRouter(UtilityTable({"cheap": 0.45, "smart": 0.92}, global_quality=0.7))
    decision = router.route(
        PromptRecord("p2", "코드 버그를 분석하고 수학적으로 증명해줘", domain="coding"),
        BudgetTier.BALANCED,
        CallHistory(),
        [CandidateModel("cheap", cost=1), CandidateModel("smart", cost=5)],
    )

    assert decision.action == "call"
    assert decision.model_id == "smart"


def test_router_selects_existing_good_output_without_new_call():
    router = LLMRouter(UtilityTable({"cheap": 0.8, "smart": 0.9}, global_quality=0.8))
    history = CallHistory(observed_outputs=[ObservedOutput("p1", "cheap", "hello", quality=0.88)])

    decision = router.route(
        PromptRecord("p1", "hello"),
        BudgetTier.FAST,
        history,
        [CandidateModel("cheap", cost=1), CandidateModel("smart", cost=10)],
    )

    assert decision.action == "select"
    assert decision.model_id == "cheap"
    assert decision.output == "hello"


def test_router_rejects_final_output_not_in_history():
    router = LLMRouter()

    with pytest.raises(ValueError, match="observed outputs"):
        router.validate_final_answer("made up", CallHistory())
