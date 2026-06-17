from llm_router.schemas import CandidateModel, ObservedOutput
from llm_router.training import UtilityTable, train_utility_table


def test_training_uses_average_quality_by_model():
    models = [
        CandidateModel(model_id="cheap", cost=1),
        CandidateModel(model_id="smart", cost=5),
    ]
    outputs = [
        ObservedOutput("p1", "cheap", "a", quality=0.5),
        ObservedOutput("p2", "cheap", "b", quality=0.7),
        ObservedOutput("p1", "smart", "c", quality=0.9),
    ]

    table = train_utility_table(models, outputs)

    assert table.expected_quality("cheap", 0.2) == 0.6
    assert table.expected_quality("smart", 0.8) == 0.9


def test_unseen_model_uses_quality_prior_then_difficulty():
    table = UtilityTable(model_quality={}, global_quality=0.55)

    assert table.expected_quality("new", 0.8, quality_prior=0.75) == 0.75
    assert table.expected_quality("new", 0.8) > table.expected_quality("new", 0.1)
