from llm_router.difficulty import score_difficulty
from llm_router.features import extract_prompt_features
from llm_router.schemas import PromptRecord


def test_easy_greeting_has_low_difficulty():
    prompt = PromptRecord(prompt_id="p1", prompt="안녕?")
    features = extract_prompt_features(prompt)

    assert features["char_count"] < 20
    assert score_difficulty(features) < 0.25


def test_code_math_multistep_prompt_has_high_difficulty():
    prompt = PromptRecord(
        prompt_id="p2",
        prompt="Python 코드의 버그를 찾고 시간복잡도를 증명한 뒤, 수식 x^2 + 3x = 0을 단계별로 풀어줘.",
        domain="coding",
        task_type="debugging",
    )
    features = extract_prompt_features(prompt)

    assert features["has_code_signal"] == 1.0
    assert features["has_math_signal"] == 1.0
    assert features["has_multistep_signal"] == 1.0
    assert score_difficulty(features) > 0.65
