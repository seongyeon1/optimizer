from __future__ import annotations

import re

from llm_router.schemas import PromptRecord


CODE_TERMS = (
    "code",
    "python",
    "javascript",
    "sql",
    "bug",
    "debug",
    "function",
    "class",
    "코드",
    "버그",
    "디버그",
    "함수",
    "시간복잡도",
)
MATH_TERMS = ("proof", "prove", "equation", "derive", "수식", "증명", "계산", "방정식", "확률")
MULTISTEP_TERMS = (
    "step by step",
    "first",
    "then",
    "after",
    "단계별",
    "먼저",
    "그 뒤",
    "그리고",
    "비교",
)
CONSTRAINT_TERMS = ("must", "only", "without", "조건", "제약", "반드시", "하지 말고")


def extract_prompt_features(prompt: PromptRecord) -> dict[str, float]:
    text = prompt.prompt
    lower = text.lower()
    char_count = len(text)
    token_count = max(1, len(re.findall(r"\S+", text)))
    korean_count = len(re.findall(r"[가-힣]", text))
    english_count = len(re.findall(r"[A-Za-z]", text))
    digit_count = len(re.findall(r"\d", text))

    features = {
        "char_count": float(char_count),
        "token_count": float(token_count),
        "korean_ratio": _safe_ratio(korean_count, char_count),
        "english_ratio": _safe_ratio(english_count, char_count),
        "digit_ratio": _safe_ratio(digit_count, char_count),
        "has_code_signal": _contains_any(lower, CODE_TERMS),
        "has_math_signal": max(
            _contains_any(lower, MATH_TERMS),
            1.0 if re.search(r"[=+\-*/^]|x\^?\d", lower) else 0.0,
        ),
        "has_multistep_signal": _contains_any(lower, MULTISTEP_TERMS),
        "has_constraint_signal": _contains_any(lower, CONSTRAINT_TERMS),
        "has_table_signal": 1.0 if "|" in text or "\t" in text else 0.0,
        "domain_coding": 1.0 if (prompt.domain or "").lower() in {"coding", "code", "programming"} else 0.0,
        "task_debugging": 1.0 if (prompt.task_type or "").lower() in {"debugging", "debug"} else 0.0,
    }
    return features


def _contains_any(text: str, terms: tuple[str, ...]) -> float:
    return 1.0 if any(term in text for term in terms) else 0.0


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
