from __future__ import annotations


def score_difficulty(features: dict[str, float]) -> float:
    length_score = min(features.get("char_count", 0.0) / 500.0, 0.25)
    token_score = min(features.get("token_count", 0.0) / 120.0, 0.15)
    signal_score = (
        0.18 * features.get("has_code_signal", 0.0)
        + 0.18 * features.get("has_math_signal", 0.0)
        + 0.16 * features.get("has_multistep_signal", 0.0)
        + 0.10 * features.get("has_constraint_signal", 0.0)
        + 0.08 * features.get("has_table_signal", 0.0)
        + 0.12 * features.get("domain_coding", 0.0)
        + 0.10 * features.get("task_debugging", 0.0)
    )
    numeric_score = min(features.get("digit_ratio", 0.0) * 0.2, 0.08)
    return max(0.0, min(1.0, length_score + token_score + signal_score + numeric_score))
