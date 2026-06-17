from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from llm_router.schemas import CandidateModel, ObservedOutput, PromptRecord


def load_prompts(path: str | Path) -> list[PromptRecord]:
    return [
        PromptRecord(
            prompt_id=str(row["prompt_id"]),
            prompt=str(row["prompt"]),
            domain=_optional_str(row.get("domain")),
            task_type=_optional_str(row.get("task_type")),
        )
        for row in _read_records(path)
    ]


def load_candidate_models(path: str | Path) -> list[CandidateModel]:
    return [
        CandidateModel(
            model_id=str(row["model_id"]),
            cost=float(row["cost"]),
            latency=_optional_float(row.get("latency")),
            family=_optional_str(row.get("family")),
            quality_prior=_optional_float(row.get("quality_prior")),
        )
        for row in _read_records(path)
    ]


def load_observed_outputs(path: str | Path) -> list[ObservedOutput]:
    return [
        ObservedOutput(
            prompt_id=str(row["prompt_id"]),
            model_id=str(row["model_id"]),
            output=str(row["output"]),
            quality=_optional_float(row.get("quality")),
        )
        for row in _read_records(path)
    ]


def write_json(path: str | Path, payload: dict[str, object]) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON payload must be an object")
    return payload


def _read_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    return list(_read_jsonl(path))


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            yield payload


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _optional_str(value: object) -> str | None:
    if value is None or value == "":
        return None
    return str(value)
