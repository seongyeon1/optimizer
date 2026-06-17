from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_router.io import (
    load_candidate_models,
    load_observed_outputs,
    load_prompts,
    read_json,
    write_json,
)
from llm_router.router import LLMRouter
from llm_router.schemas import BudgetTier, CallHistory, PromptRecord, RouteDecision
from llm_router.training import UtilityTable, train_utility_table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm-router")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--models", required=True)
    train_parser.add_argument("--outputs", required=True)
    train_parser.add_argument("--output", required=True)

    route_parser = subparsers.add_parser("route")
    route_parser.add_argument("--prompt", required=True)
    route_parser.add_argument("--tier", required=True)
    route_parser.add_argument("--models", required=True)
    route_parser.add_argument("--prompt-id", default="cli")
    route_parser.add_argument("--domain")
    route_parser.add_argument("--task-type")
    route_parser.add_argument("--utility-table")
    route_parser.add_argument("--history")

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--prompts", required=True)
    evaluate_parser.add_argument("--models", required=True)
    evaluate_parser.add_argument("--outputs", required=True)
    evaluate_parser.add_argument("--tier", required=True)
    evaluate_parser.add_argument("--utility-table")

    args = parser.parse_args(argv)
    if args.command == "train":
        return _train(args)
    if args.command == "route":
        return _route(args)
    if args.command == "evaluate":
        return _evaluate(args)
    raise ValueError(f"unsupported command: {args.command}")


def _train(args: argparse.Namespace) -> int:
    models = load_candidate_models(args.models)
    outputs = load_observed_outputs(args.outputs)
    table = train_utility_table(models, outputs)
    write_json(args.output, table.to_dict())
    return 0


def _route(args: argparse.Namespace) -> int:
    table = _load_table(args.utility_table)
    router = LLMRouter(table)
    history = _load_history(args.history)
    decision = router.route(
        PromptRecord(
            prompt_id=args.prompt_id,
            prompt=args.prompt,
            domain=args.domain,
            task_type=args.task_type,
        ),
        BudgetTier.parse(args.tier),
        history,
        load_candidate_models(args.models),
    )
    print(json.dumps(_decision_to_dict(decision), ensure_ascii=False))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    table = _load_table(args.utility_table)
    router = LLMRouter(table)
    models = load_candidate_models(args.models)
    outputs = load_observed_outputs(args.outputs)
    output_by_prompt_model = {(item.prompt_id, item.model_id): item for item in outputs}
    quality_total = 0.0
    cost_total = 0.0
    count = 0

    for prompt in load_prompts(args.prompts):
        decision = router.route(prompt, BudgetTier.parse(args.tier), CallHistory(), models)
        selected = output_by_prompt_model.get((prompt.prompt_id, decision.model_id))
        if selected is None:
            continue
        quality_total += selected.quality or 0.0
        model = next(model for model in models if model.model_id == decision.model_id)
        cost_total += model.cost
        count += 1

    payload = {
        "count": count,
        "average_quality": quality_total / count if count else 0.0,
        "total_cost": cost_total,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def _load_table(path: str | None) -> UtilityTable:
    if path is None:
        return UtilityTable(model_quality={}, global_quality=0.5)
    return UtilityTable.from_dict(read_json(path))


def _load_history(path: str | None) -> CallHistory:
    if path is None:
        return CallHistory()
    return CallHistory(observed_outputs=load_observed_outputs(Path(path)))


def _decision_to_dict(decision: RouteDecision) -> dict[str, object]:
    return {
        "action": decision.action,
        "model_id": decision.model_id,
        "output": decision.output,
        "reason": decision.reason,
        "expected_quality": decision.expected_quality,
        "expected_cost": decision.expected_cost,
    }


if __name__ == "__main__":
    raise SystemExit(main())
