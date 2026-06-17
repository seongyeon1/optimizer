# Efficient LLM Router Design

## Goal

Build a local, open-source router for the SK Telecom Efficient LLM Routing Challenge. The router chooses cost-effective candidate model outputs based on prompt difficulty, budget tier, prior call history, and candidate model metadata.

The key objective is to maximize average quality under budget constraints, with stronger emphasis on low-budget tiers:

- `Fast`: spend as little as possible, escalate only for clear high-difficulty prompts.
- `Balanced`: optimize expected quality per cost while preserving budget.
- `Premium`: prefer quality, but still avoid obviously wasteful expensive calls.

The implementation must not call external APIs or network services. It must run from local files only.

## Scope

This project will create a fresh Python package with:

- A router library.
- A CLI for training, validation, and prediction.
- A small synthetic fixture dataset for tests and examples.
- Unit tests covering feature extraction, budget behavior, routing decisions, and output constraints.
- Documentation explaining expected input files and challenge adaptation points.

The router will be designed around challenge constraints where private evaluation may reveal candidate outputs sequentially through a simulator. The interface will therefore support both:

- Offline mode: all candidate outputs and labels are available for training/evaluation.
- Online routing mode: the router chooses which model to call next or selects one of the already observed outputs.

## Recommended Approach

Use a hybrid router:

1. Rule-based prompt difficulty features.
2. Offline-learned model utility estimates from public data.
3. Tier-specific policy thresholds for escalation.

This is more competitive than a pure heuristic and less brittle than an overfitted simulator optimizer.

## Architecture

The package will live under `src/llm_router/`.

- `schemas.py`: typed dataclasses for prompts, candidate models, observed outputs, call history, route decisions, and budget tiers.
- `features.py`: deterministic prompt and metadata feature extraction.
- `difficulty.py`: converts features into an interpretable difficulty score.
- `training.py`: learns per-model utility statistics from public data.
- `policy.py`: tier-specific routing and escalation logic.
- `router.py`: public router API used by CLI and external evaluators.
- `io.py`: JSONL/CSV loading helpers.
- `cli.py`: command line entrypoint.

## Data Model

The router will accept records with these logical fields:

- Prompt:
  - `prompt_id`
  - `prompt`
  - optional `domain`
  - optional `task_type`
- Candidate model metadata:
  - `model_id`
  - `cost`
  - optional `latency`
  - optional `family`
  - optional `quality_prior`
- Candidate output:
  - `prompt_id`
  - `model_id`
  - `output`
  - optional public-training `quality`

The final answer selector must only return an output from the observed or called candidate outputs.

## Feature Design

Prompt features will be deterministic and local:

- Character and token length.
- Korean, English, code, math, and table-like content ratios.
- Question complexity markers such as multi-step wording, constraints, citations, proofs, code generation, debugging, and numerical reasoning.
- Domain and task one-hot indicators when provided.
- Candidate model cost and optional quality prior.

Difficulty scoring will remain interpretable so decisions can be audited.

## Routing Policy

The router computes expected utility for each candidate model:

```text
expected_utility = expected_quality - tier_cost_penalty * normalized_cost
```

Tier behavior:

- `Fast`: high cost penalty, strong default to the cheapest acceptable model.
- `Balanced`: moderate cost penalty, escalate for medium/high difficulty or strong expected utility gain.
- `Premium`: low cost penalty, quality-oriented but still cost-aware.

If an output has already been observed in call history, the router may select it directly when its estimated quality is sufficient for the tier. Otherwise it returns the next candidate model to call.

## Error Handling

The implementation will:

- Reject unknown budget tiers with clear errors.
- Reject decisions that reference a nonexistent model.
- Reject final answers that are not present in observed outputs.
- Fall back to cheapest candidate when metadata is sparse.
- Keep all routing deterministic unless an explicit seed is supplied.

## Testing Strategy

Use TDD for implementation. Initial tests will cover:

- Feature extraction identifies cheap/easy prompts and hard/code/math prompts.
- Difficulty score increases with multi-step and technical prompts.
- `Fast` selects a cheap model for simple prompts.
- `Balanced` escalates when difficulty is high and expected gain is meaningful.
- `Premium` prefers higher expected quality for difficult prompts.
- Final answer selection never invents outputs.
- CLI can train and predict using local fixture files.

## Deliverables

The first implementation pass will produce:

- Installable Python package metadata.
- Router source code.
- Tests and fixtures.
- README with usage examples.
- CLI commands:
  - `llm-router train`
  - `llm-router route`
  - `llm-router evaluate`

## Non-Goals

- No external LLM calls.
- No network dependency.
- No heavyweight neural model training.
- No assumptions about private simulator internals beyond the documented route/select loop.

## Open Adaptation Points

When the official public dataset format is available, only `io.py` and fixture mappings should need changes. The router API and policy behavior should remain stable.
