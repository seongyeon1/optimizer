# Efficient LLM Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Python package and CLI for compute-optimal LLM routing across Fast, Balanced, and Premium budget tiers.

**Architecture:** The router uses deterministic prompt features, an interpretable difficulty score, learned per-model utility statistics, and tier-specific policy thresholds. It supports offline training/evaluation and online route-or-select decisions without external API calls.

**Tech Stack:** Python 3.10+, standard library dataclasses/argparse/json/csv, pytest for tests.

---

## File Structure

- `pyproject.toml`: package metadata and pytest configuration.
- `README.md`: challenge-oriented usage instructions and input formats.
- `src/llm_router/__init__.py`: public package exports.
- `src/llm_router/schemas.py`: dataclasses and enums for prompts, models, outputs, history, and decisions.
- `src/llm_router/features.py`: deterministic feature extraction.
- `src/llm_router/difficulty.py`: interpretable difficulty scoring.
- `src/llm_router/training.py`: local utility table training from public labels.
- `src/llm_router/policy.py`: tier-specific route/select policy.
- `src/llm_router/router.py`: high-level API.
- `src/llm_router/io.py`: JSONL and CSV readers/writers.
- `src/llm_router/cli.py`: `train`, `route`, and `evaluate` commands.
- `tests/fixtures/*.jsonl`: small local examples.
- `tests/test_*.py`: TDD coverage.

### Task 1: Package Skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `src/llm_router/__init__.py`
- Test: `tests/test_package.py`

- [ ] **Step 1: Write the failing test**

```python
def test_package_exposes_version():
    import llm_router

    assert isinstance(llm_router.__version__, str)
    assert llm_router.__version__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_package.py -v`
Expected: FAIL because `llm_router` is not importable.

- [ ] **Step 3: Write minimal implementation**

Create `pyproject.toml` with setuptools package discovery and pytest path configuration. Create `src/llm_router/__init__.py` with `__version__ = "0.1.0"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_package.py -v`
Expected: PASS.

### Task 2: Schemas and Validation

**Files:**
- Create: `src/llm_router/schemas.py`
- Test: `tests/test_schemas.py`

- [ ] **Step 1: Write failing tests**

```python
import pytest

from llm_router.schemas import BudgetTier, CandidateModel, ObservedOutput, PromptRecord


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_schemas.py -v`
Expected: FAIL because schemas do not exist.

- [ ] **Step 3: Implement schemas**

Define `BudgetTier`, `PromptRecord`, `CandidateModel`, `ObservedOutput`, `CallHistory`, and `RouteDecision`. Validate model cost, output text, and decision actions.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_schemas.py -v`
Expected: PASS.

### Task 3: Features and Difficulty

**Files:**
- Create: `src/llm_router/features.py`
- Create: `src/llm_router/difficulty.py`
- Test: `tests/test_features_difficulty.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_features_difficulty.py -v`
Expected: FAIL because feature modules do not exist.

- [ ] **Step 3: Implement features and difficulty**

Use regex/string heuristics for length, language, code, math, table, constraints, multistep, domain, and task signals. Clamp difficulty to `[0.0, 1.0]`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_features_difficulty.py -v`
Expected: PASS.

### Task 4: Training Utility Table

**Files:**
- Create: `src/llm_router/training.py`
- Test: `tests/test_training.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_training.py -v`
Expected: FAIL because training module does not exist.

- [ ] **Step 3: Implement utility table**

Average public `quality` labels by model. Store global mean. For unseen models, use `quality_prior`; otherwise combine global quality with difficulty.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_training.py -v`
Expected: PASS.

### Task 5: Routing Policy

**Files:**
- Create: `src/llm_router/policy.py`
- Create: `src/llm_router/router.py`
- Test: `tests/test_policy_router.py`

- [ ] **Step 1: Write failing tests**

```python
import pytest

from llm_router.router import LLMRouter
from llm_router.schemas import BudgetTier, CandidateModel, CallHistory, ObservedOutput, PromptRecord
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_policy_router.py -v`
Expected: FAIL because policy/router modules do not exist.

- [ ] **Step 3: Implement policy and router**

Compute utility as `expected_quality - tier_cost_penalty * normalized_cost`. Select observed outputs above tier threshold before calling new models. Return deterministic `RouteDecision`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_policy_router.py -v`
Expected: PASS.

### Task 6: IO and CLI

**Files:**
- Create: `src/llm_router/io.py`
- Create: `src/llm_router/cli.py`
- Create: `tests/fixtures/models.jsonl`
- Create: `tests/fixtures/prompts.jsonl`
- Create: `tests/fixtures/outputs.jsonl`
- Test: `tests/test_io_cli.py`

- [ ] **Step 1: Write failing tests**

```python
import json
import subprocess
import sys
from pathlib import Path

from llm_router.io import load_candidate_models, load_observed_outputs, load_prompts


def test_jsonl_loaders_read_fixture_files():
    root = Path("tests/fixtures")

    assert len(load_prompts(root / "prompts.jsonl")) == 3
    assert len(load_candidate_models(root / "models.jsonl")) == 3
    assert len(load_observed_outputs(root / "outputs.jsonl")) >= 3


def test_cli_route_outputs_json_decision():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "llm_router.cli",
            "route",
            "--prompt",
            "안녕?",
            "--tier",
            "fast",
            "--models",
            "tests/fixtures/models.jsonl",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["action"] == "call"
    assert payload["model_id"] == "cheap"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_io_cli.py -v`
Expected: FAIL because IO, CLI, or fixtures do not exist.

- [ ] **Step 3: Implement IO, CLI, and fixtures**

Load JSONL records into schemas. CLI `route` prints one JSON decision. CLI `train` writes a utility table JSON. CLI `evaluate` computes average quality and total cost over local outputs.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_io_cli.py -v`
Expected: PASS.

### Task 7: Documentation and Full Verification

**Files:**
- Create: `README.md`

- [ ] **Step 1: Document usage**

Add install, data format, CLI examples, and challenge constraints.

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest -v`
Expected: all tests pass.

- [ ] **Step 3: Run CLI smoke test**

Run: `python -m llm_router.cli route --prompt "안녕?" --tier fast --models tests/fixtures/models.jsonl`
Expected: JSON decision selecting `cheap`.

## Self-Review

- Spec coverage: package, local/no-network behavior, tier policies, output constraint, CLI, tests, and docs are all covered.
- Placeholder scan: no `TBD` or vague implementation placeholders remain.
- Type consistency: all planned modules use `PromptRecord`, `CandidateModel`, `ObservedOutput`, `CallHistory`, `BudgetTier`, `UtilityTable`, and `RouteDecision` consistently.
- Repository note: the workspace is not currently a Git repository, so commit steps are omitted from execution unless Git is initialized later.
