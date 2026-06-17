# Efficient LLM Router

[한국어 README](README.ko.md)

Local, open-source router for the SK Telecom Efficient LLM Routing Challenge.

The router chooses which candidate model to call, or which already observed output to select, under three budget tiers:

- `fast`: cheapest acceptable model first, conservative escalation.
- `balanced`: quality/cost tradeoff with escalation for difficult prompts.
- `premium`: quality-oriented while still avoiding needless expensive calls.

The project does not call external models, APIs, or network services. It uses local metadata, local public-training outputs, and deterministic prompt features.

## Install for Local Development

```bash
python3 -m pip install -e .
```

You can also run tests directly from the source tree:

```bash
python3 -m pytest -v
```

## Data Formats

JSONL and CSV are supported for the main local inputs.

### Prompts

```json
{"prompt_id":"p1","prompt":"안녕?","domain":"chat","task_type":"greeting"}
```

Required fields:

- `prompt_id`
- `prompt`

Optional fields:

- `domain`
- `task_type`

### Candidate Models

```json
{"model_id":"cheap","cost":1,"latency":40,"family":"small","quality_prior":0.62}
```

Required fields:

- `model_id`
- `cost`

Optional fields:

- `latency`
- `family`
- `quality_prior`

### Observed Outputs

```json
{"prompt_id":"p1","model_id":"cheap","output":"안녕하세요!","quality":0.82}
```

Required fields:

- `prompt_id`
- `model_id`
- `output`

Optional fields:

- `quality`

## CLI

Train a local utility table from public labels:

```bash
PYTHONPATH=src python3 -m llm_router.cli train \
  --models tests/fixtures/models.jsonl \
  --outputs tests/fixtures/outputs.jsonl \
  --output utility-table.json
```

Route one prompt:

```bash
PYTHONPATH=src python3 -m llm_router.cli route \
  --prompt "안녕?" \
  --tier fast \
  --models tests/fixtures/models.jsonl
```

Evaluate local fixture quality and cost:

```bash
PYTHONPATH=src python3 -m llm_router.cli evaluate \
  --prompts tests/fixtures/prompts.jsonl \
  --models tests/fixtures/models.jsonl \
  --outputs tests/fixtures/outputs.jsonl \
  --tier balanced
```

## Router Behavior

The router extracts deterministic prompt features such as length, Korean/English ratio, code signals, math signals, table signals, constraints, and multistep markers. It converts those features into an interpretable difficulty score in `[0, 1]`.

For each candidate model, it estimates:

```text
expected_utility = expected_quality - tier_cost_penalty * normalized_cost
```

If call history already contains an output good enough for the tier, the router selects that output without another call. Otherwise, it returns the next model to call.

Final answer validation enforces the challenge rule that the final answer must be one of the observed candidate outputs.

## Python API

```python
from llm_router.router import LLMRouter
from llm_router.schemas import BudgetTier, CallHistory, CandidateModel, PromptRecord

router = LLMRouter()
decision = router.route(
    PromptRecord("p1", "안녕?"),
    BudgetTier.FAST,
    CallHistory(),
    [CandidateModel("cheap", cost=1), CandidateModel("smart", cost=8)],
)
print(decision)
```

## Adapting to Official Challenge Data

Keep the router API stable and adapt only the loading layer when official public data schemas are released:

- Map official prompt files into `PromptRecord`.
- Map official model metadata into `CandidateModel`.
- Map public candidate outputs and labels into `ObservedOutput`.
- Use private simulator calls to populate `CallHistory` sequentially.
