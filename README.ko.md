# Efficient LLM Router

SK텔레콤 Efficient LLM Routing Challenge를 위한 로컬 오픈소스 라우터입니다.

이 프로젝트는 프롬프트 난이도와 예산 tier에 따라 어떤 후보 모델을 호출할지, 또는 이미 관측한 후보 출력 중 어떤 답을 최종 선택할지 결정합니다.

- `fast`: 저비용 모델을 우선 사용하고, 명확히 어려운 프롬프트에서만 승급합니다.
- `balanced`: 품질과 비용의 균형을 보고, 난도가 높거나 기대 이득이 클 때 상위 모델로 승급합니다.
- `premium`: 품질을 우선하되, 명백히 불필요한 고비용 호출은 피합니다.

외부 LLM, API, 네트워크 서비스는 호출하지 않습니다. 로컬 파일의 모델 메타데이터, 공개 학습용 후보 출력, 품질 라벨, 그리고 deterministic prompt feature만 사용합니다.

## 설치

로컬 개발 모드로 설치하려면:

```bash
python3 -m pip install -e .
```

소스 트리에서 바로 테스트를 실행할 수도 있습니다.

```bash
python3 -m pytest -v
```

## 데이터 포맷

주요 입력 파일은 JSONL과 CSV를 지원합니다.

### 프롬프트

```json
{"prompt_id":"p1","prompt":"안녕?","domain":"chat","task_type":"greeting"}
```

필수 필드:

- `prompt_id`
- `prompt`

선택 필드:

- `domain`
- `task_type`

### 후보 모델 메타데이터

```json
{"model_id":"cheap","cost":1,"latency":40,"family":"small","quality_prior":0.62}
```

필수 필드:

- `model_id`
- `cost`

선택 필드:

- `latency`
- `family`
- `quality_prior`

여기서 모델은 실제 LLM weight가 아니라, 평가 환경이 제공하는 후보 모델 식별자와 비용 정보입니다. 라우터는 모델을 직접 다운로드하거나 실행하지 않고, 어떤 후보를 호출할지만 결정합니다.

### 관측된 후보 출력

```json
{"prompt_id":"p1","model_id":"cheap","output":"안녕하세요!","quality":0.82}
```

필수 필드:

- `prompt_id`
- `model_id`
- `output`

선택 필드:

- `quality`

공개 데이터에서는 후보 모델별 출력과 품질 라벨을 학습/검증에 사용할 수 있습니다. 비공개 평가에서는 시뮬레이터가 라우터가 선택한 모델의 출력만 순차적으로 제공한다고 가정합니다.

## CLI 사용법

공개 품질 라벨로 로컬 utility table을 학습합니다.

```bash
PYTHONPATH=src python3 -m llm_router.cli train \
  --models tests/fixtures/models.jsonl \
  --outputs tests/fixtures/outputs.jsonl \
  --output utility-table.json
```

프롬프트 하나를 라우팅합니다.

```bash
PYTHONPATH=src python3 -m llm_router.cli route \
  --prompt "안녕?" \
  --tier fast \
  --models tests/fixtures/models.jsonl
```

로컬 fixture 기준으로 평균 품질과 총 비용을 평가합니다.

```bash
PYTHONPATH=src python3 -m llm_router.cli evaluate \
  --prompts tests/fixtures/prompts.jsonl \
  --models tests/fixtures/models.jsonl \
  --outputs tests/fixtures/outputs.jsonl \
  --tier balanced
```

## 라우터 동작 방식

라우터는 프롬프트에서 다음과 같은 deterministic feature를 추출합니다.

- 길이와 토큰 수
- 한국어/영어/숫자 비율
- 코드 신호
- 수학 신호
- 표 형태 신호
- 제약 조건 신호
- 다단계 추론 신호
- 도메인과 태스크 타입 신호

이 feature를 바탕으로 `[0, 1]` 범위의 해석 가능한 난이도 점수를 계산합니다.

각 후보 모델에 대해서는 다음 utility를 계산합니다.

```text
expected_utility = expected_quality - tier_cost_penalty * normalized_cost
```

이미 호출 이력에 충분히 좋은 출력이 있으면 새 모델을 호출하지 않고 해당 출력을 선택합니다. 그렇지 않으면 tier 정책에 따라 다음에 호출할 후보 모델을 반환합니다.

최종 답변 검증은 챌린지 규칙에 맞춰, 반드시 관측된 후보 출력 중 하나만 최종 답으로 선택하도록 강제합니다.

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

## 공식 챌린지 데이터에 맞추기

공식 공개 데이터 스키마가 나오면 라우터 핵심 로직은 유지하고, 로딩 계층만 맞추면 됩니다.

- 공식 프롬프트 파일을 `PromptRecord`로 매핑합니다.
- 공식 모델 메타데이터를 `CandidateModel`로 매핑합니다.
- 공개 후보 출력과 품질 라벨을 `ObservedOutput`으로 매핑합니다.
- 비공개 평가 시뮬레이터가 반환하는 순차 출력을 `CallHistory`에 누적합니다.

## 현재 샘플 파일

- 후보 모델: `tests/fixtures/models.jsonl`
- 프롬프트: `tests/fixtures/prompts.jsonl`
- 후보 출력: `tests/fixtures/outputs.jsonl`

빠른 확인:

```bash
PYTHONPATH=src python3 -m llm_router.cli route \
  --prompt "Python 코드 버그를 분석하고 시간복잡도를 설명해줘" \
  --tier balanced \
  --models tests/fixtures/models.jsonl
```
