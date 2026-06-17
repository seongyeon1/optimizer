import json
import os
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
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
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
        env=env,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["action"] == "call"
    assert payload["model_id"] == "cheap"
