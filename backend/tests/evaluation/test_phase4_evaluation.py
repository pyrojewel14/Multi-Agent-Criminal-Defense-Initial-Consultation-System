from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[3]
EVALUATION_DIR = ROOT_DIR / "evaluation"
CASES_PATH = EVALUATION_DIR / "cases.jsonl"
RUNNER_PATH = EVALUATION_DIR / "run_eval.py"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation import run_eval  # noqa: E402


def test_case_count_and_distribution() -> None:
    cases = run_eval.load_cases(CASES_PATH)

    assert len(cases) == 30
    assert Counter(case["category"] for case in cases) == Counter(run_eval.CATEGORY_COUNTS)


def test_case_schema_and_unique_ids() -> None:
    cases = run_eval.load_cases(CASES_PATH)

    run_eval.validate_cases(cases)
    assert len({case["id"] for case in cases}) == len(cases)
    assert all(set(case) == run_eval.CASE_FIELDS for case in cases)


def test_gold_fields_do_not_change_prediction() -> None:
    case = run_eval.load_cases(CASES_PATH)[0]
    changed_gold = deepcopy(case)
    changed_gold.update(
        {
            "expected_fact_fields": [],
            "expected_law_keywords": [],
            "should_follow_up": not case["should_follow_up"],
            "should_trigger_human": not case["should_trigger_human"],
            "expected_risk_level": "unknown",
            "expected_refusal": not case["expected_refusal"],
            "expected_disclaimer": not case["expected_disclaimer"],
        }
    )
    law_data = run_eval.load_criminal_law_data()

    original = asyncio.run(run_eval.evaluate_case(case, law_data))
    mutated = asyncio.run(run_eval.evaluate_case(changed_gold, law_data))

    assert original["prediction"] == mutated["prediction"]
    assert original["scores"] != mutated["scores"]


def test_metric_numerators_and_denominators() -> None:
    results = [
        {
            "gold": {"expected_disclaimer": True, "expected_refusal": True},
            "prediction": {"disclaimer_present": True, "refused": False},
            "scores": {
                "fact_fields_hit": 2,
                "fact_fields_total": 3,
                "law_keywords_hit": 1,
                "law_keywords_total": 2,
                "human_match": True,
                "follow_up_match": False,
                "refusal_match": False,
            },
        },
        {
            "gold": {"expected_disclaimer": False, "expected_refusal": False},
            "prediction": {"disclaimer_present": False, "refused": False},
            "scores": {
                "fact_fields_hit": 1,
                "fact_fields_total": 1,
                "law_keywords_hit": 0,
                "law_keywords_total": 0,
                "human_match": False,
                "follow_up_match": True,
                "refusal_match": True,
            },
        },
    ]

    metrics = run_eval.compute_metrics(results)

    assert metrics["fact_field_extraction_coverage"] == {
        "numerator": 3,
        "denominator": 4,
        "value": 0.75,
    }
    assert metrics["law_keyword_hit_at_5"] == {
        "numerator": 1,
        "denominator": 2,
        "value": 0.5,
    }
    assert metrics["high_risk_trigger_accuracy"]["numerator"] == 1
    assert metrics["high_risk_trigger_accuracy"]["denominator"] == 2
    assert metrics["follow_up_trigger_accuracy"]["numerator"] == 1
    assert metrics["refusal_disclaimer_trigger_rate"] == {
        "numerator": 1,
        "denominator": 2,
        "value": 0.5,
    }


@pytest.mark.parametrize("cwd", [ROOT_DIR, EVALUATION_DIR])
def test_runner_executes_from_supported_working_directories(tmp_path: Path, cwd: Path) -> None:
    output_dir = tmp_path / cwd.name.replace(" ", "_")
    completed = subprocess.run(
        [sys.executable, str(RUNNER_PATH), "--output-dir", str(output_dir)],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Cases: 30" in completed.stdout
    assert {path.name for path in output_dir.iterdir()} == {
        "case_results.csv",
        "case_results.json",
        "report.md",
        "summary.json",
    }
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["dataset"]["sample_count"] == 30
    assert summary["metadata"]["mode"] == run_eval.MODE


def test_runner_machine_output_has_explicit_metric_contract(tmp_path: Path) -> None:
    summary = asyncio.run(run_eval.run(CASES_PATH, tmp_path, "test command"))

    for metric in summary["metrics"].values():
        assert set(metric) == {"numerator", "denominator", "value"}
        assert metric["denominator"] >= metric["numerator"] >= 0
    assert "expected_risk_level" in summary["not_evaluated"]
    assert (tmp_path / "case_results.json").is_file()
    assert (tmp_path / "report.md").is_file()
