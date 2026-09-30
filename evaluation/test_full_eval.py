"""Public corpus eval exercises real tool contract, without model calls offline."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


@pytest.mark.parametrize("demo_mode", ["on", "off"])
def test_offline_public_cases_and_distinct_evidence(tmp_path, monkeypatch, demo_mode):
    from evaluation.run_full_eval import run_evaluation

    monkeypatch.setenv("LAW_KNOWLEDGE_PROFILE", "full")
    monkeypatch.setenv("LAW_FULL_DEMO_ANNOTATIONS", demo_mode)
    result = asyncio.run(run_evaluation("offline", tmp_path / "result.json"))
    assert result["evidence_kind"] == "offline-deterministic-tools"
    assert result["llm_executed"] is False
    assert result["rag_executed"] is False
    assert result["passed"] == 8
    assert result["total"] == 8
    assert result["legal_review_completed"] is False
    assert result["annotation_mode"] == ("demo" if demo_mode == "on" else "text_only_with_six_regression")
    assert result == json.loads((tmp_path / "result.json").read_text())
