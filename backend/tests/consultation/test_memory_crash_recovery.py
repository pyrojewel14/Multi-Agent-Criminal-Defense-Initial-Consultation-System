"""强退后重建真实 SQLite 图，验证已知回执补偿及未知执行的保守边界。"""

import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

WINDOWS = [
    "input_uncommitted", "input_committed", "prepared", "running", "intake_checkpoint",
    "graph_applied", "applied", "reply_uncommitted", "reply_committed",
    "summary_unsaved", "summary_saved", "receipt_cleared",
]


@pytest.mark.parametrize("window", WINDOWS)
def test_crash_window_replay_preserves_audit_and_checkpoint(tmp_path, window):
    script = Path(__file__).with_name("memory_crash_probe.py")
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH=str(Path(__file__).resolve().parents[2]),
               MEMORY_RECENT_MESSAGES="2", MEMORY_SUMMARY_INPUT_BUDGET="4096",
               MEMORY_SUMMARY_OUTPUT_BUDGET="256")
    args = [window, str(tmp_path / "checkpoint.db"), str(tmp_path / "audit.db")]
    crashed = subprocess.run([sys.executable, str(script), "crash", *args],
                             env=env, capture_output=True, text=True, timeout=45)
    assert crashed.returncode == 86, crashed.stdout + crashed.stderr
    recovered = subprocess.run([sys.executable, str(script), "recover", *args],
                               env=env, capture_output=True, text=True, timeout=45)
    assert recovered.returncode == 0, recovered.stdout + recovered.stderr
    report = json.loads(next(line.removeprefix("CRASH_RESULT=") for line in recovered.stdout.splitlines()
                             if line.startswith("CRASH_RESULT=")))
    print("CRASH_EVIDENCE=" + json.dumps(report, ensure_ascii=False))
    before, after = report["before"], report["after"]
    unknown = window in {"input_committed", "running", "intake_checkpoint"}
    expected_receipt = {
        "input_uncommitted": None, "input_committed": None, "prepared": "prepared", "running": "running",
        "intake_checkpoint": "running", "graph_applied": "applied", "applied": "applied",
        "reply_uncommitted": "applied", "reply_committed": "applied", "summary_unsaved": "applied",
        "summary_saved": "applied", "receipt_cleared": None,
    }
    assert (before["receipt"] or {}).get("status") == expected_receipt[window]
    assert before["pending"] == (["law_ref"] if window == "intake_checkpoint" else ["fact_intake"])
    assert before["summary"]["through_sequence"] == (4 if window in {"summary_saved", "receipt_cleared"} else 2)
    assert len(before["rows"]) == (4 if window == "input_uncommitted" else
                                   6 if window in {"reply_committed", "summary_unsaved", "summary_saved", "receipt_cleared"} else 5)
    assert [row["sequence"] for row in after["rows"]] == list(range(1, len(after["rows"]) + 1))
    assert len({row["id"] for row in after["rows"]}) == len(after["rows"])
    target_inputs = [row for row in after["rows"] if row["sender_type"] == "user" and row["content"] == "合成目标陈述"]
    assert len(target_inputs) == 1
    source_id = target_inputs[0]["id"]
    target_items = [item for item in (after["case"] or {})["fields"]["behavior_sequence"]["items"]
                    if item["value"]["action"] == "合成目标陈述"]
    if unknown:
        assert all(result["error"] for result in report["results"])
        assert all("人工核验 checkpoint" in result["error"] for result in report["results"])
        assert report["extractions"] == []
        assert report["summary_batches"] == []
        assert after["pending"] == before["pending"]
        assert after["receipt"] == before["receipt"]
        assert after["case"] == before["case"]
        assert after["summary"] == before["summary"]
        assert len(target_items) == (1 if window == "intake_checkpoint" else 0)
        errors = [row for row in after["rows"] if row["metadata"].get("event") == "error_response"]
        assert len(errors) == 1
        assert "人工核验 checkpoint" in errors[0]["content"]
        assert all(result["error"] == errors[0]["content"] for result in report["results"])
        assert len(after["rows"]) == 7  # 原文五条，以及幂等的失败事件与对外错误。
        assert not any(row["sender_type"] == "agent" and row["command_id"] == target_inputs[0]["command_id"]
                       for row in after["rows"])
    else:
        assert all(result["error"] is None and result["reply"] == "合成回复" for result in report["results"])
        assert report["results"][0]["message_id"] == report["results"][1]["message_id"]
        assert after["receipt"] is None
        assert after["pending"] == ["fact_intake"]
        assert [row["content"] for row in after["rows"]] == [
            "合成种子0", "合成回复", "合成种子1", "合成回复", "合成目标陈述", "合成回复"]
        assert len(target_items) == 1
        assert target_items[0]["source_ids"] == [source_id]
        assert after["summary"]["through_sequence"] == 4
        assert after["summary"]["version"] == 2
        assert [row["sequence"] for row in after["recent"]] == [5, 6]
        assert report["extractions"] == ([["合成目标陈述"]] if window in {"input_uncommitted", "prepared"} else [])
        assert len(report["summary_batches"]) == (0 if window in {"summary_saved", "receipt_cleared"} else 1)
        if report["summary_batches"]:
            assert [row["sequence"] for row in report["summary_batches"][0]] == [3, 4]


@pytest.mark.parametrize("window", ["input_failure", "reply_failure"])
def test_known_commit_failure_keeps_same_key_retry_contract(tmp_path, window):
    script = Path(__file__).with_name("memory_crash_probe.py")
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH=str(Path(__file__).resolve().parents[2]),
               MEMORY_RECENT_MESSAGES="2", MEMORY_SUMMARY_INPUT_BUDGET="4096",
               MEMORY_SUMMARY_OUTPUT_BUDGET="256")
    result = subprocess.run([sys.executable, str(script), "failure", window,
                             str(tmp_path / "checkpoint.db"), str(tmp_path / "audit.db")],
                            env=env, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(next(line.removeprefix("CRASH_RESULT=") for line in result.stdout.splitlines()
                             if line.startswith("CRASH_RESULT=")))
    print("RETRY_EVIDENCE=" + json.dumps(report, ensure_ascii=False))
    assert report["first_error"] == {
        "error": "消息审计或执行未完成，请使用相同幂等键重试", "applied": window == "reply_failure"}
    assert all(item["error"] is None and item["reply"] == "合成回复" for item in report["results"])
    assert report["results"][0]["message_id"] == report["results"][1]["message_id"]
    assert report["extractions"] == [["合成目标陈述"]]
    assert report["after"]["receipt"] is None
    assert report["after"]["pending"] == ["fact_intake"]
    users = [row for row in report["after"]["rows"] if row["sender_type"] == "user"]
    assert [row["content"] for row in users] == ["合成种子0", "合成种子1", "合成目标陈述"]
    agent_replies = [row for row in report["after"]["rows"] if row["sender_type"] == "agent"]
    assert len(agent_replies) == 3
    items = report["after"]["case"]["fields"]["behavior_sequence"]["items"]
    target = [item for item in items if item["value"]["action"] == "合成目标陈述"]
    assert len(target) == 1
    assert target[0]["source_ids"] == [users[-1]["id"]]


@pytest.mark.parametrize("window", ["input_committed", "running"])
def test_unknown_execution_preserves_legacy_error_and_returns_current_guidance(tmp_path, window):
    script = Path(__file__).with_name("memory_crash_probe.py")
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH=str(Path(__file__).resolve().parents[2]),
               MEMORY_RECENT_MESSAGES="2", MEMORY_SUMMARY_INPUT_BUDGET="4096",
               MEMORY_SUMMARY_OUTPUT_BUDGET="256")
    with TemporaryDirectory(prefix="legacy-error-", dir=tmp_path) as directory:
        args = [window, str(Path(directory) / "checkpoint.db"), str(Path(directory) / "audit.db")]
        crashed = subprocess.run([sys.executable, str(script), "crash", *args],
                                 env=env, capture_output=True, text=True, timeout=45)
        assert crashed.returncode == 86, crashed.stdout + crashed.stderr
        recovered = subprocess.run([sys.executable, str(script), "recover_legacy_error", *args],
                                   env=env, capture_output=True, text=True, timeout=45)
        assert recovered.returncode == 0, recovered.stdout + recovered.stderr
    report = json.loads(next(line.removeprefix("CRASH_RESULT=") for line in recovered.stdout.splitlines()
                             if line.startswith("CRASH_RESULT=")))
    print("LEGACY_ERROR_EVIDENCE=" + json.dumps(report, ensure_ascii=False))
    before, after = report["before"], report["after"]
    assert len(before["rows"]) == 6
    errors = [row for row in before["rows"] if row["metadata"].get("event") == "error_response"]
    assert len(errors) == 1
    assert errors[0]["content"] == "消息处理失败，请稍后重试"
    assert errors[0]["metadata"]["request_id"] == "legacy-request"
    assert all(result["error"] == "消息执行状态未知，需人工核验 checkpoint 后修复；请勿自动重放"
               for result in report["results"])
    assert all(result["message_id"] is None and result["applied"] is False for result in report["results"])
    assert report["extractions"] == []
    assert report["summary_batches"] == []
    # 新错误追加与历史 ID 内容冲突时整个追加事务回滚；原文和图权威均不能变化。
    assert after == before
