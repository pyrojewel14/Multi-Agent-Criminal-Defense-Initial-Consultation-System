"""在独立进程中验收正式咨询接口，避免共享测试进程的单例和数据库。"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("redis-server") is None, reason="隔离验收需要本机 redis-server 可执行文件")
def test_consultation_acceptance_formal_interfaces(tmp_path):
    probe = Path(__file__).with_name("consultation_acceptance_probe.py")
    result = subprocess.run(
        [sys.executable, str(probe), "--output", str(tmp_path / "acceptance")],
        cwd=probe.parents[2], env=dict(os.environ, PYTHONNOUSERSITE="1"),
        text=True, capture_output=True, timeout=240,
    )
    logs = "\n".join(path.read_text() for path in (tmp_path / "acceptance").glob("*.log"))
    assert result.returncode == 0, result.stdout + result.stderr + logs
    evidence = json.loads((tmp_path / "acceptance" / "result.json").read_text())
    assert evidence["model_mode"] == "deterministic_gateway"
    assert evidence["real_model_calls"] == 0
    assert evidence["scenarios"]["complete"]["status"] == "completed"
    assert evidence["scenarios"]["insufficient_then_complete"]["first_pending"] == ["fact_intake"]
    assert evidence["scenarios"]["high_risk"]["alert_triggered"] is True
    assert evidence["scenarios"]["invalid_risk"]["workflow_status"] == "degraded"
    assert evidence["scenarios"]["invalid_service"]["workflow_status"] == "degraded"
    assert evidence["scenarios"]["dependency_failure"]["attempts"] == 3
    assert evidence["restart"]["final_report_preserved"] is True
    assert evidence["finite_concurrency"]["distinct_sessions"] == 2
    assert evidence["finite_concurrency"]["same_session_parallel_replay"] is True
    assert evidence["http_startup"]["shutdown_cleanup_complete"] is True
    assert evidence["http_startup"]["ready"]["api"] == "ready"
