"""Typing contracts for LangGraph workflow boundaries."""

import ast
from pathlib import Path

import pytest

from app.orchestrator import workflow


def test_workflow_does_not_suppress_type_errors_with_cast():
    """Workflow boundaries should be typed or validated without ``typing.cast``."""
    assert workflow.__file__ is not None
    tree = ast.parse(Path(workflow.__file__).read_text(encoding="utf-8"))

    cast_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "cast"
    ]

    assert cast_calls == []


def test_workflow_result_validation_rejects_invalid_state_types():
    """Replacing casts must add a real runtime check at the LangGraph boundary."""
    with pytest.raises(ValueError, match="工作流返回了无效状态"):
        workflow._validate_workflow_state({"session_id": 123})


def test_workflow_result_validation_preserves_langgraph_metadata():
    """Validation must not discard LangGraph's reserved result metadata."""
    result = {"session_id": "session-1", "__interrupt__": ["pause"]}

    assert workflow._validate_workflow_state(result) == result
