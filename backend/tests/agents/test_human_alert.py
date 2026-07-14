"""Integration tests for the HumanAlert Agent node."""

from unittest.mock import AsyncMock, patch

import pytest

from app.agents.human_alert import ALERT_MESSAGE, human_alert_node
from tests.factories import make_consultation_state


# ---------------------------------------------------------------------------
# human_alert_node – basic behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_human_alert_node_sets_current_agent():
    """human_alert_node should set current_agent to HumanAlert."""
    state = make_consultation_state(
        alert_triggered=True,
        risk_assessment={"risk_type": "SELF_INCrimination", "risk_level": "high"},
        conversation_history=[],
    )

    result = await human_alert_node(state)

    assert result.get("current_agent") == "HumanAlert"


@pytest.mark.asyncio
async def test_human_alert_node_sets_lawyer_review_needed():
    """human_alert_node should set lawyer_review_needed to True."""
    state = make_consultation_state(
        alert_triggered=True,
        risk_assessment={"risk_type": "SELF_INCrimination", "risk_level": "high"},
        conversation_history=[],
    )

    result = await human_alert_node(state)

    assert result.get("lawyer_review_needed") is True


@pytest.mark.asyncio
async def test_human_alert_node_generates_calming_message():
    """human_alert_node should produce a calming message as final_output."""
    state = make_consultation_state(
        alert_triggered=True,
        risk_assessment={"risk_type": "COLLUSION", "risk_level": "high"},
        conversation_history=[],
    )

    result = await human_alert_node(state)

    assert result.get("final_output") != ""
    # Should contain the ALERT_MESSAGE text
    assert ALERT_MESSAGE in result.get("final_output", "")


@pytest.mark.asyncio
async def test_human_alert_node_updates_conversation_history():
    """human_alert_node should append an entry to conversation_history."""
    state = make_consultation_state(
        alert_triggered=True,
        risk_assessment={"risk_type": "EVIDENCE_TAMPERING", "risk_level": "high"},
        conversation_history=[],
    )

    result = await human_alert_node(state)

    conversation_history = result.get("conversation_history", [])
    assert len(conversation_history) > 0
    # The last entry should be from HumanAlert
    last_entry = conversation_history[-1]
    assert last_entry["agent"] == "HumanAlert"
    assert last_entry["alert_triggered"] is True


@pytest.mark.asyncio
async def test_human_alert_node_preserves_alert_triggered():
    """human_alert_node should preserve the alert flag for API callers."""
    state = make_consultation_state(
        alert_triggered=True,
        risk_assessment={"risk_type": "SELF_INCrimination", "risk_level": "high"},
        conversation_history=[],
    )

    result = await human_alert_node(state)

    assert result.get("alert_triggered") is True


@pytest.mark.asyncio
async def test_human_alert_node_with_risk_assessment():
    """human_alert_node should include risk_assessment in conversation history entry."""
    risk = {"risk_type": "STRATEGY_LEAKAGE", "risk_level": "high", "details": "律师策略泄露"}
    state = make_consultation_state(
        alert_triggered=True,
        risk_assessment=risk,
        conversation_history=[],
    )

    result = await human_alert_node(state)

    last_entry = result.get("conversation_history", [])[-1]
    assert last_entry["risk_assessment"] == risk


@pytest.mark.asyncio
async def test_human_alert_node_disclaimer_injected():
    """final_output should have the disclaimer prefix injected."""
    state = make_consultation_state(
        alert_triggered=True,
        risk_assessment={"risk_type": "SELF_INCrimination", "risk_level": "high"},
        conversation_history=[],
    )

    result = await human_alert_node(state)

    final_output = result.get("final_output", "")
    assert "智能辅助生成" in final_output or "仅供参考" in final_output
