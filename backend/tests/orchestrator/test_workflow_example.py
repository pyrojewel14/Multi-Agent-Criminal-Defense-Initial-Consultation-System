"""Executable example contract for the LangGraph workflow topology."""

import pytest


@pytest.mark.asyncio
async def test_minimal_workflow_example_pauses_for_review_then_finishes():
    from examples.workflow_minimal import run_demo

    checkpoints = await run_demo()

    assert checkpoints == [
        {"stage": "reception", "current_agent": "Receptionist", "next": "fact_digger", "finished": False},
        {"stage": "review", "current_agent": "HumanReview", "next": "human_review", "finished": False},
        {"stage": "approved", "current_agent": "HumanReview", "next": None, "finished": True},
    ]
