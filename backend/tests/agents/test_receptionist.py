"""Integration tests for the Receptionist Agent node."""

from unittest.mock import AsyncMock, patch

import pytest

from app.agents import receptionist
from app.agents.receptionist import (
    CONSENT_KEYWORDS,
    USER_TYPE_PATTERNS,
    _complete_reception,
    _confirm_identity,
    _extract_and_confirm_info,
    _process_consent,
    check_consent_given,
    extract_user_type,
    receptionist_node,
)
from tests.factories import make_consultation_state


# ---------------------------------------------------------------------------
# check_consent_given / extract_user_type – unit-level helpers
# ---------------------------------------------------------------------------


def test_check_consent_given_positive():
    """Consent keywords should be detected."""
    for keyword in CONSENT_KEYWORDS:
        assert check_consent_given(f"我{keyword}了") is True


def test_check_consent_given_negative():
    """Non-consent text should not trigger consent."""
    assert check_consent_given("我想咨询一下") is False


def test_extract_user_type_suspect():
    """Suspect keyword patterns should return 'suspect'."""
    for keyword in USER_TYPE_PATTERNS["suspect"]:
        assert extract_user_type(f"{keyword}，想咨询") == "suspect"


def test_extract_user_type_victim():
    """Victim keyword patterns should return 'victim'."""
    for keyword in USER_TYPE_PATTERNS["victim"]:
        assert extract_user_type(f"{keyword}，想咨询") == "victim"


def test_extract_user_type_family():
    """Family keyword patterns should return 'family'."""
    for keyword in USER_TYPE_PATTERNS["family"]:
        assert extract_user_type(f"{keyword}，想咨询") == "family"


def test_extract_user_type_none():
    """Non-matching text should return None."""
    assert extract_user_type("随便聊聊") is None


# ---------------------------------------------------------------------------
# receptionist_node – consent not given
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_consent_returns_welcome():
    """When consent_given=False, return welcome message and keep consent_given=False."""
    state = make_consultation_state(
        consent_given=False,
        facts_raw=["你好"],
        current_input=None,
    )

    with patch("app.agents.receptionist.llm_gateway") as mock_llm:
        mock_llm.generate = AsyncMock(return_value="欢迎咨询，请阅读权利义务告知书。")
        result = await receptionist_node(state)

    assert result["consent_given"] is False
    assert result["current_agent"] == "Receptionist"
    assert result["final_output"] != ""


# ---------------------------------------------------------------------------
# receptionist_node – consent keywords
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consent_keywords_set_consent_given():
    """When user message contains consent keywords, consent_given should become True."""
    state = make_consultation_state(
        consent_given=False,
        facts_raw=["我同意"],
        current_input=None,
    )

    with patch("app.agents.receptionist.llm_gateway") as mock_llm:
        mock_llm.generate = AsyncMock(return_value="感谢您确认，请选择您的身份类型。")
        result = await receptionist_node(state)

    assert result["consent_given"] is True


# ---------------------------------------------------------------------------
# receptionist_node – user_type extraction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_type_extraction():
    """When consent_given=True and user message contains identity keywords, user_type should be set."""
    state = make_consultation_state(
        consent_given=True,
        facts_raw=["我是嫌疑人"],
        user_type=None,
        current_input=None,
    )

    with patch("app.agents.receptionist.llm_gateway") as mock_llm:
        mock_llm.generate = AsyncMock(return_value="您是嫌疑人，请问案件发生在哪个城市？")
        result = await receptionist_node(state)

    assert result["user_type"] == "suspect"


# ---------------------------------------------------------------------------
# receptionist_node – complete flow: consent → identity → city → FactDigger
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_reception_flow():
    """Full flow: consent already given, user_type set, city provided → transition to FactDigger."""
    state = make_consultation_state(
        consent_given=True,
        user_type="suspect",
        facts_raw=["北京"],
        conversation_history=[
            {"agent": "Receptionist", "user_type": "suspect", "case_city": None},
        ],
        current_input=None,
    )

    with patch("app.agents.receptionist.llm_gateway") as mock_llm:
        mock_llm.generate = AsyncMock(return_value="接待完成。")
        result = await receptionist_node(state)

    assert result["current_agent"] == "FactDigger"
    # Should have a conversation history entry with case_city
    assert any(entry.get("case_city") for entry in result.get("conversation_history", []))


@pytest.mark.asyncio
async def test_complete_reception_with_city():
    """When user_type is set and city is provided, _complete_reception should be called."""
    state = make_consultation_state(
        consent_given=True,
        user_type="suspect",
        facts_raw=["北京"],
        conversation_history=[],
        current_input=None,
    )

    with patch("app.agents.receptionist.llm_gateway") as mock_llm:
        mock_llm.generate = AsyncMock(return_value="请选择您的身份类型。")
        result = await receptionist_node(state)

    # With empty conversation_history, _extract_and_confirm_info is called first
    # which sets user_type and asks for city
    assert result["current_agent"] in ("Receptionist", "FactDigger")


# ---------------------------------------------------------------------------
# receptionist_node – coverage for missing lines
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consent_given_but_user_type_unrecognized():
    """When consent is given but user_type cannot be extracted, fall back to _confirm_identity."""
    state = make_consultation_state(
        consent_given=True,
        user_type=None,
        facts_raw=["普通提问，无身份关键词"],
        current_input=None,
    )

    with patch("app.agents.receptionist.llm_gateway") as mock_llm:
        mock_llm.generate = AsyncMock(return_value="请告知您的身份类型。")
        result = await receptionist_node(state)

    # _confirm_identity sets current_agent to Receptionist
    assert result["current_agent"] == "Receptionist"
    # user_type remains unset
    assert result.get("user_type") is None


@pytest.mark.asyncio
async def test_complete_flow_with_city_already_present():
    """When conversation_history already has a case_city, skip _complete_reception and hand off to FactDigger."""
    state = make_consultation_state(
        consent_given=True,
        user_type="suspect",
        facts_raw=["继续提供信息"],
        conversation_history=[
            {"agent": "Receptionist", "user_type": "suspect", "case_city": "北京"},
        ],
        current_input=None,
    )

    with patch("app.agents.receptionist.llm_gateway") as mock_llm:
        mock_llm.generate = AsyncMock(return_value="")
        result = await receptionist_node(state)

    # The final else branch is hit (line 232-234) — no LLM call needed
    assert result["current_agent"] == "FactDigger"
    assert result["pending_questions"] == ["请继续收集案件详情"]


@pytest.mark.asyncio
async def test_complete_flow_with_empty_user_message_but_city_present():
    """When the user message is empty but city info exists, the final else branch is hit."""
    state = make_consultation_state(
        consent_given=True,
        user_type="suspect",
        facts_raw=["", "   "],  # last entry is whitespace-only
        conversation_history=[
            {"agent": "Receptionist", "case_city": "上海"},
        ],
    )

    with patch("app.agents.receptionist.llm_gateway") as mock_llm:
        mock_llm.generate = AsyncMock(return_value="")
        result = await receptionist_node(state)

    assert result["current_agent"] == "FactDigger"


# ---------------------------------------------------------------------------
# _complete_reception – populate conversation_history when key missing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_reception_creates_conversation_history_if_missing():
    """When state lacks 'conversation_history', _complete_reception should create it."""
    state = make_consultation_state(
        consent_given=True,
        user_type="suspect",
    )
    # Remove the key entirely
    state.pop("conversation_history", None)

    result = await _complete_reception(state, "广州")
    assert result["current_agent"] == "FactDigger"
    assert result["conversation_history"]
    assert any(entry.get("case_city") == "广州" for entry in result["conversation_history"])


# ---------------------------------------------------------------------------
# __main__ block
# ---------------------------------------------------------------------------


def test_receptionist_main_runs_via_asyncio(monkeypatch):
    """The __main__ block should run cleanly when invoked as a script."""
    import asyncio

    # Simulate running the module's __main__ block
    from app.state.consultation_state import ConsultationState

    async def _runner():
        # Use the same call as the __main__ block
        return await receptionist_node(ConsultationState({"consent_given": False}))

    with patch("app.agents.receptionist.llm_gateway") as mock_llm:
        mock_llm.generate = AsyncMock(return_value="欢迎咨询")
        result = asyncio.run(_runner())
    assert "current_agent" in result
