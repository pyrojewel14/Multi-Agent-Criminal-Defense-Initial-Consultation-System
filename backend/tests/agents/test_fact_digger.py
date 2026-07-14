"""Integration tests for the FactDigger Agent node."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.agents import fact_digger
from app.agents.fact_digger import (
    COVERAGE_THRESHOLD,
    _analyze_coverage,
    _generate_fact_summary,
    _generate_follow_up_questions,
    _get_fact_value,
    _handle_first_interaction,
    _handle_high_risk_input,
    _handle_insufficient_coverage,
    _handle_sufficient_coverage,
    _load_extract_case_facts_prompt,
    _load_first_prompt,
    _load_follow_up_prompt,
    _load_summary_prompt,
    fact_digger_node,
)
from tests.factories import make_applied_law, make_consultation_state


# ---------------------------------------------------------------------------
# fact_digger_node – first interaction (no consent / no input)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_interaction_no_input():
    """When consent_given=False and no facts_raw / current_input, return first-interaction prompt."""
    state = make_consultation_state(
        consent_given=False,
        facts_raw=[],
        current_input=None,
    )

    with patch("app.agents.fact_digger.llm_gateway") as mock_llm:
        result = await fact_digger_node(state)

    assert result.get("current_agent") == "FactDigger"
    assert result.get("facts_coverage_rate") == 0.0
    # The response should contain the first-prompt guidance text
    assert result.get("final_output") != ""
    # Conversation history should have an entry from FactDigger
    assert any(
        entry.get("agent") == "FactDigger"
        for entry in result.get("conversation_history", [])
    )


# ---------------------------------------------------------------------------
# fact_digger_node – extract structured facts with consent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_structured_facts_with_consent():
    """When consent_given=True and facts_raw is non-empty, extract structured facts via LLM tool call."""
    extracted = {
        "incident_time": "2024-01-01",
        "incident_location": "北京市朝阳区",
        "consequence": "轻伤",
        "behavior_sequence": [{"actor": "张某", "action": "殴打"}],
    }

    state = make_consultation_state(
        consent_given=True,
        facts_raw=["张某在酒吧打人"],
        current_input="张某在酒吧打人",
        applied_laws=[],  # no laws yet → coverage_rate stays 0
    )

    with patch("app.agents.fact_digger.llm_gateway") as mock_llm:
        mock_llm.generate_with_tools = AsyncMock(
            return_value={
                "content": "",
                "tool_calls": [
                    {"name": "extract_case_facts", "args": extracted},
                ],
                "has_tool_call": True,
            }
        )
        mock_llm.generate = AsyncMock(return_value="mocked response")

        result = await fact_digger_node(state)

    # Structured facts should be populated from the tool call
    assert result.get("facts_structured") == extracted
    # No applied_laws → coverage_rate remains 0, agent stays FactDigger
    assert result.get("facts_coverage_rate") == 0.0
    assert result.get("current_agent") == "FactDigger"


# ---------------------------------------------------------------------------
# _analyze_coverage – various combinations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_coverage_no_laws():
    """With no applied_laws, coverage is zero."""
    result = await _analyze_coverage({}, [])
    assert result["coverage_rate"] == 0.0
    assert result["source"] == "no_laws"


@pytest.mark.asyncio
async def test_analyze_coverage_all_covered():
    """When all elements have corresponding fact values, coverage is 1.0."""
    facts = {
        "incident_time": "2024-01-01",
        "incident_location": "北京",
        "consequence": "轻伤",
        "behavior_sequence": ["殴打"],
    }
    laws = [
        {
            "elements": [
                {"name": "时间", "key": "time"},
                {"name": "地点", "key": "location"},
                {"name": "后果", "key": "consequence"},
                {"name": "行为", "key": "behavior"},
            ],
            "data_source": "json_keyword",
        }
    ]
    result = await _analyze_coverage(facts, laws)
    assert result["coverage_rate"] == 1.0
    assert result["total_elements"] == 4
    assert result["covered_elements"] == 4
    assert result["missing_elements"] == []
    assert result["source"] == "json_knowledge"


@pytest.mark.asyncio
async def test_analyze_coverage_partial():
    """When some elements are missing, coverage is partial."""
    facts = {"incident_time": "2024-01-01"}
    laws = [
        {
            "elements": [
                {"name": "时间", "key": "time"},
                {"name": "地点", "key": "location"},
            ],
            "data_source": "json_keyword",
        }
    ]
    result = await _analyze_coverage(facts, laws)
    assert result["coverage_rate"] == 0.5
    assert "地点" in result["missing_elements"]


@pytest.mark.asyncio
async def test_analyze_coverage_rag_only():
    """When only unverified RAG results exist, coverage source is rag_only."""
    laws = [
        {
            "elements": [{"name": "test", "key": "test"}],
            "data_source": "rag_unverified",
        }
    ]
    result = await _analyze_coverage({}, laws)
    assert result["source"] == "rag_only"
    assert result["coverage_rate"] == 0.0


# ---------------------------------------------------------------------------
# fact_digger_node – low coverage → follow-up questions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_low_coverage_generates_follow_up():
    """When coverage < threshold, follow-up questions should be generated."""
    facts = {"incident_time": "2024-01-01"}
    laws = [
        make_applied_law(
            elements=[
                {"name": "时间", "key": "time"},
                {"name": "地点", "key": "location"},
                {"name": "后果", "key": "consequence"},
            ],
        )
    ]

    state = make_consultation_state(
        consent_given=True,
        facts_raw=["事情发生在2024年1月1日"],
        current_input="事情发生在2024年1月1日",
        facts_structured=facts,
        applied_laws=laws,
    )

    follow_up_questions = ["请描述事件发生的地点", "请说明事件造成的后果"]

    with patch("app.agents.fact_digger.llm_gateway") as mock_llm:
        # _extract_structured_facts call
        mock_llm.generate_with_tools = AsyncMock(
            return_value={
                "content": "",
                "tool_calls": [{"name": "extract_case_facts", "args": facts}],
                "has_tool_call": True,
            }
        )
        # _generate_follow_up_questions call
        mock_llm.generate = AsyncMock(
            return_value=json.dumps({"questions": follow_up_questions})
        )

        result = await fact_digger_node(state)

    assert result.get("current_agent") == "FactDigger"
    # Should have pending questions
    assert len(result.get("pending_questions", [])) > 0
    # final_output should mention supplementary info request
    final_output = result.get("final_output", "")
    assert "补充" in final_output or "信息" in final_output


# ---------------------------------------------------------------------------
# fact_digger_node – high coverage → summary + transition
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_high_coverage_generates_summary():
    """When coverage >= threshold, generate summary and transition to RiskAssessor."""
    facts = {
        "incident_time": "2024-01-01",
        "incident_location": "北京",
        "consequence": "轻伤",
        "behavior_sequence": ["殴打"],
    }
    laws = [
        make_applied_law(
            elements=[
                {"name": "时间", "key": "time"},
                {"name": "地点", "key": "location"},
                {"name": "后果", "key": "consequence"},
                {"name": "行为", "key": "behavior"},
            ],
        )
    ]

    state = make_consultation_state(
        consent_given=True,
        facts_raw=["张某在酒吧打人造成轻伤"],
        current_input="张某在酒吧打人造成轻伤",
        facts_structured=facts,
        applied_laws=laws,
    )

    with patch("app.agents.fact_digger.llm_gateway") as mock_llm:
        mock_llm.generate_with_tools = AsyncMock(
            return_value={
                "content": "",
                "tool_calls": [{"name": "extract_case_facts", "args": facts}],
                "has_tool_call": True,
            }
        )
        # _generate_fact_summary call
        mock_llm.generate = AsyncMock(return_value="案件事实摘要：张某于2024年1月1日在北京酒吧殴打他人致轻伤。")

        result = await fact_digger_node(state)

    assert result.get("current_agent") == "RiskAssessor"
    assert "摘要" in result.get("final_output", "")


# ---------------------------------------------------------------------------
# fact_digger_node – high risk input → alert_triggered
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_high_risk_input_triggers_alert():
    """High-risk user input should set alert_triggered and route to HumanAlert."""
    state = make_consultation_state(
        consent_given=True,
        facts_raw=[],
        current_input="是我干的，帮我隐瞒一下",
        applied_laws=[],
    )

    with patch("app.agents.fact_digger.llm_gateway") as mock_llm:
        mock_llm.generate_with_tools = AsyncMock(
            return_value={
                "content": "",
                "tool_calls": [],
                "has_tool_call": False,
            }
        )
        mock_llm.generate = AsyncMock(return_value="mocked")

        result = await fact_digger_node(state)

    assert result.get("alert_triggered") is True
    assert result.get("current_agent") == "HumanAlert"


@pytest.mark.asyncio
async def test_node_appends_current_input_once_and_only_after_pii_masking():
    state = make_consultation_state(
        facts_raw=[],
        current_input="联系电话是13812345678",
        applied_laws=[],
    )
    with patch(
        "app.agents.fact_digger._extract_structured_facts",
        new_callable=AsyncMock,
        return_value={"behavior_sequence": ["咨询"]},
    ):
        result = await fact_digger_node(state)

    assert result["facts_raw"] == ["联系电话是[PHONE-MASKED]"]


@pytest.mark.asyncio
async def test_high_risk_input_is_not_persisted_before_human_alert():
    state = make_consultation_state(
        facts_raw=[],
        current_input="帮我销毁证据，电话13812345678",
    )

    result = await fact_digger_node(state)

    assert result["current_agent"] == "HumanAlert"
    assert result["facts_raw"] == []


# ---------------------------------------------------------------------------
# Prompt loaders – fallback path
# ---------------------------------------------------------------------------


class TestPromptLoadersFallback:
    def test_load_extract_case_facts_prompt_uses_loader(self):
        with patch.object(fact_digger.prompt_loader, "load", return_value="loaded-extract"):
            assert _load_extract_case_facts_prompt() == "loaded-extract"

    def test_load_extract_case_facts_prompt_fallback(self):
        with patch.object(fact_digger.prompt_loader, "load", side_effect=KeyError("nope")):
            text = _load_extract_case_facts_prompt()
        assert "incident_time" in text
        assert "behavior_sequence" in text

    def test_load_first_prompt_uses_loader(self):
        with patch.object(fact_digger.prompt_loader, "load", return_value="loaded-first"):
            assert _load_first_prompt() == "loaded-first"

    def test_load_first_prompt_fallback(self):
        with patch.object(fact_digger.prompt_loader, "load", side_effect=KeyError("nope")):
            text = _load_first_prompt()
        assert "事前" in text
        assert "事发" in text

    def test_load_follow_up_prompt_uses_loader(self):
        with patch.object(fact_digger.prompt_loader, "load", return_value="loaded-followup"):
            assert _load_follow_up_prompt() == "loaded-followup"

    def test_load_follow_up_prompt_fallback(self):
        with patch.object(fact_digger.prompt_loader, "load", side_effect=KeyError("nope")):
            text = _load_follow_up_prompt()
        assert "追问" in text

    def test_load_summary_prompt_uses_loader(self):
        with patch.object(fact_digger.prompt_loader, "load", return_value="loaded-summary"):
            assert _load_summary_prompt() == "loaded-summary"

    def test_load_summary_prompt_fallback(self):
        with patch.object(fact_digger.prompt_loader, "load", side_effect=KeyError("nope")):
            text = _load_summary_prompt()
        assert "Markdown" in text


# ---------------------------------------------------------------------------
# _handle_first_interaction
# ---------------------------------------------------------------------------


def test_handle_first_interaction_populates_state():
    """First interaction should set first-prompt text, current_agent and coverage."""
    state = make_consultation_state(conversation_history=[], consent_given=False, facts_raw=[])
    out = _handle_first_interaction(state)

    assert out.get("current_agent") == "FactDigger"
    assert out.get("facts_coverage_rate") == 0.0
    assert out.get("final_output") != ""
    assert any(
        e.get("agent") == "FactDigger"
        for e in out.get("conversation_history", [])
    )


# ---------------------------------------------------------------------------
# _handle_high_risk_input
# ---------------------------------------------------------------------------


def test_handle_high_risk_input_returns_unchanged_state():
    """If input is not high-risk, the function should return the state unchanged."""
    state = make_consultation_state()
    out = _handle_high_risk_input(state, "普通问题", state.get("conversation_history", []))

    # If not high risk, current_agent should not be set to HumanAlert
    assert out.get("current_agent") != "HumanAlert"
    assert out.get("alert_triggered") is False


# ---------------------------------------------------------------------------
# _generate_follow_up_questions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_follow_up_questions_no_missing():
    """When there are no missing elements, return empty list immediately."""
    with patch("app.agents.fact_digger.llm_gateway") as mock_llm:
        mock_llm.generate = AsyncMock()
        result = await _generate_follow_up_questions([], {})
    assert result == []
    mock_llm.generate.assert_not_called()


@pytest.mark.asyncio
async def test_generate_follow_up_questions_valid_json():
    """Valid LLM JSON should be parsed and 'questions' returned."""
    with patch("app.agents.fact_digger.llm_gateway") as mock_llm:
        mock_llm.generate = AsyncMock(
            return_value='{"questions": ["Q1", "Q2"]}'
        )
        result = await _generate_follow_up_questions(
            ["地点"], {"behavior_sequence": ["盗窃"], "consequence": ""}
        )
    assert result == ["Q1", "Q2"]


@pytest.mark.asyncio
async def test_generate_follow_up_questions_malformed_json():
    """Malformed LLM response should fall back to empty list."""
    with patch("app.agents.fact_digger.llm_gateway") as mock_llm:
        mock_llm.generate = AsyncMock(return_value="not a json response at all")
        result = await _generate_follow_up_questions(
            ["地点"], {"behavior_sequence": ["盗窃"], "consequence": ""}
        )
    assert result == []


@pytest.mark.asyncio
async def test_generate_follow_up_questions_exception():
    """When LLM raises, the function should return empty list."""
    with patch("app.agents.fact_digger.llm_gateway") as mock_llm:
        mock_llm.generate = AsyncMock(side_effect=Exception("LLM offline"))
        result = await _generate_follow_up_questions(
            ["地点"], {"behavior_sequence": ["盗窃"], "consequence": ""}
        )
    assert result == []


# ---------------------------------------------------------------------------
# _generate_fact_summary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_fact_summary_returns_text():
    """_generate_fact_summary should return the LLM text response."""
    with patch("app.agents.fact_digger.llm_gateway") as mock_llm:
        mock_llm.generate = AsyncMock(return_value="事实摘要：张某于酒吧打人。")
        result = await _generate_fact_summary(
            {"consequence": "轻伤"}, ["张某打人"]
        )
    assert "事实摘要" in result


@pytest.mark.asyncio
async def test_generate_fact_summary_masks_pii_in_legacy_raw_facts():
    with patch("app.agents.fact_digger.llm_gateway") as mock_llm:
        mock_llm.generate = AsyncMock(return_value="已脱敏摘要")
        await _generate_fact_summary(
            {"consequence": "轻伤"}, ["联系电话是13812345678"]
        )

    user_message = mock_llm.generate.await_args.kwargs["user_message"]
    assert "13812345678" not in user_message
    assert "[PHONE-MASKED]" in user_message


@pytest.mark.asyncio
async def test_generate_fact_summary_exception_returns_empty():
    """If LLM raises, return empty string."""
    with patch("app.agents.fact_digger.llm_gateway") as mock_llm:
        mock_llm.generate = AsyncMock(side_effect=Exception("LLM offline"))
        result = await _generate_fact_summary(
            {"consequence": "轻伤"}, ["张某打人"]
        )
    assert result == ""


# ---------------------------------------------------------------------------
# _analyze_coverage – weak_elements branches (empty list / false bool)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_coverage_weak_empty_list():
    """An element covered with an empty list should be marked as weak."""
    facts = {"behavior_sequence": []}  # present, but empty list
    laws = [
        {
            "elements": [{"name": "行为", "key": "behavior"}],
            "data_source": "json_keyword",
        }
    ]
    result = await _analyze_coverage(facts, laws)
    assert "行为" in result["weak_elements"]
    assert "行为" not in result["missing_elements"]


@pytest.mark.asyncio
async def test_analyze_coverage_weak_false_bool():
    """A covered element with False value should be marked as weak."""
    facts = {"surrender": False}
    laws = [
        {
            "elements": [{"name": "自首", "key": "surrender"}],
            "data_source": "json_keyword",
        }
    ]
    result = await _analyze_coverage(facts, laws)
    assert "自首" in result["weak_elements"]


# ---------------------------------------------------------------------------
# _get_fact_value
# ---------------------------------------------------------------------------


def test_get_fact_value_known_mappings():
    """Known key mappings should translate to facts_structured keys."""
    facts = {
        "incident_time": "2024-01-01",
        "incident_location": "北京",
        "parties": [],
        "behavior_sequence": ["x"],
        "consequence": "y",
        "evidence_mentioned": [],
        "arrest_status": "已羁押",
        "surrender": True,
        "victim_forgiveness": False,
        "prior_record": False,
    }
    assert _get_fact_value(facts, "time") == "2024-01-01"
    assert _get_fact_value(facts, "location") == "北京"
    assert _get_fact_value(facts, "behavior") == ["x"]
    assert _get_fact_value(facts, "surrender") is True


def test_get_fact_value_unknown_key():
    """Unknown keys should be looked up directly in the dict."""
    facts = {"custom_key": "value"}
    assert _get_fact_value(facts, "custom_key") == "value"
    assert _get_fact_value(facts, "missing") is None


# ---------------------------------------------------------------------------
# _handle_insufficient_coverage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_insufficient_coverage_generates_questions():
    """Insufficient coverage path should generate follow-ups and update pending_questions."""
    state = make_consultation_state(
        pending_questions=["existing_q"],
        conversation_history=[],
    )
    coverage = {
        "missing_elements": ["地点", "时间"],
        "covered_elements": [],
        "coverage_rate": 0.2,
        "weak_elements": [],
        "total_elements": 2,
    }
    facts = {"behavior_sequence": ["x"], "consequence": ""}

    with patch("app.agents.fact_digger.llm_gateway") as mock_llm:
        mock_llm.generate = AsyncMock(
            return_value='{"questions": ["请说明地点？", "请说明时间？"]}'
        )
        result = await _handle_insufficient_coverage(state, coverage, facts, ["existing_q"], [])

    assert result.get("current_agent") == "FactDigger"
    # Should have appended the 2 new questions to the existing 1
    assert len(result.get("pending_questions", [])) == 3
    assert "请说明地点" in result.get("final_output", "")


# ---------------------------------------------------------------------------
# _handle_sufficient_coverage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_sufficient_coverage_generates_summary():
    """Sufficient coverage should produce summary and hand off to RiskAssessor."""
    state = make_consultation_state(conversation_history=[])
    facts = {"consequence": "轻伤", "behavior_sequence": ["x"]}

    with patch("app.agents.fact_digger.llm_gateway") as mock_llm:
        mock_llm.generate = AsyncMock(return_value="案件事实摘要内容")
        result = await _handle_sufficient_coverage(state, facts, ["input1"], [])

    assert result.get("current_agent") == "RiskAssessor"
    assert "案件事实摘要" in result.get("final_output", "")
    assert any(
        e.get("agent") == "FactDigger"
        for e in result.get("conversation_history", [])
    )


# ---------------------------------------------------------------------------
# _extract_structured_facts (tested via fact_digger_node with edge responses)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_structured_facts_fallback_to_content_json():
    """When there are no tool_calls but content contains JSON, it should still parse."""
    state = make_consultation_state(
        consent_given=True,
        facts_raw=["input"],
        current_input="input",
        applied_laws=[],
    )

    extracted = {"incident_time": "2024-01-01", "consequence": "轻伤"}
    with patch("app.agents.fact_digger.llm_gateway") as mock_llm:
        mock_llm.generate_with_tools = AsyncMock(
            return_value={
                "content": json.dumps(extracted, ensure_ascii=False),
                "tool_calls": [],
                "has_tool_call": False,
            }
        )
        mock_llm.generate = AsyncMock(return_value="mocked")
        result = await fact_digger_node(state)

    assert result.get("facts_structured") == extracted


@pytest.mark.asyncio
async def test_extract_structured_facts_wrong_tool_name_warns():
    """If has_tool_call is True but the wrong tool was called, log warning and return {}."""
    state = make_consultation_state(
        consent_given=True,
        facts_raw=["input"],
        current_input="input",
        applied_laws=[],
    )

    with patch("app.agents.fact_digger.llm_gateway") as mock_llm:
        mock_llm.generate_with_tools = AsyncMock(
            return_value={
                "content": "",
                "tool_calls": [{"name": "other_tool", "args": {"foo": "bar"}}],
                "has_tool_call": True,
            }
        )
        mock_llm.generate = AsyncMock(return_value="mocked")
        result = await fact_digger_node(state)

    # No extraction happened, so facts_structured remains the default {}.
    # Then applied_laws is empty so coverage_rate stays 0.
    assert result.get("facts_structured") == {}
    assert result.get("current_agent") == "FactDigger"


@pytest.mark.asyncio
async def test_extract_structured_facts_no_result_warns():
    """When has_tool_call is False and content is empty, return {}."""
    state = make_consultation_state(
        consent_given=True,
        facts_raw=["input"],
        current_input="input",
        applied_laws=[],
    )

    with patch("app.agents.fact_digger.llm_gateway") as mock_llm:
        mock_llm.generate_with_tools = AsyncMock(
            return_value={"content": "", "tool_calls": [], "has_tool_call": False}
        )
        mock_llm.generate = AsyncMock(return_value="mocked")
        result = await fact_digger_node(state)

    assert result.get("facts_structured") == {}
    assert result.get("current_agent") == "FactDigger"


@pytest.mark.asyncio
async def test_extract_structured_facts_exception_returns_empty():
    """If the LLM raises, the function should return empty facts and continue."""
    state = make_consultation_state(
        consent_given=True,
        facts_raw=["input"],
        current_input="input",
        applied_laws=[],
    )

    with patch("app.agents.fact_digger.llm_gateway") as mock_llm:
        mock_llm.generate_with_tools = AsyncMock(side_effect=Exception("LLM offline"))
        mock_llm.generate = AsyncMock(return_value="mocked")
        result = await fact_digger_node(state)

    assert result.get("facts_structured") == {}
    # Even on extraction failure, state should be valid
    assert result.get("current_agent") in ("FactDigger", "RiskAssessor")
