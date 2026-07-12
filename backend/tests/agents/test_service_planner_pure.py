import pytest

from app.agents.service_planner import (
    _build_service_request_message,
    _extract_service_plan_structure,
    _parse_llm_response,
)


class TestParseLlmResponse:
    """Tests for _parse_llm_response."""

    def test_both_service_plan_and_report_markers(self):
        content = (
            "一些前置内容\n"
            "【服务方案开始】\n"
            "- 立即行动：申请取保候审\n"
            "# 刑事辩护初期咨询报告\n"
            "报告正文内容"
        )
        result = _parse_llm_response(content)
        assert "service_plan" in result
        assert "report_draft" in result
        assert result["report_draft"].startswith("# 刑事辩护初期咨询报告")

    def test_only_report_marker(self):
        content = "一些内容\n# 刑事辩护初期咨询报告\n报告正文"
        result = _parse_llm_response(content)
        assert result["service_plan"] == {}
        assert result["report_draft"].startswith("# 刑事辩护初期咨询报告")

    def test_no_markers(self):
        content = "这是一段没有标记的普通文本"
        result = _parse_llm_response(content)
        assert result["service_plan"] == {}
        assert result["report_draft"] == content

    def test_result_always_has_service_plan_and_report_draft_keys(self):
        result = _parse_llm_response("任意内容")
        assert "service_plan" in result
        assert "report_draft" in result


class TestExtractServicePlanStructure:
    """Tests for _extract_service_plan_structure."""

    def test_parse_typical_llm_output_with_sections(self):
        content = (
            "【服务方案】\n"
            "🔴 立即行动\n"
            "- 申请取保候审\n"
            "- 会见嫌疑人\n"
            "🟡 短期行动\n"
            "- 收集不在场证明\n"
            "🔵 后续行动\n"
            "- 制定辩护策略\n"
            "\n"
            "主要辩护策略\n"
            "- 罪轻辩护\n"
            "备选辩护策略\n"
            "- 无罪辩护\n"
            "\n"
            "侦查阶段：会见嫌疑人、了解案情\n"
            "- 固定有利证据\n"
            "审查起诉阶段：阅卷、提出辩护意见\n"
            "审判阶段：出庭辩护\n"
            "\n"
            "推荐方案：全程委托\n"
            "分阶段报价\n"
            "- 侦查阶段 3万-5万元\n"
            "- 审查起诉阶段 2万-4万元\n"
        )
        plan = _extract_service_plan_structure(content)

        # urgent_actions
        assert "申请取保候审" in plan["urgent_actions"]["immediate"]
        assert "会见嫌疑人" in plan["urgent_actions"]["immediate"]
        assert "收集不在场证明" in plan["urgent_actions"]["short_term"]
        assert "制定辩护策略" in plan["urgent_actions"]["follow_up"]

        # defense_strategies
        assert plan["defense_strategies"]["primary"] == "罪轻辩护"
        assert "无罪辩护" in plan["defense_strategies"]["alternatives"]

        # service_phases
        phases = plan["service_phases"]
        assert len(phases) >= 1
        phase_names = [p["phase"] for p in phases]
        assert "侦查阶段" in phase_names

        # fee_structure
        assert plan["fee_structure"]["recommended_plan"] != "" or plan["fee_structure"]["total_fee_range"] != ""

    def test_empty_content(self):
        plan = _extract_service_plan_structure("")
        assert plan["urgent_actions"]["immediate"] == []
        assert plan["urgent_actions"]["short_term"] == []
        assert plan["urgent_actions"]["follow_up"] == []
        assert plan["defense_strategies"]["primary"] == ""
        assert plan["service_phases"] == []
        assert plan["fee_structure"]["recommended_plan"] == ""

    def test_partial_content_only_urgent_actions(self):
        content = (
            "🔴 立即行动\n"
            "- 申请取保候审\n"
        )
        plan = _extract_service_plan_structure(content)
        assert "申请取保候审" in plan["urgent_actions"]["immediate"]
        assert plan["service_phases"] == []

    def test_fee_range_extraction(self):
        content = "费用明细\n- 侦查阶段 3万-5万元\n"
        plan = _extract_service_plan_structure(content)
        assert plan["fee_structure"]["total_fee_range"] == "3万-5万元"

    def test_default_structure_keys(self):
        plan = _extract_service_plan_structure("")
        assert "urgent_actions" in plan
        assert "defense_strategies" in plan
        assert "service_phases" in plan
        assert "fee_structure" in plan


class TestBuildServiceRequestMessage:
    """Tests for _build_service_request_message."""

    def test_message_contains_all_input_sections(self):
        facts = {"crime_type": "盗窃", "amount": "5000元"}
        laws = [{"name": "刑法第264条", "content": "盗窃罪"}]
        risk = {"predicted_sentence_range": "3年以下有期徒刑"}
        history = [{"agent": "RiskAssessor", "summary": "风险评估完成"}]

        message = _build_service_request_message(
            facts_structured=facts,
            applied_laws=laws,
            risk_assessment=risk,
            conversation_history=history,
        )

        assert "【案件基本信息】" in message
        assert "盗窃" in message
        assert "【适用法律法规】" in message
        assert "刑法第264条" in message
        assert "【风险评估结果】" in message
        assert "3年以下有期徒刑" in message
        assert "【对话历史摘要】" in message
        assert "RiskAssessor" in message
        assert "【服务方案请求】" in message

    def test_message_without_conversation_history(self):
        message = _build_service_request_message(
            facts_structured={},
            applied_laws=[],
            risk_assessment={},
            conversation_history=[],
        )
        assert "【对话历史摘要】" not in message
        assert "【服务方案请求】" in message

    def test_message_limits_history_to_last_five(self):
        history = [{"agent": f"Agent{i}", "summary": f"Summary{i}"} for i in range(10)]
        message = _build_service_request_message(
            facts_structured={},
            applied_laws=[],
            risk_assessment={},
            conversation_history=history,
        )
        assert "Agent5" in message
        assert "Agent9" in message
        # Earlier entries should not be present
        assert "Agent0" not in message
        assert "Agent4" not in message
