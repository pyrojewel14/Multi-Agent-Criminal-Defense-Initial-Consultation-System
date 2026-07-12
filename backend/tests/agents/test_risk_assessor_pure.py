import pytest

from app.agents.risk_assessor import (
    _extract_key_risks,
    _format_factor_list,
    _parse_fallback_assessment,
    format_risk_assessment_report,
)


class TestParseFallbackAssessment:
    """Tests for _parse_fallback_assessment."""

    def test_returns_default_structure_with_all_fields(self):
        result = _parse_fallback_assessment("some non-JSON text")

        assert "predicted_sentence_range" in result
        assert "mitigating_factors" in result
        assert "aggravating_factors" in result
        assert "compulsory_measure_risk" in result
        assert "evidence_risk_points" in result
        assert "procedure_risks" in result

    def test_default_predicted_sentence_range(self):
        result = _parse_fallback_assessment("anything")
        assert result["predicted_sentence_range"] == "待评估"

    def test_default_factors_are_empty_lists(self):
        result = _parse_fallback_assessment("anything")
        assert result["mitigating_factors"] == []
        assert result["aggravating_factors"] == []
        assert result["evidence_risk_points"] == []
        assert result["procedure_risks"] == []

    def test_compulsory_measure_risk_structure(self):
        result = _parse_fallback_assessment("anything")
        cm = result["compulsory_measure_risk"]
        assert cm["detention_status"] == "待确认"
        assert cm["bail_possibility"] == "待评估"
        assert cm["measure_change_space"] == "待评估"
        assert cm["prolonged_detention_risk"] == "待评估"

    def test_ignores_input_content(self):
        """The fallback function returns the same default regardless of input."""
        r1 = _parse_fallback_assessment("foo")
        r2 = _parse_fallback_assessment("bar")
        assert r1 == r2


class TestExtractKeyRisks:
    """Tests for _extract_key_risks."""

    def test_prolonged_detention_risk_high(self):
        assessment = {
            "compulsory_measure_risk": {"prolonged_detention_risk": "high"},
            "evidence_risk_points": [],
            "procedure_risks": [],
        }
        risks = _extract_key_risks(assessment)
        assert "超期羁押风险" in risks

    def test_prolonged_detention_risk_not_high(self):
        assessment = {
            "compulsory_measure_risk": {"prolonged_detention_risk": "medium"},
            "evidence_risk_points": [],
            "procedure_risks": [],
        }
        risks = _extract_key_risks(assessment)
        assert "超期羁押风险" not in risks

    def test_evidence_risk_points_count(self):
        assessment = {
            "compulsory_measure_risk": {},
            "evidence_risk_points": [{"gap": "A"}, {"gap": "B"}],
            "procedure_risks": [],
        }
        risks = _extract_key_risks(assessment)
        assert "存在2个证据风险点" in risks

    def test_no_evidence_risk_points(self):
        assessment = {
            "compulsory_measure_risk": {},
            "evidence_risk_points": [],
            "procedure_risks": [],
        }
        risks = _extract_key_risks(assessment)
        assert all("证据风险点" not in r for r in risks)

    def test_high_severity_procedure_risk(self):
        assessment = {
            "compulsory_measure_risk": {},
            "evidence_risk_points": [],
            "procedure_risks": [{"severity": "high", "type": "诉讼时效"}],
        }
        risks = _extract_key_risks(assessment)
        assert "高风险: 诉讼时效" in risks

    def test_low_severity_procedure_risk_not_included(self):
        assessment = {
            "compulsory_measure_risk": {},
            "evidence_risk_points": [],
            "procedure_risks": [{"severity": "low", "type": "管辖权"}],
        }
        risks = _extract_key_risks(assessment)
        assert all("高风险" not in r for r in risks)

    def test_empty_assessment(self):
        risks = _extract_key_risks({})
        assert risks == []

    def test_missing_optional_fields(self):
        assessment = {
            "compulsory_measure_risk": {"prolonged_detention_risk": "high"},
        }
        risks = _extract_key_risks(assessment)
        assert "超期羁押风险" in risks


class TestFormatRiskAssessmentReport:
    """Tests for format_risk_assessment_report."""

    def test_format_complete_assessment(self):
        assessment = {
            "predicted_sentence_range": "3-5年有期徒刑",
            "mitigating_factors": ["自首", "赔偿"],
            "aggravating_factors": ["累犯"],
            "compulsory_measure_risk": {
                "detention_status": "已羁押",
                "bail_possibility": "中等",
                "measure_change_space": "有限",
                "prolonged_detention_risk": "low",
            },
            "evidence_risk_points": [
                {"gap": "证人证言矛盾", "exclusion_possibility": "低"},
            ],
            "procedure_risks": [
                {"severity": "medium", "type": "管辖权", "description": "管辖权异议"},
            ],
        }
        report = format_risk_assessment_report(assessment)
        assert "3-5年有期徒刑" in report
        assert "自首" in report
        assert "累犯" in report
        assert "已羁押" in report
        assert "证人证言矛盾" in report
        assert "管辖权" in report

    def test_format_with_empty_fields(self):
        assessment = {
            "predicted_sentence_range": "待评估",
            "mitigating_factors": [],
            "aggravating_factors": [],
            "compulsory_measure_risk": {},
            "evidence_risk_points": [],
            "procedure_risks": [],
        }
        report = format_risk_assessment_report(assessment)
        assert "待评估" in report
        assert "无" in report
        assert "未发现明显证据风险点" in report
        assert "未发现明显程序风险" in report

    def test_report_contains_disclaimer(self):
        assessment = {}
        report = format_risk_assessment_report(assessment)
        assert "智能辅助生成" in report

    def test_report_has_section_headers(self):
        assessment = {}
        report = format_risk_assessment_report(assessment)
        assert "量刑预测" in report
        assert "强制措施风险" in report
        assert "证据风险点" in report
        assert "程序风险" in report


class TestFormatFactorList:
    """Tests for _format_factor_list."""

    def test_format_list_of_factors(self):
        factors = ["自首", "赔偿谅解", "初犯"]
        result = _format_factor_list(factors)
        assert "- 自首" in result
        assert "- 赔偿谅解" in result
        assert "- 初犯" in result

    def test_empty_list_returns_none(self):
        result = _format_factor_list([])
        assert result == "    无"

    def test_single_factor(self):
        result = _format_factor_list(["认罪认罚"])
        assert "- 认罪认罚" in result

    def test_indentation(self):
        result = _format_factor_list(["A", "B"])
        lines = result.split("\n")
        for line in lines:
            assert line.startswith("    ")
