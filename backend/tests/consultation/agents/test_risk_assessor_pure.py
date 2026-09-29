from app.consultation.agents.risk_assessor import (
    _extract_key_risks,
)


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
