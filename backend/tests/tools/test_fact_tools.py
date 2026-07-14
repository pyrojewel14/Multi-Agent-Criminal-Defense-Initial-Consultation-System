"""Unit tests for app/tools/fact_tools.py.

Covers:
- ``extract_case_facts`` LangChain tool behaviour: all parameters, defaults, None vs empty.
"""

from app.tools.fact_tools import extract_case_facts


# ---------------------------------------------------------------------------
# extract_case_facts – defaults and None handling
# ---------------------------------------------------------------------------


class TestExtractCaseFactsDefaults:
    def test_all_none_returns_empty_or_none_dict(self):
        """With no arguments, default values should be returned and lists should be empty."""
        result = extract_case_facts.invoke({})
        assert result["incident_time"] is None
        assert result["incident_location"] is None
        assert result["parties"] == []
        assert result["behavior_sequence"] == []
        assert result["consequence"] is None
        assert result["evidence_mentioned"] == []
        assert result["arrest_status"] is None
        assert result["surrender"] is None
        assert result["victim_forgiveness"] is None
        assert result["prior_record"] is None

    def test_invoke_via_dict_arguments(self):
        """LangChain tools support .invoke({...})."""
        out = extract_case_facts.invoke({
            "incident_time": "2024-01-01",
            "consequence": "轻伤",
        })
        assert out["incident_time"] == "2024-01-01"
        assert out["consequence"] == "轻伤"
        assert out["parties"] == []


# ---------------------------------------------------------------------------
# extract_case_facts – per-field tests
# ---------------------------------------------------------------------------


def _invoke(**kwargs):
    """Helper: call the LangChain tool with the given kwargs."""
    return extract_case_facts.invoke(kwargs)


class TestExtractCaseFactsPerField:
    def test_incident_time(self):
        out = _invoke(incident_time="2024-05-06")
        assert out["incident_time"] == "2024-05-06"

    def test_incident_location(self):
        out = _invoke(incident_location="北京市朝阳区")
        assert out["incident_location"] == "北京市朝阳区"

    def test_parties(self):
        parties = [
            {"role": "suspect", "name": "[已脱敏]-1"},
            {"role": "victim", "name": "[已脱敏]-2"},
        ]
        out = _invoke(parties=parties)
        assert out["parties"] == parties

    def test_parties_none_normalizes_to_empty(self):
        out = _invoke(parties=None)
        assert out["parties"] == []

    def test_behavior_sequence(self):
        seq = [{"actor": "张某", "action": "殴打"}]
        out = _invoke(behavior_sequence=seq)
        assert out["behavior_sequence"] == seq

    def test_behavior_sequence_none_normalizes_to_empty(self):
        out = _invoke(behavior_sequence=None)
        assert out["behavior_sequence"] == []

    def test_consequence(self):
        out = _invoke(consequence="轻伤二级")
        assert out["consequence"] == "轻伤二级"

    def test_evidence_mentioned(self):
        ev = [{"type": "监控", "description": "案发现场监控"}]
        out = _invoke(evidence_mentioned=ev)
        assert out["evidence_mentioned"] == ev

    def test_evidence_mentioned_none_normalizes_to_empty(self):
        out = _invoke(evidence_mentioned=None)
        assert out["evidence_mentioned"] == []

    def test_arrest_status(self):
        out = _invoke(arrest_status="已刑事拘留")
        assert out["arrest_status"] == "已刑事拘留"

    def test_surrender_true(self):
        out = _invoke(surrender=True)
        assert out["surrender"] is True

    def test_surrender_false(self):
        out = _invoke(surrender=False)
        assert out["surrender"] is False

    def test_victim_forgiveness_true(self):
        out = _invoke(victim_forgiveness=True)
        assert out["victim_forgiveness"] is True

    def test_victim_forgiveness_false(self):
        out = _invoke(victim_forgiveness=False)
        assert out["victim_forgiveness"] is False

    def test_prior_record_true(self):
        out = _invoke(prior_record=True)
        assert out["prior_record"] is True

    def test_prior_record_false(self):
        out = _invoke(prior_record=False)
        assert out["prior_record"] is False


# ---------------------------------------------------------------------------
# extract_case_facts – full populated payload
# ---------------------------------------------------------------------------


class TestExtractCaseFactsFullPayload:
    def test_all_fields_set(self):
        payload = {
            "incident_time": "2024-01-15",
            "incident_location": "上海市浦东新区",
            "parties": [
                {"role": "suspect", "name": "[已脱敏]-A"},
                {"role": "victim", "name": "[已脱敏]-B"},
            ],
            "behavior_sequence": [
                {"actor": "A", "action": "持刀威胁"},
                {"actor": "A", "action": "索要财物"},
            ],
            "consequence": "被害人受轻伤",
            "evidence_mentioned": [
                {"type": "监控视频", "description": "案发现场监控记录事件经过"},
                {"type": "鉴定意见", "description": "伤情鉴定意见书"},
            ],
            "arrest_status": "已逮捕",
            "surrender": False,
            "victim_forgiveness": True,
            "prior_record": False,
        }
        out = extract_case_facts.invoke(payload)
        assert out == payload

    def test_result_dict_keys(self):
        """The returned dict should always contain all 10 expected keys."""
        out = extract_case_facts.invoke({})
        assert set(out.keys()) == {
            "incident_time",
            "incident_location",
            "parties",
            "behavior_sequence",
            "consequence",
            "evidence_mentioned",
            "arrest_status",
            "surrender",
            "victim_forgiveness",
            "prior_record",
        }


# ---------------------------------------------------------------------------
# extract_case_facts – tool metadata
# ---------------------------------------------------------------------------


class TestExtractCaseFactsToolMetadata:
    def test_tool_has_name(self):
        """LangChain tools expose a 'name' attribute."""
        assert extract_case_facts.name == "extract_case_facts"

    def test_tool_has_description(self):
        """The tool description should mention fact extraction."""
        assert extract_case_facts.description
        assert "事实" in extract_case_facts.description or "提取" in extract_case_facts.description

    def test_tool_args_schema_present(self):
        """LangChain tools expose a pydantic args_schema describing parameters."""
        schema = extract_case_facts.args_schema
        # Schema is a pydantic model class
        assert schema is not None
        assert not isinstance(schema, dict)
        # The schema should declare the major fact fields
        field_names = set(schema.model_fields.keys())
        assert "incident_time" in field_names
        assert "incident_location" in field_names
        assert "parties" in field_names
        assert "behavior_sequence" in field_names
        assert "consequence" in field_names
        assert "evidence_mentioned" in field_names
        assert "arrest_status" in field_names
        assert "surrender" in field_names
        assert "victim_forgiveness" in field_names
        assert "prior_record" in field_names


# ---------------------------------------------------------------------------
# extract_case_facts – is a LangChain tool (duck-typing)
# ---------------------------------------------------------------------------


class TestExtractCaseFactsIsTool:
    def test_has_invoke(self):
        """LangChain tools expose .invoke() to run the underlying function."""
        assert callable(getattr(extract_case_facts, "invoke", None))

    def test_has_run(self):
        """Legacy LangChain tools expose .run() for string-based invocation."""
        assert callable(getattr(extract_case_facts, "run", None))

    def test_invoke_returns_dict(self):
        """A basic invoke call should return a dict (not a string or other type)."""
        out = extract_case_facts.invoke({"incident_time": "2024-06-01"})
        assert isinstance(out, dict)
        assert out["incident_time"] == "2024-06-01"
