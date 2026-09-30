"""Demo 放行须贯通读取、LawRef 和覆盖计算，不能仅改报告状态。"""

import copy
import json
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.runnables import RunnableConfig

from app.consultation.agents.legal_research import LegalToolRegistry
from app.knowledge import law_knowledge as knowledge


@pytest.fixture
def demo(monkeypatch):
    monkeypatch.setenv("LAW_KNOWLEDGE_PROFILE", "full")
    monkeypatch.setenv("LAW_FULL_DEMO_ANNOTATIONS", "on")
    knowledge.load_criminal_law_data.cache_clear()
    yield knowledge.load_criminal_law_data()
    knowledge.load_criminal_law_data.cache_clear()


@pytest.mark.asyncio
async def test_full_demo_profile_releases_reviewed_demo_into_tools(demo):
    registry = LegalToolRegistry({}, None, demo)
    await registry._search_laws("第134条")
    body = (await registry._get_article("第134条"))["article"]
    assert body["coverage_eligible"] is True
    assert "强令、组织他人违章冒险作业罪" in body["title"]
    assert body["required_elements"]
    assert body["annotation_usage"] == "demo"
    assert knowledge._build_article_index(demo)["第134条"]["annotations"]["elements"]["legal_review_status"] == "pending"
    assert sum(knowledge.is_lawref_eligible(a) for a in knowledge._build_article_index(demo).values()) == 38


def test_full_profile_needs_no_additional_review_opt_in(monkeypatch):
    monkeypatch.setenv("LAW_KNOWLEDGE_PROFILE", "full")
    monkeypatch.delenv("LAW_FULL_DEMO_ANNOTATIONS", raising=False)
    data = knowledge.load_criminal_law_data()
    assert knowledge.is_lawref_eligible(knowledge._build_article_index(data)["第133条之一"])


def test_demo_keeps_general_repealed_and_unreviewed_out(demo):
    index = knowledge._build_article_index(demo)
    for num in ("第1条", "第17条", "第149条", "第199条", "第123条"):
        assert not knowledge.is_lawref_eligible(index[num])
        assert index[num]["elements"] == []
    assert knowledge.is_lawref_eligible(index["第133条之一"])
    assert all("demo_condition" not in e["key"] for e in index["第264条"]["elements"])


def test_switching_demo_off_restores_six_without_stale_cache(demo, monkeypatch):
    monkeypatch.setenv("LAW_FULL_DEMO_ANNOTATIONS", "off")
    raw = knowledge.load_criminal_law_data()
    assert sum(knowledge.is_lawref_eligible(a) for a in knowledge._build_article_index(raw).values()) == 6
    assert knowledge._build_article_index(raw)["第133条之一"]["elements"] == []


@pytest.mark.asyncio
async def test_real_lawref_demo_has_denominator_and_no_annotation_review_block(demo):
    from app.consultation.agents.fact_digger import _analyze_coverage, fact_coverage_node
    from app.consultation.agents.law_ref import law_ref_node
    from app.consultation.workflow import check_facts_sufficient
    from tests.factories import make_consultation_state

    elements = knowledge._build_article_index(demo)["第133条之一"]["elements"]
    names = [e["name"] for e in elements]
    responses = [
        {"content": "", "tool_calls": [{"name": "search_laws", "args": {"query": "第133条之一"}}]},
        {"content": "", "tool_calls": [{"name": "get_article", "args": {"article_id": "第133条之一"}}]},
        {"content": json.dumps({"article_ids": ["第133条之一"], "matched_elements": {"第133条之一": names}, "confidence": "medium"}), "tool_calls": []},
    ]
    with patch("app.consultation.agents.legal_research.llm_gateway.generate_with_tools", new_callable=AsyncMock, side_effect=responses):
        state = await law_ref_node(make_consultation_state(user_id="", facts_structured={"behavior_sequence": ["在道路上醉酒驾驶机动车"]}))
    assert "law_search_status" in state and "applied_laws" in state and "facts_structured" in state
    assert state["law_search_status"] == "success"
    assert state["applied_laws"][0]["annotation_usage"] == "demo"
    coverage = await _analyze_coverage(state["facts_structured"], state["applied_laws"])
    assert coverage["total_elements"] > 0
    assert coverage["coverage_rate"] == 1.0
    assert coverage["degraded"] is False
    law_state = copy.deepcopy(state)
    with patch("app.consultation.agents.fact_digger.llm_gateway.generate", new_callable=AsyncMock, return_value="示例事实摘要"):
        state = await fact_coverage_node(state)
    assert state.get("fact_law_termination_reason") != "annotation_review_required"
    assert check_facts_sufficient(state) == "complete"
    from app.consultation.workflow import ConsultationOrchestrator

    compiled = ConsultationOrchestrator()._compiled_graph()
    config: RunnableConfig = {"configurable": {"thread_id": "demo-annotations-release"}}
    await compiled.aupdate_state(config, law_state, as_node="law_ref")
    with patch("app.consultation.agents.fact_digger.llm_gateway.generate", new_callable=AsyncMock, return_value="示例事实摘要"):
        result = await compiled.ainvoke(None, config, interrupt_before=["risk_assessor"])
    assert (await compiled.aget_state(config)).next == ("risk_assessor",)
    assert result["facts_coverage_rate"] == 1.0
    assert result.get("awaiting_lawyer_review") is not True


@pytest.mark.asyncio
async def test_demo_missing_branch_stays_missing_despite_overlapping_keywords(demo):
    from app.consultation.agents.fact_digger import _analyze_coverage
    from app.consultation.agents.law_ref import _build_applied_laws_from_structured

    a = knowledge._build_article_index(demo)["第133条之一"]
    law = {**a, "data_source": "json_keyword"}
    applied = _build_applied_laws_from_structured([
        {"article_number": "第133条之一", "charge_name": a["title"], "elements_matched": [], "elements_missing": [], "probability": "low", "base_sentence": a["base_sentence"]}
    ], [law])
    result = await _analyze_coverage({"behavior_sequence": ["有严重后果、情节恶劣"]}, applied)
    assert result["total_elements"] > 0
    assert result["covered_elements"] == 0
    assert result["missing_elements"]


def test_mismatched_demo_asset_fails_without_promoting(demo, monkeypatch, tmp_path):
    from pathlib import Path
    path = Path(knowledge.__file__).resolve().parents[2] / "data/law_knowledge/criminal_law_demo_annotations.json"
    asset = json.loads(path.read_text())
    asset["corpus_sha256"] = "sha256:stale"
    bad = tmp_path / "annotations.json"
    bad.write_text(json.dumps(asset))
    monkeypatch.setenv("LAW_DEMO_ANNOTATIONS_PATH", str(bad))
    with pytest.raises(knowledge.LawKnowledgeDataError):
        knowledge.load_criminal_law_data()
