"""Full text and annotation safety boundary regressions."""

import copy
import json
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.runnables import RunnableConfig

from app.consultation.agents.legal_research import LegalToolRegistry
from app.knowledge import law_knowledge as knowledge
from app.knowledge.law_retrieval import _verify_and_enrich_with_json, search_laws_by_keyword


@pytest.fixture
def full(monkeypatch):
    monkeypatch.setenv("LAW_KNOWLEDGE_PROFILE", "full")
    monkeypatch.setenv("LAW_FULL_DEMO_ANNOTATIONS", "off")
    knowledge.load_criminal_law_data.cache_clear()
    yield knowledge.load_criminal_law_data()
    knowledge.load_criminal_law_data.cache_clear()


def test_full_profile_loads_body_without_promoting_annotations(full):
    index = knowledge._build_article_index(full)
    assert len(index) == 505
    assert not knowledge.is_article_in_force(index["第199条"])
    assert not knowledge.is_lawref_eligible(index["第133条之一"])
    assert knowledge.is_lawref_eligible(index["第264条"])
    assert index["第1条"]["elements"] == []
    assert index["第1条"]["base_sentence"] == ""
    assert index["第133条之一"]["annotations"]["elements"]["review_status"] == "pending"


@pytest.mark.asyncio
async def test_text_search_and_read_outside_six_has_no_coverage_elements(full):
    with patch("app.knowledge.law_retrieval.search_laws_by_rag", new_callable=AsyncMock, return_value=[]):
        registry = LegalToolRegistry({}, None, full)
        result = await registry._search_laws("第133条之一")
        assert result["candidates"][0]["article_id"] == "第133条之一"
        body = (await registry._get_article("第133条之一"))["article"]
        assert "醉酒驾驶机动车" in body["content"]
        assert body["required_elements"] == []
        assert body["coverage_eligible"] is False
        elements = await registry._search_elements("第133条之一")
        assert elements["required_elements"] == []
        assert elements["review_status"] == "pending"
        assert (await registry._get_article("第199条"))["article"] is None


@pytest.mark.asyncio
async def test_general_and_cross_reference_body_readable(full):
    for query, wanted in [("第1条", "第一条"), ("第149条", "第一百四十九条"), ("第277条", "第二百七十七条")]:
        laws = await search_laws_by_keyword({"behavior_sequence": [query]}, full)
        assert laws[0]["article_number"] == wanted
        assert laws[0]["required_elements"] == []
    assert await search_laws_by_keyword({"behavior_sequence": ["第199条"]}, full) == []


def test_full_rag_mismatch_does_not_inherit_elements(full):
    index = knowledge._build_article_index(full)
    article = index["第264条"]
    hit = {"article_number": "第264条", "content": article["content"], "corpus_sha256": "sha256:wrong"}
    enriched = _verify_and_enrich_with_json([hit], index)
    assert enriched[0]["data_source"] == "rag_unverified"
    assert enriched[0].get("required_elements", []) == []


def test_profile_change_does_not_reuse_old_cache(monkeypatch):
    monkeypatch.setenv("LAW_KNOWLEDGE_PROFILE", "snapshot")
    knowledge.load_criminal_law_data.cache_clear()
    assert len(knowledge._build_article_index(knowledge.load_criminal_law_data())) == 6
    monkeypatch.setenv("LAW_KNOWLEDGE_PROFILE", "full")
    assert len(knowledge._build_article_index(knowledge.load_criminal_law_data())) == 505


def test_invalid_profile_fails_closed(monkeypatch):
    monkeypatch.setenv("LAW_KNOWLEDGE_PROFILE", "ful")
    knowledge.load_criminal_law_data.cache_clear()
    with pytest.raises(knowledge.LawKnowledgeDataError):
        knowledge.load_criminal_law_data()


def test_tampered_pending_annotations_fail_validation(full):
    from app.knowledge.full_law_corpus import validate_full_corpus

    changed = copy.deepcopy(full)
    article = changed["chapters"][0]["articles"][0]
    article["elements"] = [{"key": "invented", "name": "invented"}]
    with pytest.raises(knowledge.LawKnowledgeDataError):
        validate_full_corpus(changed)


def test_alternative_branches_and_reference_not_flattened(full):
    index = knowledge._build_article_index(full)
    danger = index["第133条之一"]
    assert len(danger["text_structure"]["enumerated_items"]) == 4
    assert danger["elements"] == []
    assert "第一百四十条" in index["第149条"]["text_structure"]["article_references"]
    assert index["第169条"]["annotations"]["title"]["review_status"] == "source_checked"


@pytest.mark.asyncio
async def test_real_lawref_text_only_final_cannot_create_coverage(full):
    from app.consultation.agents.fact_digger import _analyze_coverage
    from app.consultation.agents.law_ref import law_ref_node
    from tests.factories import make_consultation_state

    responses = [
        {"content": "", "tool_calls": [{"name": "search_laws", "args": {"query": "第133条之一"}}]},
        {"content": "", "tool_calls": [{"name": "get_article", "args": {"article_id": "第133条之一"}}]},
        {
            "content": json.dumps(
                {"article_ids": ["第133条之一"], "matched_elements": {"第133条之一": []}, "confidence": "low"}
            ),
            "tool_calls": [],
        },
    ]
    with (
        patch(
            "app.consultation.agents.legal_research.llm_gateway.generate_with_tools",
            new_callable=AsyncMock,
            side_effect=responses,
        ),
        patch("app.knowledge.law_retrieval.search_laws_by_rag", new_callable=AsyncMock, return_value=[]),
    ):
        state = await law_ref_node(
            make_consultation_state(facts_structured={"behavior_sequence": ["醉酒驾驶机动车"]}, applied_laws=[])
        )
    assert "applied_laws" in state and "law_text_candidates" in state and "law_search_status" in state
    assert "facts_structured" in state
    assert state["applied_laws"] == []
    assert "醉酒驾驶机动车" in state["law_text_candidates"][0]["content"]
    assert state["law_search_status"] == "text_only"
    coverage = await _analyze_coverage(state["facts_structured"], state["applied_laws"])
    assert coverage["total_elements"] == 0
    assert coverage["degraded"] is True


@pytest.mark.asyncio
async def test_full_profile_six_fixture_keeps_real_coverage_path(full):
    from app.consultation.agents.fact_digger import _analyze_coverage
    from app.consultation.agents.law_ref import law_ref_node
    from tests.factories import make_consultation_state

    responses = [
        {"content": "", "tool_calls": [{"name": "search_laws", "args": {"query": "第264条"}}]},
        {"content": "", "tool_calls": [{"name": "get_article", "args": {"article_id": "第264条"}}]},
        {
            "content": json.dumps(
                {"article_ids": ["第264条"], "matched_elements": {"第264条": []}, "confidence": "low"}
            ),
            "tool_calls": [],
        },
    ]
    with (
        patch(
            "app.consultation.agents.legal_research.llm_gateway.generate_with_tools",
            new_callable=AsyncMock,
            side_effect=responses,
        ),
        patch("app.knowledge.law_retrieval.search_laws_by_rag", new_callable=AsyncMock, return_value=[]),
    ):
        state = await law_ref_node(
            make_consultation_state(facts_structured={"behavior_sequence": ["盗窃"]}, applied_laws=[])
        )
    assert "law_search_status" in state and "applied_laws" in state and "facts_structured" in state
    assert state["law_search_status"] == "success"
    assert state["applied_laws"][0]["required_elements"]
    assert (await _analyze_coverage(state["facts_structured"], state["applied_laws"]))["total_elements"] > 0


def test_matching_rag_full_body_inherits_only_core_elements(full):
    index = knowledge._build_article_index(full)
    hits = [
        {
            "article_number": index[n]["article_number"],
            "content": index[n]["article_number"] + " " + index[n]["content"],
            "corpus_sha256": index[n]["corpus_sha256"],
            "corpus_version": index[n]["corpus_version"],
        }
        for n in ["第264条", "第133条之一"]
    ]
    laws = _verify_and_enrich_with_json(hits, index)
    assert laws[0]["data_source"] == "rag_verified"
    assert laws[0]["required_elements"]
    assert laws[1]["data_source"] == "text_only"
    assert laws[1]["required_elements"] == []
    hits[0]["content"] = "第264条 invented content"
    assert _verify_and_enrich_with_json(hits, index)[0]["data_source"] == "rag_unverified"


def test_import_reproducible_and_keeps_pending_proposals(full):
    from app.knowledge.full_law_corpus import prepare_full_corpus, review_queue

    candidate = copy.deepcopy(full)
    candidate["metadata"]["dataset_version"] = "candidate"
    for chapter in candidate["chapters"]:
        for a in chapter["articles"]:
            for name, field in [
                ("title", "title"),
                ("charges", "charge_tags"),
                ("elements", "elements"),
                ("penalty", "base_sentence"),
            ]:
                a[field] = a["annotations"][name]["value"]
    prepared = prepare_full_corpus(candidate, "sha256:test")
    assert prepared == prepare_full_corpus(candidate, "sha256:test")
    queue = review_queue(prepared)
    assert len(queue) == 505
    assert any(
        row["article_number"] == "第一百三十三条之一" and row["has_enumerated_items"] and "elements" in row["fields"]
        for row in queue
    )


def test_derived_arabic_suffix_resolves_same_body(full):
    index = knowledge._build_article_index(full)
    assert knowledge._normalize_article_number("第133条之1") == "第133条之一"
    assert "第133条之一" in index


@pytest.mark.parametrize("mutation", ["source", "derived"])
def test_full_validator_rejects_wrong_source_and_wrong_derived_number(full, mutation):
    from app.knowledge.full_law_corpus import validate_full_corpus

    broken = copy.deepcopy(full)
    if mutation == "source":
        for c in broken["chapters"]:
            for a in c["articles"]:
                if a["article_number"] == "第一条":
                    a["official_text_source"] = "wrong"
    else:
        for c in broken["chapters"]:
            for a in c["articles"]:
                if a["article_number"] == "第一百三十三条之一":
                    a["article_number"] = "第九百九十九条之一"
        broken["metadata"]["coverage"] = [a["article_number"] for c in broken["chapters"] for a in c["articles"]]
    with pytest.raises(knowledge.LawKnowledgeDataError):
        validate_full_corpus(broken)


def test_retry_reset_discards_previous_text_only_candidates():
    from app.consultation.workflow import _reset_degraded_retry_state
    from tests.factories import make_consultation_state

    state = make_consultation_state(workflow_status="degraded", law_text_candidates=[{"article_number": "第133条之一"}])
    _reset_degraded_retry_state(state)
    assert "law_text_candidates" in state
    assert state["law_text_candidates"] == []


@pytest.mark.asyncio
async def test_text_only_coverage_routes_to_human_review_without_fact_questions():
    from app.consultation.agents.fact_digger import fact_coverage_node
    from app.consultation.workflow import check_facts_sufficient, human_review_node
    from tests.factories import make_consultation_state

    state = make_consultation_state(
        facts_structured={"behavior_sequence": ["醉驾"]},
        law_search_status="text_only",
        law_text_candidates=[{"article_number": "第133条之一"}],
        applied_laws=[],
        pending_questions=[],
    )
    state = await fact_coverage_node(state)
    assert "facts_coverage_rate" in state and "fact_law_termination_reason" in state and "pending_questions" in state
    assert state["facts_coverage_rate"] == 0.0
    assert state["fact_law_termination_reason"] == "annotation_review_required"
    assert check_facts_sufficient(state) == "degraded"
    assert state["pending_questions"] == []
    state = await human_review_node(state)
    assert "final_output" in state and "awaiting_lawyer_review" in state
    assert "标注" in state["final_output"]
    assert state["awaiting_lawyer_review"] is True


@pytest.mark.asyncio
async def test_compiled_graph_text_only_goes_to_review_and_stays_waiting():
    from app.consultation.workflow import ConsultationOrchestrator
    from tests.factories import make_consultation_state

    orchestrator = ConsultationOrchestrator()
    compiled = orchestrator._compiled_graph()
    config: RunnableConfig = {"configurable": {"thread_id": "full-text-only-graph"}}
    state = make_consultation_state(
        session_id="full-text-only-graph",
        consent_given=True,
        law_search_status="text_only",
        applied_laws=[],
        law_text_candidates=[{"article_number": "第133条之一"}],
        facts_structured={"behavior_sequence": ["醉驾"]},
    )
    await compiled.aupdate_state(config, state, as_node="law_ref")
    result = await compiled.ainvoke(None, config)
    assert result["current_agent"] == "HumanReview"
    assert result["awaiting_lawyer_review"] is True
    assert result["fact_law_termination_reason"] == "annotation_review_required"
    assert result["facts_coverage_rate"] == 0.0
    assert result["law_text_candidates"][0]["article_number"] == "第133条之一"
    assert result["risk_assessment"] is None


def test_copied_public_backend_loads_without_supplier_directory(tmp_path):
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    backend = Path(knowledge.__file__).resolve().parents[2]
    destination = tmp_path / "backend"
    shutil.copytree(backend / "app", destination / "app", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    data = destination / "data/law_knowledge"
    data.mkdir(parents=True)
    for name in ["criminal_law_chapters.json", "criminal_law_full.json", "criminal_law_demo_annotations.json"]:
        shutil.copy2(backend / "data/law_knowledge" / name, data / name)
    env = {**os.environ, "PYTHONPATH": str(destination), "LAW_KNOWLEDGE_PROFILE": "full"}
    env.pop("LAW_FULL_CORPUS_PATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from app.knowledge.law_knowledge import load_criminal_law_data,_build_article_index; d=load_criminal_law_data(); i=_build_article_index(d); assert len(i)==505; assert '醉酒驾驶机动车' in i['第133条之一']['content']; print('portable-full-text-ok')",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "portable-full-text-ok" in result.stdout


@pytest.mark.asyncio
async def test_text_only_rag_keeps_retrieval_method_in_tool_observation(full):
    index = knowledge._build_article_index(full)
    a = index["第133条之一"]
    hit = {"article_number": a["article_number"], "content": a["article_number"] + " " + a["content"], "corpus_sha256": a["corpus_sha256"], "corpus_version": a["corpus_version"], "retrieval_method": "vector_index"}
    with patch("app.knowledge.law_retrieval.search_laws_by_rag", new_callable=AsyncMock, return_value=[hit]):
        result = await LegalToolRegistry({}, "user", full)._search_laws("醉酒驾驶机动车")
    candidate = next(c for c in result["candidates"] if c["article_id"] == "第133条之一")
    assert candidate["source"] == "text_only"
    assert candidate["retrieval_method"] == "vector_index"
