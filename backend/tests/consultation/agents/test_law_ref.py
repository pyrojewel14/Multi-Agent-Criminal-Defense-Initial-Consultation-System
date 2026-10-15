"""Integration tests for the LawRef Agent node."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.consultation.agents import law_ref
from app.consultation.agents.law_ref import (
    _build_applied_laws_from_structured,
    _build_element_to_law_mapping,
    _load_law_extract_prompt,
    extract_structured_laws,
    law_ref_node,
)
from app.consultation.agents.legal_research import LegalToolRegistry
from app.knowledge.law import corpus as law_knowledge
from app.knowledge.law.corpus import load_criminal_law_data
from app.knowledge.law.identifiers import _cn_to_arabic, _extract_article_number_from_text, _normalize_article_number
from app.knowledge.law.repository import _build_article_index
from tests.factories import make_consultation_state
from tests.knowledge.law.helpers import keyword_laws, patch_law_keyword, patch_law_recall, verified_laws


@pytest.fixture(autouse=True)
def scripted_law_tool_decisions():
    """为现有节点回归提供逐轮观察的确定性模型决策。"""

    async def decide(_system, message, _tools, **_kwargs):
        payload = json.loads(message)
        observations = [json.loads(item.content) for item in _kwargs.get("message_history", []) if item.type == "tool"]
        if not observations and not payload.get("read_articles"):
            facts = json.loads(payload["facts"])
            behavior = facts.get("behavior_sequence") or []
            term = behavior[0] if isinstance(behavior, list) else behavior
            if isinstance(term, dict):
                term = next((value for value in term.values() if isinstance(value, str)), "")
            return {
                "content": "",
                "tool_calls": [{"name": "search_laws", "args": {"query": str(term)}}],
                "has_tool_call": True,
            }
        if len(observations) == 1:
            candidates = observations[0]["result"].get("candidates", [])
            if candidates:
                return {
                    "content": "",
                    "tool_calls": [{"name": "get_article", "args": {"article_id": candidates[0]["article_id"]}}],
                    "has_tool_call": True,
                }
        if len(observations) >= 2 or payload.get("read_articles"):
            article = (
                payload["read_articles"][0]
                if payload.get("read_articles")
                else observations[-1]["result"].get("article")
            )
            if article:
                article_id = article["article_id"]
                elements = article["required_elements"]
                matched = [elements[0]["name"] if isinstance(elements[0], dict) else elements[0]] if elements else []
                return {
                    "content": json.dumps(
                        {
                            "article_ids": [article_id],
                            "matched_elements": {article_id: matched},
                            "confidence": "medium",
                        },
                        ensure_ascii=False,
                    ),
                    "tool_calls": [],
                    "has_tool_call": False,
                }
        return {"content": "{}", "tool_calls": [], "has_tool_call": False}

    with patch("app.consultation.agents.legal_research.llm_gateway.generate_with_tools", new=decide):
        yield


# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def explicit_snapshot_contract(monkeypatch):
    """This module tests the historical six-article and user-index contract."""
    monkeypatch.setenv("LAW_KNOWLEDGE_PROFILE", "snapshot")


# Helper: build law_data in the format load_criminal_law_data returns
# ---------------------------------------------------------------------------


def _make_law_data():
    """Return law_data dict with chapters structure matching load_criminal_law_data output."""
    return {
        "chapters": [
            {
                "chapter": "侵犯财产罪",
                "articles": [
                    {
                        "article_number": "第二百六十四条",
                        "title": "盗窃罪",
                        "content": "盗窃公私财物，数额较大的，处三年以下有期徒刑。",
                        "elements": [
                            {"name": "客体要件", "key": "property"},
                            {"name": "客观要件", "key": "behavior"},
                        ],
                        "base_sentence": "三年以下有期徒刑",
                        "charge_tags": ["盗窃", "财产犯罪"],
                        "common_keywords": ["窃取", "偷"],
                        "annotation_source": "project-maintained-v1",
                    },
                ],
            },
            {
                "chapter": "侵犯人身权利罪",
                "articles": [
                    {
                        "article_number": "第二百三十四条",
                        "title": "故意伤害罪",
                        "content": "故意伤害他人身体的，处三年以下有期徒刑。",
                        "elements": [
                            {"name": "客体要件", "key": "person"},
                            {"name": "客观要件", "key": "behavior"},
                        ],
                        "base_sentence": "三年以下有期徒刑",
                        "charge_tags": ["故意伤害", "人身权利"],
                        "common_keywords": ["殴打", "伤害"],
                        "annotation_source": "project-maintained-v1",
                    },
                ],
            },
        ]
    }


def _make_valid_law_dataset(*articles):
    """Return the versioned tracked-snapshot shape required by preflight."""
    if not articles:
        articles = (
            {
                "article_number": "第二百三十四条",
                "title": "故意伤害罪",
                "content": "故意伤害他人身体的，处三年以下有期徒刑、拘役或者管制。",
                "elements": ["故意伤害他人身体"],
                "base_sentence": "处三年以下有期徒刑、拘役或者管制",
                "charge_tags": ["故意伤害"],
                "common_keywords": ["伤害"],
                "official_text_source": "criminal-law-amendment-12",
                "annotation_source": "project-maintained-v1",
            },
        )
    return {
        "metadata": {
            "dataset_id": "criminal-law-core",
            "dataset_version": "2026-09-21.1",
            "verified_at": "2026-09-21",
            "official_text": {
                "source_id": "criminal-law-amendment-12",
                "title": "中华人民共和国刑法",
                "publisher": "上海市发展和改革委员会",
                "url": "https://fgw.sh.gov.cn/example.pdf",
                "consolidated_through": "中华人民共和国刑法修正案（十二）",
                "version_effective_from": "2024-03-01",
            },
            "redistribution_basis": {
                "title": "中华人民共和国著作权法第五条",
                "url": "https://www.npc.gov.cn/example.html",
            },
            "annotation_provenance": {
                "annotation_id": "project-maintained-v1",
                "official": False,
                "maintainer": "project maintainers",
                "purpose": "deterministic retrieval and coverage tests",
            },
            "official_text_fields": ["article_number", "content"],
            "project_annotation_fields": [
                "title",
                "elements",
                "base_sentence",
                "charge_tags",
                "common_keywords",
            ],
            "coverage": [article["article_number"] for article in articles],
            "limitations": "Not a complete criminal-law corpus or legal opinion.",
        },
        "chapters": [{"chapter": "第四章", "articles": list(articles)}],
    }


# ---------------------------------------------------------------------------
# _normalize_article_number
# ---------------------------------------------------------------------------


def test_normalize_article_number_chinese():
    """Chinese number format should be normalized to Arabic."""
    assert _normalize_article_number("第二百三十四条") == "第234条"


def test_normalize_article_number_arabic():
    """Already-Arabic format should remain unchanged."""
    assert _normalize_article_number("第234条") == "第234条"


def test_normalize_article_number_empty():
    """Empty string should return empty."""
    assert _normalize_article_number("") == ""


# ---------------------------------------------------------------------------
# search_laws_by_keyword
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_laws_by_keyword_matching():
    """Keyword search should return laws matching charge_tags or common_keywords."""
    law_data = _make_law_data()
    facts = {
        "behavior_sequence": ["盗窃"],
        "consequence": "财产损失",
    }
    results = await keyword_laws(facts, law_data)
    # Should find the theft article
    assert len(results) > 0
    assert any("盗窃" in r.get("title", "") or "盗窃" in r.get("charge_tags", []) for r in results)
    theft_law = next(result for result in results if result["article_number"] == "第二百六十四条")
    assert theft_law["required_elements"] == theft_law["elements"]


@pytest.mark.asyncio
async def test_search_laws_by_keyword_no_match():
    """Keyword search with non-matching terms should return empty list."""
    law_data = _make_law_data()
    facts = {
        "behavior_sequence": ["交通违章"],
        "consequence": "罚款",
    }
    results = await keyword_laws(facts, law_data)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_keyword_search_excludes_unreviewed_repealed_and_placeholder_articles():
    trusted = {
        "article_number": "第二百六十四条",
        "title": "盗窃罪",
        "content": "盗窃公私财物。",
        "elements": [{"key": "theft", "name": "盗窃公私财物"}],
        "base_sentence": "处三年以下有期徒刑",
        "charge_tags": ["盗窃"],
        "common_keywords": ["盗窃"],
        "annotation_source": "project-maintained-v1",
    }
    articles = [
        trusted,
        {**trusted, "article_number": "第一百三十三条之一", "annotation_source": "manual-title-v1"},
        {**trusted, "article_number": "第一百九十九条", "status": "repealed"},
        {
            **trusted,
            "article_number": "第三百八十五条",
            "elements": [{"key": "no_independent_elements", "name": "本条不设独立的构成要件要素"}],
        },
        {
            **trusted,
            "article_number": "第二百七十四条",
            "elements": [
                {"key": "a", "name": "数额特别巨大"},
                {"key": "b", "name": "数额特别巨大"},
            ],
        },
    ]
    law_data = {"chapters": [{"chapter": "第五章", "articles": articles}]}

    results = await keyword_laws({"behavior_sequence": ["盗窃"]}, law_data)

    assert [item["article_number"] for item in results] == ["第二百六十四条"]


@pytest.mark.asyncio
async def test_rag_match_does_not_promote_unreviewed_article_to_verified_candidate():
    article = {
        "article_number": "第一百三十三条之一",
        "title": "危险驾驶罪",
        "content": "在道路上驾驶机动车，追逐竞驶，情节恶劣的。",
        "elements": [{"key": "circumstances_vicious", "name": "情节恶劣"}],
        "base_sentence": "处拘役",
        "charge_tags": ["危险驾驶"],
        "common_keywords": ["醉驾"],
        "annotation_source": "manual-title-v1",
        "annotation_layer": "manual-title-v1",
    }
    law_data = {"chapters": [{"chapter": "第二章", "articles": [article]}]}
    rag_hit = {"article_number": "第133条之一", "content": article["content"], "data_source": "rag_unverified"}
    with patch_law_recall(return_value=[rag_hit]):
        registry = LegalToolRegistry({}, "user-1", law_data)
        result = await registry._search_laws("危险驾驶")

    assert result["candidates"][0]["source"] == "rag_unverified"
    assert registry.searched == {}
    assert (await registry._get_article("第133条之一"))["article"] is None


@pytest.mark.asyncio
async def test_repealed_article_is_not_a_rag_candidate():
    article = {
        "article_number": "第一百九十九条",
        "title": "删去",
        "content": "（删去）",
        "elements": [{"key": "deleted", "name": "删去"}],
        "base_sentence": "处拘役",
        "charge_tags": ["删去"],
        "common_keywords": ["删去"],
        "annotation_source": "project-maintained-v1",
        "status": "repealed",
    }
    law_data = {"chapters": [{"chapter": "第三章", "articles": [article]}]}
    rag_hit = {"article_number": "第199条", "content": "第一百九十九条（删去）", "data_source": "rag_unverified"}
    with patch_law_recall(return_value=[rag_hit]):
        registry = LegalToolRegistry({}, "user-1", law_data)
        result = await registry._search_laws("删去")

    assert result["candidates"] == []
    assert registry.searched == {}


# ---------------------------------------------------------------------------
# search_laws_by_rag
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_laws_by_rag():
    from tests.knowledge.law.helpers import make_law_service

    service, requests = make_law_service(
        _make_law_data(), laws=[{"article_number": "第264条", "content": "第二百六十四条 盗窃正文"}]
    )
    result = await service.search("查询", user_id="user-001")
    assert result.as_laws()[0]["data_source"] == "rag_verified"
    assert result.as_laws()[0]["content"] == _make_law_data()["chapters"][0]["articles"][0]["content"]
    assert requests[0].scope.user_id == "user-001"


@pytest.mark.asyncio
async def test_search_laws_by_rag_preserves_internal_retrieval_failure():
    from tests.knowledge.law.helpers import make_law_service

    service, _ = make_law_service(_make_law_data(), error=RuntimeError("retrieval unavailable"))
    result = await service.search("盗窃", user_id="user-001")
    assert result.search.candidates
    assert result.search.dependency_failed is True
    assert result.search.retrieval_status["vector"] == "failed"


@pytest.mark.asyncio
async def test_search_laws_by_rag_uses_retrieval_only_without_summary_calls():
    from tests.knowledge.law.helpers import make_law_service

    service, requests = make_law_service(
        _make_law_data(),
        laws=[{"article_number": f"第{number}条", "content": f"第{number}条 正文"} for number in (264, 265, 266)],
    )
    result = await service.search("查询", user_id="user-001")
    assert len(result.search.candidates) == 3
    assert requests[0].query == "查询"
    assert {law["article_number"] for law in result.as_laws()} == {"第二百六十四条", "第265条", "第266条"}


@pytest.mark.asyncio
async def test_search_laws_by_rag_flattens_structured_behavior_items():
    from app.knowledge.law.keyword import facts_query, facts_terms
    from tests.knowledge.law.helpers import make_law_service

    facts = {
        "behavior_sequence": [{"time": "2026年6月18日", "actor": "被调查人", "action": "徒手击打面部"}],
        "consequence": "鼻部软组织挫伤",
    }
    service, requests = make_law_service(_make_law_data())
    result = await service.search(facts_query(facts), user_id="user-001", keyword_terms=facts_terms(facts))
    assert not result.search.candidates
    assert "徒手击打面部" in requests[0].query and "鼻部软组织挫伤" in requests[0].query


@pytest.mark.asyncio
async def test_search_laws_by_rag_without_user_id_skips_retrieval():
    from tests.knowledge.law.helpers import make_law_service

    service, requests = make_law_service(_make_law_data())
    result = await service.search("盗窃", user_id=None)
    assert result.search.candidates and requests == []
    assert result.search.retrieval_status["vector"] == "skipped_unauthorized"
    assert result.as_laws()[0]["data_source"] == "json_keyword"


@pytest.mark.asyncio
async def test_search_laws_by_rag_failure():
    from tests.knowledge.law.helpers import make_law_service

    service, _ = make_law_service(_make_law_data(), error=ValueError("query dimension"))
    result = await service.search("无法识别", user_id="user-001")
    assert not result.search.candidates
    assert result.search.dependency_failed is True
    assert result.search.retrieval_status["vector_error"] == "ValueError"


# ---------------------------------------------------------------------------
# _verify_and_enrich_with_json
# ---------------------------------------------------------------------------


def test_verify_and_enrich_with_json_match():
    """RAG results with matching article numbers should be enriched."""
    rag_results = [
        {
            "article_number": "第二百六十四条",
            "title": "",
            "content": "RAG content",
            "elements": [],
            "base_sentence": "",
            "charge_tags": [],
            "common_keywords": [],
        }
    ]
    article_index = {
        "第264条": {
            "article_number": "第二百六十四条",
            "title": "盗窃罪",
            "content": "JSON content",
            "elements": ["element1"],
            "base_sentence": "三年以下",
            "charge_tags": ["盗窃"],
            "common_keywords": ["窃取"],
            "chapter": "侵犯财产罪",
            "annotation_source": "project-maintained-v1",
        }
    }
    result = verified_laws(rag_results, article_index)
    assert len(result) == 1
    assert result[0]["data_source"] == "rag_verified"
    assert result[0]["title"] == "盗窃罪"
    assert result[0]["required_elements"] == ["element1"]


def test_verify_and_enrich_with_json_no_match():
    """RAG results without matching article numbers should be marked unverified."""
    rag_results = [
        {
            "article_number": "第九百九十九条",
            "title": "",
            "content": "RAG content",
        }
    ]
    article_index = {"第264条": {"title": "盗窃罪"}}
    result = verified_laws(rag_results, article_index)
    assert len(result) == 1
    assert result[0]["data_source"] == "rag_unverified"


# ---------------------------------------------------------------------------
# _merge_and_deduplicate
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# law_ref_node – with matching charge_tags
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_law_ref_node_with_matching_facts():
    """law_ref_node should return applied_laws when facts match charge tags."""
    facts = {
        "behavior_sequence": ["盗窃"],
        "consequence": "财产损失",
    }
    state = make_consultation_state(
        facts_structured=facts,
        applied_laws=[],
    )

    with (
        patch("app.consultation.agents.law_ref.llm_gateway") as mock_llm,
        patch("app.consultation.agents.law_ref.load_criminal_law_data", return_value=_make_law_data()),
        patch_law_recall(return_value=[]),
        patch_law_keyword() as mock_keyword,
        patch("app.consultation.agents.law_ref.extract_structured_laws", new_callable=AsyncMock, return_value=[]),
    ):
        mock_keyword.return_value = [
            {
                "article_number": "第二百六十四条",
                "title": "盗窃罪",
                "content": "盗窃公私财物，数额较大的，处三年以下有期徒刑。",
                "elements": ["客体要件", "客观要件"],
                "base_sentence": "三年以下有期徒刑",
                "charge_tags": ["盗窃"],
                "common_keywords": ["窃取"],
                "chapter": "侵犯财产罪",
                "relevance_score": 3.0,
                "matched_tags": ["标签匹配: 盗窃"],
                "data_source": "json_keyword",
            }
        ]
        mock_llm.generate = AsyncMock(return_value="mocked")

        result = await law_ref_node(state)

    assert len(result.get("applied_laws", [])) > 0
    assert result.get("current_agent") == "LawRef"
    assert result.get("element_to_law_mapping") is not None


@pytest.mark.asyncio
async def test_law_ref_node_fallback_maps_real_snapshot_element_objects():
    """The real six-article snapshot must survive invalid LLM extraction fallback."""
    state = make_consultation_state(
        facts_structured={"behavior_sequence": ["殴打"], "consequence": "受伤"},
        applied_laws=[],
    )
    rag_result = [{"article_number": "第二百三十四条", "content": "故意伤害他人身体"}]
    with (
        patch_law_recall(return_value=rag_result),
        patch_law_keyword(return_value=[]),
        patch("app.consultation.agents.law_ref.extract_structured_laws", new_callable=AsyncMock, return_value=[]),
    ):
        result = await law_ref_node(state)

    assert result["current_agent"] == "LawRef"
    assert result["element_to_law_mapping"]["故意伤害他人身体"]["article_number"] == "第二百三十四条"
    assert result["applied_laws"][0]["data_source"] == "rag_verified"


@pytest.mark.asyncio
async def test_search_laws_by_keyword_accepts_structured_behavior_items():
    """对象型行为序列不应让关键词补充路径在 lower() 处崩溃。"""
    facts = {
        "behavior_sequence": [
            {"time": "2026年6月18日", "actor": "被调查人", "action": "徒手击打"},
        ],
        "consequence": "软组织挫伤",
    }
    law_data = {
        "chapters": [
            {
                "chapter": "侵犯公民人身权利罪",
                "articles": [
                    {
                        "article_number": "第二百三十四条",
                        "title": "故意伤害罪",
                        "content": "故意伤害他人身体",
                        "elements": [{"key": "harm", "name": "故意伤害他人身体"}],
                        "base_sentence": "三年以下有期徒刑",
                        "charge_tags": ["击打"],
                        "common_keywords": ["徒手击打"],
                        "annotation_source": "project-maintained-v1",
                    }
                ],
            }
        ]
    }

    results = await keyword_laws(facts, law_data)

    assert results[0]["article_number"] == "第二百三十四条"


# ---------------------------------------------------------------------------
# law_ref_node – empty facts_structured
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_law_ref_node_empty_facts():
    """When facts_structured is empty, law_ref_node should return empty applied_laws."""
    state = make_consultation_state(
        facts_structured={},
        applied_laws=[],
    )

    result = await law_ref_node(state)

    assert result.get("applied_laws") == []
    assert result.get("current_agent") == "LawRef"
    assert result["law_research"]["termination_reason"] == "missing_facts"


# ---------------------------------------------------------------------------
# law_ref_node – no match
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_law_ref_node_no_match():
    """When no laws match, applied_laws should be empty or minimal."""
    facts = {
        "behavior_sequence": ["交通违章"],
        "consequence": "罚款",
    }
    state = make_consultation_state(
        facts_structured=facts,
        applied_laws=[],
    )

    with (
        patch("app.consultation.agents.law_ref.load_criminal_law_data", return_value=_make_law_data()),
        patch_law_recall(return_value=[]),
        patch_law_keyword(return_value=[]),
        patch("app.consultation.agents.law_ref.extract_structured_laws", new_callable=AsyncMock, return_value=[]),
    ):
        result = await law_ref_node(state)

    assert result.get("applied_laws") == []
    assert result.get("current_agent") == "LawRef"


# ---------------------------------------------------------------------------
# _load_law_extract_prompt
# ---------------------------------------------------------------------------


def test_load_law_extract_prompt_uses_prompt_loader():
    """_load_law_extract_prompt should use the prompt_loader when key exists."""
    with patch.object(law_ref.prompt_loader, "load", return_value="loaded-from-yaml"):
        assert _load_law_extract_prompt() == "loaded-from-yaml"


def test_load_law_extract_prompt_falls_back_to_default():
    """_load_law_extract_prompt should return DEFAULT_LAW_EXTRACT_PROMPT when KeyError raised."""
    with patch.object(law_ref.prompt_loader, "load", side_effect=KeyError("not registered")):
        result = _load_law_extract_prompt()
    assert "刑事法律专家" in result
    assert '"charges"' in result


# ---------------------------------------------------------------------------
# load_criminal_law_data – caching and fail-fast validation
# ---------------------------------------------------------------------------


def test_load_criminal_law_data_caches_result(tmp_path, monkeypatch):
    """load_criminal_law_data should use lru_cache to return the same dict on repeated calls."""
    # Reset cache so the test doesn't leak state.
    load_criminal_law_data.cache_clear()

    sample = _make_valid_law_dataset()
    law_file = tmp_path / "criminal_law_chapters.json"
    law_file.write_text(json.dumps(sample, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(law_knowledge, "LAW_KNOWLEDGE_PATH", law_file)

    first = load_criminal_law_data()
    second = load_criminal_law_data()
    assert first is second  # lru_cache returns the exact same object
    assert "chapters" in first
    assert len(first["chapters"]) == 1
    assert first["chapters"][0]["chapter"] == "第四章"
    assert first["chapters"][0]["articles"][0]["title"] == "故意伤害罪"

    # Cleanup cache
    load_criminal_law_data.cache_clear()


def test_load_criminal_law_data_missing_file(monkeypatch):
    """Missing tracked data must stop startup instead of silently disabling validation."""
    load_criminal_law_data.cache_clear()
    missing = Path("/nonexistent/path/criminal_law_chapters.json")
    monkeypatch.setattr(law_knowledge, "LAW_KNOWLEDGE_PATH", missing)
    with pytest.raises(RuntimeError, match="法条验证数据文件不存在"):
        load_criminal_law_data()
    load_criminal_law_data.cache_clear()


def test_load_criminal_law_data_rejects_missing_metadata(monkeypatch, tmp_path):
    """An unversioned dict cannot be treated as an auditable validation snapshot."""
    load_criminal_law_data.cache_clear()
    sample = {"chapters": [{"chapter": "X", "articles": []}]}
    law_file = tmp_path / "criminal_law_chapters.json"
    law_file.write_text(json.dumps(sample), encoding="utf-8")
    monkeypatch.setattr(law_knowledge, "LAW_KNOWLEDGE_PATH", law_file)
    with pytest.raises(RuntimeError, match="metadata"):
        load_criminal_law_data()
    load_criminal_law_data.cache_clear()


def test_load_criminal_law_data_error(monkeypatch, tmp_path):
    """Corrupt JSON must fail with the affected path in the error."""
    load_criminal_law_data.cache_clear()
    bad = tmp_path / "criminal_law_chapters.json"
    bad.write_text("not valid json", encoding="utf-8")
    monkeypatch.setattr(law_knowledge, "LAW_KNOWLEDGE_PATH", bad)
    with pytest.raises(RuntimeError, match="法条验证数据不是有效 JSON"):
        load_criminal_law_data()
    load_criminal_law_data.cache_clear()


def test_load_criminal_law_data_rejects_incomplete_article(monkeypatch, tmp_path):
    """A missing provenance field must not enter the trusted JSON allowlist."""
    load_criminal_law_data.cache_clear()
    dataset = _make_valid_law_dataset()
    del dataset["chapters"][0]["articles"][0]["annotation_source"]
    law_file = tmp_path / "criminal_law_chapters.json"
    law_file.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(law_knowledge, "LAW_KNOWLEDGE_PATH", law_file)

    with pytest.raises(RuntimeError, match="annotation_source"):
        load_criminal_law_data()
    load_criminal_law_data.cache_clear()


def test_load_criminal_law_data_rejects_duplicate_article_numbers(monkeypatch, tmp_path):
    """Duplicate normalized numbers would make exact RAG verification ambiguous."""
    load_criminal_law_data.cache_clear()
    first = _make_valid_law_dataset()["chapters"][0]["articles"][0]
    duplicate = {**first, "article_number": "第234条", "title": "重复条目"}
    dataset = _make_valid_law_dataset(first, duplicate)
    law_file = tmp_path / "criminal_law_chapters.json"
    law_file.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(law_knowledge, "LAW_KNOWLEDGE_PATH", law_file)

    with pytest.raises(RuntimeError, match="重复法条编号.*第234条"):
        load_criminal_law_data()
    load_criminal_law_data.cache_clear()


def test_load_criminal_law_data_rejects_unlinked_source_id(monkeypatch, tmp_path):
    """Every article must link to the declared official and annotation sources."""
    load_criminal_law_data.cache_clear()
    dataset = _make_valid_law_dataset()
    dataset["chapters"][0]["articles"][0]["official_text_source"] = "unknown-source"
    law_file = tmp_path / "criminal_law_chapters.json"
    law_file.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(law_knowledge, "LAW_KNOWLEDGE_PATH", law_file)

    with pytest.raises(RuntimeError, match="official_text_source.*顶层来源不一致"):
        load_criminal_law_data()
    load_criminal_law_data.cache_clear()


def test_load_criminal_law_data_rejects_non_government_source_url(monkeypatch, tmp_path):
    """Official text and redistribution evidence must use auditable government URLs."""
    load_criminal_law_data.cache_clear()
    dataset = _make_valid_law_dataset()
    dataset["metadata"]["official_text"]["url"] = "https://example.com/law.pdf"
    law_file = tmp_path / "criminal_law_chapters.json"
    law_file.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(law_knowledge, "LAW_KNOWLEDGE_PATH", law_file)

    with pytest.raises(RuntimeError, match="政府 HTTPS URL"):
        load_criminal_law_data()
    load_criminal_law_data.cache_clear()


def test_tracked_law_snapshot_preflight_exposes_source_and_annotation_boundary():
    """The shipped snapshot must identify official text and non-official annotations."""
    load_criminal_law_data.cache_clear()
    data = law_knowledge.preflight_law_knowledge()

    metadata = data["metadata"]
    assert metadata["official_text"]["consolidated_through"] == "中华人民共和国刑法修正案（十二）"
    assert metadata["official_text"]["version_effective_from"] == "2024-03-01"
    assert metadata["official_text"]["url"].startswith("https://")
    assert metadata["redistribution_basis"]["url"].startswith("https://www.npc.gov.cn/")
    assert metadata["annotation_provenance"]["official"] is False
    assert metadata["official_text_fields"] == ["article_number", "content"]
    assert metadata["project_annotation_fields"] == [
        "title",
        "elements",
        "base_sentence",
        "charge_tags",
        "common_keywords",
    ]
    assert metadata["verified_at"] == "2026-09-21"
    assert set(metadata["coverage"]) == {
        "第二百三十二条",
        "第二百三十四条",
        "第二百六十三条",
        "第二百六十四条",
        "第二百六十六条",
        "第二百九十三条",
    }
    load_criminal_law_data.cache_clear()


# ---------------------------------------------------------------------------
# _cn_to_arabic
# ---------------------------------------------------------------------------


def test_cn_to_arabic_with_units():
    """Compound numbers with units should be converted."""
    assert _cn_to_arabic("二百三十四") == "234"
    assert _cn_to_arabic("一千") == "1000"


def test_cn_to_arabic_ten_special():
    """Special '十' handling — leading or standalone."""
    assert _cn_to_arabic("一十") == "10"
    assert _cn_to_arabic("十") == "10"


def test_cn_to_arabic_non_digit_input():
    """Non-digit input should be returned unchanged."""
    assert _cn_to_arabic("abc") == "abc"
    assert _cn_to_arabic("") == ""


# ---------------------------------------------------------------------------
# _normalize_article_number – fallback return path
# ---------------------------------------------------------------------------


def test_normalize_article_number_no_match_pattern():
    """When input does not match the '第X条' pattern, return stripped input."""
    assert _normalize_article_number("  random text  ") == "random text"


def test_normalize_article_number_unparseable_chinese():
    """When Chinese digits can't be parsed, return stripped input."""
    # 'foo' contains 'foo' chars, not all CN_DIGITS, so _cn_to_arabic returns the original
    # and we hit the fallback return (line 172).
    assert _normalize_article_number("第foo条") == "第foo条"


# ---------------------------------------------------------------------------
# _extract_article_number_from_text
# ---------------------------------------------------------------------------


def test_extract_article_number_from_text_chinese():
    """Extract article number from a text containing Chinese digits."""
    text = "本条依据第二百三十四条 故意伤害罪规定。"
    assert _extract_article_number_from_text(text) == "第二百三十四条"


def test_extract_article_number_from_text_arabic():
    """Extract article number from a text containing Arabic digits."""
    text = "根据第264条 盗窃公私财物。"
    assert _extract_article_number_from_text(text) == "第264条"


def test_extract_article_number_from_text_empty():
    """Empty text returns empty string."""
    assert _extract_article_number_from_text("") == ""
    assert _extract_article_number_from_text(None) == ""


def test_extract_article_number_from_text_no_match():
    """Text without a '第X条' pattern returns empty."""
    assert _extract_article_number_from_text("普通文本") == ""


# ---------------------------------------------------------------------------
# _build_article_index
# ---------------------------------------------------------------------------


def test_build_article_index_basic():
    """Build an index keyed by normalized article numbers."""
    data = _make_law_data()
    index = _build_article_index(data)
    assert "第264条" in index
    assert "第234条" in index
    assert index["第264条"]["title"] == "盗窃罪"
    assert index["第264条"]["chapter"] == "侵犯财产罪"


def test_build_article_index_empty():
    """Empty law_data returns an empty index."""
    assert _build_article_index({"chapters": []}) == {}


# ---------------------------------------------------------------------------
# search_laws_by_keyword – non-list behavior_sequence + tag score paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_laws_by_keyword_non_list_behavior():
    """When behavior_sequence is a non-list (string), it should still work."""
    law_data = _make_law_data()
    facts = {
        "behavior_sequence": "盗窃",  # string instead of list
        "consequence": "财产损失",
    }
    results = await keyword_laws(facts, law_data)
    assert len(results) > 0


@pytest.mark.asyncio
async def test_search_laws_by_keyword_common_keyword_score():
    """A keyword match only via common_keywords should score 1."""
    law_data = _make_law_data()
    facts = {
        "behavior_sequence": ["窃取"],  # matches common_keywords for 盗窃
        "consequence": "",
    }
    results = await keyword_laws(facts, law_data)
    assert len(results) > 0
    matched = results[0]
    assert matched["article_number"] == "第二百六十四条"
    # The relevance_score for a common_keyword match is 1
    assert matched["relevance_score"] >= 1


# ---------------------------------------------------------------------------
# search_laws_by_rag – document object branches, empty query
# ---------------------------------------------------------------------------


class _DocObj:
    """Helper to simulate a LangChain Document object."""

    def __init__(self, content: str):
        self.page_content = content


@pytest.mark.asyncio
async def test_search_laws_by_rag_retains_complete_unverified_body():
    from tests.knowledge.law.helpers import make_law_service

    body = "第999条 " + "未经核验的正文" * 100
    service, _ = make_law_service(_make_law_data(), laws=[{"article_number": "第999条", "content": body}])
    result = await service.search("查询", user_id="user-001")
    assert result.as_laws()[0]["content"] == body
    assert result.as_laws()[0]["data_source"] == "rag_unverified"
    assert not result.as_laws()[0]["required_elements"]


@pytest.mark.asyncio
async def test_search_laws_by_rag_typed_metadata_keeps_article_identity():
    from tests.knowledge.law.helpers import make_law_service

    service, _ = make_law_service(
        _make_law_data(), laws=[{"article_number": "第二百三十四条", "content": "第二百三十四条 故意伤害罪规定。"}]
    )
    result = await service.search("查询", user_id="user-001")
    assert result.as_laws()[0]["data_source"] == "rag_verified"
    assert result.as_laws()[0]["article_number"] == "第二百三十四条"


@pytest.mark.asyncio
async def test_search_laws_by_rag_malformed_channel_keeps_failure_visible():
    from tests.knowledge.law.helpers import make_law_service

    service, _ = make_law_service(_make_law_data(), error=TypeError("malformed stored row"))
    result = await service.search("盗窃", user_id="user-001")
    assert result.search.dependency_failed
    assert result.search.retrieval_status["vector_error"] == "TypeError"
    assert result.as_laws()[0]["data_source"] == "json_keyword"


@pytest.mark.asyncio
async def test_search_laws_by_rag_empty_query_is_explicitly_skipped():
    from app.knowledge.law.keyword import facts_query
    from tests.knowledge.law.helpers import make_law_service

    service, requests = make_law_service(_make_law_data())
    result = await service.search(facts_query({"behavior_sequence": [], "consequence": ""}), user_id="user-001")
    assert not result.search.candidates and requests == []
    assert result.search.retrieval_status["vector"] == "skipped_empty"


@pytest.mark.asyncio
async def test_search_laws_by_rag_non_list_behavior():
    from app.knowledge.law.keyword import facts_query
    from tests.knowledge.law.helpers import make_law_service

    service, requests = make_law_service(_make_law_data())
    result = await service.search(
        facts_query({"behavior_sequence": "string-behavior", "consequence": ""}), user_id="user-001"
    )
    assert not result.search.candidates
    assert requests[0].query == "string-behavior"


# ---------------------------------------------------------------------------
# extract_structured_laws
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_structured_laws_empty():
    """With no matched_laws, return empty list."""
    assert await extract_structured_laws([], {}) == []


@pytest.mark.asyncio
async def test_extract_structured_laws_valid_json_response():
    """Successful LLM JSON response should be parsed and 'charges' returned."""
    matched = [
        {
            "article_number": "第二百六十四条",
            "title": "盗窃罪",
            "content": "盗窃公私财物...",
            "elements": ["客体要件"],
            "base_sentence": "三年以下",
            "charge_tags": ["盗窃"],
        }
    ]
    llm_response = json.dumps(
        {
            "charges": [
                {
                    "charge_name": "盗窃罪",
                    "article_number": "第二百六十四条",
                    "elements_matched": ["客体要件"],
                    "elements_missing": [],
                    "base_sentence": "三年以下",
                    "probability": "high",
                }
            ],
            "procedural_notes": ["注意程序"],
        },
        ensure_ascii=False,
    )

    with patch("app.consultation.agents.law_ref.llm_gateway") as mock_llm:
        mock_llm.generate = AsyncMock(return_value=llm_response)
        charges = await extract_structured_laws(matched, {"behavior_sequence": ["盗窃"], "consequence": "损失"})

    assert len(charges) == 1
    assert charges[0]["charge_name"] == "盗窃罪"
    assert charges[0]["probability"] == "high"


@pytest.mark.asyncio
async def test_extract_structured_laws_formats_authoritative_object_elements():
    """对象形式的权威要件应按名称进入 LLM 上下文。"""
    matched = [
        {
            "article_number": "第二百六十四条",
            "title": "盗窃罪",
            "content": "盗窃公私财物。",
            "elements": [
                {"name": "客体要件", "key": "property"},
                {"name": "客观要件", "key": "behavior"},
            ],
            "base_sentence": "三年以下",
            "charge_tags": ["盗窃"],
            "data_source": "json_keyword",
        }
    ]
    captured_user_message = ""

    async def _capture_generate(system_prompt, user_message, **kwargs):
        nonlocal captured_user_message
        captured_user_message = user_message
        return '{"charges": []}'

    with patch("app.consultation.agents.law_ref.llm_gateway.generate", side_effect=_capture_generate):
        result = await extract_structured_laws(
            matched,
            {"behavior_sequence": ["盗窃"], "consequence": ""},
        )

    assert result == []
    assert "构成要件: 客体要件, 客观要件" in captured_user_message


@pytest.mark.asyncio
async def test_extract_structured_laws_no_json_in_response():
    """If LLM returns text without JSON, return empty list."""
    matched = [
        {
            "article_number": "第264条",
            "title": "盗窃罪",
            "content": "...",
            "elements": [],
            "base_sentence": "",
            "charge_tags": [],
        }
    ]
    with patch("app.consultation.agents.law_ref.llm_gateway") as mock_llm:
        mock_llm.generate = AsyncMock(return_value="抱歉，无法提取结构化信息。")
        charges = await extract_structured_laws(matched, {"behavior_sequence": ["盗窃"], "consequence": ""})

    assert charges == []


@pytest.mark.asyncio
async def test_extract_structured_laws_llm_exception():
    """If the LLM call raises, return empty list and don't propagate."""
    matched = [
        {
            "article_number": "第264条",
            "title": "盗窃罪",
            "content": "...",
            "elements": [],
            "base_sentence": "",
            "charge_tags": [],
        }
    ]
    with patch("app.consultation.agents.law_ref.llm_gateway") as mock_llm:
        mock_llm.generate = AsyncMock(side_effect=Exception("LLM offline"))
        charges = await extract_structured_laws(matched, {"behavior_sequence": ["盗窃"], "consequence": ""})

    assert charges == []


# ---------------------------------------------------------------------------
# _build_applied_laws_from_structured
# ---------------------------------------------------------------------------


def test_build_applied_laws_from_structured_with_matched_data_source():
    """Structured laws should inherit data_source from matched_laws when possible."""
    structured = [
        {
            "charge_name": "盗窃罪",
            "article_number": "第264条",
            "elements_matched": ["客体要件", "模型虚构要件"],
            "elements_missing": [],
            "base_sentence": "三年以下",
            "probability": "medium",
        }
    ]
    matched = [
        {
            "article_number": "第264条",
            "elements": ["客体要件", "主观要件"],
            "data_source": "rag_verified",
        },
    ]
    applied = _build_applied_laws_from_structured(structured, matched)
    assert len(applied) == 1
    assert applied[0]["charge_name"] == "盗窃罪"
    assert applied[0]["data_source"] == "rag_verified"
    assert applied[0]["required_elements"] == ["客体要件", "主观要件"]
    assert applied[0]["elements"] == ["客体要件", "主观要件"]
    assert applied[0]["elements_matched"] == ["客体要件"]
    assert applied[0]["elements_missing"] == ["主观要件"]
    assert set(applied[0]["elements_matched"]).isdisjoint(applied[0]["elements_missing"])
    assert set(applied[0]["elements_matched"] + applied[0]["elements_missing"]) == {
        "客体要件",
        "主观要件",
    }


def test_build_applied_laws_from_structured_no_match_falls_back():
    """When the article number is not in matched_laws, default to llm_extracted."""
    structured = [
        {
            "charge_name": "诈骗罪",
            "article_number": "第266条",
            "elements_matched": [],
            "elements_missing": [],
            "base_sentence": "三年以下",
            "probability": "low",
        }
    ]
    matched = [
        {"article_number": "第264条", "data_source": "rag_verified"},
    ]
    applied = _build_applied_laws_from_structured(structured, matched)
    assert applied[0]["data_source"] == "llm_extracted"
    assert applied[0]["required_elements"] == []


def test_build_applied_laws_from_structured_rejects_unknown_source():
    """非法来源值不能绕过来源枚举成为 coverage 候选。"""
    structured = [
        {
            "charge_name": "盗窃罪",
            "article_number": "第264条",
            "elements_matched": ["客体要件"],
            "elements_missing": [],
        }
    ]
    matched = [
        {
            "article_number": "第264条",
            "elements": ["客体要件"],
            "data_source": "json_knowledge",
        }
    ]

    applied = _build_applied_laws_from_structured(structured, matched)

    assert applied[0]["data_source"] == "llm_extracted"
    assert applied[0]["required_elements"] == []


# ---------------------------------------------------------------------------
# _build_element_to_law_mapping
# ---------------------------------------------------------------------------


def test_build_element_to_law_mapping_default_key():
    """Mapping should pick 'elements' by default."""
    laws = [
        {
            "charge_name": "盗窃罪",
            "article_number": "第264条",
            "base_sentence": "三年以下",
            "elements": ["客体要件", "客观要件"],
        }
    ]
    mapping = _build_element_to_law_mapping(laws)
    assert "客体要件" in mapping
    assert mapping["客体要件"]["charge_name"] == "盗窃罪"
    assert mapping["客体要件"]["article_number"] == "第264条"


def test_build_element_to_law_mapping_structured_key():
    """Mapping should accept 'elements_matched' for structured laws."""
    laws = [
        {
            "charge_name": "诈骗罪",
            "article_number": "第266条",
            "base_sentence": "三年以下",
            "elements_matched": ["虚构事实"],
        }
    ]
    mapping = _build_element_to_law_mapping(laws, elements_key="elements_matched")
    assert "虚构事实" in mapping
    assert mapping["虚构事实"]["charge_name"] == "诈骗罪"


# ---------------------------------------------------------------------------
# _merge_and_deduplicate – duplicate and missing-key branches
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# law_ref_node – RAG verified / keyword merged path + structured branch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_law_ref_node_rag_verified_with_json():
    """When RAG returns results and JSON knowledge has matching articles, data_source should be rag_verified."""
    facts = {
        "behavior_sequence": ["盗窃"],
        "consequence": "财产损失",
    }
    state = make_consultation_state(facts_structured=facts, applied_laws=[])

    rag_results = [
        {
            "article_number": "第二百六十四条",
            "title": "",
            "content": "RAG content",
            "elements": [],
            "base_sentence": "",
            "charge_tags": [],
            "common_keywords": [],
            "chapter": "",
            "data_source": "rag",
        }
    ]
    structured_laws = [
        {
            "charge_name": "盗窃罪",
            "article_number": "第264条",
            "elements_matched": ["客体要件"],
            "elements_missing": ["主观要件"],
            "base_sentence": "三年以下",
            "probability": "high",
        }
    ]

    with (
        patch("app.consultation.agents.law_ref.load_criminal_law_data", return_value=_make_law_data()),
        patch_law_recall(return_value=rag_results),
        patch_law_keyword(return_value=[]),
        patch(
            "app.consultation.agents.law_ref.extract_structured_laws",
            new_callable=AsyncMock,
            return_value=structured_laws,
        ),
    ):
        result = await law_ref_node(state)

    assert result.get("current_agent") == "LawRef"
    applied_laws = result.get("applied_laws", [])
    assert len(applied_laws) == 1
    # data_source comes from rag_results which was rag_verified
    assert applied_laws[0]["data_source"] == "rag_verified"
    # history entry should be appended
    assert any(entry.get("agent") == "LawRef" for entry in result.get("conversation_history", []))
    assert result.get("rag_only") is False


@pytest.mark.asyncio
async def test_law_ref_node_rag_unverified_only():
    """When RAG results are unverified and no JSON keyword matches, rag_only should be True."""
    facts = {
        "behavior_sequence": ["non-matching-term"],
        "consequence": "",
    }
    state = make_consultation_state(facts_structured=facts, applied_laws=[])

    rag_results = [
        {
            "article_number": "第999条",
            "title": "Unknown",
            "content": "...",
            "elements": [],
            "base_sentence": "",
            "charge_tags": [],
            "common_keywords": [],
            "chapter": "",
            "data_source": "rag",
        }
    ]

    with (
        patch("app.consultation.agents.law_ref.load_criminal_law_data", return_value=_make_law_data()),
        patch_law_recall(return_value=rag_results),
        patch_law_keyword(return_value=[]),
        patch("app.consultation.agents.law_ref.extract_structured_laws", new_callable=AsyncMock, return_value=[]),
    ):
        result = await law_ref_node(state)

    # 未经快照核验的候选不能由模型最终答案升级为可信法条。
    assert result.get("rag_only") is False
    assert result.get("applied_laws") == []
    assert result.get("law_search_status") == "dependency_failure"


@pytest.mark.asyncio
async def test_law_ref_node_no_existing_conversation_history():
    """When state lacks conversation_history, a new one is created and the entry is appended."""
    facts = {
        "behavior_sequence": ["交通违章"],
        "consequence": "罚款",
    }
    state = make_consultation_state(
        facts_structured=facts,
        applied_laws=[],
    )
    # Remove conversation_history from the state dict entirely
    state.pop("conversation_history", None)

    with (
        patch("app.consultation.agents.law_ref.load_criminal_law_data", return_value=_make_law_data()),
        patch_law_recall(return_value=[]),
        patch_law_keyword(return_value=[]),
        patch("app.consultation.agents.law_ref.extract_structured_laws", new_callable=AsyncMock, return_value=[]),
    ):
        result = await law_ref_node(state)

    conversation_history = result.get("conversation_history", [])
    assert conversation_history
    assert any(entry.get("agent") == "LawRef" for entry in conversation_history)


@pytest.mark.asyncio
async def test_law_ref_node_uses_user_id_for_rag_filter():
    state = make_consultation_state(
        user_id="real-user-42",
        session_id="session-must-not-be-user",
        facts_structured={"behavior_sequence": ["盗窃"], "consequence": "财产损失"},
    )

    with (
        patch("app.consultation.agents.law_ref.load_criminal_law_data", return_value=_make_law_data()),
        patch_law_recall(return_value=[]) as rag_search,
        patch("app.consultation.agents.law_ref.extract_structured_laws", new_callable=AsyncMock, return_value=[]),
    ):
        result = await law_ref_node(state)

    rag_search.assert_awaited_once_with({"behavior_sequence": ["盗窃"], "consequence": ""}, "real-user-42")
    assert result["conversation_history"][-1]["session_id"] == "session-must-not-be-user"


@pytest.mark.asyncio
async def test_law_ref_node_missing_user_id_does_not_use_session_id():
    state = make_consultation_state(
        session_id="legacy-session",
        facts_structured={"behavior_sequence": ["盗窃"], "consequence": "财产损失"},
    )
    state.pop("user_id", None)

    with (
        patch("app.consultation.agents.law_ref.load_criminal_law_data", return_value=_make_law_data()),
        patch_law_recall(return_value=[]) as rag_search,
        patch("app.consultation.agents.law_ref.extract_structured_laws", new_callable=AsyncMock, return_value=[]),
    ):
        await law_ref_node(state)

    rag_search.assert_not_awaited()
