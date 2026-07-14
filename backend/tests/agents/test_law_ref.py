"""Integration tests for the LawRef Agent node."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents import law_ref
from app.agents.law_ref import (
    _build_applied_laws_from_matched,
    _build_applied_laws_from_structured,
    _build_article_index,
    _build_element_to_law_mapping,
    _cn_to_arabic,
    _extract_article_number_from_text,
    _is_unverified_rag_result,
    _load_law_extract_prompt,
    _merge_and_deduplicate,
    _normalize_article_number,
    _verify_and_enrich_with_json,
    extract_structured_laws,
    law_ref_node,
    load_criminal_law_data,
    search_laws_by_keyword,
    search_laws_by_rag,
)
from tests.factories import make_consultation_state


# ---------------------------------------------------------------------------
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
                    },
                ],
            },
        ]
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
# _is_unverified_rag_result
# ---------------------------------------------------------------------------


def test_is_unverified_rag_result_true():
    assert _is_unverified_rag_result({"data_source": "rag_unverified"}) is True


def test_is_unverified_rag_result_false_verified():
    assert _is_unverified_rag_result({"data_source": "rag_verified"}) is False


def test_is_unverified_rag_result_false_json():
    assert _is_unverified_rag_result({"data_source": "json_keyword"}) is False


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
    results = await search_laws_by_keyword(facts, law_data)
    # Should find the theft article
    assert len(results) > 0
    assert any("盗窃" in r.get("title", "") or "盗窃" in r.get("charge_tags", []) for r in results)


@pytest.mark.asyncio
async def test_search_laws_by_keyword_no_match():
    """Keyword search with non-matching terms should return empty list."""
    law_data = _make_law_data()
    facts = {
        "behavior_sequence": ["交通违章"],
        "consequence": "罚款",
    }
    results = await search_laws_by_keyword(facts, law_data)
    assert len(results) == 0


# ---------------------------------------------------------------------------
# search_laws_by_rag
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_laws_by_rag():
    """RAG search should call RagService and return results."""
    facts = {
        "behavior_sequence": ["盗窃"],
        "consequence": "财产损失",
    }

    mock_rag_service = MagicMock()
    mock_rag_service.initialize_retriever = AsyncMock()
    mock_rag_service.get_documents_and_summary = AsyncMock(
        return_value={
            "documents": ["第二百六十四条 盗窃公私财物，数额较大的，处三年以下有期徒刑。"],
        }
    )

    with patch("app.rag.rag_service.RagService", return_value=mock_rag_service) as rag_service_class:
        results = await search_laws_by_rag(facts, "user-001")

    assert len(results) > 0
    assert results[0]["data_source"] == "rag"
    assert "第二百六十四条" in results[0].get("article_number", "") or results[0]["content"] != ""
    rag_service_class.assert_called_once_with(user_id="user-001", include_public=True)


@pytest.mark.asyncio
async def test_search_laws_by_rag_without_user_id_skips_retrieval():
    facts = {"behavior_sequence": ["盗窃"], "consequence": "财产损失"}

    with patch("app.rag.rag_service.RagService") as rag_service_class:
        results = await search_laws_by_rag(facts, None)

    assert results == []
    rag_service_class.assert_not_called()


@pytest.mark.asyncio
async def test_search_laws_by_rag_failure():
    """RAG search should return empty list on failure."""
    facts = {
        "behavior_sequence": ["盗窃"],
        "consequence": "财产损失",
    }

    with patch("app.rag.rag_service.RagService", side_effect=Exception("RAG unavailable")):
        results = await search_laws_by_rag(facts, "session-001")

    assert results == []


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
        }
    }
    result = _verify_and_enrich_with_json(rag_results, article_index)
    assert len(result) == 1
    assert result[0]["data_source"] == "rag_verified"
    assert result[0]["title"] == "盗窃罪"


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
    result = _verify_and_enrich_with_json(rag_results, article_index)
    assert len(result) == 1
    assert result[0]["data_source"] == "rag_unverified"


# ---------------------------------------------------------------------------
# _merge_and_deduplicate
# ---------------------------------------------------------------------------


def test_merge_and_deduplicate():
    """Primary results should take priority; duplicates should be removed."""
    primary = [{"article_number": "第264条", "title": "盗窃罪"}]
    secondary = [
        {"article_number": "第264条", "title": "盗窃罪（重复）"},
        {"article_number": "第234条", "title": "故意伤害罪"},
    ]
    result = _merge_and_deduplicate(primary, secondary)
    assert len(result) == 2
    # Primary version should be kept
    assert result[0]["title"] == "盗窃罪"


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

    with patch("app.agents.law_ref.llm_gateway") as mock_llm, \
         patch("app.agents.law_ref.load_criminal_law_data", return_value=_make_law_data()), \
         patch("app.agents.law_ref.search_laws_by_rag", new_callable=AsyncMock, return_value=[]), \
         patch("app.agents.law_ref.search_laws_by_keyword") as mock_keyword, \
         patch("app.agents.law_ref.extract_structured_laws", new_callable=AsyncMock, return_value=[]):

        mock_keyword.return_value = [
            {
                "article_number": "第二百六十四条",
                "title": "盗窃罪",
                "content": "盗窃公私财物...",
                "elements": ["客体要件", "客观要件"],
                "base_sentence": "三年以下有期徒刑",
                "charge_tags": ["盗窃"],
                "common_keywords": ["窃取"],
                "chapter": "侵犯财产罪",
                "relevance_score": 3.0,
                "matched_tags": ["标签匹配: 盗窃"],
            }
        ]
        mock_llm.generate = AsyncMock(return_value="mocked")

        result = await law_ref_node(state)

    assert len(result.get("applied_laws", [])) > 0
    assert result.get("current_agent") == "LawRef"
    assert result.get("element_to_law_mapping") is not None


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

    with patch("app.agents.law_ref.load_criminal_law_data", return_value=_make_law_data()), \
         patch("app.agents.law_ref.search_laws_by_rag", new_callable=AsyncMock, return_value=[]), \
         patch("app.agents.law_ref.search_laws_by_keyword", new_callable=AsyncMock, return_value=[]), \
         patch("app.agents.law_ref.extract_structured_laws", new_callable=AsyncMock, return_value=[]):

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
# load_criminal_law_data – caching, missing file, list vs dict, error path
# ---------------------------------------------------------------------------


def test_load_criminal_law_data_caches_result(tmp_path, monkeypatch):
    """load_criminal_law_data should use lru_cache to return the same dict on repeated calls."""
    # Reset cache so the test doesn't leak state.
    load_criminal_law_data.cache_clear()

    sample = [{"chapter": "第一章", "article_number": "第1条", "title": "测试罪", "content": "测试内容"}]
    law_file = tmp_path / "criminal_law_chapters.json"
    law_file.write_text(json.dumps(sample, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(law_ref, "LAW_KNOWLEDGE_PATH", law_file)

    first = load_criminal_law_data()
    second = load_criminal_law_data()
    assert first is second  # lru_cache returns the exact same object
    assert "chapters" in first
    assert len(first["chapters"]) == 1
    assert first["chapters"][0]["chapter"] == "第一章"
    assert first["chapters"][0]["articles"][0]["title"] == "测试罪"

    # Cleanup cache
    load_criminal_law_data.cache_clear()


def test_load_criminal_law_data_missing_file(monkeypatch):
    """If file is missing, return empty chapters dict and don't raise."""
    load_criminal_law_data.cache_clear()
    missing = Path("/nonexistent/path/criminal_law_chapters.json")
    monkeypatch.setattr(law_ref, "LAW_KNOWLEDGE_PATH", missing)
    result = load_criminal_law_data()
    assert result == {"chapters": []}
    load_criminal_law_data.cache_clear()


def test_load_criminal_law_data_dict_format(monkeypatch, tmp_path):
    """When JSON is a dict (not a list), use it as-is."""
    load_criminal_law_data.cache_clear()
    sample = {"chapters": [{"chapter": "X", "articles": []}]}
    law_file = tmp_path / "criminal_law_chapters.json"
    law_file.write_text(json.dumps(sample), encoding="utf-8")
    monkeypatch.setattr(law_ref, "LAW_KNOWLEDGE_PATH", law_file)
    result = load_criminal_law_data()
    assert result == sample
    load_criminal_law_data.cache_clear()


def test_load_criminal_law_data_error(monkeypatch, tmp_path):
    """If the file is corrupt, return empty chapters and don't raise."""
    load_criminal_law_data.cache_clear()
    bad = tmp_path / "criminal_law_chapters.json"
    bad.write_text("not valid json", encoding="utf-8")
    monkeypatch.setattr(law_ref, "LAW_KNOWLEDGE_PATH", bad)
    result = load_criminal_law_data()
    assert result == {"chapters": []}
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
    results = await search_laws_by_keyword(facts, law_data)
    assert len(results) > 0


@pytest.mark.asyncio
async def test_search_laws_by_keyword_common_keyword_score():
    """A keyword match only via common_keywords should score 1."""
    law_data = _make_law_data()
    facts = {
        "behavior_sequence": ["窃取"],  # matches common_keywords for 盗窃
        "consequence": "",
    }
    results = await search_laws_by_keyword(facts, law_data)
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
async def test_search_laws_by_rag_with_string_documents():
    """RAG search should accept plain string documents."""
    facts = {"behavior_sequence": ["盗窃"], "consequence": ""}

    mock_rag_service = MagicMock()
    mock_rag_service.initialize_retriever = AsyncMock()
    mock_rag_service.get_documents_and_summary = AsyncMock(
        return_value={"documents": ["第二百六十四条 关于盗窃罪。"]}
    )

    with patch("app.rag.rag_service.RagService", return_value=mock_rag_service):
        results = await search_laws_by_rag(facts, "session-001")

    assert len(results) == 1
    # article_number is stored as raw extracted text (not normalized to Arabic)
    assert "第二百六十四条" in results[0]["article_number"]
    assert results[0]["data_source"] == "rag"


@pytest.mark.asyncio
async def test_search_laws_by_rag_with_document_objects():
    """RAG search should accept objects with page_content attribute."""
    facts = {"behavior_sequence": ["盗窃"], "consequence": ""}
    docs = [_DocObj("第二百三十四条 故意伤害罪规定。")]

    mock_rag_service = MagicMock()
    mock_rag_service.initialize_retriever = AsyncMock()
    mock_rag_service.get_documents_and_summary = AsyncMock(return_value={"documents": docs})

    with patch("app.rag.rag_service.RagService", return_value=mock_rag_service):
        results = await search_laws_by_rag(facts, "session-001")

    assert len(results) == 1
    assert results[0]["data_source"] == "rag"
    assert "第二百三十四条" in results[0]["article_number"]


@pytest.mark.asyncio
async def test_search_laws_by_rag_skips_unsupported_doc_types():
    """Unsupported document types (neither string nor page_content) should be skipped."""
    facts = {"behavior_sequence": ["盗窃"], "consequence": ""}
    # An int is neither a string nor has page_content
    docs = [42, "第二百六十四条 盗窃罪内容"]

    mock_rag_service = MagicMock()
    mock_rag_service.initialize_retriever = AsyncMock()
    mock_rag_service.get_documents_and_summary = AsyncMock(return_value={"documents": docs})

    with patch("app.rag.rag_service.RagService", return_value=mock_rag_service):
        results = await search_laws_by_rag(facts, "session-001")

    # Only the string document is kept
    assert len(results) == 1
    assert "第二百六十四条" in results[0]["article_number"]


@pytest.mark.asyncio
async def test_search_laws_by_rag_empty_query_uses_fallback():
    """When query is empty, default '刑事犯罪' should be used."""
    facts = {"behavior_sequence": [], "consequence": ""}
    captured_queries = []

    class _CapturingRag:
        async def initialize_retriever(self, q):
            captured_queries.append(q)

        async def get_documents_and_summary(self, q):
            captured_queries.append(q)
            return {"documents": []}

    with patch("app.rag.rag_service.RagService", return_value=_CapturingRag()):
        await search_laws_by_rag(facts, "session-001")

    assert "刑事犯罪" in captured_queries


@pytest.mark.asyncio
async def test_search_laws_by_rag_non_list_behavior():
    """RAG search should also handle non-list behavior_sequence."""
    facts = {"behavior_sequence": "string-behavior", "consequence": ""}
    mock_rag_service = MagicMock()
    mock_rag_service.initialize_retriever = AsyncMock()
    mock_rag_service.get_documents_and_summary = AsyncMock(return_value={"documents": []})

    with patch("app.rag.rag_service.RagService", return_value=mock_rag_service):
        results = await search_laws_by_rag(facts, "session-001")

    assert results == []


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

    with patch("app.agents.law_ref.llm_gateway") as mock_llm:
        mock_llm.generate = AsyncMock(return_value=llm_response)
        charges = await extract_structured_laws(matched, {"behavior_sequence": ["盗窃"], "consequence": "损失"})

    assert len(charges) == 1
    assert charges[0]["charge_name"] == "盗窃罪"
    assert charges[0]["probability"] == "high"


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
    with patch("app.agents.law_ref.llm_gateway") as mock_llm:
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
    with patch("app.agents.law_ref.llm_gateway") as mock_llm:
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
            "elements_matched": ["客体要件"],
            "elements_missing": ["主观要件"],
            "base_sentence": "三年以下",
            "probability": "medium",
        }
    ]
    matched = [
        {"article_number": "第264条", "data_source": "rag_verified"},
    ]
    applied = _build_applied_laws_from_structured(structured, matched)
    assert len(applied) == 1
    assert applied[0]["charge_name"] == "盗窃罪"
    assert applied[0]["data_source"] == "rag_verified"
    assert applied[0]["elements"] == ["客体要件"]
    assert applied[0]["elements_missing"] == ["主观要件"]


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


# ---------------------------------------------------------------------------
# _build_applied_laws_from_matched
# ---------------------------------------------------------------------------


def test_build_applied_laws_from_matched_basic():
    """_build_applied_laws_from_matched should mirror matched_laws structure."""
    matched = [
        {
            "article_number": "第264条",
            "title": "盗窃罪",
            "elements": ["A", "B"],
            "base_sentence": "三年以下",
            "charge_tags": ["盗窃"],
            "data_source": "json_keyword",
        }
    ]
    applied = _build_applied_laws_from_matched(matched)
    assert applied[0]["charge_name"] == "盗窃罪"
    assert applied[0]["data_source"] == "json_keyword"
    assert applied[0]["elements"] == ["A", "B"]


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


def test_merge_and_deduplicate_duplicate_in_primary():
    """Primary list with duplicate keys: the first occurrence is kept, others skipped."""
    primary = [
        {"article_number": "第264条", "title": "盗窃罪"},
        {"article_number": "第264条", "title": "盗窃罪（重复）"},
    ]
    secondary = []
    result = _merge_and_deduplicate(primary, secondary)
    assert len(result) == 1
    assert result[0]["title"] == "盗窃罪"


def test_merge_and_deduplicate_no_key():
    """Laws with neither article_number nor title are appended unconditionally."""
    primary = [{"article_number": "", "title": ""}]
    secondary = []
    result = _merge_and_deduplicate(primary, secondary)
    assert len(result) == 1


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

    with patch("app.agents.law_ref.load_criminal_law_data", return_value=_make_law_data()), \
         patch("app.agents.law_ref.search_laws_by_rag", new_callable=AsyncMock, return_value=rag_results), \
         patch("app.agents.law_ref.search_laws_by_keyword", new_callable=AsyncMock, return_value=[]), \
         patch("app.agents.law_ref.extract_structured_laws", new_callable=AsyncMock, return_value=structured_laws):

        result = await law_ref_node(state)

    assert result.get("current_agent") == "LawRef"
    applied_laws = result.get("applied_laws", [])
    assert len(applied_laws) == 1
    # data_source comes from rag_results which was rag_verified
    assert applied_laws[0]["data_source"] == "rag_verified"
    # history entry should be appended
    assert any(
        entry.get("agent") == "LawRef"
        for entry in result.get("conversation_history", [])
    )
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

    with patch("app.agents.law_ref.load_criminal_law_data", return_value=_make_law_data()), \
         patch("app.agents.law_ref.search_laws_by_rag", new_callable=AsyncMock, return_value=rag_results), \
         patch("app.agents.law_ref.search_laws_by_keyword", new_callable=AsyncMock, return_value=[]), \
         patch("app.agents.law_ref.extract_structured_laws", new_callable=AsyncMock, return_value=[]):

        result = await law_ref_node(state)

    # matched_laws is non-empty (the unverified RAG result), rag_verified == 0 → rag_only True
    assert result.get("rag_only") is True


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

    with patch("app.agents.law_ref.load_criminal_law_data", return_value=_make_law_data()), \
         patch("app.agents.law_ref.search_laws_by_rag", new_callable=AsyncMock, return_value=[]), \
         patch("app.agents.law_ref.search_laws_by_keyword", new_callable=AsyncMock, return_value=[]), \
         patch("app.agents.law_ref.extract_structured_laws", new_callable=AsyncMock, return_value=[]):

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

    with patch("app.agents.law_ref.load_criminal_law_data", return_value={"chapters": []}), \
         patch("app.agents.law_ref.search_laws_by_rag", new_callable=AsyncMock, return_value=[]) as rag_search, \
         patch("app.agents.law_ref.extract_structured_laws", new_callable=AsyncMock, return_value=[]):
        result = await law_ref_node(state)

    rag_search.assert_awaited_once_with(state["facts_structured"], "real-user-42")
    assert result["conversation_history"][-1]["session_id"] == "session-must-not-be-user"


@pytest.mark.asyncio
async def test_law_ref_node_missing_user_id_does_not_use_session_id():
    state = make_consultation_state(
        session_id="legacy-session",
        facts_structured={"behavior_sequence": ["盗窃"], "consequence": "财产损失"},
    )
    state.pop("user_id", None)

    with patch("app.agents.law_ref.load_criminal_law_data", return_value={"chapters": []}), \
         patch("app.agents.law_ref.search_laws_by_rag", new_callable=AsyncMock, return_value=[]) as rag_search, \
         patch("app.agents.law_ref.extract_structured_laws", new_callable=AsyncMock, return_value=[]):
        await law_ref_node(state)

    rag_search.assert_awaited_once_with(state["facts_structured"], None)
