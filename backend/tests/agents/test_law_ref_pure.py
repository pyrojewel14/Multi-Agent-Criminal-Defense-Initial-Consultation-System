"""law_ref 模块纯函数单元测试。"""

import pytest

from app.agents.law_ref import (
    _build_article_index,
    _build_element_to_law_mapping,
    _cn_to_arabic,
    _extract_article_number_from_text,
    _is_unverified_rag_result,
    _merge_and_deduplicate,
    _normalize_article_number,
    _verify_and_enrich_with_json,
)


# ──────────────────────────── _cn_to_arabic ──────────────────────────────────


class TestCnToArabic:
    """_cn_to_arabic 测试。"""

    # --- 单个数字 ---

    def test_zero(self):
        assert _cn_to_arabic("零") == "0"

    def test_one(self):
        assert _cn_to_arabic("一") == "1"

    def test_two(self):
        assert _cn_to_arabic("二") == "2"

    def test_three(self):
        assert _cn_to_arabic("三") == "3"

    def test_four(self):
        assert _cn_to_arabic("四") == "4"

    def test_five(self):
        assert _cn_to_arabic("五") == "5"

    def test_six(self):
        assert _cn_to_arabic("六") == "6"

    def test_seven(self):
        assert _cn_to_arabic("七") == "7"

    def test_eight(self):
        assert _cn_to_arabic("八") == "8"

    def test_nine(self):
        assert _cn_to_arabic("九") == "9"

    # --- 复合数字 ---

    def test_ten(self):
        assert _cn_to_arabic("一十") == "10"

    def test_shi_alone(self):
        """'十' 单独出现应被当作 10。"""
        assert _cn_to_arabic("十") == "10"

    def test_twenty_three(self):
        assert _cn_to_arabic("二十三") == "23"

    def test_one_hundred_five(self):
        assert _cn_to_arabic("一百零五") == "105"

    def test_three_hundred_forty_two(self):
        assert _cn_to_arabic("三百四十二") == "342"

    def test_one_thousand_two_hundred_thirty_four(self):
        assert _cn_to_arabic("一千二百三十四") == "1234"

    def test_one_hundred(self):
        assert _cn_to_arabic("一百") == "100"

    def test_two_hundred(self):
        assert _cn_to_arabic("二百") == "200"

    def test_one_thousand(self):
        assert _cn_to_arabic("一千") == "1000"

    # --- 边界情况 ---

    def test_empty_string(self):
        assert _cn_to_arabic("") == ""

    def test_non_digit_chars_return_as_is(self):
        assert _cn_to_arabic("abc") == "abc"

    def test_mixed_chars_return_as_is(self):
        assert _cn_to_arabic("一百a") == "一百a"


# ──────────────────────────── _normalize_article_number ──────────────────────


class TestNormalizeArticleNumber:
    """_normalize_article_number 测试。"""

    def test_chinese_to_arabic(self):
        assert _normalize_article_number("第二百三十四条") == "第234条"

    def test_pure_digits(self):
        assert _normalize_article_number("第234条") == "第234条"

    def test_simple_chinese(self):
        assert _normalize_article_number("第三条") == "第3条"

    def test_ten(self):
        assert _normalize_article_number("第十条") == "第10条"

    def test_no_match_returns_stripped(self):
        assert _normalize_article_number("  其他  ") == "其他"

    def test_empty_string(self):
        assert _normalize_article_number("") == ""

    def test_article_twenty(self):
        assert _normalize_article_number("第二十条") == "第20条"


# ──────────────────────────── _extract_article_number_from_text ──────────────


class TestExtractArticleNumberFromText:
    """_extract_article_number_from_text 测试。"""

    def test_extract_arabic_number(self):
        assert _extract_article_number_from_text("根据第234条的规定") == "第234条"

    def test_extract_chinese_number(self):
        assert _extract_article_number_from_text("根据第二百三十四条的规定") == "第二百三十四条"

    def test_multiple_matches_returns_first(self):
        result = _extract_article_number_from_text("第1条和第2条都适用")
        assert result == "第1条"

    def test_no_match(self):
        assert _extract_article_number_from_text("没有法条编号") == ""

    def test_empty_string(self):
        assert _extract_article_number_from_text("") == ""


# ──────────────────────────── _build_article_index ───────────────────────────


class TestBuildArticleIndex:
    """_build_article_index 测试。"""

    def test_build_index_from_law_data(self):
        law_data = {
            "chapters": [
                {
                    "chapter": "总则",
                    "articles": [
                        {"article_number": "第1条", "title": "立法目的", "content": "内容1"},
                        {"article_number": "第二条", "title": "任务", "content": "内容2"},
                    ],
                },
                {
                    "chapter": "分则",
                    "articles": [
                        {"article_number": "第234条", "title": "故意伤害罪", "content": "内容3"},
                    ],
                },
            ]
        }
        index = _build_article_index(law_data)

        assert "第1条" in index
        assert "第2条" in index
        assert "第234条" in index
        assert index["第1条"]["title"] == "立法目的"
        assert index["第1条"]["chapter"] == "总则"
        assert index["第234条"]["chapter"] == "分则"

    def test_empty_chapters(self):
        index = _build_article_index({"chapters": []})
        assert index == {}

    def test_no_chapters_key(self):
        index = _build_article_index({})
        assert index == {}

    def test_empty_article_number_skipped(self):
        law_data = {
            "chapters": [
                {
                    "chapter": "总则",
                    "articles": [
                        {"article_number": "", "title": "无编号"},
                        {"article_number": "第1条", "title": "有编号"},
                    ],
                },
            ]
        }
        index = _build_article_index(law_data)
        assert len(index) == 1
        assert "第1条" in index


# ──────────────────────────── _verify_and_enrich_with_json ───────────────────


class TestVerifyAndEnrichWithJson:
    """_verify_and_enrich_with_json 测试。"""

    def test_rag_result_found_in_json(self):
        article_index = {
            "第234条": {
                "article_number": "第234条",
                "title": "故意伤害罪",
                "content": "故意伤害他人身体的...",
                "elements": ["故意", "伤害行为"],
                "base_sentence": "三年以下有期徒刑",
                "charge_tags": ["伤害"],
                "common_keywords": ["殴打"],
                "chapter": "分则",
            }
        }
        rag_results = [
            {
                "article_number": "第234条",
                "title": "",
                "content": "RAG内容",
                "elements": [],
                "base_sentence": "",
                "charge_tags": [],
                "common_keywords": [],
            }
        ]
        enriched = _verify_and_enrich_with_json(rag_results, article_index)

        assert len(enriched) == 1
        assert enriched[0]["data_source"] == "rag_verified"
        assert enriched[0]["title"] == "故意伤害罪"
        assert enriched[0]["content"] == "故意伤害他人身体的..."
        assert enriched[0]["elements"] == ["故意", "伤害行为"]
        assert enriched[0]["chapter"] == "分则"

    def test_rag_result_not_found_in_json(self):
        article_index = {"第1条": {"title": "立法目的"}}
        rag_results = [
            {
                "article_number": "第999条",
                "title": "未知",
                "content": "内容",
            }
        ]
        enriched = _verify_and_enrich_with_json(rag_results, article_index)

        assert len(enriched) == 1
        assert enriched[0]["data_source"] == "rag_unverified"
        assert enriched[0]["title"] == "未知"

    def test_empty_rag_results(self):
        enriched = _verify_and_enrich_with_json({}, {})
        assert enriched == []

    def test_chinese_article_number_normalized(self):
        """RAG 结果中法条编号为中文数字时，归一化后仍能匹配。"""
        article_index = {
            "第234条": {
                "article_number": "第234条",
                "title": "故意伤害罪",
                "content": "内容",
                "elements": [],
                "base_sentence": "",
                "charge_tags": [],
                "common_keywords": [],
                "chapter": "分则",
            }
        }
        rag_results = [
            {
                "article_number": "第二百三十四条",
                "title": "",
                "content": "RAG内容",
                "elements": [],
                "base_sentence": "",
                "charge_tags": [],
                "common_keywords": [],
            }
        ]
        enriched = _verify_and_enrich_with_json(rag_results, article_index)
        assert enriched[0]["data_source"] == "rag_verified"


# ──────────────────────────── _merge_and_deduplicate ─────────────────────────


class TestMergeAndDeduplicate:
    """_merge_and_deduplicate 测试。"""

    def test_merge_primary_and_secondary(self):
        primary = [{"article_number": "第1条", "title": "A"}]
        secondary = [{"article_number": "第2条", "title": "B"}]
        result = _merge_and_deduplicate(primary, secondary)
        assert len(result) == 2

    def test_dedup_by_article_number(self):
        primary = [{"article_number": "第1条", "title": "A"}]
        secondary = [{"article_number": "第1条", "title": "B"}]
        result = _merge_and_deduplicate(primary, secondary)
        assert len(result) == 1
        assert result[0]["title"] == "A"

    def test_primary_takes_precedence(self):
        primary = [{"article_number": "第1条", "title": "Primary", "data_source": "rag_verified"}]
        secondary = [{"article_number": "第1条", "title": "Secondary", "data_source": "json_keyword"}]
        result = _merge_and_deduplicate(primary, secondary)
        assert len(result) == 1
        assert result[0]["title"] == "Primary"

    def test_empty_lists(self):
        assert _merge_and_deduplicate([], []) == []

    def test_no_article_number_uses_title_as_key(self):
        primary = [{"article_number": "", "title": "故意伤害罪"}]
        secondary = [{"article_number": "", "title": "故意伤害罪"}]
        result = _merge_and_deduplicate(primary, secondary)
        assert len(result) == 1

    def test_no_number_no_title_added_directly(self):
        primary = [{"article_number": "", "title": "", "content": "X"}]
        result = _merge_and_deduplicate(primary, [])
        assert len(result) == 1

    def test_chinese_and_arabic_dedup(self):
        """中文和阿拉伯数字的法条编号归一化后应去重。"""
        primary = [{"article_number": "第234条", "title": "A"}]
        secondary = [{"article_number": "第二百三十四条", "title": "B"}]
        result = _merge_and_deduplicate(primary, secondary)
        assert len(result) == 1
        assert result[0]["title"] == "A"


# ──────────────────────────── _is_unverified_rag_result ──────────────────────


class TestIsUnverifiedRagResult:
    """_is_unverified_rag_result 测试。"""

    def test_unverified_returns_true(self):
        law = {"data_source": "rag_unverified"}
        assert _is_unverified_rag_result(law) is True

    def test_verified_returns_false(self):
        law = {"data_source": "rag_verified"}
        assert _is_unverified_rag_result(law) is False

    def test_json_keyword_returns_false(self):
        law = {"data_source": "json_keyword"}
        assert _is_unverified_rag_result(law) is False

    def test_no_data_source_returns_false(self):
        law = {}
        assert _is_unverified_rag_result(law) is False


# ──────────────────────────── _build_element_to_law_mapping ──────────────────


class TestBuildElementToLawMapping:
    """_build_element_to_law_mapping 测试。"""

    def test_build_mapping_with_elements(self):
        laws = [
            {
                "charge_name": "故意伤害罪",
                "article_number": "第234条",
                "elements": ["故意", "伤害行为", "轻伤以上"],
                "base_sentence": "三年以下有期徒刑",
            }
        ]
        mapping = _build_element_to_law_mapping(laws, "elements")

        assert "故意" in mapping
        assert "伤害行为" in mapping
        assert "轻伤以上" in mapping
        assert mapping["故意"]["charge_name"] == "故意伤害罪"
        assert mapping["故意"]["article_number"] == "第234条"

    def test_build_mapping_with_elements_matched_key(self):
        laws = [
            {
                "charge_name": "盗窃罪",
                "article_number": "第264条",
                "elements_matched": ["非法占有", "秘密窃取"],
                "base_sentence": "三年以下有期徒刑",
            }
        ]
        mapping = _build_element_to_law_mapping(laws, "elements_matched")

        assert "非法占有" in mapping
        assert mapping["非法占有"]["article_number"] == "第264条"

    def test_empty_laws(self):
        mapping = _build_element_to_law_mapping([], "elements")
        assert mapping == {}

    def test_law_without_elements(self):
        laws = [{"charge_name": "测试罪", "article_number": "第1条", "base_sentence": ""}]
        mapping = _build_element_to_law_mapping(laws, "elements")
        assert mapping == {}

    def test_first_occurrence_wins_for_duplicate_element(self):
        laws = [
            {"charge_name": "罪A", "article_number": "第1条", "elements": ["共同要件"], "base_sentence": "刑A"},
            {"charge_name": "罪B", "article_number": "第2条", "elements": ["共同要件"], "base_sentence": "刑B"},
        ]
        mapping = _build_element_to_law_mapping(laws, "elements")
        assert mapping["共同要件"]["charge_name"] == "罪A"

    def test_title_used_as_fallback_for_charge_name(self):
        laws = [
            {
                "title": "故意伤害罪",
                "article_number": "第234条",
                "elements": ["故意"],
                "base_sentence": "",
            }
        ]
        mapping = _build_element_to_law_mapping(laws, "elements")
        assert mapping["故意"]["charge_name"] == "故意伤害罪"
