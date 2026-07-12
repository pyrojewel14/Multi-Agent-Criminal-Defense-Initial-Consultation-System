"""验证修复后的数据流向：data_source 字段在 applied_laws 构建和覆盖率计算中的传递。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ============================================================
# 1. 测试 _is_unverified_rag_result
# ============================================================
print("=" * 60)
print("1. 测试 _is_unverified_rag_result")
print("=" * 60)

from app.agents.law_ref import _is_unverified_rag_result

test_cases = [
    ({"data_source": "rag_unverified"}, True, "未验证 RAG 结果"),
    ({"data_source": "rag_verified"}, False, "已验证 RAG 结果"),
    ({"data_source": "json_keyword"}, False, "JSON 关键词结果"),
    ({"data_source": "llm_extracted"}, False, "LLM 提取结果"),
    ({}, False, "无 data_source 字段"),
]

for law, expected, desc in test_cases:
    result = _is_unverified_rag_result(law)
    status = "PASS" if result == expected else "FAIL"
    print(f"  [{status}] {desc}: _is_unverified_rag_result({law}) = {result}, 期望 {expected}")

# ============================================================
# 2. 测试 _build_applied_laws_from_matched (回退路径)
# ============================================================
print()
print("=" * 60)
print("2. 测试 _build_applied_laws_from_matched (回退路径)")
print("=" * 60)

from app.agents.law_ref import _build_applied_laws_from_matched

matched_laws = [
    {
        "article_number": "第二百三十四条",
        "title": "故意伤害罪",
        "content": "...",
        "elements": ["故意", "伤害行为", "轻伤以上后果"],
        "base_sentence": "三年以下有期徒刑",
        "charge_tags": ["暴力犯罪"],
        "common_keywords": ["打人", "伤害"],
        "data_source": "rag_verified",
    },
    {
        "article_number": "第二百六十四条",
        "title": "盗窃罪",
        "content": "...",
        "elements": ["秘密窃取", "数额较大"],
        "base_sentence": "三年以下有期徒刑",
        "charge_tags": ["财产犯罪"],
        "common_keywords": ["偷", "盗窃"],
        "data_source": "rag_unverified",
    },
    {
        "article_number": "第二百三十二条",
        "title": "故意杀人罪",
        "content": "...",
        "elements": ["故意", "杀人行为"],
        "base_sentence": "死刑、无期徒刑或十年以上有期徒刑",
        "charge_tags": ["暴力犯罪"],
        "common_keywords": [],
        # 无 data_source 字段 — 应默认为 json_keyword
    },
]

applied_laws = _build_applied_laws_from_matched(matched_laws)

for i, law in enumerate(applied_laws):
    ds = law.get("data_source", "<缺失>")
    is_unverified = _is_unverified_rag_result(law)
    print(f"  法条 {i+1}: article_number={law['article_number']}, "
          f"charge_name={law['charge_name']}, data_source={ds}, "
          f"is_unverified={is_unverified}")

# 验证关键断言
assert applied_laws[0]["data_source"] == "rag_verified", "rag_verified 未透传"
assert applied_laws[1]["data_source"] == "rag_unverified", "rag_unverified 未透传"
assert applied_laws[2]["data_source"] == "json_keyword", "无 data_source 时未默认为 json_keyword"
print("  [PASS] 所有 data_source 透传正确")

# ============================================================
# 3. 测试 _build_applied_laws_from_structured (主路径)
# ============================================================
print()
print("=" * 60)
print("3. 测试 _build_applied_laws_from_structured (主路径)")
print("=" * 60)

from app.agents.law_ref import _build_applied_laws_from_structured

structured_laws = [
    {
        "charge_name": "故意伤害罪",
        "article_number": "第二百三十四条",
        "elements_matched": ["故意", "伤害行为"],
        "elements_missing": ["轻伤以上后果"],
        "base_sentence": "三年以下有期徒刑",
        "probability": "high",
    },
    {
        "charge_name": "盗窃罪",
        "article_number": "第二百六十四条",
        "elements_matched": ["秘密窃取"],
        "elements_missing": ["数额较大"],
        "base_sentence": "三年以下有期徒刑",
        "probability": "medium",
    },
    {
        "charge_name": "寻衅滋事罪",
        "article_number": "第二百九十三条",
        "elements_matched": ["随意殴打他人"],
        "elements_missing": ["情节恶劣"],
        "base_sentence": "五年以下有期徒刑",
        "probability": "low",
    },
]

applied_laws = _build_applied_laws_from_structured(structured_laws, matched_laws)

for i, law in enumerate(applied_laws):
    ds = law.get("data_source", "<缺失>")
    is_unverified = _is_unverified_rag_result(law)
    print(f"  法条 {i+1}: charge_name={law['charge_name']}, "
          f"article_number={law['article_number']}, data_source={ds}, "
          f"is_unverified={is_unverified}")

# 验证关键断言
assert applied_laws[0]["data_source"] == "rag_verified", \
    f"故意伤害罪应回填 rag_verified，实际为 {applied_laws[0]['data_source']}"
assert applied_laws[1]["data_source"] == "rag_unverified", \
    f"盗窃罪应回填 rag_unverified，实际为 {applied_laws[1]['data_source']}"
assert applied_laws[2]["data_source"] == "llm_extracted", \
    f"寻衅滋事罪无匹配来源，应为 llm_extracted，实际为 {applied_laws[2]['data_source']}"
print("  [PASS] 所有 data_source 回填正确")

# ============================================================
# 4. 测试 _analyze_coverage (fact_digger 覆盖率计算)
# ============================================================
print()
print("=" * 60)
print("4. 测试 _analyze_coverage (覆盖率计算)")
print("=" * 60)

import asyncio
from app.agents.fact_digger import _analyze_coverage

facts_structured = {
    "behavior_sequence": ["推搡对方", "对方倒地"],
    "consequence": "对方轻伤二级",
    "surrender": True,
    "arrest_status": "取保候审",
}

# 场景 A: 混合来源 — rag_verified + rag_unverified + json_keyword
mixed_applied_laws = [
    {
        "charge_name": "故意伤害罪",
        "article_number": "第234条",
        "elements": ["故意", "伤害行为", "轻伤以上后果"],
        "base_sentence": "三年以下有期徒刑",
        "data_source": "rag_verified",
    },
    {
        "charge_name": "盗窃罪",
        "article_number": "第264条",
        "elements": [],  # 未验证结果，elements 为空
        "base_sentence": "",
        "data_source": "rag_unverified",
    },
    {
        "charge_name": "故意杀人罪",
        "article_number": "第232条",
        "elements": ["故意", "杀人行为"],
        "base_sentence": "死刑",
        "data_source": "json_keyword",
    },
]

result_a = asyncio.run(_analyze_coverage(facts_structured, mixed_applied_laws))
print(f"  场景 A (混合来源):")
print(f"    source={result_a['source']}")
print(f"    json_law_count={result_a.get('json_law_count', 'N/A')}")
print(f"    rag_count={result_a.get('rag_count', 'N/A')}")
print(f"    total_elements={result_a['total_elements']}")
print(f"    covered_elements={result_a['covered_elements']}")
print(f"    coverage_rate={result_a['coverage_rate']:.2f}")
print(f"    missing_elements={result_a['missing_elements']}")

# 验证：rag_unverified 应被过滤，只有 rag_verified 和 json_keyword 参与
assert result_a.get("rag_count", 0) == 1, f"应过滤出 1 个 rag_unverified，实际 rag_count={result_a.get('rag_count')}"
assert result_a.get("json_law_count", 0) == 2, f"应有 2 个可靠结果，实际 json_law_count={result_a.get('json_law_count')}"
print("  [PASS] 场景 A: rag_unverified 被正确过滤")

# 场景 B: 全部为 rag_unverified
rag_only_laws = [
    {
        "charge_name": "未知罪名",
        "article_number": "",
        "elements": [],
        "base_sentence": "",
        "data_source": "rag_unverified",
    },
]

result_b = asyncio.run(_analyze_coverage(facts_structured, rag_only_laws))
print(f"  场景 B (全部 rag_unverified):")
print(f"    source={result_b['source']}")
print(f"    rag_count={result_b.get('rag_count', 'N/A')}")
print(f"    coverage_rate={result_b['coverage_rate']:.2f}")

assert result_b["source"] == "rag_only", f"应返回 rag_only，实际为 {result_b['source']}"
print("  [PASS] 场景 B: 全部 rag_unverified 时正确返回 rag_only")

# 场景 C: 无 applied_laws
result_c = asyncio.run(_analyze_coverage(facts_structured, []))
print(f"  场景 C (无 applied_laws):")
print(f"    source={result_c['source']}")
print(f"    coverage_rate={result_c['coverage_rate']:.2f}")

assert result_c["source"] == "no_laws", f"应返回 no_laws，实际为 {result_c['source']}"
print("  [PASS] 场景 C: 无 applied_laws 时正确返回 no_laws")

# ============================================================
# 总结
# ============================================================
print()
print("=" * 60)
print("ALL TESTS PASSED - 数据流向验证通过")
print("=" * 60)
