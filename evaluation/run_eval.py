#!/usr/bin/env python3
"""运行 Phase 4 小规模、可复现的离线基线评估。"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import re
import shlex
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.agents.human_alert import human_alert_node  # noqa: E402
from app.agents.law_ref import (  # noqa: E402
    _normalize_article_number,
    load_criminal_law_data,
    search_laws_by_keyword,
)
from app.orchestrator.workflow import check_facts_sufficient  # noqa: E402
from app.security.disclaimer import DISCLAIMER_PREFIX, disclaimer  # noqa: E402
from app.security.sensitive_filter import detect_high_risk  # noqa: E402


MODE = "offline-deterministic-baseline"
DEFAULT_CASES_PATH = Path(__file__).resolve().with_name("cases.jsonl")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().with_name("results")
TOP_K = 5

CATEGORY_COUNTS = {
    "ordinary_consultation": 10,
    "missing_facts": 8,
    "high_risk_human": 6,
    "ambiguous_refusal": 6,
}
CASE_FIELDS = {
    "id",
    "category",
    "input",
    "expected_fact_fields",
    "expected_law_keywords",
    "should_follow_up",
    "should_trigger_human",
    "expected_risk_level",
    "expected_refusal",
    "expected_disclaimer",
}
RISK_LEVELS = {"low", "medium", "high", "unknown"}
FACT_FIELDS = {
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
FOLLOW_UP_FIELDS = (
    "incident_time",
    "incident_location",
    "behavior_sequence",
    "consequence",
    "evidence_mentioned",
    "arrest_status",
)
TIME_TERMS = (
    "昨晚",
    "昨天",
    "上周末",
    "上周",
    "前天晚上",
    "前天",
    "三个月前",
    "两天前",
    "上个月",
    "去年",
    "今年四月",
    "今年",
    "凌晨",
    "晚上",
)
LOCATION_ALIASES = {
    "网上": "网络",
    "网络": "网络",
    "超市": "超市",
    "酒吧": "酒吧",
    "小区门口": "小区门口",
    "道路": "道路",
    "酒店房间": "酒店房间",
    "办公室": "办公室",
    "公司": "公司",
    "商场": "商场",
    "夜宵店": "夜宵店",
    "派出所": "派出所",
}
EVIDENCE_TERMS = (
    "监控",
    "诊断证明",
    "聊天记录",
    "银行流水",
    "转账流水",
    "行车记录仪",
    "事故认定书",
    "检测报告",
    "账本",
    "转账记录",
    "短信",
    "录音",
    "维修发票",
)
ARREST_TERMS = (
    "刑事拘留",
    "已被公安传唤",
    "尚未被传唤",
    "取保候审",
    "已被留置",
    "尚未到案",
    "警方已经立案",
    "已经报警",
    "到派出所投案",
)
CONSEQUENCE_PATTERN = re.compile(
    r"(?:\d+[万千百]?元|轻伤|重伤|死亡|致死|损失|骗走|转走|追回|"
    r"没有发生事故|血液酒精|获利|砸坏|毁坏|被扣押)"
)
OUT_OF_SCOPE_TERMS = ("天气", "租房", "押金", "民事纠纷")
LEGAL_CONTEXT_TERMS = ("犯罪", "刑事", "判几年", "公安", "报警", "案情", "无罪")
CERTAINTY_PATTERNS = (
    re.compile(r"保证.*无罪"),
    re.compile(r"不需要.*案情.*直接.*结论"),
)


def load_cases(path: Path = DEFAULT_CASES_PATH) -> list[dict[str, Any]]:
    """读取 JSONL，并保留行号以便报告格式错误。"""
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 格式错误: {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"case 必须是对象: {path}:{line_number}")
            cases.append(value)
    return cases


def validate_cases(cases: list[dict[str, Any]]) -> None:
    """校验 Phase 4 数据规模、分布和统一 schema。"""
    errors: list[str] = []
    if len(cases) != 30:
        errors.append(f"样本数应为 30，实际为 {len(cases)}")

    counts = Counter(case.get("category") for case in cases)
    if dict(counts) != CATEGORY_COUNTS:
        errors.append(f"类别分布应为 {CATEGORY_COUNTS}，实际为 {dict(counts)}")

    ids = [case.get("id") for case in cases]
    if len(ids) != len(set(ids)):
        errors.append("case ID 必须唯一")

    for index, case in enumerate(cases, 1):
        case_id = case.get("id", f"line-{index}")
        if set(case) != CASE_FIELDS:
            errors.append(f"{case_id}: schema 字段不一致")
        if not isinstance(case.get("id"), str) or not case.get("id"):
            errors.append(f"{case_id}: id 必须是非空字符串")
        if not isinstance(case.get("input"), str) or not case.get("input", "").strip():
            errors.append(f"{case_id}: input 必须是非空字符串")
        for list_field in ("expected_fact_fields", "expected_law_keywords"):
            value = case.get(list_field)
            if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
                errors.append(f"{case_id}: {list_field} 必须是字符串列表")
        unknown_fact_fields = set(case.get("expected_fact_fields", [])) - FACT_FIELDS
        if unknown_fact_fields:
            errors.append(f"{case_id}: 未知事实字段 {sorted(unknown_fact_fields)}")
        for bool_field in (
            "should_follow_up",
            "should_trigger_human",
            "expected_refusal",
            "expected_disclaimer",
        ):
            if not isinstance(case.get(bool_field), bool):
                errors.append(f"{case_id}: {bool_field} 必须是布尔值")
        if case.get("expected_risk_level") not in RISK_LEVELS:
            errors.append(f"{case_id}: expected_risk_level 值无效")

    if errors:
        raise ValueError("评估集校验失败:\n- " + "\n- ".join(errors))


def _first_term(text: str, terms: tuple[str, ...]) -> str | None:
    return next((term for term in terms if term in text), None)


def _law_terms(law_data: dict[str, Any]) -> list[str]:
    terms: set[str] = set()
    for chapter in law_data.get("chapters", []):
        for article in chapter.get("articles", []):
            title = str(article.get("title", ""))
            if title:
                terms.add(title.removesuffix("罪"))
            terms.update(str(item) for item in article.get("charge_tags", []) if item)
            terms.update(str(item) for item in article.get("common_keywords", []) if item)
    return sorted((term for term in terms if len(term) >= 2), key=lambda item: (-len(item), item))


def extract_facts_baseline(input_text: str, law_data: dict[str, Any]) -> dict[str, Any]:
    """仅从 input 提取有限事实，作为可审计的离线 baseline。"""
    facts: dict[str, Any] = {}

    time_value = _first_term(input_text, TIME_TERMS)
    if time_value:
        facts["incident_time"] = time_value

    for source, normalized in LOCATION_ALIASES.items():
        if source in input_text:
            facts["incident_location"] = normalized
            break

    behavior = [term for term in _law_terms(law_data) if term in input_text]
    if behavior:
        facts["behavior_sequence"] = behavior[:8]

    if CONSEQUENCE_PATTERN.search(input_text):
        facts["consequence"] = input_text

    evidence = [term for term in EVIDENCE_TERMS if term in input_text]
    if evidence:
        facts["evidence_mentioned"] = evidence

    arrest_status = _first_term(input_text, ARREST_TERMS)
    if arrest_status:
        facts["arrest_status"] = arrest_status

    if "投案" in input_text or "自首" in input_text:
        facts["surrender"] = True
    if "谅解" in input_text:
        facts["victim_forgiveness"] = "未谅解" not in input_text
    if "无前科" in input_text or "有前科" in input_text:
        facts["prior_record"] = "有前科" in input_text

    return facts


def _is_present(value: Any) -> bool:
    return value is not None and value != "" and value != []


def _should_refuse_baseline(
    input_text: str,
    facts: dict[str, Any],
    retrieved_laws: list[dict[str, Any]],
) -> bool:
    """识别明显超范围或无法形成刑事咨询输入的请求。"""
    compact = re.sub(r"\s+", "", input_text)
    if len(compact) <= 4:
        return True
    if any(term in input_text for term in OUT_OF_SCOPE_TERMS):
        return True
    if any(pattern.search(input_text) for pattern in CERTAINTY_PATTERNS):
        return True
    has_legal_context = any(term in input_text for term in LEGAL_CONTEXT_TERMS)
    return not retrieved_laws and not facts and not has_legal_context


def _follow_up_prediction(facts: dict[str, Any]) -> tuple[bool, float, str]:
    covered = sum(1 for field in FOLLOW_UP_FIELDS if _is_present(facts.get(field)))
    coverage_rate = covered / len(FOLLOW_UP_FIELDS)
    route = check_facts_sufficient(
        {
            "facts_coverage_rate": coverage_rate,
            "fact_law_loop_count": 0,
            "alert_triggered": False,
        }
    )
    return route == "loop", coverage_rate, route


async def predict_input(
    input_text: str,
    law_data: dict[str, Any],
    top_k: int = TOP_K,
) -> dict[str, Any]:
    """只接收被评估 input，返回与 gold 隔离的预测。"""
    high_risk, risk_type = detect_high_risk(input_text)
    if high_risk:
        alert_state: dict[str, Any] = {
            "session_id": "evaluation-offline",
            "user_id": "evaluation-offline",
            "conversation_history": [],
            "risk_assessment": {"risk_type": risk_type, "risk_level": "high"},
        }
        alert_result = await human_alert_node(alert_state)
        output = str(alert_result.get("final_output", ""))
        return {
            "facts_structured": {},
            "retrieved_laws": [],
            "should_follow_up": False,
            "should_trigger_human": True,
            "risk_type": risk_type,
            "refused": False,
            "disclaimer_present": output.startswith(DISCLAIMER_PREFIX),
            "action": "human_alert",
            "baseline_fact_coverage": None,
            "workflow_route": "alert",
        }

    facts = extract_facts_baseline(input_text, law_data)
    matched_laws = await search_laws_by_keyword(facts, law_data)
    retrieved_laws = [
        {
            "article_number": law.get("article_number", ""),
            "title": law.get("title", ""),
            "charge_tags": law.get("charge_tags", []),
            "common_keywords": law.get("common_keywords", []),
        }
        for law in matched_laws[:top_k]
    ]
    refused = _should_refuse_baseline(input_text, facts, retrieved_laws)
    should_follow_up, coverage_rate, route = _follow_up_prediction(facts)
    if refused:
        should_follow_up = False
        route = "refuse"
        response = "当前输入不足以形成刑事初期咨询判断，无法继续给出结论。"
        action = "refuse"
    elif should_follow_up:
        response = "为了更准确地分析案件，请补充时间、地点、行为、后果、证据和当前程序状态。"
        action = "follow_up"
    else:
        response = "已形成离线基线结构化结果，后续结论仍需律师确认。"
        action = "continue"
    output = disclaimer.inject(response)

    return {
        "facts_structured": facts,
        "retrieved_laws": retrieved_laws,
        "should_follow_up": should_follow_up,
        "should_trigger_human": False,
        "risk_type": "",
        "refused": refused,
        "disclaimer_present": output.startswith(DISCLAIMER_PREFIX),
        "action": action,
        "baseline_fact_coverage": coverage_rate,
        "workflow_route": route,
    }


def _keyword_hit(keyword: str, laws: list[dict[str, Any]]) -> bool:
    normalized_keyword = _normalize_article_number(keyword)
    if normalized_keyword != keyword or re.fullmatch(r"第\d+条(?:之.+)?", keyword):
        return any(
            _normalize_article_number(str(law.get("article_number", ""))) == normalized_keyword
            for law in laws
        )

    needle = keyword.casefold()
    for law in laws:
        values = [
            law.get("title", ""),
            *law.get("charge_tags", []),
            *law.get("common_keywords", []),
        ]
        if any(needle in str(value).casefold() for value in values):
            return True
    return False


def score_case(case: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    """仅在预测完成后读取 gold，并计算单 case 得分。"""
    expected_fields = case["expected_fact_fields"]
    facts = prediction["facts_structured"]
    missing_fields = [field for field in expected_fields if not _is_present(facts.get(field))]

    expected_keywords = case["expected_law_keywords"]
    missed_keywords = [
        keyword for keyword in expected_keywords if not _keyword_hit(keyword, prediction["retrieved_laws"])
    ]

    return {
        "fact_fields_hit": len(expected_fields) - len(missing_fields),
        "fact_fields_total": len(expected_fields),
        "missing_fact_fields": missing_fields,
        "law_keywords_hit": len(expected_keywords) - len(missed_keywords),
        "law_keywords_total": len(expected_keywords),
        "missed_law_keywords": missed_keywords,
        "human_match": prediction["should_trigger_human"] == case["should_trigger_human"],
        "follow_up_match": prediction["should_follow_up"] == case["should_follow_up"],
        "refusal_match": prediction["refused"] == case["expected_refusal"],
        "disclaimer_match": (
            prediction["disclaimer_present"] if case["expected_disclaimer"] else not prediction["disclaimer_present"]
        ),
    }


async def evaluate_case(case: dict[str, Any], law_data: dict[str, Any]) -> dict[str, Any]:
    """先以 input 预测，再在独立步骤读取 gold 评分。"""
    prediction = await predict_input(case["input"], law_data)
    scores = score_case(case, prediction)
    return {
        "id": case["id"],
        "category": case["category"],
        "input": case["input"],
        "gold": {
            key: case[key]
            for key in sorted(CASE_FIELDS - {"id", "category", "input"})
        },
        "prediction": prediction,
        "scores": scores,
    }


def _metric(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """按显式分子、分母聚合 Phase 4 指标。"""
    fact_numerator = sum(result["scores"]["fact_fields_hit"] for result in results)
    fact_denominator = sum(result["scores"]["fact_fields_total"] for result in results)
    law_numerator = sum(result["scores"]["law_keywords_hit"] for result in results)
    law_denominator = sum(result["scores"]["law_keywords_total"] for result in results)
    human_numerator = sum(bool(result["scores"]["human_match"]) for result in results)
    follow_up_numerator = sum(bool(result["scores"]["follow_up_match"]) for result in results)
    refusal_numerator = sum(bool(result["scores"]["refusal_match"]) for result in results)
    disclaimer_applicable = [result for result in results if result["gold"]["expected_disclaimer"]]
    disclaimer_numerator = sum(bool(result["prediction"]["disclaimer_present"]) for result in disclaimer_applicable)
    refusal_positive = [result for result in results if result["gold"]["expected_refusal"]]
    refusal_positive_numerator = sum(bool(result["prediction"]["refused"]) for result in refusal_positive)

    return {
        "fact_field_extraction_coverage": _metric(fact_numerator, fact_denominator),
        "law_keyword_hit_at_5": _metric(law_numerator, law_denominator),
        "high_risk_trigger_accuracy": _metric(human_numerator, len(results)),
        "follow_up_trigger_accuracy": _metric(follow_up_numerator, len(results)),
        "refusal_trigger_accuracy": _metric(refusal_numerator, len(results)),
        "disclaimer_trigger_rate": _metric(disclaimer_numerator, len(disclaimer_applicable)),
        "refusal_disclaimer_trigger_rate": _metric(
            refusal_positive_numerator + disclaimer_numerator,
            len(refusal_positive) + len(disclaimer_applicable),
        ),
    }


def collect_failures(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """按指标收集失败 case 和可审计原因。"""
    failures: dict[str, list[dict[str, Any]]] = {
        "fact_field_extraction": [],
        "law_keyword_hit_at_5": [],
        "high_risk_trigger": [],
        "follow_up_trigger": [],
        "refusal_trigger": [],
        "disclaimer_trigger": [],
    }
    for result in results:
        scores = result["scores"]
        prediction = result["prediction"]
        gold = result["gold"]
        if scores["missing_fact_fields"]:
            failures["fact_field_extraction"].append(
                {"id": result["id"], "reason": f"缺失字段: {', '.join(scores['missing_fact_fields'])}"}
            )
        if scores["missed_law_keywords"]:
            failures["law_keyword_hit_at_5"].append(
                {"id": result["id"], "reason": f"未命中: {', '.join(scores['missed_law_keywords'])}"}
            )
        if not scores["human_match"]:
            failures["high_risk_trigger"].append(
                {
                    "id": result["id"],
                    "reason": (
                        f"gold={gold['should_trigger_human']}, "
                        f"prediction={prediction['should_trigger_human']}, risk_type={prediction['risk_type'] or '-'}"
                    ),
                }
            )
        if not scores["follow_up_match"]:
            failures["follow_up_trigger"].append(
                {
                    "id": result["id"],
                    "reason": f"gold={gold['should_follow_up']}, prediction={prediction['should_follow_up']}",
                }
            )
        if not scores["refusal_match"]:
            failures["refusal_trigger"].append(
                {
                    "id": result["id"],
                    "reason": f"gold={gold['expected_refusal']}, prediction={prediction['refused']}",
                }
            )
        if not scores["disclaimer_match"]:
            failures["disclaimer_trigger"].append(
                {
                    "id": result["id"],
                    "reason": f"gold={gold['expected_disclaimer']}, prediction={prediction['disclaimer_present']}",
                }
            )
    return failures


def build_diagnostics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """输出高风险触发混淆矩阵，显式展示 false positive。"""
    true_positive = sum(
        result["gold"]["should_trigger_human"]
        and result["prediction"]["should_trigger_human"]
        for result in results
    )
    true_negative = sum(
        not result["gold"]["should_trigger_human"]
        and not result["prediction"]["should_trigger_human"]
        for result in results
    )
    false_positive = sum(
        not result["gold"]["should_trigger_human"]
        and result["prediction"]["should_trigger_human"]
        for result in results
    )
    false_negative = sum(
        result["gold"]["should_trigger_human"]
        and not result["prediction"]["should_trigger_human"]
        for result in results
    )
    negative_count = true_negative + false_positive
    return {
        "high_risk_confusion_matrix": {
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
        },
        "high_risk_false_positive_rate": _metric(false_positive, negative_count),
    }


def _format_percent(metric: dict[str, Any]) -> str:
    value = metric["value"]
    return "N/A" if value is None else f"{value * 100:.1f}%"


def _dataset_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_summary(
    cases_path: Path,
    results: list[dict[str, Any]],
    command: str,
) -> dict[str, Any]:
    metrics = compute_metrics(results)
    return {
        "metadata": {
            "mode": MODE,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "top_k": TOP_K,
            "cases_path": str(cases_path.resolve()),
            "cases_sha256": _dataset_sha256(cases_path),
        },
        "dataset": {
            "sample_count": len(results),
            "category_distribution": dict(Counter(result["category"] for result in results)),
        },
        "metrics": metrics,
        "diagnostics": build_diagnostics(results),
        "not_evaluated": {
            "expected_risk_level": (
                "RiskAssessor 依赖 LLM，当前无稳定离线预测接口；gold 字段保留但不计算风险等级准确率。"
            ),
            "live_llm_rag": "本模式不调用 LLM、Ollama、向量库或 reranker。",
        },
        "failures": collect_failures(results),
    }


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(results: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "category",
        "action",
        "predicted_fact_fields",
        "retrieved_articles",
        "missing_fact_fields",
        "missed_law_keywords",
        "human_match",
        "follow_up_match",
        "refusal_match",
        "disclaimer_match",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            prediction = result["prediction"]
            scores = result["scores"]
            writer.writerow(
                {
                    "id": result["id"],
                    "category": result["category"],
                    "action": prediction["action"],
                    "predicted_fact_fields": "|".join(prediction["facts_structured"]),
                    "retrieved_articles": "|".join(
                        str(law.get("article_number", "")) for law in prediction["retrieved_laws"]
                    ),
                    "missing_fact_fields": "|".join(scores["missing_fact_fields"]),
                    "missed_law_keywords": "|".join(scores["missed_law_keywords"]),
                    "human_match": scores["human_match"],
                    "follow_up_match": scores["follow_up_match"],
                    "refusal_match": scores["refusal_match"],
                    "disclaimer_match": scores["disclaimer_match"],
                }
            )


def write_report(summary: dict[str, Any], path: Path) -> None:
    metadata = summary["metadata"]
    dataset = summary["dataset"]
    metrics = summary["metrics"]
    diagnostics = summary["diagnostics"]
    failures = summary["failures"]
    lines = [
        "# Phase 4 Evaluation Report",
        "",
        "## Run",
        "",
        f"- Mode: `{metadata['mode']}`",
        f"- Time (UTC): `{metadata['generated_at']}`",
        f"- Command: `{metadata['command']}`",
        f"- Cases: {dataset['sample_count']}",
        f"- Distribution: `{json.dumps(dataset['category_distribution'], ensure_ascii=False)}`",
        f"- Dataset SHA-256: `{metadata['cases_sha256']}`",
        "",
        "## Metrics",
        "",
        "| Metric | Result | Numerator | Denominator |",
        "| --- | ---: | ---: | ---: |",
    ]
    metric_labels = {
        "fact_field_extraction_coverage": "Fact field extraction coverage",
        "law_keyword_hit_at_5": "Law keyword hit@5",
        "high_risk_trigger_accuracy": "High-risk trigger accuracy",
        "follow_up_trigger_accuracy": "Follow-up trigger accuracy",
        "refusal_trigger_accuracy": "Refusal trigger accuracy",
        "disclaimer_trigger_rate": "Disclaimer trigger rate",
        "refusal_disclaimer_trigger_rate": "Refusal/disclaimer trigger rate",
    }
    for key, label in metric_labels.items():
        metric = metrics[key]
        lines.append(
            f"| {label} | {_format_percent(metric)} | {metric['numerator']} | {metric['denominator']} |"
        )

    confusion = diagnostics["high_risk_confusion_matrix"]
    false_positive_rate = diagnostics["high_risk_false_positive_rate"]
    lines.extend(
        [
            "",
            "## High-risk Diagnostics",
            "",
            (
                "- Confusion matrix: "
                f"TP={confusion['true_positive']}, TN={confusion['true_negative']}, "
                f"FP={confusion['false_positive']}, FN={confusion['false_negative']}"
            ),
            (
                "- False-positive rate: "
                f"{_format_percent(false_positive_rate)} "
                f"({false_positive_rate['numerator']}/{false_positive_rate['denominator']})"
            ),
        ]
    )

    lines.extend(["", "## Failures", ""])
    any_failure = False
    for metric_name, entries in failures.items():
        if not entries:
            continue
        any_failure = True
        lines.append(f"### {metric_name}")
        lines.append("")
        lines.extend(f"- `{entry['id']}`: {entry['reason']}" for entry in entries)
        lines.append("")
    if not any_failure:
        lines.append("- None.")

    lines.extend(
        [
            "## Boundaries",
            "",
            "- Facts are predicted by a deterministic lexical baseline that receives only `input`; it is not the LLM-backed FactDigger extractor.",
            "- Law candidates come from the real `search_laws_by_keyword()` implementation and the local JSON law knowledge base, using only baseline-predicted facts.",
            "- High-risk predictions use the real `detect_high_risk()` function; positive cases render the real HumanAlert response.",
            "- Follow-up routing reuses `check_facts_sufficient()`, but its coverage value comes from six fixed baseline fields rather than LLM/legal-element coverage.",
            "- Disclaimer output uses the real `DisclaimerService`; refusal classification is a deterministic baseline because the project has no standalone refusal node.",
            "- `expected_risk_level` is not evaluated because RiskAssessor requires an LLM and has no stable deterministic offline interface.",
            "- This small offline baseline does not establish open-domain accuracy, live RAG recall, real-user effectiveness, performance, or production stability.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run(cases_path: Path, output_dir: Path, command: str) -> dict[str, Any]:
    cases = load_cases(cases_path)
    validate_cases(cases)
    law_data = load_criminal_law_data()
    results = [await evaluate_case(case, law_data) for case in cases]
    summary = build_summary(cases_path, results, command)

    write_json(summary, output_dir / "summary.json")
    write_json(results, output_dir / "case_results.json")
    write_csv(results, output_dir / "case_results.csv")
    write_report(summary, output_dir / "report.md")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = " ".join(shlex.quote(item) for item in ["python", *sys.argv])
    summary = asyncio.run(run(args.cases, args.output_dir, command))
    print(f"Mode: {summary['metadata']['mode']}")
    print(f"Cases: {summary['dataset']['sample_count']}")
    for name, metric in summary["metrics"].items():
        print(
            f"{name}: {_format_percent(metric)} "
            f"({metric['numerator']}/{metric['denominator']})"
        )
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
