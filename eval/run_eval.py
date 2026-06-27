#!/usr/bin/env python3
"""Offline MVP evaluation for the criminal defense consultation system."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.agents.fact_digger import _analyze_coverage  # noqa: E402
from app.agents.law_ref import (  # noqa: E402
    _build_applied_laws_from_matched,
    _normalize_article_number,
    load_criminal_law_data,
    search_laws_by_keyword,
)
from app.security.sensitive_filter import detect_high_risk, mask_pii  # noqa: E402


DEFAULT_CASES_PATH = ROOT_DIR / "eval" / "cases.jsonl"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "eval" / "results"


@dataclass
class EvalResult:
    case_id: str
    category: str
    expected_articles: list[str]
    retrieved_articles: list[str]
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    law_hit: bool
    fact_required_coverage: float
    element_coverage: float
    element_source: str
    high_risk_expected: bool
    high_risk_detected: bool
    high_risk_match: bool
    risk_type: str
    pii_mask_expected: bool
    pii_masked: bool
    pii_mask_match: bool
    pass_eval: bool


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return cases


def normalize_article(article: str) -> str:
    return _normalize_article_number(article)


def recall_at_k(expected: list[str], retrieved: list[str], k: int) -> float:
    if not expected:
        return 1.0
    expected_set = {normalize_article(item) for item in expected}
    retrieved_set = {normalize_article(item) for item in retrieved[:k]}
    return len(expected_set & retrieved_set) / len(expected_set)


def reciprocal_rank(expected: list[str], retrieved: list[str]) -> float:
    if not expected:
        return 1.0
    expected_set = {normalize_article(item) for item in expected}
    for idx, article in enumerate(retrieved, 1):
        if normalize_article(article) in expected_set:
            return 1.0 / idx
    return 0.0


def required_fact_coverage(facts: dict[str, Any], required_keys: list[str]) -> float:
    if not required_keys:
        return 1.0

    covered = 0
    for key in required_keys:
        value = facts.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, list) and not value:
            continue
        covered += 1
    return covered / len(required_keys)


def did_mask_pii(text: str) -> bool:
    return mask_pii(text) != text


async def evaluate_case(case: dict[str, Any], law_data: dict[str, Any]) -> EvalResult:
    facts = case.get("facts_structured", {})
    expected_articles = case.get("expected_articles", [])

    matched_laws = await search_laws_by_keyword(facts, law_data)
    retrieved_articles = [law.get("article_number", "") for law in matched_laws if law.get("article_number")]

    applied_laws = _build_applied_laws_from_matched(matched_laws[:5])
    coverage = await _analyze_coverage(facts, applied_laws)

    high_risk_detected, risk_type = detect_high_risk(case.get("user_input", ""))
    pii_masked = did_mask_pii(case.get("user_input", ""))

    recall1 = recall_at_k(expected_articles, retrieved_articles, 1)
    recall3 = recall_at_k(expected_articles, retrieved_articles, 3)
    recall5 = recall_at_k(expected_articles, retrieved_articles, 5)
    fact_cov = required_fact_coverage(facts, case.get("required_fact_keys", []))
    element_cov = float(coverage.get("coverage_rate", 0.0))

    high_risk_expected = bool(case.get("expected_high_risk", False))
    pii_mask_expected = bool(case.get("expected_pii_mask", False))
    law_hit = recall5 >= 1.0
    high_risk_match = high_risk_detected == high_risk_expected
    pii_mask_match = pii_masked == pii_mask_expected

    pass_eval = (
        law_hit
        and fact_cov >= 0.6
        and high_risk_match
        and pii_mask_match
    )

    return EvalResult(
        case_id=case.get("case_id", ""),
        category=case.get("category", ""),
        expected_articles=expected_articles,
        retrieved_articles=retrieved_articles[:5],
        recall_at_1=recall1,
        recall_at_3=recall3,
        recall_at_5=recall5,
        mrr=reciprocal_rank(expected_articles, retrieved_articles),
        law_hit=law_hit,
        fact_required_coverage=fact_cov,
        element_coverage=element_cov,
        element_source=str(coverage.get("source", "")),
        high_risk_expected=high_risk_expected,
        high_risk_detected=high_risk_detected,
        high_risk_match=high_risk_match,
        risk_type=risk_type,
        pii_mask_expected=pii_mask_expected,
        pii_masked=pii_masked,
        pii_mask_match=pii_mask_match,
        pass_eval=pass_eval,
    )


def write_csv(results: list[EvalResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(EvalResult.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = result.__dict__.copy()
            row["expected_articles"] = "|".join(row["expected_articles"])
            row["retrieved_articles"] = "|".join(row["retrieved_articles"])
            writer.writerow(row)


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def write_report(results: list[EvalResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    total = len(results)
    pass_rate = mean([1.0 if item.pass_eval else 0.0 for item in results]) if results else 0.0
    recall1 = mean([item.recall_at_1 for item in results]) if results else 0.0
    recall3 = mean([item.recall_at_3 for item in results]) if results else 0.0
    recall5 = mean([item.recall_at_5 for item in results]) if results else 0.0
    mrr = mean([item.mrr for item in results]) if results else 0.0
    fact_cov = mean([item.fact_required_coverage for item in results]) if results else 0.0
    element_cov = mean([item.element_coverage for item in results]) if results else 0.0

    high_risk_cases = [item for item in results if item.high_risk_expected]
    high_risk_recall = (
        mean([1.0 if item.high_risk_detected else 0.0 for item in high_risk_cases])
        if high_risk_cases
        else 1.0
    )
    high_risk_negative_cases = [item for item in results if not item.high_risk_expected]
    high_risk_false_positive_rate = (
        mean([1.0 if item.high_risk_detected else 0.0 for item in high_risk_negative_cases])
        if high_risk_negative_cases
        else 0.0
    )
    pii_cases = [item for item in results if item.pii_mask_expected]
    pii_recall = mean([1.0 if item.pii_masked else 0.0 for item in pii_cases]) if pii_cases else 1.0
    pii_negative_cases = [item for item in results if not item.pii_mask_expected]
    pii_false_positive_rate = (
        mean([1.0 if item.pii_masked else 0.0 for item in pii_negative_cases])
        if pii_negative_cases
        else 0.0
    )

    lines = [
        "# Evaluation MVP Report",
        "",
        "## Summary",
        "",
        f"- Cases: {total}",
        f"- Pass rate: {pct(pass_rate)}",
        f"- Law Recall@1 / @3 / @5: {pct(recall1)} / {pct(recall3)} / {pct(recall5)}",
        f"- Law MRR: {mrr:.3f}",
        f"- Required fact coverage: {pct(fact_cov)}",
        f"- Raw legal element coverage: {pct(element_cov)}",
        f"- High-risk trigger recall: {pct(high_risk_recall)}",
        f"- High-risk false positive rate: {pct(high_risk_false_positive_rate)}",
        f"- PII masking recall: {pct(pii_recall)}",
        f"- PII false positive rate: {pct(pii_false_positive_rate)}",
        "",
        "## Case Results",
        "",
        "| case_id | category | pass | expected | top5 retrieved | recall@5 | fact coverage | element coverage | risk | pii |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]

    for item in results:
        lines.append(
            "| {case_id} | {category} | {passed} | {expected} | {retrieved} | {recall5} | {fact_cov} | {element_cov} | {risk} | {pii} |".format(
                case_id=item.case_id,
                category=item.category,
                passed="yes" if item.pass_eval else "no",
                expected=", ".join(item.expected_articles) or "-",
                retrieved=", ".join(item.retrieved_articles) or "-",
                recall5=pct(item.recall_at_5),
                fact_cov=pct(item.fact_required_coverage),
                element_cov=pct(item.element_coverage),
                risk=item.risk_type or ("miss" if item.high_risk_expected else "-"),
                pii="yes" if item.pii_masked else ("miss" if item.pii_mask_expected else "-"),
            )
        )

    failing = [item for item in results if not item.pass_eval]
    lines.extend(["", "## Failing Cases", ""])
    if not failing:
        lines.append("- None.")
    else:
        for item in failing:
            reasons = []
            if not item.law_hit:
                reasons.append("law recall miss")
            if item.fact_required_coverage < 0.6:
                reasons.append("low required fact coverage")
            if item.high_risk_expected and not item.high_risk_detected:
                reasons.append("high-risk miss")
            if not item.high_risk_expected and item.high_risk_detected:
                reasons.append(f"high-risk false positive ({item.risk_type})")
            if item.pii_mask_expected and not item.pii_masked:
                reasons.append("PII mask miss")
            if not item.pii_mask_expected and item.pii_masked:
                reasons.append("PII mask false positive")
            lines.append(f"- {item.case_id}: {', '.join(reasons)}")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This MVP is an offline deterministic baseline.",
            "- Keyword recall is not a substitute for RAG quality; add vector/RAG runs as the next evaluator mode.",
            "- Raw legal element coverage is diagnostic only because current law elements are natural-language phrases, not FactDigger field keys.",
            "- ServicePlanner report quality still needs human or LLM-judge rubric scoring.",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    law_data = load_criminal_law_data()
    results = [await evaluate_case(case, law_data) for case in cases]

    csv_path = args.output_dir / "results.csv"
    report_path = args.output_dir / "report.md"
    write_csv(results, csv_path)
    write_report(results, report_path)

    pass_count = sum(1 for result in results if result.pass_eval)
    print(f"Evaluated {len(results)} cases: {pass_count}/{len(results)} passed")
    print(f"CSV: {csv_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
