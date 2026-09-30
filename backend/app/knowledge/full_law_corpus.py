"""全量正文与待复核标注的独立版本契约。"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any

from app.knowledge.law_knowledge import (
    LAW_KNOWLEDGE_PATH,
    LawKnowledgeDataError,
    _build_article_index,
    _normalize_article_number,
    _require_government_https_url,
    is_article_in_force,
)

FULL_CORPUS_PATH = LAW_KNOWLEDGE_PATH.with_name("criminal_law_full.json")
DERIVED_COUNTS = {
    17: 1,
    37: 1,
    120: 6,
    133: 2,
    134: 1,
    135: 1,
    139: 1,
    142: 1,
    162: 2,
    169: 1,
    175: 1,
    177: 1,
    185: 1,
    205: 1,
    210: 1,
    219: 1,
    224: 1,
    234: 1,
    236: 1,
    244: 1,
    253: 1,
    260: 1,
    262: 2,
    276: 1,
    280: 2,
    284: 1,
    286: 1,
    287: 2,
    291: 2,
    293: 1,
    299: 1,
    307: 1,
    308: 1,
    334: 1,
    336: 1,
    342: 1,
    344: 1,
    355: 1,
    388: 1,
    390: 1,
    399: 1,
    408: 1,
}
FIELDS = {"title": "title", "charges": "charge_tags", "elements": "elements", "penalty": "base_sentence"}
TITLE_SOURCE = "https://www.court.gov.cn/fabu/xiangqing/424452.html"


def sha256(value: bytes) -> str:
    """生成带算法前缀的内容指纹。"""
    return "sha256:" + hashlib.sha256(value).hexdigest()


def text_structure(content: str) -> dict[str, Any]:
    """保留列举项与跨条引用原文；语法提取不解释各分支法律关系。"""
    items = re.findall(r"（[一二三四五六七八九十]+）[^（]+", content)
    refs = re.findall(r"第[一二三四五六七八九十百千零\d]+条(?:之[一二三四五六七八九十\d]+)?", content)
    return {"review_status": "syntax_only", "enumerated_items": items, "article_references": list(dict.fromkeys(refs))}


def prepare_full_corpus(candidate: dict[str, Any], candidate_sha256: str) -> dict[str, Any]:
    """隔离供应数据中的提议标注，只有与固定六条资产一致的标注可进入旧覆盖契约。"""
    data = copy.deepcopy(candidate)
    metadata = data["metadata"]
    baseline_raw = LAW_KNOWLEDGE_PATH.read_bytes()
    baseline = _build_article_index(json.loads(baseline_raw))
    metadata.update(
        {
            "schema_version": "full-text-v1",
            "dataset_version": candidate["metadata"]["dataset_version"] + ".integration1",
            "candidate_sha256": candidate_sha256,
            "regression_snapshot_sha256": sha256(baseline_raw),
            "legal_review_completed": False,
            "limitations": "政府整合版正文的程序导入；未完成逐条法律人工审查。六条项目标注仅沿用固定回归契约，其余标注不得用于覆盖分母。适用日期表示整合版本时点，不是个案行为时法选择。",
        }
    )
    metadata.pop("verified_at", None)
    metadata["imported_at"] = "2026-09-30"
    metadata["annotation_provenance"]["placeholders"] = (
        "占位只保留在 annotations.value 中作为待复核提议，不提供给运行时法律分析。"
    )
    for chapter in data["chapters"]:
        for article in chapter["articles"]:
            number = _normalize_article_number(article["article_number"])
            core = baseline.get(number)
            inherited = core is not None and core["content"] == article["content"]
            period = {
                "consolidation_effective_from": metadata["official_text"]["version_effective_from"],
                "effective_to": None,
                "scope": "consolidated_version_not_individual_article_commencement",
            }
            article["status"] = "active" if is_article_in_force(article) else "repealed"
            article["text_provenance"] = {
                "source_id": metadata["official_text"]["source_id"],
                "url": metadata["official_text"]["url"],
                "review_status": "source_attributed",
                "validation_status": "program_checked",
                "legal_review_status": "pending",
                "content_sha256": sha256(article["content"].encode()),
                "applicability": period,
            }
            article["annotations"] = {}
            for name, field in FIELDS.items():
                value = copy.deepcopy(article[field])
                status = "project_regression" if inherited else "pending"
                source = (
                    "project-maintained-v1"
                    if inherited
                    else ("manual-title-v1" if name in {"title", "charges"} else "auto-derived-v1")
                )
                if number == "第169条" and name == "title":
                    status, source = "source_checked", TITLE_SOURCE
                article["annotations"][name] = {
                    "value": value,
                    "review_status": status,
                    "source": source,
                    "reviewed_by": None,
                    "legal_review_status": "pending",
                    "applicability": period,
                }
                if not inherited:
                    article[field] = [] if field in {"elements", "charge_tags"} else ""
            article["display_title"] = article["annotations"]["title"]["value"]
            article["common_keywords"] = core["common_keywords"] if inherited and core is not None else []
            article["annotation_layer"] = "project-maintained-v1" if inherited else "pending"
            article["annotation_source"] = "project-maintained-v1" if inherited else "pending"
            article["text_structure"] = text_structure(article["content"])
    validate_full_corpus(data)
    return data


def validate_full_corpus(data: dict[str, Any]) -> None:
    """校验结构、来源、条号和标注隔离；不将校验结果当成人工法律复核。"""
    try:
        meta = data["metadata"]
        if meta["schema_version"] != "full-text-v1" or meta["legal_review_completed"] is not False:
            raise ValueError("schema/review")
        _require_government_https_url(meta["official_text"], "url", "metadata.official_text")
        _require_government_https_url(meta["redistribution_basis"], "url", "metadata.redistribution_basis")
        baseline_raw = LAW_KNOWLEDGE_PATH.read_bytes()
        if meta["regression_snapshot_sha256"] != sha256(baseline_raw):
            raise ValueError("regression snapshot hash")
        baseline = _build_article_index(json.loads(baseline_raw))
        articles = [a for c in data["chapters"] for a in c["articles"]]
        numbers = [_normalize_article_number(a["article_number"]) for a in articles]
        if (
            len(articles) != 505
            or len(set(numbers)) != 505
            or meta["coverage"] != [a["article_number"] for a in articles]
        ):
            raise ValueError("coverage")
        expected = {f"第{i}条" for i in range(1, 453)}
        expected.update(
            f"第{number}条之{'一二三四五六'[i]}" for number, count in DERIVED_COUNTS.items() for i in range(count)
        )
        if expected != set(numbers) or any(
            not re.fullmatch(r"第\d+条(?:之[一二三四五六七八九十]+)?", n) for n in numbers
        ):
            raise ValueError("article numbers")
        for a, n in zip(articles, numbers):
            p = a["text_provenance"]
            if not a["content"].strip() or p["content_sha256"] != sha256(a["content"].encode()):
                raise ValueError("content hash")
            if (
                a["official_text_source"] != p["source_id"]
                or p["source_id"] != meta["official_text"]["source_id"]
                or p["url"] != meta["official_text"]["url"]
            ):
                raise ValueError("text source")
            if p["legal_review_status"] != "pending" or p["review_status"] != "source_attributed":
                raise ValueError("text review status")
            if (a["status"] == "repealed") != (n == "第199条") or a["status"] not in {"active", "repealed"}:
                raise ValueError("in force status")
            if a["text_structure"] != text_structure(a["content"]):
                raise ValueError("text structure")
            core = baseline.get(n)
            inherited = core is not None and core["content"] == a["content"]
            for name, field in FIELDS.items():
                annotation = a["annotations"][name]
                if not annotation["source"] or annotation["legal_review_status"] != "pending":
                    raise ValueError("annotation source/review")
                if annotation["applicability"] != p["applicability"]:
                    raise ValueError("annotation applicability")
                if inherited and core is not None:
                    if (
                        a[field] != core[field]
                        or annotation["value"] != core[field]
                        or annotation["review_status"] != "project_regression"
                    ):
                        raise ValueError("regression annotation changed")
                elif a[field] != ([] if field in {"elements", "charge_tags"} else ""):
                    raise ValueError("pending annotation exposed")
                elif annotation["review_status"] != (
                    "source_checked" if n == "第169条" and name == "title" else "pending"
                ):
                    raise ValueError("annotation promotion")
            if n == "第169条" and (
                a["annotations"]["title"]["value"] != "徇私舞弊低价折股、出售公司、企业资产罪"
                or a["annotations"]["title"]["source"] != TITLE_SOURCE
            ):
                raise ValueError("article 169 title source")
            if a["annotation_layer"] != ("project-maintained-v1" if inherited else "pending"):
                raise ValueError("annotation layer")
    except (KeyError, TypeError, ValueError, OSError, AttributeError) as exc:
        raise LawKnowledgeDataError(f"全量语料契约失败: {exc}") from exc


def review_queue(data: dict[str, Any]) -> list[dict[str, Any]]:
    """按条号和字段生成复核清单，保留分支与跨条关系的待核实原因。"""
    queue = []
    for chapter in data["chapters"]:
        for a in chapter["articles"]:
            fields = ["content"] + [
                name for name, annotation in a["annotations"].items() if annotation["review_status"] == "pending"
            ]
            queue.append(
                {
                    "article_number": a["article_number"],
                    "fields": fields,
                    "status": "pending",
                    "content_sha256": a["text_provenance"]["content_sha256"],
                    "has_enumerated_items": bool(a["text_structure"]["enumerated_items"]),
                    "article_references": a["text_structure"]["article_references"],
                    "reason": "正文逐条比对、罪名名录、完整要件及替代/加重分支、法定刑跨条适用均需人工复核",
                }
            )
    return queue
