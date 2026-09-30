"""将基础校对后的项目标注用于演示，不要求律师确认或改写官方正文。"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from app.knowledge.full_law_corpus import FIELDS, sha256
from app.knowledge.law_knowledge import LawKnowledgeDataError, _normalize_article_number, _require_government_https_url

DEMO_LAYER = "demo-reviewed-v1"


def apply_demo_annotations(corpus: dict[str, Any], path: Path) -> dict[str, Any]:
    """加载演示标注并核对版本、正文和原标注，避免建议串到另一条文。"""
    try:
        asset = json.loads(path.read_text(encoding="utf-8"))
        if (asset["schema_version"] != "demo-annotations-v1" or asset["official"] is not False
                or asset["usage"] != "demo" or asset["legal_review_completed"] is not False
                or asset["corpus_sha256"] != corpus["metadata"]["corpus_sha256"]):
            raise ValueError("演示标注版本不匹配")
        annotation_id = asset["annotation_id"]
        if not isinstance(annotation_id, str) or not annotation_id.strip():
            raise ValueError("缺少演示标注版本")
        data = copy.deepcopy(corpus)
        index = {_normalize_article_number(a["article_number"]): a for c in data["chapters"] for a in c["articles"]}
        seen = set()
        for record in asset["records"]:
            number = _normalize_article_number(record["article_number"])
            article = index[number]
            if number in seen or article["status"] != "active" or article["annotation_layer"] != "pending":
                raise ValueError("条文重复、失效或与旧快照标注冲突")
            seen.add(number)
            if (record["review_status"] != "demo_ready" or not record["reviewed_by"]
                    or record["content_sha256"] != sha256(article["content"].encode())
                    or record["original_annotations_sha256"] != sha256(json.dumps(article["annotations"], ensure_ascii=False, sort_keys=True).encode())):
                raise ValueError("演示标注未校对或正文绑定失效")
            if record["proposal"]["elements"]["kind"] != "branch_proposal":
                raise ValueError("一般规定不能作为独立罪名")
            if not record["sources"]:
                raise ValueError("缺少校对来源")
            for source in record["sources"]:
                _require_government_https_url(source, "url", "demo.sources")
            elements = record["elements"]
            if (not isinstance(elements, list) or not elements
                    or any(not isinstance(e, dict) or not e.get("key") or not e.get("name") for e in elements)
                    or len({e["key"] for e in elements}) != len(elements)
                    or any(e["key"] == "no_independent_elements" for e in elements)):
                raise ValueError("演示要件为空或含占位")
            if (not isinstance(record["title"], str) or not record["title"].strip()
                    or not isinstance(record["base_sentence"], str) or not record["base_sentence"].strip()
                    or not isinstance(record["charge_tags"], list) or not record["charge_tags"]):
                raise ValueError("演示标注字段缺失")
            article["original_annotations"] = article["annotations"]
            article["annotations"] = copy.deepcopy(article["annotations"])
            for name, field in FIELDS.items():
                article[field] = copy.deepcopy(record[field])
                article["annotations"][name].update({"value": copy.deepcopy(record[field]), "review_status": "demo_ready",
                                                    "reviewed_by": record["reviewed_by"], "source": annotation_id})
            article.update({"display_title": record["title"], "annotation_layer": DEMO_LAYER,
                            "annotation_source": annotation_id, "annotation_usage": "demo",
                            "demo_review_status": "demo_ready", "demo_proposal": record["proposal"]})
        data["metadata"]["demo_annotations"] = {"annotation_id": annotation_id, "released_articles": len(seen),
                                               "usage": "demo", "official": False, "legal_review_completed": False}
        return data
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise LawKnowledgeDataError(f"演示标注加载失败: {exc}") from exc
