"""刑法正文与固定验证快照的加载、校验和共享条文索引。"""

import copy
import hashlib
import json
import os
import re
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

from app.infrastructure.logging import get_logger

from .identifiers import _normalize_article_number
from .models import LawKnowledgeDataError
from .repository import _build_article_index, is_article_in_force

_logger = get_logger("LawKnowledge")
LAW_KNOWLEDGE_PATH = Path(__file__).resolve().parents[3] / "data" / "law_knowledge" / "criminal_law_chapters.json"


def load_criminal_law_data(*, profile: str | None = None, env: Mapping[str, str] | None = None) -> Dict[str, Any]:
    """默认加载全量正文及演示标注；显式 snapshot 保留六条回归资产。"""
    values = dict(os.environ if env is None else env)
    profile = profile if profile is not None else values.get("LAW_KNOWLEDGE_PROFILE", "full")
    if profile not in {"snapshot", "full"}:
        raise LawKnowledgeDataError(f"未知法条配置: {profile}")
    path = LAW_KNOWLEDGE_PATH
    if profile == "full":
        path = Path(values.get("LAW_FULL_CORPUS_PATH", str(path.with_name("criminal_law_full.json"))))
    try:
        stat = path.stat()
    except OSError as exc:
        raise LawKnowledgeDataError(f"法条验证数据文件不存在: {path}") from exc
    data = _load_law_path(str(path.resolve()), stat.st_mtime_ns, stat.st_size, profile)
    if profile == "full":
        demo_mode = values.get("LAW_FULL_DEMO_ANNOTATIONS", "on")
        if demo_mode not in {"on", "off"}:
            raise LawKnowledgeDataError(f"未知演示标注配置: {demo_mode}")
        if demo_mode == "on":
            from app.knowledge.law.annotations import apply_demo_annotations

            demo_path = Path(values.get("LAW_DEMO_ANNOTATIONS_PATH", str(LAW_KNOWLEDGE_PATH.with_name("criminal_law_demo_annotations.json"))))
            return apply_demo_annotations(data, demo_path)
    return data


@lru_cache(maxsize=4)
def _load_law_path(path: str, mtime_ns: int, size: int, profile: str) -> Dict[str, Any]:
    """加载并校验版本化的刑事法律验证快照（带缓存）。"""
    selected_path = Path(path)
    if not selected_path.is_file():
        raise LawKnowledgeDataError(f"法条验证数据文件不存在: {LAW_KNOWLEDGE_PATH}")

    try:
        with selected_path.open("r", encoding="utf-8") as file:
            raw_data = file.read()
            data = json.loads(raw_data)
    except json.JSONDecodeError as exc:
        raise LawKnowledgeDataError(f"法条验证数据不是有效 JSON: {LAW_KNOWLEDGE_PATH}: {exc}") from exc
    except OSError as exc:
        raise LawKnowledgeDataError(f"无法读取法条验证数据: {LAW_KNOWLEDGE_PATH}: {exc}") from exc

    if profile == "full":
        validate_full_corpus(data)
        data["metadata"]["corpus_sha256"] = "sha256:" + hashlib.sha256(raw_data.encode("utf-8")).hexdigest()
    else:
        _validate_law_knowledge_data(data)
    _logger.info("【load_criminal_law_data】成功加载法条验证快照，共 %d 章", len(data["chapters"]))
    return data


load_criminal_law_data.cache_clear = _load_law_path.cache_clear


def _require_non_empty_string(container: Dict[str, Any], field: str, location: str) -> str:
    """读取必填非空字符串，并在错误中保留字段位置。"""
    value = container.get(field)
    if not isinstance(value, str) or not value.strip():
        raise LawKnowledgeDataError(f"法条验证数据字段缺失或为空: {location}.{field}")
    return value


def _require_government_https_url(container: Dict[str, Any], field: str, location: str) -> str:
    """要求来源证据指向 HTTPS 的中国政府域名。"""
    value = _require_non_empty_string(container, field, location)
    parsed = urlparse(value)
    hostname = parsed.hostname or ""
    if parsed.scheme != "https" or not (hostname == "gov.cn" or hostname.endswith(".gov.cn")):
        raise LawKnowledgeDataError(f"{location}.{field} 必须是政府 HTTPS URL")
    return value


def _validate_law_knowledge_data(data: Any) -> None:
    """校验官方文本来源与项目维护标注相互分离的数据契约。"""
    if not isinstance(data, dict):
        raise LawKnowledgeDataError("法条验证数据顶层必须是包含 metadata 和 chapters 的对象")

    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        raise LawKnowledgeDataError("法条验证数据缺少 metadata 对象")
    for field in ("dataset_id", "dataset_version", "verified_at", "limitations"):
        _require_non_empty_string(metadata, field, "metadata")

    official_text = metadata.get("official_text")
    if not isinstance(official_text, dict):
        raise LawKnowledgeDataError("法条验证数据缺少 metadata.official_text 对象")
    for field in ("source_id", "title", "publisher", "consolidated_through", "version_effective_from"):
        _require_non_empty_string(official_text, field, "metadata.official_text")
    _require_government_https_url(official_text, "url", "metadata.official_text")

    redistribution_basis = metadata.get("redistribution_basis")
    if not isinstance(redistribution_basis, dict):
        raise LawKnowledgeDataError("法条验证数据缺少 metadata.redistribution_basis 对象")
    _require_non_empty_string(redistribution_basis, "title", "metadata.redistribution_basis")
    _require_government_https_url(redistribution_basis, "url", "metadata.redistribution_basis")

    annotation_provenance = metadata.get("annotation_provenance")
    if not isinstance(annotation_provenance, dict) or annotation_provenance.get("official") is not False:
        raise LawKnowledgeDataError("metadata.annotation_provenance.official 必须明确为 false")
    for field in ("annotation_id", "maintainer", "purpose"):
        _require_non_empty_string(annotation_provenance, field, "metadata.annotation_provenance")

    if metadata.get("official_text_fields") != ["article_number", "content"]:
        raise LawKnowledgeDataError("metadata.official_text_fields 必须只声明 article_number 和 content")
    if metadata.get("project_annotation_fields") != [
        "title",
        "elements",
        "base_sentence",
        "charge_tags",
        "common_keywords",
    ]:
        raise LawKnowledgeDataError("metadata.project_annotation_fields 与运行时项目标注字段不一致")

    chapters = data.get("chapters")
    if not isinstance(chapters, list) or not chapters:
        raise LawKnowledgeDataError("法条验证数据 chapters 必须是非空数组")

    seen_numbers: set[str] = set()
    actual_numbers: list[str] = []
    for chapter_index, chapter in enumerate(chapters):
        location = f"chapters[{chapter_index}]"
        if not isinstance(chapter, dict):
            raise LawKnowledgeDataError(f"法条验证数据 {location} 必须是对象")
        _require_non_empty_string(chapter, "chapter", location)
        articles = chapter.get("articles")
        if not isinstance(articles, list) or not articles:
            raise LawKnowledgeDataError(f"法条验证数据 {location}.articles 必须是非空数组")

        for article_index, article in enumerate(articles):
            article_location = f"{location}.articles[{article_index}]"
            if not isinstance(article, dict):
                raise LawKnowledgeDataError(f"法条验证数据 {article_location} 必须是对象")
            for field in (
                "article_number",
                "title",
                "content",
                "base_sentence",
                "official_text_source",
                "annotation_source",
            ):
                _require_non_empty_string(article, field, article_location)
            for field in ("elements", "charge_tags", "common_keywords"):
                value = article.get(field)
                if not isinstance(value, list) or not value:
                    raise LawKnowledgeDataError(f"法条验证数据字段缺失或为空: {article_location}.{field}")

            if article["official_text_source"] != official_text["source_id"]:
                raise LawKnowledgeDataError(f"{article_location}.official_text_source 与顶层来源不一致")
            if article["annotation_source"] != annotation_provenance["annotation_id"]:
                raise LawKnowledgeDataError(f"{article_location}.annotation_source 与顶层来源不一致")

            article_number = article["article_number"]
            normalized_number = _normalize_article_number(article_number)
            if normalized_number in seen_numbers:
                raise LawKnowledgeDataError(f"法条验证数据存在重复法条编号: {normalized_number}")
            seen_numbers.add(normalized_number)
            actual_numbers.append(article_number)

    coverage = metadata.get("coverage")
    if not isinstance(coverage, list) or coverage != actual_numbers:
        raise LawKnowledgeDataError("metadata.coverage 必须按快照顺序完整列出全部法条编号")


def preflight_law_knowledge() -> Dict[str, Any]:
    """显式执行法条验证快照预检，供应用启动阶段 fail-fast。"""
    data = load_criminal_law_data()
    _logger.info(
        "【preflight_law_knowledge】法条验证快照通过: %s@%s",
        data["metadata"]["dataset_id"],
        data["metadata"]["dataset_version"],
    )
    return data


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
