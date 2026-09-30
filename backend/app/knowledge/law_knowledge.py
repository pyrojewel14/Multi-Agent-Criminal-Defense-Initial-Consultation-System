"""刑法正文与固定验证快照的加载、校验和共享条文索引。"""

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

from app.infrastructure.logging import get_logger

_logger = get_logger("LawKnowledge")
LAW_KNOWLEDGE_PATH = Path(__file__).resolve().parents[2] / "data" / "law_knowledge" / "criminal_law_chapters.json"


class LawKnowledgeDataError(RuntimeError):
    """表示 tracked 法条验证快照缺失、损坏或不符合审计契约。"""


def load_criminal_law_data() -> Dict[str, Any]:
    """按显式配置选择资产，缓存键包含路径和文件版本以避免跨配置复用。"""
    profile = os.getenv("LAW_KNOWLEDGE_PROFILE", "snapshot")
    if profile not in {"snapshot", "full"}:
        raise LawKnowledgeDataError(f"未知法条配置: {profile}")
    path = LAW_KNOWLEDGE_PATH
    if profile == "full":
        path = Path(os.getenv("LAW_FULL_CORPUS_PATH", str(path.with_name("criminal_law_full.json"))))
    try:
        stat = path.stat()
    except OSError as exc:
        raise LawKnowledgeDataError(f"法条验证数据文件不存在: {path}") from exc
    data = _load_law_path(str(path.resolve()), stat.st_mtime_ns, stat.st_size, profile)
    if profile == "full":
        demo_mode = os.getenv("LAW_FULL_DEMO_ANNOTATIONS", "on")
        if demo_mode not in {"on", "off"}:
            raise LawKnowledgeDataError(f"未知演示标注配置: {demo_mode}")
        if demo_mode == "on":
            from app.knowledge.demo_law_annotations import apply_demo_annotations

            demo_path = Path(os.getenv("LAW_DEMO_ANNOTATIONS_PATH", str(LAW_KNOWLEDGE_PATH.with_name("criminal_law_demo_annotations.json"))))
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
        from app.knowledge.full_law_corpus import validate_full_corpus
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


_CN_NUM_MAP = {
    "零": "0",
    "一": "1",
    "二": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
}
_CN_DIGITS = set("零一二三四五六七八九十百千")


def _cn_to_arabic(cn: str) -> str:
    """将中文数字转换为阿拉伯数字。"""
    if not cn or not all(char in _CN_DIGITS for char in cn):
        return cn
    result = 0
    current = 0
    for char in cn:
        if char in _CN_NUM_MAP:
            current = int(_CN_NUM_MAP[char])
        elif char == "十":
            result += (current or 1) * 10
            current = 0
        elif char == "百":
            result += current * 100
            current = 0
        elif char == "千":
            result += current * 1000
            current = 0
    return str(result + current)


def _normalize_article_number(article_number: str) -> str:
    """统一中文数字和阿拉伯数字法条编号的表示。"""
    if not article_number:
        return ""
    match = re.match(r"第(.+?)条(之[一二三四五六七八九十百千零\d]+)?", article_number)
    if not match:
        return article_number.strip()
    number, suffix = match.group(1), match.group(2) or ""
    if suffix:
        suffix = "之" + ({str(i): v for i, v in enumerate("零一二三四五六七八九十")}.get(suffix[1:], suffix[1:]))
    if number.isdigit():
        return f"第{number}条{suffix}"
    arabic = _cn_to_arabic(number)
    return f"第{arabic}条{suffix}" if arabic != number else article_number.strip()


def _build_article_index(law_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """构建归一化法条编号到验证快照条目的索引。"""
    index = {}
    for chapter in law_data.get("chapters", []):
        for article in chapter.get("articles", []):
            normalized = _normalize_article_number(article.get("article_number", ""))
            if normalized:
                index[normalized] = {**article, "chapter": chapter.get("chapter", "")}
                if law_data.get("metadata", {}).get("schema_version") == "full-text-v1":
                    index[normalized]["corpus_sha256"] = law_data["metadata"].get("corpus_sha256", "")
                    index[normalized]["corpus_version"] = law_data["metadata"]["dataset_version"]
    return index


def _element_name(element: Any) -> str:
    """提取项目标注的构成要件名称。"""
    if isinstance(element, dict):
        return str(element.get("name", ""))
    return str(element)


def is_article_in_force(article: Dict[str, Any]) -> bool:
    """排除快照中明确标记失效或正文已删去的条文。"""
    return article.get("status") in (None, "active") and article.get("content", "").strip() not in (
        "（删去）",
        "(删去)",
    )


def is_lawref_eligible(article: Dict[str, Any]) -> bool:
    """允许旧回归或已基础校对的演示标注进入 LawRef，排除占位要件。"""
    if not is_article_in_force(article):
        return False

    layer = article.get("annotation_layer", article.get("annotation_source"))
    if layer not in {"project-maintained-v1", "demo-reviewed-v1"} or not article.get("annotation_source"):
        return False
    if layer == "demo-reviewed-v1" and (
        article.get("annotation_usage") != "demo" or article.get("demo_review_status") != "demo_ready"
    ):
        return False

    sentence = article.get("base_sentence", "")
    if not isinstance(sentence, str) or not sentence.strip() or sentence.strip().startswith("（本条不"):
        return False

    elements = article.get("elements")
    if not isinstance(elements, list) or not elements:
        return False
    names = [_element_name(element).strip() for element in elements]
    if any(not name for name in names) or len(names) != len(set(names)):
        return False
    if any(isinstance(element, dict) and element.get("key") == "no_independent_elements" for element in elements):
        return False
    return True
