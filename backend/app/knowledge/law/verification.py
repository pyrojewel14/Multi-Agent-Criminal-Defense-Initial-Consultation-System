"""核验法律正文与标注资格，相关性分数不参与资格判断。"""

from dataclasses import replace
from typing import Any, Dict

from app.knowledge.retrieval.models import ChannelResult, RecallPool, RetrievalCandidate

from .identifiers import _normalize_article_number, extract_article_number
from .models import LawVerification, VerifiedLawPool, plain
from .repository import LawRepository, is_article_in_force


def _element_name(element: Any) -> str:
    """提取项目标注的构成要件名称。"""
    if isinstance(element, dict):
        return str(element.get("name", ""))
    return str(element)


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


class LawVerifier:
    """逐个核验原始候选，为代表选择提供请求局部资格映射。"""

    def __init__(self, repository: LawRepository):
        self.repository = repository

    def _verify(self, candidate: RetrievalCandidate, *, channel: str) -> tuple[RetrievalCandidate, LawVerification] | None:
        metadata = plain(candidate.metadata)
        number = _normalize_article_number(metadata.get("article_number") or metadata.get("article_id") or extract_article_number(candidate.content))
        article = self.repository.article_index.get(number)
        if article is not None and not is_article_in_force(article):
            return None
        keyword = channel in {"keyword", "exact"}
        full = bool(article and article.get("text_provenance"))
        bound = article is not None
        if bound and keyword:
            bound = candidate.content == article["content"]
        elif bound and full:
            bound = (metadata.get("corpus_sha256") == article.get("corpus_sha256")
                     and metadata.get("corpus_version") == article.get("corpus_version")
                     and metadata.get("content_sha256") == article["text_provenance"]["content_sha256"]
                     and candidate.content == f"{article['article_number']} {article['content']}")
        eligible = bool(bound and is_lawref_eligible(article))
        source = ("json_keyword" if keyword else "rag_verified") if eligible else ("text_only" if bound and full else "rag_unverified")
        provenance = {key: metadata.get(key) for key in ("corpus_version", "corpus_sha256", "content_sha256", "official_text_source",
                       "text_provenance", "source", "text_review_status", "legal_review_status")}
        provenance["verified"] = bound
        required = tuple(article.get("elements", ())) if eligible else ()
        verification = LawVerification(number, source, eligible, required, provenance)
        enriched = {**metadata, "article_number": number, "data_source": source,
                    "coverage_eligible": eligible, "elements": plain(required), "required_elements": plain(required)}
        content = candidate.content
        if bound:
            enriched.update(plain(article))
            enriched.update(data_source=source, coverage_eligible=eligible, elements=plain(required), required_elements=plain(required))
            content = article["content"]
        return replace(candidate, content=content, metadata=enriched), verification

    def verify_pool(self, pool: RecallPool) -> VerifiedLawPool:
        """核验完整通道池，不融合、不评分、不提前截断。"""
        verifications, channels = {}, {}
        for name, channel in pool.channels.items():
            candidates = []
            for candidate in channel.candidates:
                verified = self._verify(candidate, channel=name)
                if verified is not None:
                    enriched, verification = verified
                    candidates.append(enriched)
                    verifications[candidate.candidate_id] = verification
            channels[name] = ChannelResult(name, tuple(candidates), channel.scores, channel.status)
        return VerifiedLawPool(RecallPool(channels, pool.retrieval_status, pool.dependency_failed), verifications)
