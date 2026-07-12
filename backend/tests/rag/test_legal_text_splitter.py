"""Unit tests for ``app.rag.legal_text_splitter.LegalArticleSplitter``.

These tests exercise the splitter against a variety of legal text shapes
(single article, multiple articles, long article, JSON-structured legal docs,
plain-text fallbacks) without requiring a real LLM or database.
"""

import json

import pytest
from langchain_core.documents import Document

from app.rag.legal_text_splitter import LegalArticleSplitter


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def splitter():
    """Default splitter used in most tests."""
    return LegalArticleSplitter(chunk_size=2000, chunk_overlap=200)


@pytest.fixture
def small_splitter():
    """Splitter with a tiny chunk_size that forces the long-article path."""
    return LegalArticleSplitter(chunk_size=50, chunk_overlap=10)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_defaults(self):
        s = LegalArticleSplitter()
        assert s.chunk_size == 2000
        assert s.chunk_overlap == 200
        assert s.preserve_structure is True
        assert s.enable_metadata_extraction is True
        assert s.article_pattern is not None
        assert s.chapter_pattern is not None
        assert s.paragraph_pattern is not None

    def test_custom_values(self):
        s = LegalArticleSplitter(
            chunk_size=512,
            chunk_overlap=64,
            preserve_structure=False,
            enable_metadata_extraction=False,
        )
        assert s.chunk_size == 512
        assert s.chunk_overlap == 64
        assert s.preserve_structure is False
        assert s.enable_metadata_extraction is False


# ---------------------------------------------------------------------------
# split_text / split_text_sync
# ---------------------------------------------------------------------------


class TestSplitText:
    def test_split_text_single_article(self, splitter):
        text = "第一条 故意伤害他人身体的，处三年以下有期徒刑。"
        chunks = splitter.split_text(text)
        assert len(chunks) >= 1
        assert any("第一条" in c for c in chunks)

    def test_split_text_sync_matches_async(self, splitter):
        text = "第一条 故意伤害他人身体的，处三年以下有期徒刑。\n第二条 盗窃公私财物，数额较大的。"
        async_chunks = splitter.split_text(text)
        sync_chunks = splitter.split_text_sync(text)
        assert async_chunks == sync_chunks

    def test_split_text_empty(self, splitter):
        # The splitter always yields at least one chunk (the original document
        # returned from _split_by_articles when no article pattern matches).
        result = splitter.split_text("")
        assert result == [""]


# ---------------------------------------------------------------------------
# _split_by_articles
# ---------------------------------------------------------------------------


class TestSplitByArticles:
    def test_single_article(self, splitter):
        text = "第一百条 盗窃公私财物，数额较大的。"
        doc = Document(page_content=text, metadata={})
        chunks = splitter._split_by_articles(doc)
        assert len(chunks) == 1
        assert chunks[0].metadata["article_number"] == "第一百条"
        assert chunks[0].metadata["chunk_type"] == "article"

    def test_multiple_articles(self, splitter):
        text = "第一条 故意伤害他人身体的。\n第二条 盗窃公私财物，数额较大的。\n第三条 诈骗公私财物，数额较大的。"
        doc = Document(page_content=text, metadata={"src": "test"})
        chunks = splitter._split_by_articles(doc)
        assert len(chunks) == 3
        article_numbers = [c.metadata["article_number"] for c in chunks]
        assert "第一条" in article_numbers
        assert "第二条" in article_numbers
        assert "第三条" in article_numbers
        # Inherited metadata should be present in every chunk
        for c in chunks:
            assert c.metadata.get("src") == "test"

    def test_no_article_match_returns_original(self, splitter):
        text = "这是一段不含法条编号的普通文本。"
        doc = Document(page_content=text, metadata={})
        chunks = splitter._split_by_articles(doc)
        assert chunks == [doc]

    def test_numeric_article_pattern(self, splitter):
        text = "第10条 故意伤害他人身体的。"
        doc = Document(page_content=text, metadata={})
        chunks = splitter._split_by_articles(doc)
        assert len(chunks) == 1
        assert chunks[0].metadata["article_number"] == "第10条"

    def test_metadata_extraction_disabled(self):
        s = LegalArticleSplitter(enable_metadata_extraction=False)
        text = "第一条 故意伤害他人身体的，致人重伤的。"
        doc = Document(page_content=text, metadata={})
        chunks = s._split_by_articles(doc)
        # No "legal_category" / "article_type" should be set
        assert "legal_category" not in chunks[0].metadata

    def test_long_article_triggers_split(self, small_splitter):
        # Create an article that is clearly larger than chunk_size
        body = "。" .join([f"这是第{i}句话" for i in range(20)])
        text = f"第一条 {body}"
        doc = Document(page_content=text, metadata={})
        chunks = small_splitter._split_by_articles(doc)
        # _split_long_article returns paragraph chunks
        assert all(c.metadata.get("is_partial_article") for c in chunks)


# ---------------------------------------------------------------------------
# _split_long_article
# ---------------------------------------------------------------------------


class TestSplitLongArticle:
    def test_produces_paragraph_chunk(self, small_splitter):
        # The long-article path always yields at least one paragraph chunk
        # (with the is_partial_article flag set).
        text = "第一条 这是第一句。这是第二句。这是第三句。"
        chunks = small_splitter._split_long_article(text, "第一条", {"article_number": "第一条"})
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata["is_partial_article"] is True
            assert chunk.metadata["chunk_type"] == "paragraph"
            assert "paragraph_index" in chunk.metadata

    def test_metadata_includes_base_fields(self, small_splitter):
        text = "第一条 句子一。句子二。句子三。句子四。句子五。"
        base_meta = {"article_number": "第一条", "src": "test"}
        chunks = small_splitter._split_long_article(text, "第一条", base_meta)
        for chunk in chunks:
            assert chunk.metadata.get("article_number") == "第一条"
            assert chunk.metadata.get("src") == "test"

    def test_chunk_indices_sequential(self, small_splitter):
        text = "第一条 句子一。句子二。句子三。句子四。句子五。句子六。句子七。"
        chunks = small_splitter._split_long_article(text, "第一条", {})
        indices = [c.metadata["paragraph_index"] for c in chunks]
        assert indices == list(range(len(chunks)))


# ---------------------------------------------------------------------------
# _split_structured_doc / _create_chunk_from_item
# ---------------------------------------------------------------------------


class TestSplitStructuredDoc:
    def test_law_text_format_dict(self, splitter):
        data = {"law_text": "第一条 故意伤害。", "article": "第一条", "charge": "故意伤害罪"}
        doc = Document(page_content=json.dumps(data, ensure_ascii=False), metadata={"src": "json"})
        chunks = splitter._split_structured_doc(doc)
        assert len(chunks) == 1
        assert chunks[0].page_content == "第一条 故意伤害。"
        assert chunks[0].metadata["article_number"] == "第一条"
        assert chunks[0].metadata["chunk_type"] == "legal_unit"

    def test_law_text_format_list(self, splitter):
        data = [
            {"law_text": "第一条 故意伤害。", "article": "第一条"},
            {"law_text": "第二条 盗窃财物。", "article": "第二条"},
        ]
        doc = Document(page_content=json.dumps(data, ensure_ascii=False), metadata={})
        chunks = splitter._split_structured_doc(doc)
        assert len(chunks) == 2
        article_numbers = [c.metadata["article_number"] for c in chunks]
        assert article_numbers == ["第一条", "第二条"]

    def test_content_format(self, splitter):
        data = {"title": "盗窃罪", "content": "盗窃公私财物，数额较大的。第二百六十四条 盗窃公私财物。"}
        doc = Document(page_content=json.dumps(data, ensure_ascii=False), metadata={})
        chunks = splitter._split_structured_doc(doc)
        assert len(chunks) == 1
        assert chunks[0].metadata["title"] == "盗窃罪"
        assert chunks[0].metadata["chunk_type"] == "legal_article"
        assert chunks[0].metadata["source_format"] == "content_format"
        assert chunks[0].metadata["article_number"] == "第二百六十四条"

    def test_content_format_no_article(self, splitter):
        data = {"title": "无名条款", "content": "第一行内容"}
        doc = Document(page_content=json.dumps(data, ensure_ascii=False), metadata={})
        chunks = splitter._split_structured_doc(doc)
        # Falls back to the first-line extraction path
        assert len(chunks) == 1
        assert chunks[0].metadata["article_number"] == "第一行内容"

    def test_invalid_json_falls_back_to_articles(self, splitter):
        # Begin with "{" so _is_structured_legal_doc returns True (tentatively),
        # then JSON parsing fails and we fall back to _split_by_articles.
        content = "{ not valid json\n第一条 这是第一条内容"
        doc = Document(page_content=content, metadata={})
        chunks = splitter._split_structured_doc(doc)
        # Should fall back to article splitting
        assert any("第一条" in c.page_content for c in chunks)

    def test_unknown_format_returns_original_doc(self, splitter):
        data = [{"foo": "bar"}, {"baz": "qux"}]
        doc = Document(page_content=json.dumps(data), metadata={})
        # No "law_text" / "content" key, so all items return None, chunks empty,
        # and the function returns [doc]
        chunks = splitter._split_structured_doc(doc)
        assert chunks == [doc]

    def test_create_chunk_from_law_text_empty_returns_none(self, splitter):
        result = splitter._create_chunk_from_law_text_format(
            {"law_text": "   ", "article": "第一条"}, {}
        )
        assert result is None

    def test_create_chunk_from_content_empty_returns_none(self, splitter):
        result = splitter._create_chunk_from_content_format(
            {"content": "", "title": "title"}, {}
        )
        assert result is None

    def test_create_chunk_from_item_non_dict_returns_none(self, splitter):
        result = splitter._create_chunk_from_item("not a dict", {})
        assert result is None


# ---------------------------------------------------------------------------
# _split_by_paragraphs (fallback path)
# ---------------------------------------------------------------------------


class TestSplitByParagraphs:
    def test_single_paragraph(self, splitter):
        doc = Document(page_content="这是一段普通文本。", metadata={})
        chunks = splitter._split_by_paragraphs(doc)
        # _split_by_paragraphs may either return [doc] (if no double-newline)
        # or a list of one paragraph chunk
        assert len(chunks) >= 1

    def test_multiple_paragraphs(self, splitter):
        text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
        doc = Document(page_content=text, metadata={"src": "x"})
        chunks = splitter._split_by_paragraphs(doc)
        # All chunks should be of chunk_type paragraph
        for c in chunks:
            assert c.metadata.get("chunk_type") == "paragraph"

    def test_long_paragraph_not_split_within(self, small_splitter):
        # _split_by_paragraphs only splits on paragraph boundaries (blank
        # lines). A single very long paragraph is kept whole in one chunk.
        para = "字" * 200
        doc = Document(page_content=para, metadata={})
        chunks = small_splitter._split_by_paragraphs(doc)
        assert len(chunks) == 1
        assert chunks[0].page_content == para

    def test_empty_paragraphs_skipped(self, splitter):
        text = "段落1\n\n\n\n段落2\n\n段落3"
        doc = Document(page_content=text, metadata={})
        chunks = splitter._split_by_paragraphs(doc)
        # All non-empty content is captured
        joined = "".join(c.page_content for c in chunks)
        assert "段落1" in joined
        assert "段落2" in joined
        assert "段落3" in joined


# ---------------------------------------------------------------------------
# split_documents (top-level dispatch)
# ---------------------------------------------------------------------------


class TestSplitDocuments:
    def test_dispatch_structured_json(self, splitter):
        data = json.dumps([{"law_text": "第一条 内容。", "article": "第一条"}], ensure_ascii=False)
        doc = Document(page_content=data, metadata={})
        result = splitter.split_documents([doc])
        assert len(result) == 1
        assert result[0].metadata["chunk_type"] == "legal_unit"

    def test_dispatch_articles(self, splitter):
        doc = Document(page_content="第一条 内容。\n第二条 内容二。", metadata={})
        result = splitter.split_documents([doc])
        assert len(result) == 2
        for c in result:
            assert c.metadata["chunk_type"] == "article"

    def test_dispatch_paragraphs(self, splitter):
        doc = Document(page_content="普通段落1\n\n普通段落2", metadata={})
        result = splitter.split_documents([doc])
        assert all(c.metadata["chunk_type"] == "paragraph" for c in result)

    def test_split_documents_sync_matches_async(self, splitter):
        text = "第一条 内容。\n第二条 内容二。"
        doc = Document(page_content=text, metadata={})
        assert splitter.split_documents([doc]) == splitter.split_documents_sync([doc])

    def test_split_documents_multiple_docs(self, splitter):
        docs = [
            Document(page_content="第一条 内容。", metadata={}),
            Document(page_content="第二条 内容二。", metadata={}),
        ]
        result = splitter.split_documents(docs)
        # At least 2 chunks (one per article)
        assert len(result) >= 2


# ---------------------------------------------------------------------------
# _extract_legal_metadata
# ---------------------------------------------------------------------------


class TestExtractLegalMetadata:
    def test_aggravated_article_type(self, splitter):
        # 致人重伤 should yield "aggravated"
        result = splitter._extract_legal_metadata("致人重伤的处三年以上十年以下。", {})
        assert result["article_type"] == "aggravated"

    def test_most_severe_article_type(self, splitter):
        # Use text without matching the earlier "standard" pattern (which
        # contains "处...有期徒刑"). The first matching pattern in
        # sentence_patterns wins via the break statement.
        result = splitter._extract_legal_metadata("致人死亡特别恶劣。", {})
        assert result["article_type"] == "most_severe"

    def test_first_aggravated_match_wins(self, splitter):
        # death_related overrides any other "category" classification
        result = splitter._extract_legal_metadata("致人死亡，致人重伤的特殊情况。", {})
        assert result["legal_category"] == "death_related"

    def test_property_crime_detection(self, splitter):
        result = splitter._extract_legal_metadata("盗窃公私财物数额较大。", {})
        assert result["legal_category"] == "property_crime"

    def test_danger_related_detection(self, splitter):
        result = splitter._extract_legal_metadata("危害公共安全的行为。", {})
        assert result["legal_category"] == "danger_related"


# ---------------------------------------------------------------------------
# _extract_article_from_content
# ---------------------------------------------------------------------------


class TestExtractArticleFromContent:
    def test_with_article_number(self, splitter):
        content = "第二百三十二条 故意杀人的，处死刑。"
        result = splitter._extract_article_from_content(content)
        assert result["article_number"] == "第二百三十二条"
        assert result["article_text"] == content

    def test_without_article_number_first_line(self, splitter):
        content = "这是一段普通内容。\n没有法条编号。"
        result = splitter._extract_article_from_content(content)
        # article_number should equal the first line (truncated to 20 chars)
        assert result["article_number"] == "这是一段普通内容。"
        assert result["article_text"] == content

    def test_long_first_line_truncated(self, splitter):
        first_line = "一二三四五六七八九十一二三四五六七八九十"
        content = f"{first_line}\nsecond line"
        result = splitter._extract_article_from_content(content)
        assert len(result["article_number"]) == 20
