"""Unit tests for ``app.rag.text_spliter.AsyncTextSplitter``.

The splitter delegates to ``RecursiveCharacterTextSplitter``; we exercise the
asyncio wrappers, the optional embedding-based optimisation, and the sync
variants. The embedding model is mocked.
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from app.rag.text_spliter import AsyncTextSplitter


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_uses_default_separators(self):
        s = AsyncTextSplitter(chunk_size=100, chunk_overlap=20)
        assert s.chunk_size == 100
        assert s.chunk_overlap == 20
        assert s.embedding_model is None
        assert s.splitter is not None

    def test_custom_separators_override_default(self):
        s = AsyncTextSplitter(chunk_size=100, chunk_overlap=10, separators=["|", "@"])
        assert s.separators == ["|", "@"]


# ---------------------------------------------------------------------------
# split_text / split_text_sync
# ---------------------------------------------------------------------------


class TestSplitText:
    @pytest.mark.asyncio
    async def test_splits_text(self):
        s = AsyncTextSplitter(chunk_size=10, chunk_overlap=2)
        text = "第一段内容\n\n第二段内容"
        chunks = await s.split_text(text)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_no_embedding_no_optimization(self):
        s = AsyncTextSplitter(chunk_size=10, chunk_overlap=2)
        # Should not call embedding_model since it's None
        chunks = await s.split_text("a b c")
        assert isinstance(chunks, list)

    def test_split_text_sync(self):
        s = AsyncTextSplitter(chunk_size=10, chunk_overlap=2)
        text = "段落一\n\n段落二"
        chunks = s.split_text_sync(text)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# split_documents / split_documents_sync
# ---------------------------------------------------------------------------


class TestSplitDocuments:
    @pytest.mark.asyncio
    async def test_splits_documents(self):
        s = AsyncTextSplitter(chunk_size=10, chunk_overlap=2)
        docs = [Document(page_content="段一内容" * 5, metadata={"id": 1})]
        result = await s.split_documents(docs)
        assert isinstance(result, list)

    def test_split_documents_sync(self):
        s = AsyncTextSplitter(chunk_size=10, chunk_overlap=2)
        docs = [Document(page_content="段一内容" * 5, metadata={"id": 1})]
        result = s.split_documents_sync(docs)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _optimize_chunks (async) with mock embeddings
# ---------------------------------------------------------------------------


class TestOptimizeChunksAsync:
    @pytest.mark.asyncio
    async def test_merges_similar_chunks(self):
        emb = MagicMock()
        # Same vector for everything -> similarity = 1.0 > 0.7 -> merge all
        emb.embed_query = MagicMock(return_value=[1.0, 0.0])
        s = AsyncTextSplitter(chunk_size=10, chunk_overlap=2, embedding_model=emb)
        chunks = ["段一", "段二", "段三"]
        optimized = await s._optimize_chunks(chunks)
        # All merged into one
        assert len(optimized) == 1
        assert "段一" in optimized[0]
        assert "段三" in optimized[0]

    @pytest.mark.asyncio
    async def test_keeps_dissimilar_chunks_separate(self):
        emb = MagicMock()
        # We need side effects for 4 calls:
        # (chunks[0], chunks[1]), (merged=chunks[0]+chunks[1], chunks[2]), ...
        # Actually only the (current_chunk, chunks[i]) comparisons fire.
        # 3 chunks -> 2 comparisons -> 2 pairs -> 4 calls.
        emb.embed_query = MagicMock(side_effect=[
            [1.0, 0.0], [0.0, 1.0],   # 1st comparison: orthogonal -> no merge
            [0.0, 1.0], [0.0, 1.0],   # 2nd comparison: identical -> merge
        ])
        s = AsyncTextSplitter(chunk_size=10, chunk_overlap=2, embedding_model=emb)
        chunks = ["A", "B", "C"]
        optimized = await s._optimize_chunks(chunks)
        # First and second: orthogonal -> not merged. Second and third: identical -> merged
        assert len(optimized) == 2


# ---------------------------------------------------------------------------
# _optimize_chunks_sync
# ---------------------------------------------------------------------------


class TestOptimizeChunksSync:
    def test_merges_similar(self):
        emb = MagicMock()
        emb.embed_query = MagicMock(return_value=[1.0, 0.0])
        s = AsyncTextSplitter(chunk_size=10, chunk_overlap=2, embedding_model=emb)
        chunks = ["段一", "段二"]
        optimized = s._optimize_chunks_sync(chunks)
        assert len(optimized) == 1

    def test_no_embedding_skips_optimization(self):
        s = AsyncTextSplitter(chunk_size=10, chunk_overlap=2)
        chunks = ["a", "b"]
        # _optimize_chunks_sync unconditionally iterates; no embedding means
        # _calculate_similarity_sync returns 0.0 -> no merge
        optimized = s._optimize_chunks_sync(chunks)
        assert optimized == ["a", "b"]


# ---------------------------------------------------------------------------
# _calculate_similarity / _calculate_similarity_sync / _cosine_similarity
# ---------------------------------------------------------------------------


class TestSimilarity:
    def test_cosine_similarity_identical_vectors(self):
        s = AsyncTextSplitter()
        sim = s._cosine_similarity([1.0, 0.0], [1.0, 0.0])
        assert sim == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal_vectors(self):
        s = AsyncTextSplitter()
        sim = s._cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert sim == pytest.approx(0.0)

    def test_cosine_similarity_zero_vector(self):
        s = AsyncTextSplitter()
        sim = s._cosine_similarity([0.0, 0.0], [1.0, 0.0])
        assert sim == 0.0

    def test_calculate_similarity_sync_no_embedding(self):
        s = AsyncTextSplitter()
        assert s._calculate_similarity_sync("a", "b") == 0.0

    def test_calculate_similarity_sync_with_embedding(self):
        emb = MagicMock()
        emb.embed_query = MagicMock(side_effect=[[1.0, 0.0], [1.0, 0.0]])
        s = AsyncTextSplitter(embedding_model=emb)
        sim = s._calculate_similarity_sync("a", "b")
        assert sim == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_calculate_similarity_no_embedding(self):
        s = AsyncTextSplitter()
        sim = await s._calculate_similarity("a", "b")
        assert sim == 0.0

    @pytest.mark.asyncio
    async def test_calculate_similarity_with_embedding(self):
        emb = MagicMock()
        emb.embed_query = MagicMock(side_effect=[[1.0, 0.0], [0.0, 1.0]])
        s = AsyncTextSplitter(embedding_model=emb)
        sim = await s._calculate_similarity("a", "b")
        assert sim == pytest.approx(0.0)
