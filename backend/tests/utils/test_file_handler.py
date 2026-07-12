"""Unit tests for ``app.utils.file_handler``.

Tests use ``tmp_path`` to create small text/PDF/Word/Markdown/PPT/JSON fixtures
on the fly. We only test the behaviour of the loader functions, not the
internal library implementations, by mocking the underlying LangChain
``*Loader`` classes where helpful.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.utils import file_handler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_text(path, content="hello world"):
    path.write_text(content, encoding="utf-8")
    return path


def _write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# get_file_md5_hex
# ---------------------------------------------------------------------------


class TestGetFileMd5Hex:
    @pytest.mark.asyncio
    async def test_returns_md5_of_file(self, tmp_path):
        p = _write_text(tmp_path / "f.txt", "abc")
        result = await file_handler.get_file_md5_hex(str(p))
        assert result == "900150983cd24fb0d6963f7d28e17f72"  # md5("abc")

    @pytest.mark.asyncio
    async def test_missing_path_returns_empty(self, tmp_path):
        result = await file_handler.get_file_md5_hex(str(tmp_path / "missing.txt"))
        assert result == ""

    @pytest.mark.asyncio
    async def test_path_is_dir_returns_empty(self, tmp_path):
        result = await file_handler.get_file_md5_hex(str(tmp_path))
        assert result == ""


class TestGetFileMd5HexSync:
    def test_returns_md5_of_file(self, tmp_path):
        p = _write_text(tmp_path / "f.txt", "abc")
        assert file_handler.get_file_md5_hex_sync(str(p)) == "900150983cd24fb0d6963f7d28e17f72"

    def test_missing_path_returns_empty(self, tmp_path):
        assert file_handler.get_file_md5_hex_sync(str(tmp_path / "missing.txt")) == ""

    def test_path_is_dir_returns_empty(self, tmp_path):
        assert file_handler.get_file_md5_hex_sync(str(tmp_path)) == ""


# ---------------------------------------------------------------------------
# listdir_allowed_type
# ---------------------------------------------------------------------------


class TestListdirAllowedType:
    @pytest.mark.asyncio
    async def test_returns_matching_files(self, tmp_path):
        _write_text(tmp_path / "a.txt", "x")
        _write_text(tmp_path / "b.md", "x")
        (tmp_path / "c.bin").write_bytes(b"x")
        result = await file_handler.listdir_allowed_type(str(tmp_path), (".txt", ".md"))
        names = sorted(os.path.basename(p) for p in result)
        assert names == ["a.txt", "b.md"]

    @pytest.mark.asyncio
    async def test_missing_dir_returns_empty_tuple(self, tmp_path):
        result = await file_handler.listdir_allowed_type(str(tmp_path / "nope"), (".txt",))
        assert result == ()

    @pytest.mark.asyncio
    async def test_path_is_file_returns_empty_tuple(self, tmp_path):
        p = _write_text(tmp_path / "f.txt")
        result = await file_handler.listdir_allowed_type(str(p), (".txt",))
        assert result == ()


# ---------------------------------------------------------------------------
# txt_loader
# ---------------------------------------------------------------------------


class TestTxtLoader:
    @pytest.mark.asyncio
    async def test_loads_utf8(self, tmp_path):
        p = _write_text(tmp_path / "a.txt", "utf8 content")
        with patch("app.utils.file_handler.TextLoader") as tl:
            tl.return_value.load = MagicMock(return_value=[Document(page_content="utf8 content")])
            result = await file_handler.txt_loader(str(p))
        assert len(result) == 1
        assert result[0].page_content == "utf8 content"

    @pytest.mark.asyncio
    async def test_falls_back_to_gbk(self, tmp_path):
        p = _write_text(tmp_path / "a.txt", "abc")
        with patch("app.utils.file_handler.TextLoader") as tl:
            # First call (utf-8) raises, second (gbk) returns
            tl.return_value.load = MagicMock(side_effect=[UnicodeDecodeError("u", b"", 0, 1, "x"), [Document(page_content="gbk content")]])
            result = await file_handler.txt_loader(str(p))
        assert result[0].page_content == "gbk content"

    @pytest.mark.asyncio
    async def test_returns_empty_on_double_failure(self, tmp_path):
        p = _write_text(tmp_path / "a.txt", "abc")
        with patch("app.utils.file_handler.TextLoader") as tl:
            tl.return_value.load = MagicMock(side_effect=Exception("nope"))
            result = await file_handler.txt_loader(str(p))
        assert result == []


class TestTxtLoaderSync:
    def test_loads_utf8(self, tmp_path):
        p = _write_text(tmp_path / "a.txt", "utf8 content")
        with patch("app.utils.file_handler.TextLoader") as tl:
            tl.return_value.load = MagicMock(return_value=[Document(page_content="utf8 content")])
            result = file_handler.txt_loader_sync(str(p))
        assert result[0].page_content == "utf8 content"

    def test_falls_back_to_gbk(self, tmp_path):
        p = _write_text(tmp_path / "a.txt", "abc")
        with patch("app.utils.file_handler.TextLoader") as tl:
            tl.return_value.load = MagicMock(side_effect=[UnicodeDecodeError("u", b"", 0, 1, "x"), [Document(page_content="gbk content")]])
            result = file_handler.txt_loader_sync(str(p))
        assert result[0].page_content == "gbk content"

    def test_returns_empty_on_double_failure(self, tmp_path):
        p = _write_text(tmp_path / "a.txt", "abc")
        with patch("app.utils.file_handler.TextLoader") as tl:
            tl.return_value.load = MagicMock(side_effect=Exception("nope"))
            assert file_handler.txt_loader_sync(str(p)) == []


# ---------------------------------------------------------------------------
# pdf_loader
# ---------------------------------------------------------------------------


class TestPdfLoader:
    @pytest.mark.asyncio
    async def test_with_password_uses_pypdf(self, tmp_path):
        p = _write_text(tmp_path / "a.pdf", "binary")
        with patch("app.utils.file_handler.PyPDFLoader") as pl:
            pl.return_value.load = MagicMock(return_value=[Document(page_content="pdf")])
            result = await file_handler.pdf_loader(str(p), password="secret")
        assert result[0].page_content == "pdf"
        pl.assert_called_once()

    @pytest.mark.asyncio
    async def test_unstructured_success(self, tmp_path):
        p = _write_text(tmp_path / "a.pdf", "binary")
        with patch("app.utils.file_handler.UnstructuredPDFLoader") as ul, \
             patch("app.utils.file_handler.PyPDFLoader") as pl:
            ul.return_value.load = MagicMock(return_value=[Document(page_content="unstructured")])
            result = await file_handler.pdf_loader(str(p))
        assert result[0].page_content == "unstructured"
        pl.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_back_to_pypdf(self, tmp_path):
        p = _write_text(tmp_path / "a.pdf", "binary")
        with patch("app.utils.file_handler.UnstructuredPDFLoader") as ul, \
             patch("app.utils.file_handler.PyPDFLoader") as pl:
            ul.return_value.load = MagicMock(side_effect=Exception("unstructured failed"))
            pl.return_value.load = MagicMock(return_value=[Document(page_content="pypdf")])
            result = await file_handler.pdf_loader(str(p))
        assert result[0].page_content == "pypdf"

    @pytest.mark.asyncio
    async def test_unstructured_empty_falls_back(self, tmp_path):
        p = _write_text(tmp_path / "a.pdf", "binary")
        with patch("app.utils.file_handler.UnstructuredPDFLoader") as ul, \
             patch("app.utils.file_handler.PyPDFLoader") as pl:
            # unstructured returns an empty list of docs -> falls through
            ul.return_value.load = MagicMock(return_value=[])
            pl.return_value.load = MagicMock(return_value=[Document(page_content="pypdf")])
            result = await file_handler.pdf_loader(str(p))
        assert result[0].page_content == "pypdf"


class TestPdfLoaderSync:
    def test_with_password(self, tmp_path):
        p = _write_text(tmp_path / "a.pdf", "binary")
        with patch("app.utils.file_handler.PyPDFLoader") as pl:
            pl.return_value.load = MagicMock(return_value=[Document(page_content="pdf")])
            result = file_handler.pdf_loader_sync(str(p), password="x")
        assert result[0].page_content == "pdf"

    def test_unstructured_success(self, tmp_path):
        p = _write_text(tmp_path / "a.pdf", "binary")
        with patch("app.utils.file_handler.UnstructuredPDFLoader") as ul, \
             patch("app.utils.file_handler.PyPDFLoader") as pl:
            ul.return_value.load = MagicMock(return_value=[Document(page_content="unstructured")])
            result = file_handler.pdf_loader_sync(str(p))
        assert result[0].page_content == "unstructured"
        pl.assert_not_called()

    def test_falls_back_to_pypdf(self, tmp_path):
        p = _write_text(tmp_path / "a.pdf", "binary")
        with patch("app.utils.file_handler.UnstructuredPDFLoader") as ul, \
             patch("app.utils.file_handler.PyPDFLoader") as pl:
            ul.return_value.load = MagicMock(side_effect=Exception("x"))
            pl.return_value.load = MagicMock(return_value=[Document(page_content="pypdf")])
            result = file_handler.pdf_loader_sync(str(p))
        assert result[0].page_content == "pypdf"

    def test_unstructured_empty_falls_back(self, tmp_path):
        p = _write_text(tmp_path / "a.pdf", "binary")
        with patch("app.utils.file_handler.UnstructuredPDFLoader") as ul, \
             patch("app.utils.file_handler.PyPDFLoader") as pl:
            ul.return_value.load = MagicMock(return_value=[])
            pl.return_value.load = MagicMock(return_value=[Document(page_content="pypdf")])
            result = file_handler.pdf_loader_sync(str(p))
        assert result[0].page_content == "pypdf"


# ---------------------------------------------------------------------------
# word_loader
# ---------------------------------------------------------------------------


class TestWordLoader:
    @pytest.mark.asyncio
    async def test_loads(self, tmp_path):
        p = _write_text(tmp_path / "a.docx", "fake")
        with patch("app.utils.file_handler.TextLoader") as tl:
            tl.return_value.load = MagicMock(return_value=[Document(page_content="word")])
            result = await file_handler.word_loader(str(p))
        assert result[0].page_content == "word"

    @pytest.mark.asyncio
    async def test_returns_empty_on_error(self, tmp_path):
        p = _write_text(tmp_path / "a.docx", "fake")
        with patch("app.utils.file_handler.TextLoader") as tl:
            tl.return_value.load = MagicMock(side_effect=Exception("x"))
            result = await file_handler.word_loader(str(p))
        assert result == []


class TestWordLoaderSync:
    def test_loads(self, tmp_path):
        p = _write_text(tmp_path / "a.docx", "fake")
        with patch("app.utils.file_handler.TextLoader") as tl:
            tl.return_value.load = MagicMock(return_value=[Document(page_content="word")])
            result = file_handler.word_loader_sync(str(p))
        assert result[0].page_content == "word"

    def test_returns_empty_on_error(self, tmp_path):
        p = _write_text(tmp_path / "a.docx", "fake")
        with patch("app.utils.file_handler.TextLoader") as tl:
            tl.return_value.load = MagicMock(side_effect=Exception("x"))
            assert file_handler.word_loader_sync(str(p)) == []


# ---------------------------------------------------------------------------
# markdown_loader
# ---------------------------------------------------------------------------


class TestMarkdownLoader:
    @pytest.mark.asyncio
    async def test_loads(self, tmp_path):
        p = _write_text(tmp_path / "a.md", "# title\n")
        with patch("app.utils.file_handler.UnstructuredMarkdownLoader") as ml:
            ml.return_value.load = MagicMock(return_value=[Document(page_content="md")])
            result = await file_handler.markdown_loader(str(p))
        assert result[0].page_content == "md"

    @pytest.mark.asyncio
    async def test_returns_empty_on_error(self, tmp_path):
        p = _write_text(tmp_path / "a.md", "# title\n")
        with patch("app.utils.file_handler.UnstructuredMarkdownLoader") as ml:
            ml.return_value.load = MagicMock(side_effect=Exception("x"))
            result = await file_handler.markdown_loader(str(p))
        assert result == []


class TestMarkdownLoaderSync:
    def test_loads(self, tmp_path):
        p = _write_text(tmp_path / "a.md", "# title\n")
        with patch("app.utils.file_handler.UnstructuredMarkdownLoader") as ml:
            ml.return_value.load = MagicMock(return_value=[Document(page_content="md")])
            result = file_handler.markdown_loader_sync(str(p))
        assert result[0].page_content == "md"

    def test_returns_empty_on_error(self, tmp_path):
        p = _write_text(tmp_path / "a.md", "# title\n")
        with patch("app.utils.file_handler.UnstructuredMarkdownLoader") as ml:
            ml.return_value.load = MagicMock(side_effect=Exception("x"))
            assert file_handler.markdown_loader_sync(str(p)) == []


# ---------------------------------------------------------------------------
# ppt_loader
# ---------------------------------------------------------------------------


class TestPptLoader:
    @pytest.mark.asyncio
    async def test_loads(self, tmp_path):
        p = _write_text(tmp_path / "a.pptx", "fake")
        with patch("app.utils.file_handler.UnstructuredPowerPointLoader") as pl:
            pl.return_value.load = MagicMock(return_value=[Document(page_content="ppt")])
            result = await file_handler.ppt_loader(str(p))
        assert result[0].page_content == "ppt"

    @pytest.mark.asyncio
    async def test_returns_empty_on_error(self, tmp_path):
        p = _write_text(tmp_path / "a.pptx", "fake")
        with patch("app.utils.file_handler.UnstructuredPowerPointLoader") as pl:
            pl.return_value.load = MagicMock(side_effect=Exception("x"))
            result = await file_handler.ppt_loader(str(p))
        assert result == []


class TestPptLoaderSync:
    def test_loads(self, tmp_path):
        p = _write_text(tmp_path / "a.pptx", "fake")
        with patch("app.utils.file_handler.UnstructuredPowerPointLoader") as pl:
            pl.return_value.load = MagicMock(return_value=[Document(page_content="ppt")])
            result = file_handler.ppt_loader_sync(str(p))
        assert result[0].page_content == "ppt"

    def test_returns_empty_on_error(self, tmp_path):
        p = _write_text(tmp_path / "a.pptx", "fake")
        with patch("app.utils.file_handler.UnstructuredPowerPointLoader") as pl:
            pl.return_value.load = MagicMock(side_effect=Exception("x"))
            assert file_handler.ppt_loader_sync(str(p)) == []


# ---------------------------------------------------------------------------
# json_loader
# ---------------------------------------------------------------------------


class TestJsonLoader:
    @pytest.mark.asyncio
    async def test_uses_jsonloader(self, tmp_path):
        p = _write_json(tmp_path / "a.json", [{"content": "x"}, {"content": "y"}])
        with patch("app.utils.file_handler.JSONLoader") as jl:
            jl.return_value.load = MagicMock(return_value=[Document(page_content="x"), Document(page_content="y")])
            result = await file_handler.json_loader(str(p))
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_returns_empty_on_error(self, tmp_path):
        p = _write_json(tmp_path / "a.json", [{"content": "x"}])
        with patch("app.utils.file_handler.JSONLoader") as jl:
            jl.return_value.load = MagicMock(side_effect=Exception("x"))
            result = await file_handler.json_loader(str(p))
        assert result == []


class TestJsonLoaderSync:
    def test_list_of_dicts_with_content(self, tmp_path):
        p = _write_json(tmp_path / "a.json", [
            {"content": "first", "title": "A"},
            {"content": "second", "title": "B"},
        ])
        result = file_handler.json_loader_sync(str(p))
        assert len(result) == 2
        assert result[0].page_content == "first"
        assert result[0].metadata.get("title") == "A"
        assert result[0].metadata.get("line") == 1
        assert result[1].metadata.get("line") == 2

    def test_single_dict(self, tmp_path):
        p = _write_json(tmp_path / "a.json", {"content": "only one"})
        result = file_handler.json_loader_sync(str(p))
        assert len(result) == 1
        assert result[0].page_content == "only one"

    def test_dict_without_content_returns_empty(self, tmp_path):
        p = _write_json(tmp_path / "a.json", {"foo": "bar"})
        result = file_handler.json_loader_sync(str(p))
        assert result == []

    def test_custom_content_field(self, tmp_path):
        p = _write_json(tmp_path / "a.json", {"text": "hello"})
        result = file_handler.json_loader_sync(str(p), content_field="text")
        assert len(result) == 1
        assert result[0].page_content == "hello"

    def test_returns_empty_on_error(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json", encoding="utf-8")
        result = file_handler.json_loader_sync(str(p))
        assert result == []
