import asyncio
import hashlib
import json
import os
import sys

import aiofiles
from langchain_community.document_loaders import (
    JSONLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPDFLoader,
    UnstructuredPowerPointLoader,
)
from langchain_core.documents import Document

from app.utils.logger import get_logger
from app.utils.path_tool import get_abstract_path

_logger = get_logger("FileHandler")


class FontBBoxStreamFilter:
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        if "FontBBox from font descriptor" not in data:
            self.stream.write(data)

    def flush(self):
        self.stream.flush()


sys.stderr = FontBBoxStreamFilter(sys.stderr)


async def get_file_md5_hex(file_path: str) -> str:
    """获取文件的 MD5 值。

    Args:
        file_path: 文件路径。

    Returns:
        MD5 哈希值字符串，失败时返回空字符串。
    """
    abs_file_path = get_abstract_path(file_path) if not os.path.isabs(file_path) else file_path

    if not os.path.exists(abs_file_path):
        _logger.error("MD5 计算文件路径不存在: %s", abs_file_path)
        return ""

    if not os.path.isfile(abs_file_path):
        _logger.error("MD5 计算路径不是文件: %s", abs_file_path)
        return ""

    md5_object = hashlib.md5()
    chunk_size = 1024
    try:
        async with aiofiles.open(abs_file_path, "rb") as f:
            while chunk := await f.read(chunk_size):
                md5_object.update(chunk)
    except Exception as e:
        _logger.error("MD5 计算读取文件出错: %s, %s", abs_file_path, e)
        return ""

    return md5_object.hexdigest()


async def listdir_allowed_type(path: str, allowed_types: tuple[str, ...]) -> tuple:
    """获取指定目录下所有允许的文件类型。

    Args:
        path: 目录路径。
        allowed_types: 允许的文件类型元组。

    Returns:
        符合条件的文件路径列表。
    """
    abs_path = get_abstract_path(path) if not os.path.isabs(path) else path

    if not os.path.exists(abs_path):
        _logger.error("文件列表目录路径不存在: %s", abs_path)
        return ()

    if not os.path.isdir(abs_path):
        _logger.error("文件列表路径不是目录: %s", abs_path)
        return ()

    file_list = []
    for f in await asyncio.to_thread(os.listdir, abs_path):
        if f.endswith(allowed_types):
            file_path = os.path.join(abs_path, f)
            file_list.append(file_path)

    return tuple(file_list)


async def pdf_loader(file_path: str, password: str | None = None) -> list[Document]:
    """加载 PDF 文件内容（支持包含图片和文字的混合 PDF）。

    Args:
        file_path: PDF 文件路径。
        password: PDF 密码（如果有）。

    Returns:
        PDF 文件内容列表。
    """
    abs_file_path = get_abstract_path(file_path) if not os.path.isabs(file_path) else file_path

    if password:
        loader = PyPDFLoader(abs_file_path, password=password)
        return await asyncio.to_thread(loader.load)

    try:
        loader = UnstructuredPDFLoader(abs_file_path)
        docs = await asyncio.to_thread(loader.load)
        if docs and any(len(doc.page_content.strip()) > 0 for doc in docs):
            return docs
    except Exception as e:
        _logger.warning("PDF 加载 UnstructuredPDFLoader 失败，尝试 PyPDFLoader: %s", e)

    loader = PyPDFLoader(abs_file_path)
    return await asyncio.to_thread(loader.load)


async def txt_loader(file_path: str) -> list[Document]:
    """加载 TXT 文件内容。

    Args:
        file_path: TXT 文件路径。

    Returns:
        TXT 文件内容列表。
    """
    abs_file_path = get_abstract_path(file_path) if not os.path.isabs(file_path) else file_path

    encodings = ["utf-8", "gbk"]
    for encoding in encodings:
        try:
            loader = TextLoader(abs_file_path, encoding=encoding)
            return await asyncio.to_thread(loader.load)
        except Exception as e:
            _logger.warning("文本文件加载编码失败: encoding=%s, path=%s, error=%s", encoding, abs_file_path, e)
            continue
    return []


async def word_loader(file_path: str) -> list[Document]:
    """加载 WORD 文件内容。

    Args:
        file_path: WORD 文件路径。

    Returns:
        WORD 文件内容列表。
    """
    abs_file_path = get_abstract_path(file_path) if not os.path.isabs(file_path) else file_path
    try:
        loader = TextLoader(abs_file_path, encoding="utf-8")
        return await asyncio.to_thread(loader.load)
    except Exception as e:
        _logger.error("WORD 文件加载失败: %s, %s", abs_file_path, e)
        return []


async def markdown_loader(file_path: str) -> list[Document]:
    """加载 Markdown 文件内容。

    Args:
        file_path: Markdown 文件路径。

    Returns:
        Markdown 文件内容列表。
    """
    abs_file_path = get_abstract_path(file_path) if not os.path.isabs(file_path) else file_path
    try:
        loader = UnstructuredMarkdownLoader(abs_file_path, mode="single")
        return await asyncio.to_thread(loader.load)
    except Exception as e:
        _logger.error("Markdown 文件加载失败: %s, %s", abs_file_path, e)
        return []


async def ppt_loader(file_path: str) -> list[Document]:
    """加载 PPT/PPTX 文件内容。

    Args:
        file_path: PPT 文件路径。

    Returns:
        PPT 文件内容列表。
    """
    abs_file_path = get_abstract_path(file_path) if not os.path.isabs(file_path) else file_path
    try:
        loader = UnstructuredPowerPointLoader(abs_file_path, mode="single")
        return await asyncio.to_thread(loader.load)
    except Exception as e:
        _logger.error("PPT 文件加载失败: %s, %s", abs_file_path, e)
        return []


async def json_loader(file_path: str, jq_schema: str = ".[] | .content") -> list[Document]:
    """加载 JSON 文件内容（使用 LangChain Community JSONLoader）。

    Args:
        file_path: JSON 文件路径。
        jq_schema: jq 模式，用于从 JSON 中提取内容。默认提取 .content 字段。
                   对于刑法_cleaned.json，使用 ".[] | .content" 提取所有条文的 content 字段。

    Returns:
        JSON 文件内容列表。
    """
    abs_file_path = get_abstract_path(file_path) if not os.path.isabs(file_path) else file_path
    try:
        loader = JSONLoader(file_path=abs_file_path, jq_schema=jq_schema)
        return loader.load()
    except Exception as e:
        _logger.error("JSON 文件加载失败: %s, %s", abs_file_path, e)
        return []


def get_file_md5_hex_sync(file_path: str) -> str:
    """同步获取文件的 MD5 值（用于多线程场景）。


    Args:
        file_path: 文件路径。

    Returns:
        MD5 哈希值字符串，失败时返回空字符串。
    """
    abs_file_path = get_abstract_path(file_path) if not os.path.isabs(file_path) else file_path

    if not os.path.exists(abs_file_path):
        _logger.error("MD5 计算文件路径不存在: %s", abs_file_path)
        return ""

    if not os.path.isfile(abs_file_path):
        _logger.error("MD5 计算路径不是文件: %s", abs_file_path)
        return ""

    md5_object = hashlib.md5()
    chunk_size = 1024
    try:
        with open(abs_file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                md5_object.update(chunk)
    except Exception as e:
        _logger.error("MD5 计算读取文件出错: %s, %s", abs_file_path, e)
        return ""

    return md5_object.hexdigest()


def pdf_loader_sync(file_path: str, password: str | None = None) -> list[Document]:
    """同步加载 PDF 文件内容（用于多线程场景，支持包含图片和文字的混合 PDF）。

    Args:
        file_path: PDF 文件路径。
        password: PDF 密码（如果有）。

    Returns:
        PDF 文件内容列表。
    """
    abs_file_path = get_abstract_path(file_path) if not os.path.isabs(file_path) else file_path

    if password:
        loader = PyPDFLoader(abs_file_path, password=password)
        return loader.load()

    try:
        loader = UnstructuredPDFLoader(abs_file_path)
        docs = loader.load()
        if docs and any(len(doc.page_content.strip()) > 0 for doc in docs):
            return docs
    except Exception as e:
        _logger.warning("PDF 加载 UnstructuredPDFLoader 失败，尝试 PyPDFLoader: %s", e)

    loader = PyPDFLoader(abs_file_path)
    return loader.load()


def txt_loader_sync(file_path: str) -> list[Document]:
    """同步加载 TXT 文件内容（用于多线程场景）。

    Args:
        file_path: TXT 文件路径。

    Returns:
        TXT 文件内容列表。
    """
    abs_file_path = get_abstract_path(file_path) if not os.path.isabs(file_path) else file_path

    encodings = ["utf-8", "gbk"]
    for encoding in encodings:
        try:
            loader = TextLoader(abs_file_path, encoding=encoding)
            return loader.load()
        except Exception as e:
            _logger.warning("文本文件加载编码失败: encoding=%s, path=%s, error=%s", encoding, abs_file_path, e)
            continue
    return []


def word_loader_sync(file_path: str) -> list[Document]:
    """同步加载 WORD 文件内容（用于多线程场景）。

    Args:
        file_path: WORD 文件路径。

    Returns:
        WORD 文件内容列表。
    """
    abs_file_path = get_abstract_path(file_path) if not os.path.isabs(file_path) else file_path
    try:
        loader = TextLoader(abs_file_path, encoding="utf-8")
        return loader.load()
    except Exception as e:
        _logger.error("WORD 文件加载失败: %s, %s", abs_file_path, e)
        return []


def markdown_loader_sync(file_path: str) -> list[Document]:
    """同步加载 Markdown 文件内容（用于多线程场景）。

    Args:
        file_path: Markdown 文件路径。

    Returns:
        Markdown 文件内容列表。
    """
    abs_file_path = get_abstract_path(file_path) if not os.path.isabs(file_path) else file_path
    try:
        loader = UnstructuredMarkdownLoader(abs_file_path, mode="single")
        return loader.load()
    except Exception as e:
        _logger.error("Markdown 文件加载失败: %s, %s", abs_file_path, e)
        return []


def ppt_loader_sync(file_path: str) -> list[Document]:
    """同步加载 PPT/PPTX 文件内容（用于多线程场景）。

    Args:
        file_path: PPT 文件路径。

    Returns:
        PPT 文件内容列表。
    """
    abs_file_path = get_abstract_path(file_path) if not os.path.isabs(file_path) else file_path
    try:
        loader = UnstructuredPowerPointLoader(abs_file_path, mode="single")
        return loader.load()
    except Exception as e:
        _logger.error("PPT 文件加载失败: %s, %s", abs_file_path, e)
        return []


def json_loader_sync(file_path: str, content_field: str = "content") -> list[Document]:
    """同步加载 JSON 文件内容（用于多线程场景）。

    Args:
        file_path: JSON 文件路径。
        content_field: JSON 中包含内容的字段名，默认提取 "content" 字段。

    Returns:
        JSON 文件内容列表。
    """
    abs_file_path = get_abstract_path(file_path) if not os.path.isabs(file_path) else file_path
    try:
        with open(abs_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            documents = []
            
            if isinstance(data, list):
                for idx, item in enumerate(data):
                    if content_field in item:
                        text = item[content_field]
                        metadata = {k: v for k, v in item.items() if k != content_field}
                        metadata['line'] = idx + 1
                        documents.append(Document(page_content=text, metadata=metadata))
            elif isinstance(data, dict) and content_field in data:
                documents.append(Document(page_content=data[content_field], metadata={}))
            
            return documents
    except Exception as e:
        _logger.error("JSON 文件加载失败: %s, %s", abs_file_path, e)
        return []
