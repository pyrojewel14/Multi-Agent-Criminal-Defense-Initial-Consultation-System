"""仅加载、切片及准备 metadata，不持有存储或模型。"""

import asyncio
from pathlib import Path

from langchain_core.documents import Document

from . import loaders


class DocumentProcessor:
    """由组装层注入切片器，文件生命周期由服务层负责。"""

    def __init__(self, splitter):
        self.splitter = splitter

    def get_file_document_sync(self, read_path: str) -> list[Document]:
        """按文件扩展名选择现有加载能力。"""
        loader = {
            ".txt": loaders.txt_loader_sync, ".pdf": loaders.pdf_loader_sync,
            ".md": loaders.markdown_loader_sync, ".pptx": loaders.ppt_loader_sync,
            ".docx": loaders.word_loader_sync, ".json": loaders.json_loader_sync,
        }.get(Path(read_path).suffix.lower())
        return [] if loader is None else loader(read_path)

    async def get_file_document(self, read_path: str) -> list[Document]:
        """在线程池加载文件，避免阻塞事件循环。"""
        return await asyncio.to_thread(self.get_file_document_sync, read_path)

    def split_documents_sync(self, documents: list[Document]) -> list[Document]:
        """保留注入切片器的既有切片行为。"""
        return self.splitter.split_documents_sync(documents)

    def prepare_sync(self, read_path: str, *, filename: str, user_id: str, md5: str,
                     is_public: bool = False) -> list[Document]:
        """准备独立分块并覆盖上传身份字段，不执行去重或写入。"""
        documents = self.get_file_document_sync(read_path)
        if not documents:
            raise ValueError("文件加载为空")
        chunks = self.split_documents_sync(documents)
        if not chunks:
            raise ValueError("切片结果为空")
        return [Document(page_content=doc.page_content, metadata={
            **doc.metadata, "source": filename, "user_id": user_id, "original_filename": filename,
            "md5": md5, "is_public": is_public,
        }) for doc in chunks]
