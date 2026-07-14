import asyncio
import hashlib
import os
import tempfile

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.rag.legal_text_splitter import LegalArticleSplitter
from app.rag.text_spliter import AsyncTextSplitter
from app.utils.config import chroma_config
from app.utils.factory import embed_model
from app.utils.file_handler import (
    get_file_md5_hex,
    json_loader,
    json_loader_sync,
    listdir_allowed_type,
    markdown_loader,
    markdown_loader_sync,
    pdf_loader,
    pdf_loader_sync,
    ppt_loader,
    ppt_loader_sync,
    txt_loader,
    txt_loader_sync,
    word_loader,
    word_loader_sync,
)
from app.utils.logger import get_logger

_logger = get_logger("DocProcessor")


class DocumentProcessor:
    """文档处理器"""

    def __init__(self, vectors_store: Chroma, md5_store):
        self.vectors_store = vectors_store
        self.md5_store = md5_store

        splitter_type = chroma_config.get("splitter_type", "recursive")

        if splitter_type == "legal_article":
            self.spliter = LegalArticleSplitter(
                chunk_size=chroma_config.get("chunk_size", 2000),
                chunk_overlap=chroma_config.get("chunk_overlap", 200),
                preserve_structure=chroma_config.get("preserve_legal_structure", True),
                enable_metadata_extraction=chroma_config.get("enable_metadata_extraction", True),
            )
            _logger.info("使用法律专用文本分割器 (LegalArticleSplitter)")
        else:
            self.spliter = AsyncTextSplitter(
                chunk_size=chroma_config["chunk_size"],
                chunk_overlap=chroma_config["chunk_overlap"],
                separators=chroma_config["separators"],
                embedding_model=embed_model,
            )
            _logger.info("使用通用文本分割器 (AsyncTextSplitter)")

    async def get_file_document(self, read_path: str) -> list[Document]:
        """异步加载文件。

        Args:
            read_path: 文件路径。

        Returns:
            文档列表。
        """
        if read_path.endswith(".txt"):
            return await txt_loader(read_path)
        elif read_path.endswith(".pdf"):
            return await pdf_loader(read_path)
        elif read_path.endswith(".md"):
            return await markdown_loader(read_path)
        elif read_path.endswith(".pptx"):
            return await ppt_loader(read_path)
        elif read_path.endswith(".docx"):
            return await word_loader(read_path)
        elif read_path.endswith(".json"):
            return await json_loader(read_path)
        else:
            return []

    def get_file_document_sync(self, read_path: str) -> list[Document]:
        """同步加载文件（用于多线程场景）。

        Args:
            read_path: 文件路径。

        Returns:
            文档列表。
        """
        if read_path.endswith(".txt"):
            return txt_loader_sync(read_path)
        elif read_path.endswith(".pdf"):
            return pdf_loader_sync(read_path)
        elif read_path.endswith(".md"):
            return markdown_loader_sync(read_path)
        elif read_path.endswith(".pptx"):
            return ppt_loader_sync(read_path)
        elif read_path.endswith(".docx"):
            return word_loader_sync(read_path)
        elif read_path.endswith(".json"):
            return json_loader_sync(read_path)
        else:
            return []

    def split_documents_sync(self, documents: list[Document]) -> list[Document]:
        """同步分割文档（用于多线程场景）。

        Args:
            documents: 文档列表。

        Returns:
            分割后的文档列表。
        """
        return self.spliter.split_documents_sync(documents)

    async def _split_documents(self, documents: list[Document]) -> list[Document]:
        """在线程中调用两类分割器共有的同步接口。"""
        return await asyncio.to_thread(self.spliter.split_documents_sync, documents)

    async def get_document(
        self,
        files: list | None = None,
        user_id: str | None = None,
        is_public: bool = False,
        progress_callback=None,
    ):
        """处理文档并将其转为向量存入向量数据库。

        Args:
            files: 上传的文件列表，如果为 None 则从数据文件夹读取。
            user_id: 用户 ID，用于标记文档的所有者。
            is_public: 是否为公共文档，公共文档可供所有用户检索。
            progress_callback: 进度回调函数，用于实时返回处理进度。
        """
        file_paths = []
        file_names = {}

        if files:
            for file in files:
                temp_file_path = await asyncio.to_thread(
                    tempfile.NamedTemporaryFile, delete=False, suffix=os.path.splitext(file.filename)[1]
                )
                content = await file.read()
                await asyncio.to_thread(temp_file_path.write, content)
                file_paths.append(temp_file_path.name)
                file_names[temp_file_path.name] = file.filename
        else:
            allowed_file_path: tuple[str] = await listdir_allowed_type(
                chroma_config["data_path"], tuple(chroma_config["allow_knowledge_file_types"])
            )
            file_paths = list(allowed_file_path)

        for idx, file_path in enumerate(file_paths):
            filename = file_names.get(file_path, os.path.basename(file_path))

            md5_hex = await get_file_md5_hex(file_path)
            if await self.md5_store.check_md5_hex(md5_hex, user_id):
                if progress_callback:
                    await progress_callback(
                        {"step": "skipping", "filename": filename, "message": f"文件 {filename} 已存在，跳过"}
                    )
                _logger.info("文件 MD5 已存在，跳过: path=%s, md5=%s", file_path, md5_hex)
                if files:
                    try:
                        os.unlink(file_path)
                    except:
                        pass
                continue

            try:
                if progress_callback:
                    await progress_callback(
                        {"step": "loading", "filename": filename, "message": f"正在加载文档 {filename}..."}
                    )
                _logger.info("开始加载文档: %s", filename)

                try:
                    document: list[Document] = await self.get_file_document(file_path)
                except Exception as e:
                    import traceback

                    _logger.error("get_file_document 出错: %s\n%s", file_path, traceback.format_exc())
                    if progress_callback:
                        await progress_callback(
                            {
                                "step": "error",
                                "filename": filename,
                                "message": f"文件 {filename} 加载出错: {str(e)}",
                                "error_message": str(e),
                            }
                        )
                    if files:
                        try:
                            os.unlink(file_path)
                        except Exception:
                            pass
                    continue

                if not document:
                    if progress_callback:
                        await progress_callback(
                            {
                                "step": "error",
                                "filename": filename,
                                "message": f"文件 {filename} 加载内容为空，跳过",
                                "error_message": "文件内容为空",
                            }
                        )
                    _logger.error("文件加载内容为空，跳过: path=%s", file_path)
                    if files:
                        try:
                            os.unlink(file_path)
                        except Exception as e:
                            pass
                    continue

                if progress_callback:
                    await progress_callback(
                        {"step": "splitting", "filename": filename, "message": f"正在切分文档 {filename}..."}
                    )
                _logger.info("开始切分文档: %s", filename)

                document = await self._split_documents(document)
                if not document:
                    if progress_callback:
                        await progress_callback(
                            {
                                "step": "error",
                                "filename": filename,
                                "message": f"文件 {filename} 切分内容为空，跳过",
                                "error_message": "文档切分后为空",
                            }
                        )
                    _logger.error("文档切分内容为空，跳过: path=%s", file_path)
                    if files:
                        try:
                            os.unlink(file_path)
                        except:
                            pass
                    continue

                if progress_callback:
                    await progress_callback(
                        {"step": "storing", "filename": filename, "message": f"正在存储向量 {filename}..."}
                    )
                _logger.info("开始存储向量: %s, count=%d", filename, len(document))

                if user_id:
                    for doc in document:
                        doc.metadata["user_id"] = user_id

                for doc in document:
                    doc.metadata["original_filename"] = filename
                    doc.metadata["md5"] = md5_hex
                    doc.metadata["is_public"] = is_public

                existing_chunk_md5s = await self.md5_store.get_all_chunk_md5(user_id)
                unique_docs = []
                for doc in document:
                    chunk_md5 = hashlib.md5(doc.page_content[:200].encode()).hexdigest()
                    if chunk_md5 not in existing_chunk_md5s:
                        existing_chunk_md5s.add(chunk_md5)
                        unique_docs.append(doc)
                        await self.md5_store.save_chunk_md5(chunk_md5, user_id, md5_hex)
                    else:
                        _logger.debug("Chunk MD5 已存在，跳过: %s...", doc.page_content[:40].replace("\n", " "))

                if not unique_docs:
                    _logger.warning("所有 chunk 均已存在，跳过存储: %s", filename)
                    if progress_callback:
                        await progress_callback(
                            {
                                "step": "skipping",
                                "filename": filename,
                                "message": f"文件 {filename} 所有内容均已存在，跳过",
                            }
                        )
                    if files:
                        try:
                            os.unlink(file_path)
                        except:
                            pass
                    continue

                _logger.info("存储向量: %s, 总数=%d, 唯一=%d", filename, len(document), len(unique_docs))

                await asyncio.to_thread(self.vectors_store.add_documents, unique_docs)

                original_filename = file_names.get(file_path, filename) if files else filename
                await self.md5_store.save_md5_hex(md5_hex, filename, original_filename, user_id)

                if progress_callback:
                    await progress_callback(
                        {"step": "completed", "filename": filename, "message": f"文件 {filename} 处理完成"}
                    )
                _logger.info("文件 MD5 已保存: path=%s, md5=%s", file_path, md5_hex)

                if files:
                    try:
                        os.unlink(file_path)
                    except:
                        pass

            except Exception as e:
                if progress_callback:
                    await progress_callback(
                        {
                            "step": "error",
                            "filename": filename,
                            "message": f"文件 {filename} 处理失败",
                            "error_message": str(e),
                        }
                    )
                _logger.error("文件处理出错: path=%s, error=%s", file_path, e)
                if files:
                    try:
                        os.unlink(file_path)
                    except:
                        pass
                continue
