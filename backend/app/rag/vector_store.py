import asyncio
import os

from chromadb.api.types import Where
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.utils.config import chroma_config
from app.utils.factory import embed_model
from app.utils.logger import get_logger
from app.utils.path_tool import get_abstract_path

from .document_handler import DocumentProcessor
from .md5_manager import MD5Store
from .retrievers import EmptyRetriever
from .retrievers.hybrid_retriever import HybridRetriever

_logger = get_logger("VectorStore")

_vector_store_instance = None


def get_vector_store() -> "VectorStoreService":
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStoreService()
    return _vector_store_instance


class VectorStoreService:
    """向量数据库服务"""

    def __init__(self):
        import chromadb
        import chromadb.api.shared_system_client as _ssc
        from chromadb.config import Settings

        persist_dir = get_abstract_path(chroma_config["persist_directory"])
        os.makedirs(persist_dir, exist_ok=True)

        _chroma_client = self._create_chroma_client(persist_dir)

        self.vectors_store = Chroma(
            client=_chroma_client,
            collection_name=chroma_config["collection_name"],
            embedding_function=embed_model,
        )

        self.md5_store = MD5Store()
        self.hybrid_retriever = HybridRetriever(self.vectors_store)
        self.document_processor = DocumentProcessor(self.vectors_store, self.md5_store)

    @staticmethod
    def _create_chroma_client(persist_dir: str):
        import chromadb
        import chromadb.api.shared_system_client as _ssc
        from chromadb.config import Settings

        for attempt in range(3):
            _ssc.SharedSystemClient._identifier_to_system.clear()
            try:
                return chromadb.PersistentClient(
                    path=persist_dir,
                    settings=Settings(anonymized_telemetry=False),
                )
            except KeyError:
                if attempt == 2:
                    raise
                _logger.warning(
                    "ChromaDB PersistentClient KeyError (attempt %d/3), retrying...",
                    attempt + 1,
                )

    async def get_bm25_retriever(self, user_id: str | None = None):
        """获取 BM25 检索器。

        Args:
            user_id: 用户 ID。

        Returns:
            BM25 检索器实例。
        """
        return await self.hybrid_retriever.get_bm25_retriever(user_id)

    async def _get_all_documents(self) -> list[Document]:
        """获取向量库中的所有文档。

        Returns:
            文档列表。
        """
        return await self.hybrid_retriever._get_all_documents()

    async def get_retriever(
        self,
        query: str | None = None,
        user_id: str | None = None,
        include_public: bool = False,
    ):
        """获取检索器。

        Args:
            query: 查询语句。
            user_id: 用户 ID。
            include_public: 是否包含公共文档。

        Returns:
            检索器实例。
        """
        return await self.hybrid_retriever.get_retriever(query, user_id, include_public)

    @staticmethod
    def get_dynamic_weights(query: str | None = None):
        """获取动态权重。

        Args:
            query: 查询语句。

        Returns:
            权重列表。
        """
        return HybridRetriever.get_dynamic_weights(query)

    async def check_md5_hex(self, md5_for_check: str, user_id: str | None = None) -> bool:
        """检查 MD5 是否存在。

        Args:
            md5_for_check: 要检查的 MD5 值。
            user_id: 用户 ID。

        Returns:
            是否存在。
        """
        return await self.md5_store.check_md5_hex(md5_for_check, user_id)

    async def save_md5_hex(
        self,
        md5_hex: str,
        filename: str | None = None,
        original_filename: str | None = None,
        user_id: str | None = None,
    ):
        """保存 MD5 值。

        Args:
            md5_hex: MD5 值。
            filename: 文件名。
            original_filename: 原始文件名。
            user_id: 用户 ID。
        """
        await self.md5_store.save_md5_hex(md5_hex, filename, original_filename, user_id)

    def save_md5_hex_sync(
        self,
        md5_hex: str,
        filename: str | None = None,
        original_filename: str | None = None,
        user_id: str | None = None,
    ):
        """同步保存 MD5 值。

        Args:
            md5_hex: MD5 值。
            filename: 文件名。
            original_filename: 原始文件名。
            user_id: 用户 ID。
        """
        self.md5_store.save_md5_hex_sync(md5_hex, filename, original_filename, user_id)

    async def delete_all_documents(self):
        """删除向量库中的所有文档和 MD5 记录。

        仅供超管使用，用于清空整个知识库。
        """
        try:
            await asyncio.to_thread(self.vectors_store.delete, where={})
            await self.md5_store.clear_all()
            _logger.info("已清空所有知识库文档和 MD5 记录")
        except Exception as e:
            _logger.error("清空所有文档出错: error=%s", e)
            raise

    async def delete_user_documents(self, user_id: str):
        """删除指定用户的所有文档（包括 MD5 记录）。

        Args:
            user_id: 用户 ID。
        """
        try:
            await self.delete_user_md5(user_id, delete_documents=True)
        except Exception as e:
            _logger.error("删除用户文档出错: user_id=%s, error=%s", user_id, e)
            raise

    async def delete_user_md5(self, user_id: str, delete_documents: bool = True):
        """删除指定用户的 MD5 记录。

        Args:
            user_id: 用户 ID。
            delete_documents: 是否同时删除向量数据库中的文档（默认 True）。
        """
        try:
            if delete_documents:
                await asyncio.to_thread(self.vectors_store.delete, where={"user_id": user_id})
                _logger.info("已删除用户的所有文档: user_id=%s", user_id)

            await self.md5_store.delete_user_md5(user_id)
        except Exception as e:
            _logger.error("删除用户 MD5 记录出错: user_id=%s, error=%s", user_id, e)

    async def delete_by_filename(self, user_id: str, filename: str, delete_documents: bool = True):
        """通过文件名删除 MD5 记录及其对应的知识库内容。

        Args:
            user_id: 用户 ID。
            filename: 要删除的文件名。
            delete_documents: 是否同时删除向量数据库中的对应文档（默认 True）。

        Returns:
            是否成功删除。
        """
        try:
            md5_to_delete = await self.md5_store.delete_by_filename(user_id, filename)
            if md5_to_delete is None:
                _logger.warning("文件不存在于用户 MD5 记录中: user_id=%s, filename=%s", user_id, filename)
                return False

            _logger.info("已删除文件 MD5 记录: user_id=%s, filename=%s", user_id, filename)

            if delete_documents:
                where_clause = {"$and": [{"user_id": user_id}, {"md5": md5_to_delete}]}
                await asyncio.to_thread(self.vectors_store.delete, where=where_clause)
                _logger.info("已删除文件对应文档: user_id=%s, filename=%s", user_id, filename)

            return True

        except Exception as e:
            _logger.error("删除文件出错: user_id=%s, filename=%s, error=%s", user_id, filename, e)
            return False

    async def delete_single_md5(self, user_id: str, md5_to_delete: str, delete_documents: bool = True):
        """删除单个 MD5 记录及其对应的知识库内容。

        Args:
            user_id: 用户 ID。
            md5_to_delete: 要删除的 MD5 值。
            delete_documents: 是否同时删除向量数据库中的对应文档（默认 True）。

        Returns:
            是否成功删除。
        """
        try:
            success = await self.md5_store.delete_single_md5(user_id, md5_to_delete)
            if not success:
                _logger.warning("MD5 记录不存在: user_id=%s, md5=%s", user_id, md5_to_delete)
                return False

            _logger.info("已删除 MD5 记录: user_id=%s, md5=%s", user_id, md5_to_delete)

            if delete_documents:
                where_clause = {"$and": [{"user_id": user_id}, {"md5": md5_to_delete}]}
                await asyncio.to_thread(self.vectors_store.delete, where=where_clause)
                _logger.info("已删除 MD5 对应文档: user_id=%s, md5=%s", user_id, md5_to_delete)

            return True

        except Exception as e:
            _logger.error("删除 MD5 记录出错: user_id=%s, md5=%s, error=%s", user_id, md5_to_delete, e)
            return False

    async def get_md5_info(self, user_id: str, md5_value: str):
        """获取 MD5 对应的文档信息。

        Args:
            user_id: 用户 ID。
            md5_value: MD5 值。

        Returns:
            MD5 信息字典，不存在返回 None。
        """
        try:
            return await self.md5_store.get_md5_info(user_id, md5_value)
        except Exception as e:
            _logger.error("获取 MD5 信息出错: user_id=%s, md5=%s, error=%s", user_id, md5_value, e)
            return None

    async def get_all_md5_records(self, user_id: str):
        """获取用户的所有 MD5 记录。

        Args:
            user_id: 用户 ID。

        Returns:
            MD5 记录列表。
        """
        try:
            records = await self.md5_store.get_all_md5_records(user_id)
            _logger.info("获取用户 MD5 记录: user_id=%s, count=%d", user_id, len(records))
            return records
        except Exception as e:
            _logger.error("获取用户 MD5 记录出错: user_id=%s, error=%s", user_id, e)
            return []

    async def get_user_documents(self, user_id: str | None = None):
        """获取用户的知识库文档列表。

        Args:
            user_id: 用户 ID，如果为 None 则获取所有文档。

        Returns:
            文档信息列表，包含文件名、文档数量、预览等信息。
        """
        try:
            where_clause: Where | None = {"user_id": user_id} if user_id else None
            all_docs = await asyncio.to_thread(
                self.vectors_store.get, include=["documents", "metadatas"], where=where_clause
            )

            docs_info = {}

            for i, doc_id in enumerate(all_docs["ids"]):
                metadata = all_docs["metadatas"][i] if i < len(all_docs["metadatas"]) else {}
                content = all_docs["documents"][i] if i < len(all_docs["documents"]) else ""

                filename = metadata.get("source", metadata.get("filename", "unknown"))
                if isinstance(filename, str) and "\\" in filename:
                    filename = os.path.basename(filename)

                original_filename = metadata.get("original_filename", filename)
                if filename not in docs_info:
                    docs_info[filename] = {
                        "id": doc_id,
                        "filename": filename,
                        "original_filename": original_filename,
                        "user_id": metadata.get("user_id"),
                        "chunk_count": 0,
                        "preview": "",
                        "created_at": metadata.get("created_at"),
                    }

                docs_info[filename]["chunk_count"] += 1

                if not docs_info[filename]["preview"] and content:
                    preview_length = 100
                    docs_info[filename]["preview"] = content[:preview_length] + (
                        "..." if len(content) > preview_length else ""
                    )

            result = list(docs_info.values())
            _logger.info("获取用户知识库文档: user_id=%s, count=%d", user_id, len(result))
            return result

        except Exception as e:
            _logger.error("获取用户知识库文档出错: user_id=%s, error=%s", user_id, e)
            raise

    async def get_document_detail(self, user_id: str, filename: str):
        """获取文档的详细内容。

        Args:
            user_id: 用户 ID。
            filename: 文件名。

        Returns:
            文档详情信息，包含完整内容。
        """
        try:
            where_clause: Where = {"user_id": user_id}
            all_docs = await asyncio.to_thread(
                self.vectors_store.get, include=["documents", "metadatas"], where=where_clause
            )

            doc_info = None
            full_content = []
            chunk_count = 0

            for i, doc_id in enumerate(all_docs["ids"]):
                metadata = all_docs["metadatas"][i] if i < len(all_docs["metadatas"]) else {}
                content = all_docs["documents"][i] if i < len(all_docs["documents"]) else ""

                source = metadata.get("source", metadata.get("filename", ""))
                if isinstance(source, str):
                    source_name = os.path.basename(source)
                else:
                    source_name = str(source)

                if source_name == filename:
                    if not doc_info:
                        doc_info = {
                            "id": doc_id,
                            "filename": filename,
                            "user_id": metadata.get("user_id"),
                            "chunk_count": 0,
                            "content": "",
                            "created_at": metadata.get("created_at"),
                        }
                    chunk_count += 1
                    full_content.append(content)

            if doc_info:
                doc_info["chunk_count"] = chunk_count
                doc_info["content"] = "\n".join(full_content)

            _logger.info("获取文档详情: filename=%s, chunk_count=%d", filename, chunk_count)
            return doc_info

        except Exception as e:
            _logger.error("获取文档详情出错: filename=%s, error=%s", filename, e)
            raise

    async def get_document_chunks(self, user_id: str, filename: str):
        """获取文档的所有切片信息。

        Args:
            user_id: 用户 ID。
            filename: 文件名。

        Returns:
            切片列表信息。
        """
        try:
            where_clause: Where = {"user_id": user_id}
            all_docs = await asyncio.to_thread(
                self.vectors_store.get, include=["documents", "metadatas"], where=where_clause
            )

            chunks = []
            chunk_index = 0

            for i, doc_id in enumerate(all_docs["ids"]):
                metadata = all_docs["metadatas"][i] if i < len(all_docs["metadatas"]) else {}
                content = all_docs["documents"][i] if i < len(all_docs["documents"]) else ""

                source = metadata.get("source", metadata.get("filename", ""))
                if isinstance(source, str):
                    source_name = os.path.basename(source)
                else:
                    source_name = str(source)

                if source_name == filename:
                    chunks.append({"chunk_id": doc_id, "index": chunk_index, "content": content, "metadata": metadata})
                    chunk_index += 1

            result = {"filename": filename, "total_chunks": len(chunks), "chunks": chunks}

            _logger.info("获取文档切片: filename=%s, total_chunks=%d", filename, len(chunks))
            return result

        except Exception as e:
            _logger.error("获取文档切片出错: filename=%s, error=%s", filename, e)
            raise

    async def get_file_document(self, read_path: str) -> list[Document]:
        """获取文件文档。

        Args:
            read_path: 文件路径。

        Returns:
            文档列表。
        """
        return await self.document_processor.get_file_document(read_path)

    def get_file_document_sync(self, read_path: str) -> list[Document]:
        """同步获取文件文档。

        Args:
            read_path: 文件路径。

        Returns:
            文档列表。
        """
        return self.document_processor.get_file_document_sync(read_path)

    def split_documents_sync(self, documents: list[Document]) -> list[Document]:
        """同步分割文档。

        Args:
            documents: 文档列表。

        Returns:
            分割后的文档列表。
        """
        return self.document_processor.split_documents_sync(documents)

    async def get_document(
        self,
        files: list | None = None,
        user_id: str | None = None,
        is_public: bool = False,
        progress_callback=None,
    ):
        """获取文档。

        Args:
            files: 上传的文件列表。
            user_id: 用户 ID。
            is_public: 是否为公共文档。
            progress_callback: 进度回调函数。
        """
        await self.document_processor.get_document(files, user_id, is_public, progress_callback)


if __name__ == "__main__":

    async def main():
        store = get_vector_store()
        await store.get_document()

        retriever = await store.get_retriever()
        results = await retriever.ainvoke("扫地")
        print(f"检索结果数量: {len(results)}")
        for result in results:
            print(result)

    asyncio.run(main())
