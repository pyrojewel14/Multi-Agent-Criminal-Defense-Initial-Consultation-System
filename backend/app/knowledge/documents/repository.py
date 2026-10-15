"""协调文档管理与去重记录，不执行模型推理或检索。"""

import asyncio

from app.knowledge.storage.chroma import ChromaStore

from .fingerprints import FingerprintStore


def _filename(metadata):
    """优先使用上传身份；旧记录兼容两种路径分隔符。"""
    value = metadata.get("original_filename") or metadata.get("filename") or metadata.get("source") or "unknown"
    return str(value).replace("\\", "/").rsplit("/", 1)[-1]


class DocumentRepository:
    """只访问显式绑定的文档库；用户读取始终包含所有者范围。"""

    def __init__(self, store: ChromaStore, fingerprints: FingerprintStore):
        self.store = store
        self.fingerprints = fingerprints

    async def _user_rows(self, user_id: str):
        """缺少身份时返回空，禁止退化为全库读取。"""
        if not user_id:
            return ()
        return await asyncio.to_thread(self.store.get_by_metadata, where={"user_id": user_id})

    async def get_user_documents(self, user_id: str):
        """按上传文件名分组并保留现有文档列表字段。"""
        docs_info = {}
        for row in await self._user_rows(user_id):
            metadata, content = row.metadata, row.content
            filename = _filename(metadata)
            if filename not in docs_info:
                docs_info[filename] = {
                    "id": row.row_id,
                    "filename": filename,
                    "original_filename": metadata.get("original_filename", filename),
                    "user_id": metadata.get("user_id"),
                    "chunk_count": 0,
                    "preview": "",
                    "created_at": metadata.get("created_at"),
                }
            docs_info[filename]["chunk_count"] += 1
            if not docs_info[filename]["preview"] and content:
                docs_info[filename]["preview"] = content[:100] + ("..." if len(content) > 100 else "")
        return list(docs_info.values())

    async def get_document_chunks(self, user_id: str, filename: str):
        """使用与列表一致的文件身份匹配分块并保留响应结构。"""
        chunks = []
        for row in await self._user_rows(user_id):
            if _filename(row.metadata) == filename:
                chunks.append(
                    {
                        "chunk_id": row.row_id,
                        "index": len(chunks),
                        "content": row.content,
                        "metadata": dict(row.metadata),
                    }
                )
        return {"filename": filename, "total_chunks": len(chunks), "chunks": chunks}

    async def get_document_detail(self, user_id: str, filename: str):
        """从用户范围内的分块组装原有详情字段。"""
        result = await self.get_document_chunks(user_id, filename)
        if not result["chunks"]:
            return None
        first = result["chunks"][0]
        return {
            "id": first["chunk_id"],
            "filename": filename,
            "user_id": first["metadata"].get("user_id"),
            "chunk_count": result["total_chunks"],
            "content": "\n".join(chunk["content"] for chunk in result["chunks"]),
            "created_at": first["metadata"].get("created_at"),
        }

    async def delete_all_documents(self):
        """仅由管理员入口调用显式全库清理能力。"""
        await asyncio.to_thread(self.store.clear_all_rows)
        await self.fingerprints.clear_all()

    async def delete_user_documents(self, user_id: str):
        """删除用户文档与记录，拒绝缺失所有者。"""
        await self.delete_user_md5(user_id, True)

    async def delete_user_md5(self, user_id: str, delete_documents: bool = True):
        """按调用方要求清理用户记录及其文档。"""
        if not user_id:
            raise ValueError("user_id is required")
        if delete_documents:
            await asyncio.to_thread(self.store.delete_rows, where={"user_id": user_id})
        await self.fingerprints.delete_user_md5(user_id)

    async def delete_single_md5(self, user_id: str, md5_value: str, delete_documents: bool = True):
        """先确认用户记录存在，再清理对应分块及记录。"""
        record = await self.fingerprints.get_md5_info(user_id, md5_value)
        if record is None:
            return False
        if delete_documents:
            await asyncio.to_thread(self.store.delete_rows, where={"$and": [{"user_id": user_id}, {"md5": md5_value}]})
        return await self.fingerprints.delete_single_md5(user_id, md5_value)

    async def delete_by_filename(self, user_id: str, filename: str, delete_documents: bool = True):
        """保持文件名删除接口，所有存储删除均限定用户与 MD5。"""
        for record in await self.fingerprints.get_all_md5_records(user_id):
            if record.get("filename", record.get("original_filename")) == filename:
                return await self.delete_single_md5(user_id, record["md5"], delete_documents)
        return False

    async def get_md5_info(self, user_id: str, md5_value: str):
        """读取用户的既有 MD5 记录。"""
        return await self.fingerprints.get_md5_info(user_id, md5_value)

    async def get_all_md5_records(self, user_id: str):
        """读取用户的既有 MD5 记录列表。"""
        return await self.fingerprints.get_all_md5_records(user_id)
