import hashlib
import asyncio
import json
import os
import shutil
from datetime import datetime

import aiofiles
from aiofiles import os as aio_os

from app.infrastructure.logging import get_logger

_logger = get_logger("MD5Store")


class FingerprintStore:
    """MD5 存储管理器"""

    def __init__(self, base_dir: str):
        self.base_dir = os.fspath(base_dir)

    def _get_md5_store_dir(self, user_id: str | None = None) -> str:
        """获取 MD5 存储目录。

        Args:
            user_id: 用户 ID，为 None 时返回公共目录。

        Returns:
            MD5 存储目录路径。
        """
        if user_id:
            return os.path.join(self.base_dir, "user_md5", user_id)
        else:
            return os.path.join(self.base_dir, "public_md5")

    async def check_md5_hex(self, md5_for_check: str, user_id: str | None = None) -> bool:
        """异步检查 MD5 是否存在。

        Args:
            md5_for_check: 要检查的 MD5 值。
            user_id: 用户 ID，为 None 时检查公共知识库。

        Returns:
            是否存在。
        """
        md5_dir = self._get_md5_store_dir(user_id)
        md5_path = os.path.join(md5_dir, "md5_hex_store.txt")

        if not await aio_os.path.exists(md5_dir):
            await aio_os.makedirs(md5_dir, exist_ok=True)
            async with aiofiles.open(md5_path, "w", encoding="utf-8"):
                pass
            return False

        if not await aio_os.path.exists(md5_path):
            async with aiofiles.open(md5_path, "w", encoding="utf-8"):
                pass
            return False

        try:
            async with aiofiles.open(md5_path, "r", encoding="utf-8") as f:
                async for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("{"):
                        try:
                            data = json.loads(line)
                            if data.get("md5") == md5_for_check:
                                return True
                        except BaseException:
                            if line == md5_for_check:
                                return True
                    else:
                        if line == md5_for_check:
                            return True
            return False
        except Exception as e:
            _logger.error("检查 MD5 时出错: %s", e)
            return False

    async def save_md5_hex(
        self,
        md5_hex: str,
        filename: str | None = None,
        original_filename: str | None = None,
        user_id: str | None = None,
    ):
        """异步保存 MD5。

        Args:
            md5_hex: 要保存的 MD5 值。
            filename: 文件名（可选）。
            original_filename: 原始文件名（可选）。
            user_id: 用户 ID，为 None 时保存到公共知识库。
        """
        md5_dir = self._get_md5_store_dir(user_id)
        md5_path = os.path.join(md5_dir, "md5_hex_store.txt")

        if not await aio_os.path.exists(md5_dir):
            await aio_os.makedirs(md5_dir, exist_ok=True)

        data = {
            "md5": md5_hex,
            "filename": filename,
            "original_filename": original_filename,
            "upload_time": datetime.now().isoformat(),
        }

        async with aiofiles.open(md5_path, "a", encoding="utf-8") as f:
            await f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def save_md5_hex_sync(
        self,
        md5_hex: str,
        filename: str | None = None,
        original_filename: str | None = None,
        user_id: str | None = None,
    ):
        """同步保存 MD5（用于多线程场景）。

        Args:
            md5_hex: 要保存的 MD5 值。
            filename: 文件名（可选）。
            original_filename: 原始文件名（可选）。
            user_id: 用户 ID，为 None 时保存到公共知识库。
        """
        md5_dir = self._get_md5_store_dir(user_id)
        md5_path = os.path.join(md5_dir, "md5_hex_store.txt")

        if not os.path.exists(md5_dir):
            os.makedirs(md5_dir, exist_ok=True)

        data = {
            "md5": md5_hex,
            "filename": filename,
            "original_filename": original_filename,
            "upload_time": datetime.now().isoformat(),
        }

        with open(md5_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    async def _read_md5_records(self, user_id: str | None = None) -> tuple:
        """读取用户的 MD5 记录文件。

        Args:
            user_id: 用户 ID，为 None 时读取公共知识库。

        Returns:
            (file_path, records 列表)，每条记录为 dict。
        """
        md5_dir = self._get_md5_store_dir(user_id)
        md5_path = os.path.join(md5_dir, "md5_hex_store.txt")

        if not await aio_os.path.exists(md5_path):
            return md5_path, []

        records = []
        async with aiofiles.open(md5_path, "r", encoding="utf-8") as f:
            async for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("{"):
                    try:
                        records.append(json.loads(line))
                    except BaseException:
                        records.append({"md5": line, "filename": None, "original_filename": None, "upload_time": None})
                else:
                    records.append({"md5": line, "filename": None, "original_filename": None, "upload_time": None})
        return md5_path, records

    async def _write_md5_records(self, md5_path: str, records: list):
        """写入 MD5 记录文件，空列表时自动清理文件及目录。

        Args:
            md5_path: 文件路径。
            records: 记录列表。
        """
        if not records:
            md5_dir = os.path.dirname(md5_path)
            if await aio_os.path.exists(md5_path):
                await aio_os.remove(md5_path)
            if await aio_os.path.exists(md5_dir):
                try:
                    await aio_os.rmdir(md5_dir)
                except OSError:
                    pass
            return

        async with aiofiles.open(md5_path, "w", encoding="utf-8") as f:
            for record in records:
                await f.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def delete_user_md5(self, user_id: str):
        """删除用户的整个 MD5 记录目录。

        Args:
            user_id: 用户 ID。
        """
        md5_dir = self._get_md5_store_dir(user_id)
        if await aio_os.path.exists(md5_dir):
            await asyncio.to_thread(shutil.rmtree, md5_dir)
        _logger.info("已删除用户 MD5 记录: user_id=%s", user_id)

    async def delete_by_filename(self, user_id: str, filename: str):
        """通过文件名删除 MD5 记录。

        Args:
            user_id: 用户 ID。
            filename: 文件名。

        Returns:
            被删记录的 md5 值，不存在返回 None。
        """
        md5_path, records = await self._read_md5_records(user_id)
        if not records:
            return None

        found_md5 = None
        remaining = []
        for record in records:
            record_filename = record.get("filename", record.get("original_filename"))
            if record_filename == filename:
                found_md5 = record.get("md5")
            else:
                remaining.append(record)

        if found_md5 is None:
            return None

        await self._delete_document_chunks(user_id, found_md5)
        await self._write_md5_records(md5_path, remaining)
        _logger.info("已删除文件 MD5 记录: user_id=%s, filename=%s", user_id, filename)
        return found_md5

    async def delete_single_md5(self, user_id: str, md5_to_delete: str) -> bool:
        """删除单个 MD5 记录。

        Args:
            user_id: 用户 ID。
            md5_to_delete: 要删除的 MD5 值。

        Returns:
            是否成功删除。
        """
        md5_path, records = await self._read_md5_records(user_id)
        if not records:
            return False

        remaining = [r for r in records if r.get("md5") != md5_to_delete]
        if len(remaining) == len(records):
            return False

        await self._delete_document_chunks(user_id, md5_to_delete)
        await self._write_md5_records(md5_path, remaining)
        _logger.info("已删除 MD5 记录: user_id=%s, md5=%s", user_id, md5_to_delete)
        return True

    async def _delete_document_chunks(self, user_id: str, doc_md5: str):
        """删除目标文档关联的 chunk 指纹，保留其他文档及旧格式行。"""
        directory = os.path.join(self._get_md5_store_dir(user_id), "chunk_md5")
        path = os.path.join(directory, "chunks.txt")
        if not await aio_os.path.exists(path):
            return
        remaining = []
        async with aiofiles.open(path, "r", encoding="utf-8") as stream:
            async for line in stream:
                try:
                    record = json.loads(line)
                except (ValueError, TypeError):
                    remaining.append(line)
                    continue
                if not isinstance(record, dict) or record.get("doc_md5") != doc_md5:
                    remaining.append(line)
        if remaining:
            async with aiofiles.open(path, "w", encoding="utf-8") as stream:
                await stream.writelines(remaining)
        else:
            await aio_os.remove(path)
            try:
                await aio_os.rmdir(directory)
            except OSError:
                pass

    async def rollback_upload(self, user_id: str, doc_md5: str):
        """清理尚未完成上传的文件和分块指纹，保留其他文档与旧格式行。"""
        path, records = await self._read_md5_records(user_id)
        # 两种记录分别尝试清理；任何失败仍交由服务层报告。
        results = await asyncio.gather(
            self._delete_document_chunks(user_id, doc_md5),
            self._write_md5_records(path, [record for record in records if record.get("md5") != doc_md5]),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def get_md5_info(self, user_id: str, md5_value: str):
        """获取 MD5 对应的文档信息。

        Args:
            user_id: 用户 ID。
            md5_value: MD5 值。

        Returns:
            MD5 信息字典，不存在返回 None。
        """
        _, records = await self._read_md5_records(user_id)
        for record in records:
            if record.get("md5") == md5_value:
                return record
        return None

    async def get_all_md5_records(self, user_id: str) -> list:
        """获取用户的所有 MD5 记录。

        Args:
            user_id: 用户 ID。

        Returns:
            MD5 记录列表。
        """
        _, records = await self._read_md5_records(user_id)
        return records

    async def check_chunk_md5(self, chunk_md5: str, user_id: str) -> bool:
        """检查 chunk 内容 MD5 是否已存在（去重用）。

        Args:
            chunk_md5: chunk 内容前200字符的 MD5。
            user_id: 用户 ID。

        Returns:
            是否已存在。
        """
        md5_dir = self._get_md5_store_dir(user_id)
        chunk_md5_dir = os.path.join(md5_dir, "chunk_md5")
        chunk_md5_file = os.path.join(chunk_md5_dir, "chunks.txt")

        if not await aio_os.path.exists(chunk_md5_file):
            return False

        try:
            async with aiofiles.open(chunk_md5_file, "r", encoding="utf-8") as f:
                async for line in f:
                    if line.strip() == chunk_md5:
                        return True
            return False
        except Exception as e:
            _logger.error("检查 chunk MD5 时出错: %s", e)
            return False

    async def save_chunk_md5(self, chunk_md5: str, user_id: str, doc_md5: str | None = None):
        """保存 chunk 内容 MD5。

        Args:
            chunk_md5: chunk 内容前200字符的 MD5。
            user_id: 用户 ID。
            doc_md5: 所属文档的 MD5（可选，用于追溯）。
        """
        md5_dir = self._get_md5_store_dir(user_id)
        chunk_md5_dir = os.path.join(md5_dir, "chunk_md5")

        if not await aio_os.path.exists(chunk_md5_dir):
            await aio_os.makedirs(chunk_md5_dir, exist_ok=True)

        chunk_md5_file = os.path.join(chunk_md5_dir, "chunks.txt")

        data = {"chunk_md5": chunk_md5, "doc_md5": doc_md5, "save_time": datetime.now().isoformat()}

        async with aiofiles.open(chunk_md5_file, "a", encoding="utf-8") as f:
            await f.write(json.dumps(data, ensure_ascii=False) + "\n")

    async def get_all_chunk_md5(self, user_id: str, *, exclude_doc_md5: str | None = None) -> set:
        """获取用户的所有 chunk MD5（用于快速查重）。

        Args:
            user_id: 用户 ID。
            exclude_doc_md5: 重试未完成上传时忽略该文档的残留指纹。

        Returns:
            chunk MD5 值集合。
        """
        md5_dir = self._get_md5_store_dir(user_id)
        chunk_md5_file = os.path.join(md5_dir, "chunk_md5", "chunks.txt")

        if not await aio_os.path.exists(chunk_md5_file):
            return set()

        chunk_md5s = set()
        try:
            async with aiofiles.open(chunk_md5_file, "r", encoding="utf-8") as f:
                async for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if exclude_doc_md5 is not None and data.get("doc_md5") == exclude_doc_md5:
                            continue
                        chunk_md5s.add(data.get("chunk_md5", ""))
                    except BaseException:
                        pass
            return chunk_md5s
        except Exception as e:
            _logger.error("读取 chunk MD5 时出错: %s", e)
            return set()

    async def clear_all(self):
        """清空所有 MD5 记录（用于超管清空整个知识库）。

        删除用户 MD5 目录和公共 MD5 目录，并重建空目录以确保后续正常使用。
        """
        user_md5_dir = os.path.join(self.base_dir, "user_md5")
        public_md5_dir = os.path.join(self.base_dir, "public_md5")

        if os.path.exists(user_md5_dir):
            try:
                await asyncio.to_thread(shutil.rmtree, user_md5_dir)
                os.makedirs(user_md5_dir, exist_ok=True)
                _logger.info("已清空并重建用户 MD5 目录")
            except Exception as e:
                _logger.error("清空用户 MD5 目录时出错: %s", e)

        if os.path.exists(public_md5_dir):
            try:
                await asyncio.to_thread(shutil.rmtree, public_md5_dir)
                os.makedirs(public_md5_dir, exist_ok=True)
                _logger.info("已清空并重建公共 MD5 目录")
            except Exception as e:
                _logger.error("清空公共 MD5 目录时出错: %s", e)

        _logger.info("已完成所有 MD5 记录清空")


async def get_file_md5_hex(file_path: str) -> str:
    """获取文件的 MD5 值。

    Args:
        file_path: 文件路径。

    Returns:
        MD5 哈希值字符串，失败时返回空字符串。
    """
    abs_file_path = os.path.abspath(file_path)

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


def get_file_md5_hex_sync(file_path: str) -> str:
    """同步获取文件的 MD5 值（用于多线程场景）。


    Args:
        file_path: 文件路径。

    Returns:
        MD5 哈希值字符串，失败时返回空字符串。
    """
    abs_file_path = os.path.abspath(file_path)

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
