import asyncio
import time
import json
import magic
import os
import tempfile
from typing import List, Optional, Dict, Any, AsyncGenerator
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from fastapi import HTTPException, UploadFile

from app.utils.logger import get_logger
from app.rag.vector_store import get_vector_store, VectorStoreService
from app.rag.task_queue import TaskQueue
from app.rag.sse_models import SSEEvent, SliceResult
from app.utils.file_handler import get_file_md5_hex_sync


ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.md', '.pptx', '.docx', '.json'}
ALLOWED_MIME_TYPES = {
    'application/pdf', 'text/plain', 'text/markdown',
    'application/vnd.ms-powerpoint',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/json'
}
MAX_FILE_SIZE = 20 * 1024 * 1024
MAX_FOLDER_SIZE = 200 * 1024 * 1024


_logger = get_logger("KnowledgeService")


@dataclass
class ProcessingState:
    total_files: int = 0
    total_valid: int = 0
    sliced_count: int = 0
    written_count: int = 0
    success_count: int = 0
    failed_count: int = 0
    slice_success_count: int = 0

    def current_progress(self) -> int:
        if self.total_valid == 0:
            return 0
        slice_progress = (self.sliced_count / self.total_valid) * 60
        write_progress = (self.written_count / self.total_valid) * 40
        return int(min(99, slice_progress + write_progress))


def _sync_slice_file(file_content: bytes, filename: str, file_index: int, user_id: str, is_public: bool, queue: TaskQueue):
    """在 ThreadPoolExecutor 中执行的同步切片函数。"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        try:
            store = get_vector_store()
            documents = store.get_file_document_sync(temp_file_path)
            if not documents:
                queue.put(SliceResult.error_result(file_index=file_index, filename=filename, error="文件加载为空"))
                return

            split_docs = store.split_documents_sync(documents)
            if not split_docs:
                queue.put(SliceResult.error_result(file_index=file_index, filename=filename, error="切片结果为空"))
                return

            md5_hex = get_file_md5_hex_sync(temp_file_path)
            for doc in split_docs:
                doc.metadata['user_id'] = user_id
                doc.metadata['original_filename'] = filename
                doc.metadata['md5'] = md5_hex
                doc.metadata['is_public'] = is_public

            queue.put(SliceResult.success_result(
                file_index=file_index, filename=filename, documents=split_docs, md5=md5_hex
            ))
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    except Exception as e:
        _logger.error("SSE 上传切片文件出错: filename=%s, error=%s", filename, e)
        queue.put(SliceResult.error_result(file_index=file_index, filename=filename, error=str(e)))


class KnowledgeService:
    """知识库管理服务"""

    async def handle_add_vector_single(self, file: UploadFile, user_id: str, is_public: bool = False) -> str:
        """处理添加单个向量逻辑。

        Args:
            file: 上传的文件。
            user_id: 用户 ID。
            is_public: 是否为公共文档。

        Returns:
            文件名。
        """
        store = get_vector_store()

        if file.size > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="文件大小不能超过 20MB")

        content = await file.read()
        await file.seek(0)

        mime = magic.Magic(mime=True)
        file_type = mime.from_buffer(content)

        file_extension = os.path.splitext(file.filename)[1].lower()

        if file_type not in ALLOWED_MIME_TYPES and file_extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"文件类型不支持，目前支持 PDF、TXT、Markdown、PPTX、DOCX 文件类型。检测到的文件类型: {file_type}，扩展名: {file_extension}"
            )

        await store.get_document(files=[file], user_id=user_id, is_public=is_public)
        return file.filename

    async def handle_add_vector_multiple(self, files: List[UploadFile], user_id: str, is_public: bool = False) -> List[str]:
        """处理添加多个向量逻辑。

        Args:
            files: 上传的文件列表。
            user_id: 用户 ID。
            is_public: 是否为公共文档。

        Returns:
            文件名列表。
        """
        total_size = 0
        for file in files:
            total_size += file.size or 0

        if total_size > MAX_FOLDER_SIZE:
            raise HTTPException(status_code=400, detail="文件总大小不能超过 200MB")

        start_time = time.time()
        results = []
        for file in files:
            try:
                await self.handle_add_vector_single(file, user_id, is_public)
                results.append(file.filename)
            except Exception as e:
                _logger.error("添加向量处理文件出错: filename=%s, error=%s", file.filename, e)
                raise

        end_time = time.time()
        _logger.info("添加向量完成: count=%d, time=%.2f 秒", len(results), end_time - start_time)

        return results

    def _yield_start_event(self, total_files: int) -> str:
        """SSE 事件：开始处理，通知前端文件总数。"""
        return SSEEvent(
            event_type='start', total_files=total_files, message='开始处理文件...', progress=0
        ).to_sse()

    def _yield_size_error_event(self) -> str:
        """SSE 事件：文件总大小超限错误。"""
        return SSEEvent(
            event_type='error', message='文件总大小不能超过 200MB',
            error_message='文件总大小不能超过 200MB'
        ).to_sse()

    def _yield_validation_error_event(
        self, current_index: int, total_files: int, filename: str,
        file_type: str, file_extension: str, failed_count: int
    ) -> str:
        """SSE 事件：单个文件 MIME 类型验证失败。"""
        return SSEEvent(
            event_type='error', file_index=current_index, total_files=total_files,
            filename=filename, step='validation',
            message=f'文件 {filename} 类型不支持',
            error_message=f'文件类型: {file_type}，扩展名: {file_extension}',
            progress=int(current_index / total_files * 100),
            failed_count=failed_count
        ).to_sse()

    def _yield_slicing_completed_event(self, result: SliceResult, state: ProcessingState) -> str:
        """SSE 事件：单个文件多线程切片完成，准备写入向量库。"""
        return SSEEvent(
            event_type='slicing_completed', file_index=result.file_index,
            total_files=state.total_files, filename=result.filename,
            chunk_count=result.chunk_count, step='slicing',
            message=f'文件 {result.filename} 切片完成，共 {result.chunk_count} 个切片',
            progress=state.current_progress(),
            success_count=state.success_count, failed_count=state.failed_count,
            slice_success_count=state.slice_success_count
        ).to_sse()

    def _yield_writing_event(self, result: SliceResult, state: ProcessingState) -> str:
        """SSE 事件：开始将切片结果写入向量数据库。"""
        return SSEEvent(
            event_type='writing', file_index=result.file_index,
            total_files=state.total_files, filename=result.filename,
            step='writing', message=f'正在写入向量 {result.filename}...',
            progress=state.current_progress(),
            success_count=state.success_count, failed_count=state.failed_count,
            slice_success_count=state.slice_success_count
        ).to_sse()

    def _yield_completed_event(self, result: SliceResult, state: ProcessingState) -> str:
        """SSE 事件：单个文件全部处理完成（切片+写入成功）。"""
        return SSEEvent(
            event_type='completed', file_index=result.file_index,
            total_files=state.total_files, filename=result.filename,
            step='completed', message=f'文件 {result.filename} 处理完成',
            progress=state.current_progress(),
            success_count=state.success_count, failed_count=state.failed_count,
            slice_success_count=state.slice_success_count
        ).to_sse()

    def _yield_write_error_event(self, result: SliceResult, state: ProcessingState, error: str) -> str:
        """SSE 事件：切片结果写入向量数据库时发生异常。"""
        return SSEEvent(
            event_type='error', file_index=result.file_index,
            total_files=state.total_files, filename=result.filename,
            step='writing', message=f'文件 {result.filename} 写入失败',
            error_message=error,
            progress=state.current_progress(),
            success_count=state.success_count, failed_count=state.failed_count,
            slice_success_count=state.slice_success_count
        ).to_sse()

    def _yield_slice_error_event(self, result: SliceResult, state: ProcessingState) -> str:
        """SSE 事件：单个文件切片阶段失败（文件损坏/格式不支持等）。"""
        return SSEEvent(
            event_type='error', file_index=result.file_index,
            total_files=state.total_files, filename=result.filename,
            step='slicing', message=f'文件 {result.filename} 切片失败',
            error_message=result.error,
            progress=state.current_progress(),
            success_count=state.success_count, failed_count=state.failed_count,
            slice_success_count=state.slice_success_count
        ).to_sse()

    def _yield_finish_event(self, start_time: float, total_files: int, success_count: int, failed_count: int) -> str:
        """SSE 事件：所有文件处理结束，汇总统计信息。"""
        total_time = round(time.time() - start_time, 2)
        return SSEEvent(
            event_type='finish', total_files=total_files,
            success_count=success_count, failed_count=failed_count,
            message=f'处理完成，耗时 {total_time} 秒', progress=100
        ).to_sse()

    async def _validate_and_read_files(
        self, files: List[UploadFile]
    ) -> tuple[List[dict], List[str], int]:
        """阶段 1: 读取文件内容并验证总大小；阶段 2: 逐一验证文件 MIME 类型。

        Args:
            files: 上传的文件列表。

        Returns:
            (有效文件列表, SSE 错误事件列表, 总文件数)。
        """
        total_files = len(files)
        total_size = 0
        files_content = []
        error_events: List[str] = []

        for file in files:
            content = await file.read()
            files_content.append({'file': file, 'content': content})
            total_size += len(content)
            await file.seek(0)

        if total_size > MAX_FOLDER_SIZE:
            _logger.error("SSE 上传文件总大小超过限制: total_size=%.2fMB, limit=200MB", total_size / (1024 * 1024))
            return [], [self._yield_size_error_event()], total_files

        mime = magic.Magic(mime=True)
        valid_files = []
        current_index = 1
        failed_count = 0

        for file_info in files_content:
            file = file_info['file']
            content = file_info['content']
            file_type = mime.from_buffer(content)
            file_extension = os.path.splitext(file.filename)[1].lower()

            if file_type not in ALLOWED_MIME_TYPES and file_extension not in ALLOWED_EXTENSIONS:
                failed_count += 1
                error_events.append(self._yield_validation_error_event(
                    current_index, total_files, file.filename,
                    file_type, file_extension, failed_count
                ))
                _logger.warning("SSE 上传文件类型验证失败: filename=%s, file_type=%s, extension=%s", file.filename, file_type, file_extension)
            else:
                valid_files.append({
                    'content': content,
                    'filename': file.filename,
                    'file_index': current_index
                })
                _logger.debug("SSE 上传文件类型验证通过: filename=%s", file.filename)
            current_index += 1

        return valid_files, error_events, total_files

    def _start_slicing(
        self, valid_files: List[dict], user_id: str, is_public: bool = False
    ) -> tuple[TaskQueue, ThreadPoolExecutor, list]:
        """启动多线程切片，返回 (队列, 执行器, future 列表)。

        Args:
            valid_files: 有效文件列表。
            user_id: 用户 ID。
            is_public: 是否为公共文档。

        Returns:
            (队列, 执行器, future 列表)。
        """
        queue = TaskQueue(maxsize=10)
        queue.set_total_count(len(valid_files))

        slice_tasks = [
            (info['content'], info['filename'], info['file_index'], user_id, is_public)
            for info in valid_files
        ]

        max_workers = min(len(slice_tasks), max(1, os.cpu_count() or 1))
        _logger.info("SSE 上传切片阶段使用线程数: %d", max_workers)

        executor = ThreadPoolExecutor(max_workers=max_workers)
        futures = [executor.submit(_sync_slice_file, *args, queue) for args in slice_tasks]

        return queue, executor, futures

    async def _process_slice_results(
        self, queue: TaskQueue, valid_count: int, store: VectorStoreService,
        state: ProcessingState, user_id: str, is_public: bool = False
    ) -> AsyncGenerator[str, None]:
        """消费切片队列 → 写入向量库 → yield SSE 进度事件。

        Args:
            queue: 任务队列。
            valid_count: 有效文件数量。
            store: 向量存储服务。
            state: 处理状态。
            user_id: 用户 ID。
            is_public: 是否为公共文档。
        """
        while state.written_count < valid_count:
            try:
                result = queue.get(block=True, timeout=0.1)

                state.sliced_count += 1

                if result.success:
                    state.slice_success_count += 1

                    yield self._yield_slicing_completed_event(result, state)

                    try:
                        yield self._yield_writing_event(result, state)

                        await asyncio.to_thread(store.vectors_store.add_documents, result.documents)
                        await store.save_md5_hex(result.md5, result.filename, result.filename, user_id)

                        state.success_count += 1
                        state.written_count += 1

                        yield self._yield_completed_event(result, state)
                        _logger.info("SSE 上传文件写入完成: filename=%s", result.filename)

                    except Exception as e:
                        state.written_count += 1
                        state.failed_count += 1
                        _logger.error("SSE 上传写入文件出错: filename=%s, error=%s", result.filename, e)
                        yield self._yield_write_error_event(result, state, str(e))

                else:
                    state.written_count += 1
                    state.failed_count += 1
                    _logger.error("SSE 上传切片文件失败: filename=%s, error=%s", result.filename, result.error)
                    yield self._yield_slice_error_event(result, state)

                queue.task_done()

            except Exception:
                continue

    async def handle_add_vector_multiple_stream(
        self,
        files: List[UploadFile],
        user_id: str,
        is_public: bool = False
    ) -> AsyncGenerator[str, None]:
        """处理多个文件上传并返回流式进度（多线程切片 + 单线程串行写入）。

        Args:
            files: 上传的文件列表。
            user_id: 用户 ID。
            is_public: 是否为公共文档。

        Yields:
            SSE 事件字符串。
        """
        total_files = len(files)
        _logger.info("SSE 上传开始处理: count=%d, user_id=%s", total_files, user_id)

        yield self._yield_start_event(total_files)

        valid_files, error_events, _ = await self._validate_and_read_files(files)
        for event in error_events:
            yield event

        if not valid_files:
            _logger.info("SSE 上传无有效文件可处理")
            return

        start_time = time.time()
        state = ProcessingState(
            total_files=total_files,
            total_valid=len(valid_files)
        )

        queue, executor, _ = self._start_slicing(valid_files, user_id, is_public)

        store = get_vector_store()
        async for sse in self._process_slice_results(queue, len(valid_files), store, state, user_id, is_public):
            yield sse

        executor.shutdown(wait=True)

        _logger.info(
            "SSE 上传文件处理完成: total=%d, success=%d, failed=%d, time=%.2f 秒",
            total_files, state.success_count, state.failed_count, time.time() - start_time
        )

        yield self._yield_finish_event(start_time, total_files, state.success_count, state.failed_count)

    def _calculate_progress(self, sliced_count: int, written_count: int, total: int) -> int:
        """计算处理进度。

        Args:
            sliced_count: 已切片数量。
            written_count: 已写入数量。
            total: 总数。

        Returns:
            进度百分比。
        """
        if total == 0:
            return 0
        slice_progress = (sliced_count / total) * 60
        write_progress = (written_count / total) * 40
        return int(min(99, slice_progress + write_progress))

    async def clean_all(self) -> None:
        """处理清空所有知识库文档（供超管使用）。

        删除向量库中的所有文档和 MD5 记录。
        """
        store = get_vector_store()
        await store.delete_all_documents()

    async def clean_user_upload(self, user_id: str) -> None:
        """处理删除用户上传的所有向量逻辑。

        Args:
            user_id: 用户 ID。
        """
        store = get_vector_store()
        await store.delete_user_documents(user_id)

    async def handle_clear_all_md5(self, delete_documents: bool = True) -> None:
        """处理清空所有 MD5 记录（供超管使用）。

        Args:
            delete_documents: 是否同时删除知识库文档（默认 True）。
        """
        store = get_vector_store()
        if delete_documents:
            await store.delete_all_documents()
        else:
            await store.md5_store.clear_all()
        _logger.info("已清空所有 MD5 记录（delete_documents=%s）", delete_documents)

    async def handle_clear_user_md5(self, user_id: str, delete_documents: bool = True) -> None:
        """清空用户 MD5 记录。

        Args:
            user_id: 用户 ID。
            delete_documents: 是否同时删除知识库文档（默认 True）。
        """
        store = get_vector_store()
        await store.delete_user_md5(user_id, delete_documents)
        if delete_documents:
            _logger.info("知识库清空用户 MD5 记录和文档: user_id=%s", user_id)
        else:
            _logger.info("知识库清空用户 MD5 记录（保留文档）: user_id=%s", user_id)

    async def handle_delete_single_md5(self, user_id: str, md5_value: str, delete_documents: bool = True) -> bool:
        """删除单个 MD5 记录。

        Args:
            user_id: 用户 ID。
            md5_value: MD5 值。
            delete_documents: 是否同时删除知识库文档（默认 True）。

        Returns:
            是否成功删除。
        """
        store = get_vector_store()
        success = await store.delete_single_md5(user_id, md5_value, delete_documents)
        if success:
            _logger.info("知识库删除 MD5 记录: user_id=%s, md5=%s", user_id, md5_value)
        else:
            _logger.warning("知识库删除 MD5 记录失败: user_id=%s, md5=%s", user_id, md5_value)
        return success

    async def handle_delete_by_filename(self, user_id: str, filename: str, delete_documents: bool = True) -> bool:
        """通过文件名删除 MD5 记录。

        Args:
            user_id: 用户 ID。
            filename: 文件名。
            delete_documents: 是否同时删除知识库文档（默认 True）。

        Returns:
            是否成功删除。
        """
        store = get_vector_store()
        success = await store.delete_by_filename(user_id, filename, delete_documents)
        if success:
            _logger.info("知识库删除文件: user_id=%s, filename=%s", user_id, filename)
        else:
            _logger.warning("知识库删除文件失败: user_id=%s, filename=%s", user_id, filename)
        return success

    async def handle_get_md5_info(self, user_id: str, md5_value: str):
        """获取 MD5 对应的文档信息。

        Args:
            user_id: 用户 ID。
            md5_value: MD5 值。

        Returns:
            MD5 信息字典。
        """
        store = get_vector_store()
        return await store.get_md5_info(user_id, md5_value)

    async def handle_get_all_md5_records(self, user_id: str):
        """获取用户的所有 MD5 记录。

        Args:
            user_id: 用户 ID。

        Returns:
            MD5 记录列表。
        """
        store = get_vector_store()
        return await store.get_all_md5_records(user_id)

    async def handle_get_user_knowledge(self, user_id: str) -> list:
        """获取用户的知识库文档列表。

        Args:
            user_id: 用户 ID。

        Returns:
            文档信息列表。
        """
        store = get_vector_store()
        documents = await store.get_user_documents(user_id)
        _logger.info("知识库获取用户文档: user_id=%s, count=%d", user_id, len(documents))
        return documents

    async def handle_get_document_detail(self, user_id: str, filename: str) -> dict:
        """获取文档的详细内容。

        Args:
            user_id: 用户 ID。
            filename: 文件名。

        Returns:
            文档详情信息。
        """
        store = get_vector_store()
        document = await store.get_document_detail(user_id, filename)
        if not document:
            raise HTTPException(status_code=404, detail=f"文档 {filename} 不存在")
        _logger.info("知识库获取文档详情: filename=%s", filename)
        return document

    async def handle_get_document_chunks(self, user_id: str, filename: str) -> dict:
        """获取文档切片信息。

        Args:
            user_id: 用户 ID。
            filename: 文件名。

        Returns:
            切片信息字典。
        """
        store = get_vector_store()
        chunks = await store.get_document_chunks(user_id, filename)
        if chunks['total_chunks'] == 0:
            raise HTTPException(status_code=404, detail=f"文档 {filename} 不存在或没有切片")
        _logger.info("知识库获取文档切片: filename=%s, count=%d", filename, chunks['total_chunks'])
        return chunks


def get_knowledge_service() -> KnowledgeService:
    """获取知识库服务实例（用于依赖注入）。

    Returns:
        KnowledgeService 实例。
    """
    return KnowledgeService()
