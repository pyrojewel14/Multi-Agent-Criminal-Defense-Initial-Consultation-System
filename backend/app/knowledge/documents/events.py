"""定义上传进度事件和工作线程交回的切片结果。"""

import json
from dataclasses import asdict, dataclass
from typing import Optional

EVENT_RESPONSE = "response"
EVENT_ERROR = "error"
EVENT_DONE = "done"


@dataclass
class SSEEvent:
    """保存上传阶段、文件身份和进度计数，供前端增量展示。"""

    event_type: str
    message: str
    total_files: int = 0
    file_index: Optional[int] = None
    filename: Optional[str] = None
    step: Optional[str] = None
    progress: int = 0
    success_count: int = 0
    failed_count: int = 0
    slice_success_count: int = 0
    error_message: Optional[str] = None
    chunk_count: Optional[int] = None

    def to_sse(self) -> str:
        """编码为 progress 事件；具体业务阶段保留在 JSON 的 event_type 中。"""
        payload = {k: v for k, v in asdict(self).items() if v is not None}
        return f"event: progress\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


class SliceResult:
    """在线程间传递单个文件的切片成功结果或错误信息。"""

    def __init__(self):
        self.file_index: int = 0
        self.filename: str = ""
        self.documents: list = []
        self.md5: str = ""
        self.success: bool = False
        self.error: Optional[str] = None
        self.chunk_count: int = 0

    @classmethod
    def success_result(cls, file_index: int, filename: str, documents: list, md5: str) -> "SliceResult":
        """保存分块及文件指纹，供后续异步写入阶段使用。"""
        result = cls()
        result.file_index = file_index
        result.filename = filename
        result.documents = documents
        result.md5 = md5
        result.success = True
        result.chunk_count = len(documents)
        return result

    @classmethod
    def error_result(cls, file_index: int, filename: str, error: str) -> "SliceResult":
        """保存文件身份与切片错误，不附带可写入的分块。"""
        result = cls()
        result.file_index = file_index
        result.filename = filename
        result.success = False
        result.error = error
        return result

    def to_dict(self) -> dict:
        """导出结果字段，documents 保留原始分块对象。"""
        return {
            "file_index": self.file_index,
            "filename": self.filename,
            "documents": self.documents,
            "md5": self.md5,
            "success": self.success,
            "error": self.error,
            "chunk_count": self.chunk_count,
        }
