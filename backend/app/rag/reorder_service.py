import os
from typing import Any, Dict, List

from dotenv import load_dotenv

from app.rag.reranker import RerankerConfig, RerankerFactory
from app.utils.logger import get_logger

load_dotenv()

_logger = get_logger("RAG.Reorder")


def check_and_download_model(config: RerankerConfig) -> str:
    """检查并下载重排序模型。

    Args:
        config: 重排序模型配置。

    Returns:
        模型本地路径。

    Raises:
        RuntimeError: 模型下载失败时抛出。
    """
    local_path = config.local_path

    if os.path.exists(os.path.join(local_path, "config.json")):
        _logger.info("【check_and_download_model】本地模型已存在: %s", local_path)
        return local_path

    _logger.warning("【check_and_download_model】本地模型未找到: %s", local_path)
    _logger.info("【check_and_download_model】开始从魔搭社区下载模型: %s", config.model_name)

    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        from modelscope import snapshot_download

        downloaded_path = snapshot_download(
            model_id=config.model_name,
            cache_dir=config.cache_dir,
        )

        _logger.info("【check_and_download_model】模型下载完成，保存路径: %s", downloaded_path)
        return downloaded_path

    except Exception as e:
        _logger.error("【check_and_download_model】模型下载失败: %s", str(e))
        raise RuntimeError(f"模型下载失败: {str(e)}") from e


class ReorderService:
    """文档重排序服务。

    使用重排序模型对检索结果进行相关性评分和重排序。
    """

    def __init__(self, reranker_type: str = "causal_lm", config: RerankerConfig = None):
        """初始化重排序服务。

        Args:
            reranker_type: 重排序器类型。
            config: 重排序模型配置。
        """
        self.config = config or RerankerConfig.from_env()
        self.reranker_type = reranker_type
        self._reranker = RerankerFactory.create(self.config, reranker_type)
        _logger.info("【__init__】ReorderService 初始化完成: type=%s, model=%s", reranker_type, self.config.model_name)

    async def reorder_documents(self, query: str, documents: List[str], thinking_callback=None) -> Dict[str, Any]:
        """对文档进行重排序。

        Args:
            query: 查询语句。
            documents: 文档列表。
            thinking_callback: 思考过程回调函数。

        Returns:
            包含重排序结果的字典，格式为：
            {"success": bool, "documents": List[Dict], "error": str}
        """
        try:
            _logger.debug("【reorder_documents】开始重排序: query=%s, doc_count=%d", query[:50], len(documents))
            result = await self._reranker.rerank(query, documents, thinking_callback)

            if result["success"] and result["documents"]:
                sorted_docs = result["documents"]
                _logger.info("【reorder_documents】=== 排序结果 ===")
                for i, doc_info in enumerate(sorted_docs, 1):
                    score = doc_info.get("similarity", 0)
                    content_preview = doc_info.get("document", "")[:80]
                    _logger.info("【reorder_documents】#%d 分数: %.4f | 内容: %s...", i, score, content_preview)
                _logger.info("【reorder_documents】=== 共 %d 个文档 ===", len(sorted_docs))

            _logger.info(
                "【reorder_documents】重排序完成: success=%s, doc_count=%d", result["success"], len(result["documents"])
            )
            return result
        except Exception as e:
            error_msg = str(e)
            _logger.error("【reorder_documents】重排序失败: %s", error_msg)
            return {"success": False, "documents": [], "error": error_msg}

    def readiness(self) -> Dict[str, Any]:
        """返回重排序器状态，不触发模型懒加载。"""
        state = self._reranker.readiness()
        return {**state, "type": self.reranker_type}

    def close(self) -> None:
        """释放重排序器持有的执行器资源。"""
        self._reranker.close()

    @staticmethod
    async def format_reorder_result(sorted_docs: List[Dict]) -> str:
        """格式化重排序结果。

        Args:
            sorted_docs: 重排序后的文档列表。

        Returns:
            格式化后的字符串。
        """
        formatted = "重排序后的文档列表：\n"
        for i, doc in enumerate(sorted_docs, 1):
            formatted += f"{i}. 相似度: {doc.get('similarity', 0):.4f}\n"
            formatted += f"   内容: {doc.get('document', '')}\n\n"
        return formatted


def create_default_reranker() -> ReorderService:
    """创建默认的重排序服务。

    Returns:
        ReorderService 实例。
    """
    return ReorderService()


reorder_service = create_default_reranker()


if __name__ == "__main__":
    import asyncio

    config = RerankerConfig.from_env()
    check_and_download_model(config)

    service = ReorderService(reranker_type="causal_lm", config=config)
    result = asyncio.run(service.reorder_documents("你好", ["你好", "你好吗", "你好吗？"]))
    print(result)
