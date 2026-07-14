import asyncio
import hashlib

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langsmith import traceable

from app.rag.reorder_service import reorder_service
from app.rag.vector_store import get_vector_store
from app.utils.factory import chat_model
from app.utils.logger import get_logger
from app.utils.prompt_loader import prompt_loader

_logger = get_logger("RagService")


def _configure_hyde_model(model):
    """限制本地 Ollama 的 HyDE 输出规模。

    Qwen thinking 模式可能生成远超检索所需的长文本，导致请求长期占用模型。
    仅为 Ollama 的 HyDE 调用关闭 reasoning 并限制生成长度，其他模型保持原行为。
    """
    if model.__class__.__module__.startswith("langchain_ollama"):
        return model.model_copy(update={"reasoning": False, "num_predict": 64})
    return model


def _deduplicate_documents(documents: list) -> list:
    seen_hashes = set()
    unique_docs = []
    for doc in documents:
        content_hash = hashlib.md5(doc.page_content[:200].encode()).hexdigest()
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_docs.append(doc)
    if len(unique_docs) < len(documents):
        _logger.info("文档去重: %d -> %d", len(documents), len(unique_docs))
    return unique_docs


class RagService:
    def __init__(self, user_id: str | None = None, thinking_callback=None, include_public: bool = True):
        self.vector_store = get_vector_store()
        self.retriever = None
        self.user_id = user_id
        self.include_public = include_public
        self.prompt_text = prompt_loader.load("rag_summary_prompt")
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.chat_model = chat_model
        self.hyde_model = _configure_hyde_model(self.chat_model)
        self.chain = self._init_chain()
        self.hyde_prompt_template = PromptTemplate.from_template(
            "基于以下问题生成一段不超过 80 个汉字的法律检索假设文本，包含可能的行为、罪名和法条术语；"
            "不要输出分析过程：\n\n问题：{query}\n\n假设文本："
        )
        self.thinking_callback = thinking_callback

    async def initialize_retriever(self, query: str | None = None):
        """初始化检索器。

        Args:
            query: 查询语句，用于动态调整权重。
        """
        if self.retriever is None:
            weights = self.vector_store.get_dynamic_weights(query)

            if self.thinking_callback:
                await self.thinking_callback(
                    {
                        "type": "thinking",
                        "stage": "retrieval",
                        "content": f"初始化检索器（向量权重: {weights[0]:.1f}, BM25权重: {weights[1]:.1f}）",
                        "details": {"vector_weight": weights[0], "bm25_weight": weights[1]},
                    }
                )

            self.retriever = await self.vector_store.get_retriever(query, self.user_id, self.include_public)

    def _init_chain(self):
        """初始化链。"""
        chain = self.prompt_template | self.chat_model | StrOutputParser()
        return chain

    @traceable
    async def generate_hypothetical_document(self, query: str) -> str:
        """使用 HyDE 技术生成假设性文档。

        Args:
            query: 用户查询。

        Returns:
            假设性文档内容。
        """
        try:
            hyde_chain = self.hyde_prompt_template | self.hyde_model | StrOutputParser()
            hypothetical_doc = await hyde_chain.ainvoke({"query": query})
            _logger.info("HyDE 生成假设性文档: %s", hypothetical_doc[:100])
            return hypothetical_doc
        except Exception as e:
            _logger.error("HyDE 生成假设性文档失败: %s", e)
            return query

    @traceable
    async def retrieve_document(self, query: str) -> list:
        """使用 HyDE 技术从向量数据库里检索文档。

        Args:
            query: 查询语句。

        Returns:
            检索到的文档列表。
        """
        if not self.user_id:
            _logger.warning("HyDE user_id 为空，不进行任何检索")
            return []

        try:
            if self.retriever is None:
                await self.initialize_retriever(query)

            _logger.info("HyDE 开始处理查询: %s", query[:50])

            if self.thinking_callback:
                await self.thinking_callback(
                    {"type": "thinking", "stage": "hyde", "content": f"正在基于查询生成假设性文档..."}
                )

            hypothetical_doc = await self.generate_hypothetical_document(query)

            if self.thinking_callback:
                await self.thinking_callback(
                    {
                        "type": "thinking",
                        "stage": "hyde",
                        "content": f"假设性文档生成完成",
                        "details": {
                            "hypothetical_doc_preview": hypothetical_doc[:200] + "..."
                            if len(hypothetical_doc) > 200
                            else hypothetical_doc
                        },
                    }
                )

            _logger.info("HyDE 使用假设性文档进行检索")

            if self.thinking_callback:
                await self.thinking_callback(
                    {"type": "thinking", "stage": "retrieval", "content": "正在向量数据库中检索相关文档..."}
                )

            hyde_retriever = await self.vector_store.get_retriever(hypothetical_doc, self.user_id, self.include_public)
            documents = await hyde_retriever.ainvoke(hypothetical_doc)
            documents = _deduplicate_documents(documents)
            _logger.info("HyDE 检索到 %d 个相关文档", len(documents))
            for i, doc in enumerate(documents, 1):
                source = doc.metadata.get("original_filename", doc.metadata.get("source", "?"))
                preview = doc.page_content[:80].replace("\n", " ")
                _logger.info("  [%d] %s | %s...", i, source, preview)

            if self.thinking_callback:
                doc_previews = []
                for i, doc in enumerate(documents, 1):
                    preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                    doc_previews.append(
                        {
                            "index": i,
                            "preview": preview,
                            "source": doc.metadata.get("original_filename", doc.metadata.get("source", "unknown")),
                        }
                    )
                await self.thinking_callback(
                    {
                        "type": "thinking",
                        "stage": "retrieval",
                        "content": f"检索到 {len(documents)} 个相关文档",
                        "details": {"documents": doc_previews},
                    }
                )

            return documents
        except Exception as e:
            _logger.error("HyDE 检索文档失败: %s", e)
            return []

    @traceable
    async def reorder_documents(self, query: str, documents: list) -> list:
        """对文档进行重排序。

        Args:
            query: 查询语句。
            documents: 文档列表。

        Returns:
            重排序后的文档列表。
        """
        if self.thinking_callback:
            await self.thinking_callback(
                {"type": "thinking", "stage": "reorder", "content": f"正在对 {len(documents)} 个文档进行重排序..."}
            )

        result = await reorder_service.reorder_documents(query, documents, thinking_callback=self.thinking_callback)
        if result["success"]:
            reordered_documents = [doc.get("document", "") for doc in result["documents"]]
            _logger.info("文档重排序成功，返回 %d 个文档", len(reordered_documents))

            if self.thinking_callback:
                score_details = []
                for i, doc in enumerate(result["documents"], 1):
                    score_details.append(
                        {
                            "rank": i,
                            "score": round(doc.get("similarity", 0), 4),
                            "preview": doc.get("document", "")[:100] + "..."
                            if len(doc.get("document", "")) > 100
                            else doc.get("document", ""),
                        }
                    )
                await self.thinking_callback(
                    {
                        "type": "thinking",
                        "stage": "reorder",
                        "content": f"重排序完成，返回 {len(reordered_documents)} 个文档",
                        "details": {"scores": score_details},
                    }
                )

            return reordered_documents
        else:
            _logger.warning("重排序失败: %s", result["error"])
            return documents

    @traceable
    async def get_documents_and_summary(self, query: str) -> dict:
        """获取文档列表和摘要。

        Args:
            query: 查询语句。

        Returns:
            包含文档列表和摘要的字典。
        """
        if not self.user_id:
            _logger.warning("user_id 为空，不返回任何文档")
            return {"documents": [], "summary": "抱歉，我没有找到相关的信息。"}

        try:
            documents = await self.retrieve_document(query)

            document_contents = [doc.page_content for doc in documents]

            reordered_documents = await self.reorder_documents(query, document_contents)

            if not reordered_documents:
                return {"documents": [], "summary": "抱歉，我没有找到相关的信息。"}

            try:
                individual_summaries = []
                max_documents = 3

                if self.thinking_callback:
                    await self.thinking_callback(
                        {
                            "type": "thinking",
                            "stage": "summarize",
                            "content": f"正在对前 {min(max_documents, len(reordered_documents))} 个最相关文档进行总结...",
                        }
                    )

                async def summarize_document(i, doc):
                    _logger.debug("正在总结第 %d 个文档", i)
                    if self.thinking_callback:
                        await self.thinking_callback(
                            {"type": "thinking", "stage": "summarize", "content": f"正在总结第 {i} 个文档..."}
                        )
                    single_context = f"【参考资料{i}】:{doc}\n"
                    import time

                    start_time = time.time()
                    single_summary = await asyncio.wait_for(
                        self.chain.ainvoke({"input": query, "context": single_context}), timeout=30.0
                    )
                    end_time = time.time()
                    _logger.debug("第 %d 个文档总结耗时: %.2f 秒", i, end_time - start_time)
                    return single_summary

                tasks = []
                for i, doc in enumerate(reordered_documents[:max_documents], 1):
                    tasks.append(summarize_document(i, doc))

                import time

                start_time = time.time()
                individual_summaries = await asyncio.gather(*tasks)
                end_time = time.time()
                _logger.info("所有文档总结完成，总耗时: %.2f 秒", end_time - start_time)

                if len(individual_summaries) == 1:
                    _logger.info("生成摘要成功（单文档）")
                    return {"documents": reordered_documents, "summary": individual_summaries[0]}

                combined_context = "以下是多个文档的摘要，请综合这些信息生成最终的回答：\n\n"
                for i, summary in enumerate(individual_summaries, 1):
                    combined_context += f"【文档{i}摘要】:{summary}\n\n"

                _logger.debug("合并摘要完成，开始生成最终总结")

                if self.thinking_callback:
                    await self.thinking_callback(
                        {"type": "thinking", "stage": "summarize", "content": "正在综合多个文档生成最终回答..."}
                    )

                final_summary = await asyncio.wait_for(
                    self.chain.ainvoke({"input": query, "context": combined_context}), timeout=30.0
                )

                _logger.info("生成摘要成功（多文档合并）")
                return {"documents": reordered_documents, "summary": final_summary}
            except asyncio.TimeoutError:
                _logger.error("生成摘要超时")
                return {"documents": reordered_documents, "summary": "抱歉，生成摘要超时，请稍后再试。"}
        except Exception as e:
            _logger.error("生成摘要失败: %s", e)
            return {"documents": [], "summary": "抱歉，处理您的请求时出现了错误。"}

    @traceable
    async def rag_summary(self, query: str) -> str:
        """RAG 摘要生成。

        Args:
            query: 查询语句。

        Returns:
            摘要文本。
        """
        result = await self.get_documents_and_summary(query)
        return result.get("summary", "抱歉，处理您的请求时出现了错误。")


if __name__ == "__main__":
    import asyncio

    async def main():
        service = RagService(user_id="41109725c0f540fea7ecdf047f5597b6")
        await service.initialize_retriever()
        result = await service.rag_summary("防卫过当致人死亡适用于哪条法律")
        print(result)

    asyncio.run(main())
