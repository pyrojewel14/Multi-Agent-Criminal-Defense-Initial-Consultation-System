"""摘要只消费统一检索结果，不构建召回、HyDE 或重排能力。"""

import asyncio

from app.knowledge.retrieval.models import SearchResult


class DocumentSummaryService:
    """注入接受 input/context 的摘要链，保留现有摘要响应结构。"""

    def __init__(self, model, *, timeout_s: float = 30.0, thinking_callback=None):
        self.model = model
        self.timeout_s = timeout_s
        self.thinking_callback = thinking_callback

    async def get_documents_and_summary(self, query: str, result: SearchResult) -> dict:
        """使用已有检索结果生成摘要，最多并发总结前三条，再按需合并。

        每次模型调用独立使用 timeout_s；超时保留候选，其他异常返回空文档列表。
        """
        documents = list(result.candidates)
        if not documents:
            return {"documents": [], "summary": "抱歉，我没有找到相关的信息。"}
        try:

            async def summarize(index, doc):
                """发送可选进度通知，并在单次超时预算内总结一条候选。"""
                if self.thinking_callback:
                    await self.thinking_callback(
                        {"type": "thinking", "stage": "summarize", "content": f"正在总结第 {index} 个文档..."}
                    )
                return await asyncio.wait_for(
                    self.model.ainvoke({"input": query, "context": f"【参考资料{index}】:{doc.content}\n"}),
                    timeout=self.timeout_s,
                )

            summaries = await asyncio.gather(*(summarize(i, doc) for i, doc in enumerate(documents[:3], 1)))
            if len(summaries) == 1:
                summary = summaries[0]
            else:
                # 只综合已生成的摘要，不再次召回或重排文档。
                context = "以下是多个文档的摘要，请综合这些信息生成最终的回答：\n\n"
                context += "".join(f"【文档{i}摘要】:{summary}\n\n" for i, summary in enumerate(summaries, 1))
                summary = await asyncio.wait_for(
                    self.model.ainvoke({"input": query, "context": context}), timeout=self.timeout_s
                )
            return {"documents": documents, "summary": summary}
        except asyncio.TimeoutError:
            return {"documents": documents, "summary": "抱歉，生成摘要超时，请稍后再试。"}
        except Exception:
            return {"documents": [], "summary": "抱歉，处理您的请求时出现了错误。"}

    async def rag_summary(self, query: str, result: SearchResult) -> str:
        """返回已选检索结果的摘要文本。"""
        return (await self.get_documents_and_summary(query, result))["summary"]
