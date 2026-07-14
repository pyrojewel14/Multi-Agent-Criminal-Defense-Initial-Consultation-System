#!/usr/bin/env python3
"""
测试脚本：验证刑法条文检索功能
使用客户用户查询"防卫过当致人死亡"相关内容
"""

import asyncio
import sys
sys.path.insert(0, '/Users/clearguo/Desktop/Multi-Agent Criminal Defense Initial Consultation System/backend')

from app.rag.rag_service import RagService
from app.utils.logger import get_logger

_logger = get_logger("TestRetrieval")


async def test_retrieval():
    """测试检索功能"""
    
    print("=" * 60)
    print("测试：检索刑法中关于'防卫过当致人死亡'的条文")
    print("=" * 60)
    
    # 使用客户用户 ID（之前创建的 client1）
    client_user_id = "e40b8861-6577-4d13-9e78-453b720eaadb"
    
    print(f"\n用户信息:")
    print(f"  - 用户ID: {client_user_id}")
    print(f"  - 用户类型: client (客户)")
    print(f"  - 是否包含公共文档: True")
    
    # 创建 RAG 服务（客户可以检索公共文档）
    rag_service = RagService(user_id=client_user_id, include_public=True)
    
    # 测试查询
    query = "防卫过当致人死亡"
    
    print(f"\n查询内容: {query}")
    print(f"预期结果: 应能检索到刑法第20条关于正当防卫和防卫过当的内容")
    
    try:
        # 初始化检索器
        print("\n正在初始化检索器...")
        await rag_service.initialize_retriever(query)
        assert rag_service.retriever is not None
        
        # 执行检索
        print("正在执行检索...")
        results = rag_service.retriever.invoke(query, top_k=5)
        
        print("\n" + "=" * 60)
        print(f"检索结果: 共找到 {len(results)} 条相关内容")
        print("=" * 60)
        
        if results:
            for idx, result in enumerate(results, 1):
                print(f"\n--- 结果 {idx} ---")
                print(f"内容: {result.page_content[:200]}...")
                print(f"来源: {result.metadata.get('source', 'unknown')}")
                if 'article_number' in result.metadata:
                    print(f"条文: {result.metadata['article_number']}")
        else:
            print("\n❌ 未能检索到任何内容！")
            print("可能原因:")
            print("  1. 刑法_cleaned.json 未正确上传")
            print("  2. 向量数据库中缺少相关向量")
            print("  3. 检索参数设置不当")
            
    except Exception as e:
        print(f"\n❌ 检索过程出错: {e}")
        import traceback
        traceback.print_exc()


async def test_vector_store_count():
    """测试向量数据库中的文档数量"""
    print("\n" + "=" * 60)
    print("测试：检查向量数据库中的文档数量")
    print("=" * 60)
    
    from app.rag.vector_store import get_vector_store
    
    vector_store = get_vector_store()
    
    # 使用 get 方法获取所有文档
    try:
        all_docs = vector_store.vectors_store.get(include=["documents", "metadatas"])
        
        print(f"\n向量数据库中总文档数: {len(all_docs.get('documents', []))}")
        
        documents = all_docs.get('documents', [])
        metadatas = all_docs.get('metadatas', [])
        
        # 检查是否有公共文档
        public_docs = [doc for doc, meta in zip(documents, metadatas) 
                       if meta.get('is_public', False)]
        
        print(f"公共文档数: {len(public_docs)}")
        
        # 检查第二十条相关文档
        article_20_docs = [doc for doc in documents 
                           if '第二十条' in doc or '防卫' in doc]
        
        print(f"包含'防卫'关键词的文档数: {len(article_20_docs)}")
        
        if article_20_docs:
            print("\n第二十条相关内容预览:")
            for doc in article_20_docs[:3]:
                print(f"\n---")
                print(f"{doc[:150]}...")
    except Exception as e:
        print(f"\n❌ 获取文档出错: {e}")
        import traceback
        traceback.print_exc()


async def test_client_can_access_public_docs():
    """测试客户用户是否能访问公共文档"""
    print("\n" + "=" * 60)
    print("测试：客户用户访问公共文档")
    print("=" * 60)
    
    from app.rag.vector_store import get_vector_store
    
    # 客户用户 ID
    client_user_id = "e40b8861-6577-4d13-9e78-453b720eaadb"
    
    # 获取向量存储
    vector_store_service = get_vector_store()
    vector_store = vector_store_service.vectors_store
    
    query = "正当防卫"
    
    print(f"\n客户用户 ({client_user_id}) 检索: '{query}'")
    
    try:
        # 直接使用向量检索（默认包含公共文档）
        # 使用相似度检索
        results = vector_store.similarity_search_with_score(
            query=query,
            k=5,
            filter=None  # 不限制，过滤在检索层面处理
        )
        
        print(f"\n检索结果: {len(results)} 条")
        
        if results:
            print("\n检索到的内容:")
            for idx, (doc, score) in enumerate(results, 1):
                print(f"\n{idx}. 相似度: {score:.4f}")
                print(f"   内容: {doc.page_content[:150]}...")
                print(f"   metadata: {doc.metadata}")
        else:
            print("\n❌ 客户用户无法检索到公共文档！")
            
    except Exception as e:
        print(f"\n❌ 出错: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """主函数"""
    print("\n" + "=" * 60)
    print(" 刑法条文检索功能测试")
    print("=" * 60)
    
    # 测试1: 检查向量数据库
    await test_vector_store_count()
    
    # 测试2: 测试客户用户检索
    await test_client_can_access_public_docs()
    
    # 测试3: 测试具体查询
    await test_retrieval()
    
    print("\n" + "=" * 60)
    print(" 测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
