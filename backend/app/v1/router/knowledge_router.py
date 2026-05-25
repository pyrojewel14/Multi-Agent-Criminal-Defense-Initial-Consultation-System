from typing import List

from fastapi import Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRouter

from app.core.rate_limit import rate_limit
from app.core.success_response import success_response
from app.schemas.models import (
    DocumentChunksResponse,
    KnowledgeDocumentDetail,
    KnowledgeListResponse,
    MD5ListResponse,
    MD5Record,
)
from app.security.rbac import RoleChecker, get_current_user
from app.v1.service.knowledge_service import KnowledgeService, get_knowledge_service

knowledge_router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@knowledge_router.post("/add/single")
async def add_vector_single(
    file: UploadFile = File(...),
    is_public: bool = Query(default=False, description="是否设为公共文档，公共文档可供所有用户检索"),
    current_user: dict = Depends(RoleChecker(["admin"])),
    knowledge_service: KnowledgeService = Depends(get_knowledge_service),
    _: None = Depends(rate_limit(limit=5, window=60)),
):
    """上传文件到知识库，仅超管可操作。

    Args:
        is_public: 是否设为公共文档，设为公共后可供所有用户检索。
    """
    user_id = current_user["user_id"]
    filename = await knowledge_service.handle_add_vector_single(file, user_id, is_public)
    visibility = "公共" if is_public else "私人"
    return success_response(message=f"文件 {filename} 已成功上传为{visibility}文档并存储到向量数据库")


@knowledge_router.post("/add/multiple")
async def add_vector_multiple(
    files: List[UploadFile] = File(..., description="要上传的文件列表"),
    is_public: bool = Query(default=False, description="是否设为公共文档，公共文档可供所有用户检索"),
    current_user: dict = Depends(RoleChecker(["admin"])),
    knowledge_service: KnowledgeService = Depends(get_knowledge_service),
    _: None = Depends(rate_limit(limit=3, window=60)),
):
    """上传多个文件到知识库，仅超管可操作。

    Args:
        is_public: 是否设为公共文档，设为公共后可供所有用户检索。
    """
    user_id = current_user["user_id"]
    filenames = await knowledge_service.handle_add_vector_multiple(files, user_id, is_public)
    visibility = "公共" if is_public else "私人"
    return success_response(message=f"文件 {filenames} 已成功上传为{visibility}文档并存储到向量数据库")


@knowledge_router.post("/add/multiple/stream")
async def add_vector_multiple_stream(
    files: List[UploadFile] = File(..., description="要上传的文件列表"),
    is_public: bool = Query(default=False, description="是否设为公共文档，公共文档可供所有用户检索"),
    current_user: dict = Depends(RoleChecker(["admin"])),
    knowledge_service: KnowledgeService = Depends(get_knowledge_service),
    _: None = Depends(rate_limit(limit=3, window=60)),
):
    """上传多个文件，流式返回处理进度，仅超管可操作。

    Args:
        is_public: 是否设为公共文档，设为公共后可供所有用户检索。
    """
    user_id = current_user["user_id"]
    return StreamingResponse(
        knowledge_service.handle_add_vector_multiple_stream(files, user_id, is_public),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "Access-Control-Allow-Origin": "*"},
    )


@knowledge_router.delete("/clean")
async def clean_all_vectors(
    current_user: dict = Depends(RoleChecker(["admin"])),
    knowledge_service: KnowledgeService = Depends(get_knowledge_service),
):
    """清空所有知识库文档，仅超管可操作。"""
    await knowledge_service.clean_all()
    return success_response(message="已成功清空所有知识库文档")


@knowledge_router.delete("/md5/clear")
async def clear_all_md5(
    delete_documents: bool = True,
    current_user: dict = Depends(RoleChecker(["admin"])),
    knowledge_service: KnowledgeService = Depends(get_knowledge_service),
):
    """清空所有 MD5 记录，仅超管可操作。

    Args:
        delete_documents: 是否同时删除知识库文档（默认 True）。
    """
    await knowledge_service.handle_clear_all_md5(delete_documents)
    if delete_documents:
        return success_response(message="已成功清空所有 MD5 记录和知识库文档")
    else:
        return success_response(message="已成功清空所有 MD5 记录（保留知识库文档）")


@knowledge_router.delete("/md5/delete/{md5_value}")
async def delete_single_md5(
    md5_value: str,
    delete_documents: bool = True,
    current_user: dict = Depends(RoleChecker(["admin"])),
    knowledge_service: KnowledgeService = Depends(get_knowledge_service),
):
    """删除单个 MD5 记录及其对应的知识库内容，仅超管可操作。

    Args:
        md5_value: 要删除的 MD5 值。
        delete_documents: 是否同时删除知识库文档（默认 True）。
    """
    user_id = current_user["user_id"]
    success = await knowledge_service.handle_delete_single_md5(user_id, md5_value, delete_documents)
    if success:
        if delete_documents:
            return success_response(message=f"已成功删除 MD5 记录 {md5_value} 及其对应的知识库文档")
        else:
            return success_response(message=f"已成功删除 MD5 记录 {md5_value}（保留知识库文档）")
    else:
        raise HTTPException(status_code=404, detail=f"MD5 记录 {md5_value} 不存在")


@knowledge_router.delete("/delete/filename")
async def delete_by_filename(
    filename: str,
    delete_documents: bool = True,
    current_user: dict = Depends(RoleChecker(["admin"])),
    knowledge_service: KnowledgeService = Depends(get_knowledge_service),
):
    """通过文件名删除 MD5 记录及其对应的知识库文档，仅超管可操作。

    Args:
        filename: 要删除的文件名。
        delete_documents: 是否同时删除知识库文档（默认 True）。
    """
    user_id = current_user["user_id"]
    success = await knowledge_service.handle_delete_by_filename(user_id, filename, delete_documents)
    if success:
        if delete_documents:
            return success_response(message=f"已成功删除文件 {filename} 的 MD5 记录及其对应的知识库文档")
        else:
            return success_response(message=f"已成功删除文件 {filename} 的 MD5 记录（保留知识库文档）")
    else:
        raise HTTPException(status_code=404, detail=f"文件 {filename} 不存在")


@knowledge_router.get("/md5/list", response_model=MD5ListResponse)
async def get_all_md5_records(
    current_user: dict = Depends(RoleChecker(["admin"])),
    knowledge_service: KnowledgeService = Depends(get_knowledge_service),
    _: None = Depends(rate_limit(limit=10, window=60)),
):
    """获取所有 MD5 记录，仅超管可操作。"""
    user_id = current_user["user_id"]
    records = await knowledge_service.handle_get_all_md5_records(user_id)
    return success_response(data=MD5ListResponse(records=records, total_count=len(records)))


@knowledge_router.get("/md5/{md5_value}", response_model=MD5Record)
async def get_md5_info(
    md5_value: str,
    current_user: dict = Depends(RoleChecker(["admin"])),
    knowledge_service: KnowledgeService = Depends(get_knowledge_service),
    _: None = Depends(rate_limit(limit=10, window=60)),
):
    """获取 MD5 对应的文档信息，仅超管可操作。

    Args:
        md5_value: MD5 值。
    """
    user_id = current_user["user_id"]
    md5_info = await knowledge_service.handle_get_md5_info(user_id, md5_value)
    if md5_info:
        return success_response(data=md5_info)
    else:
        raise HTTPException(status_code=404, detail=f"MD5 记录 {md5_value} 不存在")


@knowledge_router.get("/list", response_model=KnowledgeListResponse)
async def get_all_knowledge_list(
    current_user: dict = Depends(RoleChecker(["admin"])),
    knowledge_service: KnowledgeService = Depends(get_knowledge_service),
    _: None = Depends(rate_limit(limit=10, window=60)),
):
    """获取所有知识库文档列表，仅超管可操作。"""
    user_id = current_user["user_id"]
    documents = await knowledge_service.handle_get_user_knowledge(user_id)
    return success_response(data=KnowledgeListResponse(documents=documents, total_count=len(documents)))


@knowledge_router.get("/detail", response_model=KnowledgeDocumentDetail)
async def get_document_detail(
    filename: str,
    current_user: dict = Depends(RoleChecker(["admin"])),
    knowledge_service: KnowledgeService = Depends(get_knowledge_service),
    _: None = Depends(rate_limit(limit=10, window=60)),
):
    """获取文档详情内容，仅超管可操作。"""
    user_id = current_user["user_id"]
    document = await knowledge_service.handle_get_document_detail(user_id, filename)
    return success_response(data=document)


@knowledge_router.get("/chunks", response_model=DocumentChunksResponse)
async def get_document_chunks(
    filename: str,
    current_user: dict = Depends(RoleChecker(["admin"])),
    knowledge_service: KnowledgeService = Depends(get_knowledge_service),
    _: None = Depends(rate_limit(limit=10, window=60)),
):
    """获取文档切片信息，仅超管可操作。"""
    user_id = current_user["user_id"]
    chunks = await knowledge_service.handle_get_document_chunks(user_id, filename)
    return success_response(data=chunks)
