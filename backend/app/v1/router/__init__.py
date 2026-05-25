from fastapi import APIRouter

from app.v1.router.consultation import router as consultation_router
from app.v1.router.lawyer import lawyer_session_router
from app.v1.router.auth import auth_router
from app.v1.router.users import user_router
from app.v1.router.lawyers import lawyer_management_router
from app.v1.router.consultation_history import consultation_router as consultation_history_router
from app.v1.router.knowledge_router import knowledge_router

api_router = APIRouter()

api_router.include_router(consultation_router)
api_router.include_router(lawyer_session_router)
api_router.include_router(auth_router)
api_router.include_router(user_router)
api_router.include_router(lawyer_management_router)
api_router.include_router(consultation_history_router)
api_router.include_router(knowledge_router)

__all__ = [
    "api_router",
    "consultation_router",
    "lawyer_session_router",
    "auth_router",
    "user_router",
    "lawyer_management_router",
    "consultation_history_router",
    "knowledge_router",
]