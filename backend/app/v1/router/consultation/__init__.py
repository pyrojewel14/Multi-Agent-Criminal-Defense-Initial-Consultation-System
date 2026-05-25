from fastapi import APIRouter

from app.v1.router.consultation.routes import router as http_router
from app.v1.router.consultation.websocket import ws_router

router = APIRouter()
router.include_router(http_router)
router.include_router(ws_router)

__all__ = ["router"]
