from fastapi import APIRouter

from app.api.v1.routers.consultation.routes import router as http_router
from app.api.v1.routers.consultation.websocket import ws_router

router = APIRouter()
router.include_router(http_router)
router.include_router(ws_router)

__all__ = ["router"]
