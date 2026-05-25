import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.orchestrator.workflow import orchestrator
from app.utils.logger import get_logger
from app.v1.router.consultation.constants import HIGH_RISK_ALERT_MESSAGE
from app.v1.service import consultation_service

_logger = get_logger("Router.Consultation.WebSocket")

ws_router = APIRouter()


class ConnectionManager:
    """WebSocket 连接管理器"""

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        """建立 WebSocket 连接"""
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        self.active_connections[session_id].append(websocket)
        _logger.info("【connect】WebSocket连接建立: session_id=%s", session_id)

    def disconnect(self, websocket: WebSocket, session_id: str) -> None:
        """断开 WebSocket 连接"""
        if session_id in self.active_connections:
            if websocket in self.active_connections[session_id]:
                self.active_connections[session_id].remove(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
        _logger.info("【disconnect】WebSocket连接断开: session_id=%s", session_id)

    async def send_to_client(self, websocket: WebSocket, message: dict) -> None:
        """向客户端发送消息"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            _logger.error("【send_to_client】发送消息失败: %s", e)

    async def broadcast(self, session_id: str, message: dict) -> None:
        """广播消息到指定会话的所有连接"""
        if session_id in self.active_connections:
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    _logger.error("【broadcast】广播消息失败: %s", e)


manager = ConnectionManager()


@ws_router.websocket("/api/v1/sessions/{session_id}/ws")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket 通信端点

    建立 WebSocket 连接，支持双向实时消息通信和心跳机制。

    Args:
        websocket: WebSocket 连接对象
        session_id: 会话ID
    """
    await manager.connect(websocket, session_id)

    heartbeat_task = None
    state = None

    try:
        state = await consultation_service.get_session_state(session_id)

        if not state:
            await websocket.send_json({
                "type": "error",
                "content": "会话不存在或已过期",
            })
            await websocket.close()
            return

        heartbeat_task = asyncio.create_task(_heartbeat_loop(websocket, session_id))

        await websocket.send_json({
            "type": "ack",
            "content": "连接已建立",
            "session_id": session_id,
            "current_agent": state.get("current_agent", "Receptionist"),
        })

        while True:
            data = await websocket.receive_text()

            try:
                message_data = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "content": "无效的JSON格式",
                })
                continue

            message_type = message_data.get("type", "message")

            if message_type == "heartbeat":
                await websocket.send_json({
                    "type": "heartbeat_ack",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                continue

            if message_type == "message":
                content = message_data.get("content", "")

                # 每轮重新获取最新状态，避免使用过期的局部变量
                state = await consultation_service.get_session_state(session_id)
                if not state:
                    await websocket.send_json({
                        "type": "error",
                        "content": "会话不存在或已过期",
                    })
                    break

                if not state.get("consent_given") and "同意" not in content and "确认" not in content:
                    await websocket.send_json({
                        "type": "message",
                        "agent_name": "Receptionist",
                        "content": "请先回复'同意'确认您已阅读并理解权利义务告知。",
                        "session_id": session_id,
                    })
                    continue

                current_agent = state.get("current_agent", "Receptionist")

                # 通过 resume_workflow 恢复，LangGraph 条件边自动路由
                result = await consultation_service.process_message(session_id, content, state, current_agent)

                if result.error:
                    _logger.error("【websocket_endpoint】消息处理异常: %s", result.error)
                    await websocket.send_json({
                        "type": "error",
                        "content": "消息处理失败，请稍后重试",
                    })
                    continue

                if result.alert_triggered:
                    await websocket.send_json({
                        "type": "alert",
                        "content": HIGH_RISK_ALERT_MESSAGE,
                        "agent_name": current_agent,
                        "session_id": session_id,
                    })
                else:
                    await websocket.send_json({
                        "type": "message",
                        "agent_name": current_agent,
                        "content": result.response_content,
                        "session_id": session_id,
                    })

                    await websocket.send_json({
                        "type": "ack",
                        "content": "消息已处理",
                        "agent_name": current_agent,
                    })

    except WebSocketDisconnect:
        _logger.info("【websocket_endpoint】客户端断开连接: session_id=%s", session_id)
    except Exception as e:
        _logger.error("【websocket_endpoint】WebSocket异常: session_id=%s, error=%s", session_id, str(e))
    finally:
        if heartbeat_task:
            heartbeat_task.cancel()

        manager.disconnect(websocket, session_id)


async def _heartbeat_loop(websocket: WebSocket, _session_id: str) -> None:
    """WebSocket 心跳循环

    定期发送心跳包以保持连接活跃。

    Args:
        websocket: WebSocket 连接对象
        session_id: 会话ID
    """
    while True:
        try:
            await asyncio.sleep(30)
            await websocket.send_json({
                "type": "heartbeat",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        except Exception:
            break
