"""Phase 6 的真实 SQLite 落库和律师分配同步契约。"""

from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models.user import Base, Consultation, ConsultationMessage, User, UserRole
from app.v1.service import consultation_service
from tests.factories import make_consultation_state


@pytest.mark.asyncio
async def test_consultation_and_messages_are_really_persisted_in_sqlite():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    async with session_factory() as db:
        user = User(
            id="phase6-client",
            username="phase6-client",
            password_hash="test-only-hash",
            role=UserRole.CLIENT,
        )
        db.add(user)
        await db.commit()

        consultation_id = await consultation_service.create_consultation_record(
            "workflow-session-id", "phase6-client", "suspect", db
        )
        message_id = await consultation_service.save_message_to_db(
            consultation_id=consultation_id,
            session_id="workflow-session-id",
            content="我需要咨询",
            sender_type="user",
            sender_id="phase6-client",
            db=db,
        )

        consultation = await db.scalar(
            select(Consultation).where(Consultation.id == consultation_id)
        )
        message = await db.scalar(
            select(ConsultationMessage).where(ConsultationMessage.id == message_id)
        )

    await engine.dispose()

    assert consultation is not None
    assert consultation.client_id == "phase6-client"
    assert consultation.user_type == "suspect"
    assert message is not None
    assert message.consultation_id == consultation_id
    assert message.content == "我需要咨询"


@pytest.mark.asyncio
async def test_assignment_sync_updates_checkpoint_memory_and_cache():
    state = make_consultation_state(
        session_id="workflow-session",
        consultation_id="consultation-1",
        lawyer_id=None,
    )
    updated = dict(state, lawyer_id="lawyer-1")
    with patch.object(
        consultation_service.orchestrator,
        "get_active_sessions",
        return_value={"workflow-session": state},
    ), patch.object(
        consultation_service.orchestrator,
        "update_workflow_state",
        new_callable=AsyncMock,
        return_value=updated,
    ) as update_state, patch.object(
        consultation_service,
        "persist_state",
        new_callable=AsyncMock,
    ) as persist_state:
        session_id = await consultation_service.assign_lawyer_to_active_session(
            "consultation-1", "lawyer-1"
        )

    assert session_id == "workflow-session"
    update_state.assert_awaited_once_with(
        "workflow-session", {"lawyer_id": "lawyer-1"}
    )
    persist_state.assert_awaited_once_with("workflow-session", updated)
