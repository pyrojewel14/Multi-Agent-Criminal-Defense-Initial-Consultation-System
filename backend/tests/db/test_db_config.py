"""Unit tests for ``app.db.db_config``.

Covers ``init_db``, ``close_db``, ``check_database_connection`` and the
``get_db`` async generator dependency. The actual SQLAlchemy engine is mocked
so tests run without a live MySQL/SQLite server.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.db import db_config
from app.db.db_config import (
    AsyncSessionLocal,
    close_db,
    get_db,
    init_db,
)


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------


class TestInitDb:
    @pytest.mark.asyncio
    async def test_init_db_creates_all_tables(self):
        """``init_db`` opens a connection and runs ``Base.metadata.create_all``."""
        with patch.object(db_config, "async_engine") as mock_engine:
            mock_conn = AsyncMock()
            mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_conn.__aexit__ = AsyncMock(return_value=None)
            mock_conn.run_sync = AsyncMock()
            result = MagicMock()
            result.fetchall.return_value = [(0, "workflow_session_id")]
            mock_conn.execute = AsyncMock(return_value=result)

            mock_engine.begin.return_value = mock_conn

            await init_db()

        mock_engine.begin.assert_called_once()
        mock_conn.run_sync.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_init_db_adds_workflow_identity_to_existing_sqlite_database(self, tmp_path):
        """既有 SQLite 表会增量获得工作流关联列和唯一索引。"""
        database_path = tmp_path / "legacy.db"
        engine = create_async_engine(f"sqlite+aiosqlite:///{database_path}")
        try:
            async with engine.begin() as conn:
                await conn.execute(text("CREATE TABLE consultations (id VARCHAR(36) PRIMARY KEY)"))

            with patch.object(db_config, "async_engine", engine):
                await init_db()

            async with engine.connect() as conn:
                columns = (await conn.execute(text("PRAGMA table_info(consultations)"))).fetchall()
                indexes = (await conn.execute(text("PRAGMA index_list(consultations)"))).fetchall()

            assert "workflow_session_id" in {row[1] for row in columns}
            workflow_index = next(
                row for row in indexes if row[1] == "ix_consultations_workflow_session_id"
            )
            assert workflow_index[2] == 1
        finally:
            await engine.dispose()


# ---------------------------------------------------------------------------
# get_db dependency
# ---------------------------------------------------------------------------


class TestGetDb:
    @pytest.mark.asyncio
    async def test_get_db_yields_session_and_commits(self):
        """Normal flow: yield session and commit on clean exit."""
        session = MagicMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        session.close = AsyncMock()

        # ``AsyncSessionLocal()`` returns a context manager whose ``__aenter__``
        # yields the session.
        sessionmaker = MagicMock()
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=session)
        cm.__aexit__ = AsyncMock(return_value=None)
        sessionmaker.return_value = cm

        with patch.object(db_config, "AsyncSessionLocal", sessionmaker):
            gen = get_db()
            value = await gen.__anext__()
            assert value is session
            # Simulate clean completion
            try:
                await gen.__anext__()
            except StopAsyncIteration:
                pass

        session.commit.assert_awaited_once()
        session.rollback.assert_not_awaited()
        session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_db_rolls_back_on_exception(self):
        """When the route body raises, the session is rolled back then re-raises."""
        session = MagicMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        session.close = AsyncMock()

        sessionmaker = MagicMock()
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=session)
        cm.__aexit__ = AsyncMock(return_value=None)
        sessionmaker.return_value = cm

        with patch.object(db_config, "AsyncSessionLocal", sessionmaker):
            gen = get_db()
            await gen.__anext__()

            with pytest.raises(RuntimeError, match="boom"):
                await gen.athrow(RuntimeError("boom"))

        session.rollback.assert_awaited_once()
        session.commit.assert_not_awaited()
        session.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# check_database_connection
# ---------------------------------------------------------------------------


class TestCheckDatabaseConnection:
    @pytest.mark.asyncio
    async def test_returns_true_when_query_succeeds(self):
        with patch.object(db_config, "async_engine") as mock_engine:
            mock_conn = AsyncMock()
            mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_conn.__aexit__ = AsyncMock(return_value=None)
            mock_conn.execute = AsyncMock()
            mock_engine.connect.return_value = mock_conn

            result = await db_config.check_database_connection()

        assert result is True
        mock_conn.execute.assert_awaited()

    @pytest.mark.asyncio
    async def test_returns_false_when_query_raises(self):
        with patch.object(db_config, "async_engine") as mock_engine:
            mock_conn = AsyncMock()
            mock_conn.__aenter__ = AsyncMock(side_effect=RuntimeError("connection refused"))
            mock_conn.__aexit__ = AsyncMock(return_value=None)
            mock_engine.connect.return_value = mock_conn

            result = await db_config.check_database_connection()

        assert result is False


# ---------------------------------------------------------------------------
# close_db
# ---------------------------------------------------------------------------


class TestCloseDb:
    @pytest.mark.asyncio
    async def test_close_db_disposes_engine(self):
        with patch.object(db_config, "async_engine") as mock_engine:
            mock_engine.dispose = AsyncMock()

            await close_db()

        mock_engine.dispose.assert_awaited_once()


# ---------------------------------------------------------------------------
# Module-level imports
# ---------------------------------------------------------------------------


class TestModuleImports:
    def test_async_session_local_is_sessionmaker(self):
        """``AsyncSessionLocal`` is configured as a sessionmaker for the engine."""
        assert db_config.AsyncSessionLocal is not None
        assert db_config.async_engine is not None

    def test_database_url_uses_aiosqlite(self):
        """The default URL should be a SQLite aiosqlite URL."""
        assert "aiosqlite" in db_config.ASYNC_DATABASE_URL
