"""Application startup preflight tests."""

import pytest

import main


@pytest.mark.asyncio
async def test_lifespan_runs_law_data_preflight_before_external_services(monkeypatch):
    """Invalid tracked law data must fail before database or Redis startup."""
    events = []

    def _preflight():
        events.append("preflight")

    async def _init_db():
        events.append("database")

    async def _init_redis():
        events.append("redis")

    async def _close_service():
        return None

    monkeypatch.setattr(main, "preflight_law_knowledge", _preflight, raising=False)
    monkeypatch.setattr(main, "init_db", _init_db)
    monkeypatch.setattr(main, "init_redis", _init_redis)
    monkeypatch.setattr(main, "close_db", _close_service)
    monkeypatch.setattr(main, "close_redis", _close_service)

    async with main.lifespan(main.app):
        pass

    assert events == ["preflight", "database", "redis"]
