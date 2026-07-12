"""Unit tests for ``app.rag.md5_manager.md5_store.MD5Store``.

Tests isolate the store by pointing ``MD5Store.base_dir`` at a per-test
``tmp_path`` so no real data is touched. The store is a thin wrapper over
asynchronous file IO, so we use ``asyncio_mode = auto`` (configured in
``pyproject.toml``) to run async tests.
"""

import json
import os

import pytest

from app.rag.md5_manager.md5_store import MD5Store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    """Build an ``MD5Store`` that writes inside ``tmp_path``."""
    s = MD5Store()
    s.base_dir = str(tmp_path)
    return s


# ---------------------------------------------------------------------------
# _get_md5_store_dir
# ---------------------------------------------------------------------------


class TestGetMd5StoreDir:
    def test_user_dir_includes_user_id(self, store):
        d = store._get_md5_store_dir("user-1")
        assert d.endswith(os.path.join("user_md5", "user-1"))

    def test_public_dir(self, store):
        d = store._get_md5_store_dir(None)
        assert d.endswith("public_md5")


# ---------------------------------------------------------------------------
# check_md5_hex / save_md5_hex
# ---------------------------------------------------------------------------


class TestCheckAndSaveMd5Hex:
    @pytest.mark.asyncio
    async def test_save_then_check_user(self, store):
        assert await store.check_md5_hex("abc123", "user-1") is False
        await store.save_md5_hex("abc123", filename="a.txt", original_filename="orig.txt", user_id="user-1")
        assert await store.check_md5_hex("abc123", "user-1") is True

    @pytest.mark.asyncio
    async def test_check_different_users_isolated(self, store):
        await store.save_md5_hex("aaa", user_id="user-1")
        assert await store.check_md5_hex("aaa", "user-2") is False

    @pytest.mark.asyncio
    async def test_save_public(self, store):
        await store.save_md5_hex("public-md5")
        assert await store.check_md5_hex("public-md5", None) is True

    @pytest.mark.asyncio
    async def test_legacy_plain_text_md5_line(self, store, tmp_path):
        # Pre-populate with a legacy "plain" MD5 line (no JSON envelope)
        md5_dir = store._get_md5_store_dir("user-x")
        os.makedirs(md5_dir, exist_ok=True)
        md5_path = os.path.join(md5_dir, "md5_hex_store.txt")
        with open(md5_path, "w", encoding="utf-8") as f:
            f.write("legacy-md5\n")
        assert await store.check_md5_hex("legacy-md5", "user-x") is True

    @pytest.mark.asyncio
    async def test_check_md5_malformed_json_line_falls_back_to_equality(self, store):
        md5_dir = store._get_md5_store_dir("user-y")
        os.makedirs(md5_dir, exist_ok=True)
        md5_path = os.path.join(md5_dir, "md5_hex_store.txt")
        with open(md5_path, "w", encoding="utf-8") as f:
            f.write("{broken json\n")
        # Should not raise; falls back to plain string comparison
        assert await store.check_md5_hex("{broken json", "user-y") is True

    def test_save_md5_hex_sync(self, store, tmp_path):
        store.save_md5_hex_sync("sync-md5", filename="f.txt", original_filename="o.txt", user_id="sync-user")
        md5_path = os.path.join(store._get_md5_store_dir("sync-user"), "md5_hex_store.txt")
        assert os.path.exists(md5_path)
        with open(md5_path, "r", encoding="utf-8") as f:
            data = json.loads(f.read().strip())
        assert data["md5"] == "sync-md5"
        assert data["filename"] == "f.txt"
        assert data["original_filename"] == "o.txt"


# ---------------------------------------------------------------------------
# _read_md5_records / _write_md5_records
# ---------------------------------------------------------------------------


class TestReadWriteRecords:
    @pytest.mark.asyncio
    async def test_read_empty(self, store):
        path, records = await store._read_md5_records("no-such-user")
        assert records == []
        assert path.endswith("md5_hex_store.txt")

    @pytest.mark.asyncio
    async def test_write_empty_records_removes_file(self, store):
        await store.save_md5_hex("m1", user_id="user-w")
        path, _ = await store._read_md5_records("user-w")
        await store._write_md5_records(path, [])
        # File should be removed (and dir cleaned up)
        assert not os.path.exists(path)

    @pytest.mark.asyncio
    async def test_write_records_round_trip(self, store):
        await store.save_md5_hex("m1", filename="a.txt", user_id="user-r")
        path, records = await store._read_md5_records("user-r")
        await store._write_md5_records(path, records + [{"md5": "m2", "filename": "b.txt"}])
        _, records2 = await store._read_md5_records("user-r")
        md5s = [r["md5"] for r in records2]
        assert "m1" in md5s
        assert "m2" in md5s


# ---------------------------------------------------------------------------
# delete_user_md5 / delete_by_filename / delete_single_md5
# ---------------------------------------------------------------------------


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_user_md5_removes_file_and_dir(self, store):
        await store.save_md5_hex("m1", user_id="user-d")
        await store.delete_user_md5("user-d")
        # File should be gone
        md5_path = os.path.join(store._get_md5_store_dir("user-d"), "md5_hex_store.txt")
        assert not os.path.exists(md5_path)

    @pytest.mark.asyncio
    async def test_delete_user_md5_missing_no_error(self, store):
        # Should be a no-op, not raise
        await store.delete_user_md5("never-existed")

    @pytest.mark.asyncio
    async def test_delete_by_filename_returns_md5(self, store):
        await store.save_md5_hex("m-file-1", filename="a.txt", user_id="user-d2")
        await store.save_md5_hex("m-file-2", filename="b.txt", user_id="user-d2")
        removed = await store.delete_by_filename("user-d2", "a.txt")
        assert removed == "m-file-1"
        # Second record still present
        records = await store.get_all_md5_records("user-d2")
        assert [r["md5"] for r in records] == ["m-file-2"]

    @pytest.mark.asyncio
    async def test_delete_by_filename_falls_back_to_original_filename(self, store):
        # Source code uses dict.get("filename", dict.get("original_filename")),
        # so original_filename is only consulted when the "filename" key is
        # *absent*. Passing only original_filename explicitly is not enough -
        # the call must also omit the filename key.
        await store.save_md5_hex("m-x", user_id="user-d3")  # filename=None, original_filename=None
        removed = await store.delete_by_filename("user-d3", "orig.txt")
        # The record's filename is None, so deletion is a no-op
        assert removed is None

    @pytest.mark.asyncio
    async def test_delete_by_filename_missing_returns_none(self, store):
        await store.save_md5_hex("m-x", filename="a.txt", user_id="user-d4")
        result = await store.delete_by_filename("user-d4", "not-present.txt")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_by_filename_no_records_returns_none(self, store):
        result = await store.delete_by_filename("ghost", "anything.txt")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_single_md5_success(self, store):
        await store.save_md5_hex("m1", user_id="user-s")
        await store.save_md5_hex("m2", user_id="user-s")
        assert await store.delete_single_md5("user-s", "m1") is True
        records = await store.get_all_md5_records("user-s")
        assert [r["md5"] for r in records] == ["m2"]

    @pytest.mark.asyncio
    async def test_delete_single_md5_not_found(self, store):
        await store.save_md5_hex("m1", user_id="user-s2")
        assert await store.delete_single_md5("user-s2", "not-here") is False

    @pytest.mark.asyncio
    async def test_delete_single_md5_no_records(self, store):
        assert await store.delete_single_md5("ghost", "anything") is False


# ---------------------------------------------------------------------------
# get_md5_info / get_all_md5_records
# ---------------------------------------------------------------------------


class TestGetInfo:
    @pytest.mark.asyncio
    async def test_get_md5_info_returns_record(self, store):
        await store.save_md5_hex("m-info", filename="a.txt", user_id="user-i")
        info = await store.get_md5_info("user-i", "m-info")
        assert info is not None
        assert info["md5"] == "m-info"
        assert info["filename"] == "a.txt"

    @pytest.mark.asyncio
    async def test_get_md5_info_missing_returns_none(self, store):
        await store.save_md5_hex("m-info", user_id="user-i2")
        assert await store.get_md5_info("user-i2", "no-such") is None

    @pytest.mark.asyncio
    async def test_get_md5_info_no_records_returns_none(self, store):
        assert await store.get_md5_info("ghost", "any") is None

    @pytest.mark.asyncio
    async def test_get_all_md5_records(self, store):
        await store.save_md5_hex("a", user_id="user-a")
        await store.save_md5_hex("b", user_id="user-a")
        records = await store.get_all_md5_records("user-a")
        assert {r["md5"] for r in records} == {"a", "b"}

    @pytest.mark.asyncio
    async def test_get_all_md5_records_empty(self, store):
        assert await store.get_all_md5_records("ghost") == []


# ---------------------------------------------------------------------------
# check_chunk_md5 / save_chunk_md5 / get_all_chunk_md5
# ---------------------------------------------------------------------------


class TestChunkMd5:
    @pytest.mark.asyncio
    async def test_check_missing_chunk(self, store):
        assert await store.check_chunk_md5("c1", "user-c") is False

    @pytest.mark.asyncio
    async def test_check_chunk_md5_prepopulated_plain_line(self, store):
        # check_chunk_md5 expects plain lines (line.strip() == chunk_md5).
        # Pre-populate the chunk file directly with a plain line.
        chunk_dir = os.path.join(store._get_md5_store_dir("user-c-pre"), "chunk_md5")
        os.makedirs(chunk_dir, exist_ok=True)
        chunk_file = os.path.join(chunk_dir, "chunks.txt")
        with open(chunk_file, "w", encoding="utf-8") as f:
            f.write("c1\n")
        assert await store.check_chunk_md5("c1", "user-c-pre") is True
        assert await store.check_chunk_md5("c2", "user-c-pre") is False

    @pytest.mark.asyncio
    async def test_save_chunk_md5_appends_to_file(self, store):
        await store.save_chunk_md5("c1", "user-c", doc_md5="d1")
        chunk_file = os.path.join(
            store._get_md5_store_dir("user-c"), "chunk_md5", "chunks.txt"
        )
        with open(chunk_file, "r", encoding="utf-8") as f:
            line = f.readline().strip()
        data = json.loads(line)
        assert data["chunk_md5"] == "c1"
        assert data["doc_md5"] == "d1"

    @pytest.mark.asyncio
    async def test_get_all_chunk_md5(self, store):
        await store.save_chunk_md5("c1", "user-c2", doc_md5="d1")
        await store.save_chunk_md5("c2", "user-c2", doc_md5="d1")
        result = await store.get_all_chunk_md5("user-c2")
        assert result == {"c1", "c2"}

    @pytest.mark.asyncio
    async def test_get_all_chunk_md5_empty(self, store):
        assert await store.get_all_chunk_md5("no-user") == set()

    @pytest.mark.asyncio
    async def test_check_chunk_md5_with_unrelated_plain_line(self, store, tmp_path):
        chunk_dir = os.path.join(store._get_md5_store_dir("user-x"), "chunk_md5")
        os.makedirs(chunk_dir, exist_ok=True)
        chunk_file = os.path.join(chunk_dir, "chunks.txt")
        with open(chunk_file, "w", encoding="utf-8") as f:
            f.write("some-other-md5\n")
        # The plain "some-other-md5" line is not the chunk we're checking
        assert await store.check_chunk_md5("not-json-line", "user-x") is False

    @pytest.mark.asyncio
    async def test_get_all_chunk_md5_malformed_lines_skipped(self, store, tmp_path):
        chunk_dir = os.path.join(store._get_md5_store_dir("user-y"), "chunk_md5")
        os.makedirs(chunk_dir, exist_ok=True)
        chunk_file = os.path.join(chunk_dir, "chunks.txt")
        with open(chunk_file, "w", encoding="utf-8") as f:
            f.write("garbage-no-json\n")
            f.write(json.dumps({"chunk_md5": "ok", "doc_md5": "d", "save_time": "t"}, ensure_ascii=False) + "\n")
        result = await store.get_all_chunk_md5("user-y")
        assert result == {"ok"}


# ---------------------------------------------------------------------------
# clear_all
# ---------------------------------------------------------------------------


class TestClearAll:
    @pytest.mark.asyncio
    async def test_clear_all_removes_user_and_public(self, store):
        await store.save_md5_hex("u1", user_id="u-clear")
        await store.save_md5_hex("p1")
        await store.clear_all()
        assert await store.get_all_md5_records("u-clear") == []
        assert await store.get_all_md5_records(None) == []

    @pytest.mark.asyncio
    async def test_clear_all_with_no_dirs(self, store):
        # Should not raise even when nothing exists
        await store.clear_all()


# ---------------------------------------------------------------------------
# save_md5_hex uses asyncio to read and write
# ---------------------------------------------------------------------------


class TestSaveUsesDatetime:
    @pytest.mark.asyncio
    async def test_save_records_upload_time(self, store):
        await store.save_md5_hex("ts", user_id="u-time")
        records = await store.get_all_md5_records("u-time")
        assert len(records) == 1
        assert records[0]["upload_time"] is not None
