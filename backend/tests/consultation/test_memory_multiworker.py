"""两个真实 HTTP 进程共享全新业务库与 checkpoint 库的受控并发验收。"""
import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

unsafe_probe = pytest.mark.skipif(os.getenv("MEMORY_RUN_UNSAFE_MULTIWORKER_PROBE") != "1",
    reason="绕过正式 lifespan 的已知不安全多 worker 探针；仅显式启用以重现历史缺陷")


class Workers:
    def __init__(self, root, stores=None):
        self.root = root
        self.stores = stores or root
        self.root.mkdir(parents=True, exist_ok=True)
        self.stores.mkdir(parents=True, exist_ok=True)
        self.processes = []
        self.clients = []
        self.logs = []
        self.requests = []
        self.identities = []

    def start(self, count=2, formal=False):
        for index in range(count):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            log = (self.root / f"worker-{index}.log").open("w")
            self.logs.append(log)
            env = dict(os.environ, MEMORY_RECENT_MESSAGES="2", PYTHONNOUSERSITE="1",
                       DATABASE_PATH=str(self.stores / "audit.db"),
                       MEMORY_PROBE_FORMAL_LIFESPAN="1" if formal else "0",
                       PYTHONPATH=str(Path(__file__).resolve().parents[2]))
            process = subprocess.Popen([sys.executable, str(Path(__file__).with_name("memory_multiworker_probe.py")),
                str(self.stores / "audit.db"), str(self.stores / "checkpoint.db"), str(self.root), str(index), str(port)],
                env=env, stdout=log, stderr=subprocess.STDOUT)
            self.processes.append(process)
            client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=20, trust_env=False)
            self.clients.append(client)
            deadline = time.monotonic() + 50
            while True:
                assert process.poll() is None, (self.root / f"worker-{index}.log").read_text()
                try:
                    response = client.get("/probe/ready")
                    if response.status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                assert time.monotonic() < deadline, "worker startup timeout"
                time.sleep(0.05)
        self.identities = [client.get("/probe/ready").json() for client in self.clients]
        assert len({identity["pid"] for identity in self.identities}) == count
        self.evidence("identities", self.identities)
        return self

    def close(self):
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
        for process in self.processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for client in self.clients:
            client.close()
        for log in self.logs:
            log.close()

    def create(self, worker=0):
        client = self.clients[worker]
        result = client.post("/api/v1/sessions", json={"user_type": "suspect"})
        assert result.status_code == 200, result.text
        sid = result.json()["session_id"]
        consent = client.post(f"/api/v1/sessions/{sid}/confirm-consent",
                              json={"session_id": sid, "consent_given": True, "idempotency_key": "consent",
                                    "consent_timestamp": "2026-10-13T00:00:00Z", "consent_version": "synthetic-v1"})
        assert consent.status_code == 200, consent.text
        return sid

    def message(self, worker, sid, key, content=None):
        entry = {"worker": worker, "session_id": sid, "key": key, "started": time.monotonic()}
        self.requests.append(entry)
        response = self.clients[worker].post(f"/api/v1/sessions/{sid}/message",
            json={"session_id": sid, "content": content or f"合成陈述-{key}", "idempotency_key": key})
        entry.update(finished=time.monotonic(), status=response.status_code, body=response.json())
        return response

    def snapshot(self, sid, worker=1):
        result = self.clients[worker].get(f"/probe/snapshot/{sid}")
        assert result.status_code == 200, result.text
        return result.json()

    def arm(self, worker, stage, name):
        result = self.clients[worker].post("/probe/arm", json={stage: name})
        assert result.status_code == 200

    def entered(self, name, worker):
        deadline = time.monotonic() + 15
        while not (self.root / f"{name}.{worker}.entered").exists():
            assert time.monotonic() < deadline, f"gate not reached: {name}/{worker}"
            time.sleep(0.01)

    def release(self, name):
        (self.root / f"{name}.release").touch()

    def evidence(self, name, payload):
        (self.root / f"{name}.json").write_text(json.dumps({"workers": self.identities,
            "requests": self.requests, "result": payload}, ensure_ascii=False, indent=2))


@pytest.fixture
def workers(tmp_path):
    cluster = Workers(tmp_path)
    try:
        yield cluster.start()
    finally:
        cluster.close()


def assert_memory(snapshot, inputs):
    assert snapshot["pending"] == ["fact_intake"]
    state = snapshot["state"]
    assert state.get("message_audit") is None
    raw = snapshot["raw"]
    assert len({row["id"] for row in raw}) == len(raw)
    assert [row["sequence"] for row in raw] == list(range(1, len(raw) + 1))
    fields = state["memory"]["case"]["fields"]
    actions = [item["value"]["action"] for item in fields["behavior_sequence"]["items"]]
    assert set(actions) == set(inputs)
    users = [row for row in raw if row["sender_type"] == "user" and row["content"] in inputs]
    assert {row["content"] for row in users} == set(inputs)
    assert len(users) == len(inputs)
    assert all(row["record_kind"] == "external" for row in users)
    recent = state["memory"]["recent"]
    assert len(recent) == 2
    assert [row["id"] for row in recent] == [row["id"] for row in raw[-2:]]
    summary = state["memory"]["summary"]
    assert summary["through_sequence"] == raw[-3]["sequence"]
    assert summary["version"] >= 1
    assert summary["status"] == "ready"
    assert summary["text"]
    assert not any(row["record_kind"] == "internal" for row in raw)


@unsafe_probe
def test_different_sessions_concurrent_and_worker_exit_takeover(workers):
    first, second = workers.create(), workers.create(1)
    workers.arm(0, "extract", "different")
    with ThreadPoolExecutor(2) as pool:
        held = pool.submit(workers.message, 0, first, "first")
        workers.entered("different", 0)
        other = workers.message(1, second, "second")
        assert other.status_code == 200, other.text
        workers.release("different")
        assert held.result().status_code == 200
    events_before_exit = [client.get("/probe/events").json() for client in workers.clients]
    workers.processes[0].terminate()
    workers.processes[0].wait(timeout=10)
    replay = workers.message(1, first, "first")
    continued = workers.message(1, first, "continued")
    assert replay.status_code == continued.status_code == 200
    for sid, texts in [(first, ["合成陈述-first", "合成陈述-continued"]), (second, ["合成陈述-second"])]:
        snapshot = workers.snapshot(sid)
        workers.evidence(sid, snapshot)
        assert_memory(snapshot, texts)
    assert not any("database is locked" in path.read_text() for path in workers.root.glob("worker-*.log"))
    events = events_before_exit + [workers.clients[1].get("/probe/events").json()]
    workers.evidence("storage-events", events)
    assert not any(event["event"] == "storage_error" for batch in events for event in batch)


@pytest.mark.parametrize("same_key", [False, True])
@unsafe_probe
def test_same_session_concurrent_commands_are_safe(workers, same_key):
    sid = workers.create()
    for worker in range(2):
        workers.arm(worker, "raw_flush", "same")
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(workers.message, 0, sid, "one")
        workers.entered("same", 0)
        second = pool.submit(workers.message, 1, sid, "one" if same_key else "two")
        workers.entered("same", 1)
        workers.release("same")
        results = [first.result(), second.result()]
    snapshot = workers.snapshot(sid)
    workers.evidence("same-key" if same_key else "same-session", {
        "responses": [{"status": response.status_code, "body": response.json()} for response in results],
        "snapshot": snapshot,
        "events": [client.get("/probe/events").json() for client in workers.clients]})
    assert [response.status_code for response in results] == [200, 200]
    if same_key:
        assert results[0].json()["message_id"] == results[1].json()["message_id"]
    assert_memory(snapshot, ["合成陈述-one"] if same_key else ["合成陈述-one", "合成陈述-two"])


@unsafe_probe
def test_summary_competition_preserves_newer_case_and_cursor(workers):
    sid = workers.create()
    assert workers.message(0, sid, "seed").status_code == 200
    workers.arm(0, "summary", "summary")
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(workers.message, 0, sid, "older")
        workers.entered("summary", 0)
        replay = workers.message(1, sid, "older")
        assert replay.status_code == 200, replay.text
        second = workers.message(1, sid, "newer")
        assert second.status_code == 200, second.text
        before = workers.snapshot(sid)
        workers.release("summary")
        assert first.result().status_code == 200
    after = workers.snapshot(sid)
    workers.evidence("summary-race", {"before": before, "after": after,
        "events": [client.get("/probe/events").json() for client in workers.clients]})
    assert after["state"]["memory"]["summary"]["through_sequence"] >= before["state"]["memory"]["summary"]["through_sequence"]
    assert after["state"]["memory"]["summary"]["version"] >= before["state"]["memory"]["summary"]["version"]
    assert_memory(after, ["合成陈述-seed", "合成陈述-older", "合成陈述-newer"])


def test_formal_http_lifespan_rejects_second_worker_then_recovers_after_kill(tmp_path):
    stores = tmp_path / "stores"
    owner = Workers(tmp_path / "owner", stores)
    contender = Workers(tmp_path / "contender", stores)
    replacement = Workers(tmp_path / "replacement", stores)
    try:
        owner.start(count=1, formal=True)
        assert owner.identities[0]["formal_main_lifespan"] is True
        sid = owner.create()
        original = owner.message(0, sid, "before-kill")
        assert original.status_code == 200, original.text
        before = owner.snapshot(sid, worker=0)
        with pytest.raises(AssertionError, match="single worker"):
            contender.start(count=1, formal=True)
        assert contender.processes[0].poll() is not None
        assert owner.clients[0].get("/probe/ready").status_code == 200
        owner.processes[0].kill()
        owner.processes[0].wait(timeout=10)
        replacement.start(count=1, formal=True)
        assert replacement.identities[0]["pid"] != owner.identities[0]["pid"]
        replay = replacement.message(0, sid, "before-kill")
        assert replay.status_code == 200, replay.text
        assert replay.json()["message_id"] == original.json()["message_id"]
        assert replacement.clients[0].get("/probe/events").json() == []
        assert replacement.snapshot(sid, worker=0) == before
        continued = replacement.message(0, sid, "after-kill")
        assert continued.status_code == 200, continued.text
        after = replacement.snapshot(sid, worker=0)
        assert_memory(after, ["合成陈述-before-kill", "合成陈述-after-kill"])
        events = replacement.clients[0].get("/probe/events").json()
        assert not any(event["event"] == "storage_error" for event in events)
        replacement.evidence("formal-recovery", {"owner": owner.identities, "before": before, "after": after,
                                                  "events": events})
        assert not any("database is locked" in path.read_text() for path in tmp_path.glob("*/*.log"))
    finally:
        owner.close()
        contender.close()
        replacement.close()


GUARD_SCRIPT = '''
import sys
from app.infrastructure.database.worker_ownership import sqlite_worker_ownership
try:
    with sqlite_worker_ownership(*sys.argv[1:]):
        print("held", flush=True)
        sys.stdin.readline()
except RuntimeError as exc:
    print(str(exc), flush=True)
    sys.exit(23)
'''


def start_guard(*paths):
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    return subprocess.Popen([sys.executable, "-c", GUARD_SCRIPT, *map(str, paths)], env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def stop_guard(process):
    if process.poll() is None:
        process.communicate("release\n", timeout=10)
    else:
        process.communicate(timeout=10)


@pytest.mark.parametrize("shared", ["raw", "checkpoint", "both", "alias", "relative"])
def test_ownership_rejects_shared_resource_across_processes(tmp_path, monkeypatch, shared):
    audit, checkpoint = tmp_path / "audit.db", tmp_path / "checkpoint.db"
    first = start_guard(audit, checkpoint)
    second = None
    try:
        assert first.stdout.readline().strip() == "held", first.stderr.read()
        if shared == "alias":
            alias = tmp_path / "alias"
            alias.symlink_to(tmp_path, target_is_directory=True)
            paths = [alias / "sub" / ".." / "audit.db", tmp_path / "other-checkpoint.db"]
        elif shared == "relative":
            monkeypatch.chdir(tmp_path)
            paths = [Path("audit.db"), Path("other-checkpoint.db")]
        else:
            paths = [audit if shared in {"raw", "both"} else tmp_path / "other-audit.db",
                     checkpoint if shared in {"checkpoint", "both"} else tmp_path / "other-checkpoint.db"]
        second = start_guard(*reversed(paths))
        output, error = second.communicate("release\n", timeout=10)
        assert second.returncode == 23, (output, error)
        assert "single worker" in output
        assert str(tmp_path) not in output + error
    finally:
        stop_guard(first)
        if second and second.poll() is None:
            stop_guard(second)
    # 锁文件保留同一 inode，正常退出后新的进程可重新占用。
    third = start_guard(audit, checkpoint)
    try:
        assert third.stdout.readline().strip() == "held", third.stderr.read()
    finally:
        stop_guard(third)


def test_ownership_independent_stores_and_sigkill_release(tmp_path):
    first = start_guard(tmp_path / "a.db", tmp_path / "a-checkpoint.db")
    second = start_guard(tmp_path / "b.db", tmp_path / "b-checkpoint.db")
    replacement = None
    try:
        assert first.stdout.readline().strip() == "held", first.stderr.read()
        assert second.stdout.readline().strip() == "held", second.stderr.read()
        first.kill()
        first.wait(timeout=10)
        replacement = start_guard(tmp_path / "a.db", tmp_path / "a-checkpoint.db")
        assert replacement.stdout.readline().strip() == "held", replacement.stderr.read()
    finally:
        stop_guard(first)
        stop_guard(second)
        if replacement:
            stop_guard(replacement)


def test_ownership_partial_failure_releases_first_path(tmp_path):
    occupied = start_guard(tmp_path / "z.db")
    try:
        assert occupied.stdout.readline().strip() == "held", occupied.stderr.read()
        partial = start_guard(tmp_path / "a.db", tmp_path / "z.db")
        output, error = partial.communicate("release\n", timeout=10)
        assert partial.returncode == 23, (output, error)
        released = start_guard(tmp_path / "a.db")
        try:
            assert released.stdout.readline().strip() == "held", released.stderr.read()
        finally:
            stop_guard(released)
    finally:
        stop_guard(occupied)


@pytest.mark.asyncio
async def test_main_lifespan_owns_actual_stores_before_initialization(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    import main
    from app.infrastructure.database.readiness import SQLiteReadiness
    from app.infrastructure.database.worker_ownership import sqlite_worker_ownership

    raw, checkpoint = tmp_path / "actual-raw.db", tmp_path / "actual-checkpoint.db"
    monkeypatch.setattr(main.database_module, "DATABASE_PATH", str(raw))
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "misleading-env.db"))
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_DB_PATH", str(checkpoint))
    preflight = Mock()
    init_db = AsyncMock(side_effect=RuntimeError("synthetic init failure"))
    monkeypatch.setattr(main, "preflight_law_knowledge", preflight)
    monkeypatch.setattr(main, "init_db", init_db)
    for name in ["init_redis", "close_redis", "close_db"]:
        monkeypatch.setattr(main, name, AsyncMock())
    for name in ["reorder_service", "chat_model_factory", "embed_model_factory"]:
        monkeypatch.setattr(main, name, SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(main, "orchestrator", SimpleNamespace(
        configure_checkpointer=lambda *args, **kwargs: None, mark_checkpointer_closed=lambda: None))
    monkeypatch.setattr(main, "checkpoint_readiness", SQLiteReadiness())
    occupied = start_guard(raw)
    try:
        assert occupied.stdout.readline().strip() == "held", occupied.stderr.read()
        with pytest.raises(RuntimeError, match="single worker"):
            async with main.lifespan(main.app):
                pytest.fail("shared raw database must reject before yield")
        preflight.assert_not_called()
        init_db.assert_not_awaited()
        assert not checkpoint.exists()
    finally:
        stop_guard(occupied)
    preflight.side_effect = RuntimeError("synthetic preflight failure")
    with pytest.raises(RuntimeError, match="synthetic preflight failure"):
        async with main.lifespan(main.app):
            pytest.fail("preflight must fail")
    with sqlite_worker_ownership(raw, checkpoint):
        pass
    preflight.side_effect = None
    with pytest.raises(RuntimeError, match="synthetic init failure"):
        async with main.lifespan(main.app):
            pytest.fail("initialization must fail")
    # 初始化异常必须释放两库，且锁文件不能通过 unlink 重建。
    with sqlite_worker_ownership(raw, checkpoint):
        pass
    init_db.side_effect = None
    async with main.lifespan(main.app):
        competitor = start_guard(checkpoint)
        output, error = competitor.communicate("release\n", timeout=10)
        assert competitor.returncode == 23, (output, error)
    with sqlite_worker_ownership(raw, checkpoint):
        pass


def test_ownership_unsupported_platform_fails_closed(tmp_path, monkeypatch):
    from app.infrastructure.database import worker_ownership

    monkeypatch.setattr(worker_ownership, "fcntl", None)
    with pytest.raises(RuntimeError, match="single worker"):
        with worker_ownership.sqlite_worker_ownership(tmp_path / "audit.db"):
            pytest.fail("unsupported locking must fail closed")


def test_ownership_lock_io_failure_hides_path_and_releases_partial_locks(tmp_path, monkeypatch):
    from app.infrastructure.database import worker_ownership

    real_flock = worker_ownership.fcntl.flock
    attempts = 0

    def fail_second(descriptor, operation):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise OSError(f"synthetic lock failure: {tmp_path}")
        return real_flock(descriptor, operation)

    monkeypatch.setattr(worker_ownership.fcntl, "flock", fail_second)
    with pytest.raises(RuntimeError, match="single worker") as failure:
        with worker_ownership.sqlite_worker_ownership(tmp_path / "a.db", tmp_path / "z.db"):
            pytest.fail("lock I/O failure must reject startup")
    assert str(tmp_path) not in str(failure.value)
    monkeypatch.setattr(worker_ownership.fcntl, "flock", real_flock)
    with worker_ownership.sqlite_worker_ownership(tmp_path / "a.db", tmp_path / "z.db"):
        pass


def test_memory_databases_do_not_create_disk_locks_but_empty_arguments_reject(tmp_path, monkeypatch):
    from app.infrastructure.database.worker_ownership import sqlite_worker_ownership

    monkeypatch.chdir(tmp_path)
    with sqlite_worker_ownership(":memory:", Path(":memory:")):
        assert list(tmp_path.iterdir()) == []
    with pytest.raises(RuntimeError, match="single worker"):
        with sqlite_worker_ownership():
            pytest.fail("no database arguments must reject")
    with sqlite_worker_ownership(":memory:", "shared.db"):
        competitor = start_guard("shared.db", ":memory:")
        output, error = competitor.communicate("release\n", timeout=10)
        assert competitor.returncode == 23, (output, error)
    assert not (tmp_path / ":memory:.worker.lock").exists()


@pytest.mark.asyncio
async def test_main_memory_checkpoint_does_not_claim_restart_recovery_or_pollute_cwd(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import main
    from app.consultation.workflow import ConsultationOrchestrator
    from app.infrastructure.database.readiness import SQLiteReadiness

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main.database_module, "DATABASE_PATH", ":memory:")
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_DB_PATH", ":memory:")
    monkeypatch.setattr(main, "preflight_law_knowledge", lambda: None)
    for name in ["init_db", "init_redis", "close_redis", "close_db"]:
        monkeypatch.setattr(main, name, AsyncMock())
    for name in ["reorder_service", "chat_model_factory", "embed_model_factory"]:
        monkeypatch.setattr(main, name, SimpleNamespace(close=lambda: None))
    graph = ConsultationOrchestrator()
    monkeypatch.setattr(main, "orchestrator", graph)
    monkeypatch.setattr(main, "checkpoint_readiness", SQLiteReadiness())
    async with main.lifespan(main.app):
        assert graph.checkpoint_persistence == "process"
        assert graph.can_resume_after_restart is False
        assert list(tmp_path.iterdir()) == []
    assert list(tmp_path.iterdir()) == []
