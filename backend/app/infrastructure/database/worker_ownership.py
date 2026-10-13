"""本机 SQLite 单 worker 所有权；不替代跨主机协调或消息事务。"""

import os
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
except ImportError:
    fcntl = None


_LOCK_ERROR = "SQLite 存储要求 single worker；无法取得独占所有权，启动已拒绝。"


@contextmanager
def sqlite_worker_ownership(*database_paths):
    """按规范化路径分别持有非阻塞文件锁，整个应用生命周期内禁止第二所有者。

    相对路径和符号链接归一化后排序，任一路径争用或不支持锁均拒绝启动。
    精确的 :memory: 是各进程独立的 SQLite，只跳过该资源；无参数仍拒绝。
    正常退出、初始化失败或进程终止由关闭描述符释放锁；锁文件不能删除，
    否则旧所有者与新所有者可能锁住不同 inode。仅适用于可靠支持 flock 的
    本机共享文件系统；硬链接、不同挂载视图和绕过本入口的程序不在保证内。
    """
    if not database_paths:
        raise RuntimeError(_LOCK_ERROR)
    descriptors = []
    try:
        try:
            paths = sorted({Path(path).resolve() for path in database_paths if os.fspath(path) != ":memory:"})
            if paths and fcntl is None:
                raise RuntimeError(_LOCK_ERROR)
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                descriptor = os.open(str(path) + ".worker.lock", os.O_CREAT | os.O_RDWR, 0o600)
                descriptors.append(descriptor)
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, TypeError, ValueError, RuntimeError):
            # 固定错误正文和抑制异常链，避免启动日志泄露存储绝对路径。
            raise RuntimeError(_LOCK_ERROR) from None
        yield
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
