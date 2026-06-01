from __future__ import annotations

import os
import pickle
import select
import shutil
import signal
import socket
import struct
import subprocess
import tempfile
import textwrap
import threading
import time
import uuid
import weakref
from contextlib import suppress
from pathlib import Path
from typing import Any

import dill

from .kernel import KERNEL_BOOTSTRAP, KERNEL_LOOP
from .prefix import PREFIX


DEFAULT_PREINSTALL: tuple[str, ...] = ("numpy", "requests", "dill", "pydantic")

_SHUTDOWN_WAIT = 2.0
_REAP_WAIT = 1.0


class _CellTimeout(Exception):
    """Internal — caught by `execute_cell` to drive SIGINT/SIGKILL escalation."""


def _compute_child_pythonpath() -> str:
    """Build PYTHONPATH so the kernel child can import `plm.*` and `AFramework.*`."""
    here = Path(__file__).resolve()
    # plm/repl/session.py → parents[2] is the repo root.
    repo_root = here.parents[2]
    entries = [str(repo_root)]
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        entries.append(existing)
    return os.pathsep.join(entries)


def _assemble_kernel_script(prefix_text: str) -> str:
    """Glue bootstrap + prefix + loop into one `.repl_kernel.py` script."""
    body = prefix_text + "\n" + KERNEL_LOOP
    indented = textwrap.indent(body, "    ")
    return (
        KERNEL_BOOTSTRAP
        + "\ntry:\n"
        + indented
        + "\nexcept BaseException:\n"
        + "    import traceback as _repl_tb_mod\n"
        + "    try:\n"
        + "        _repl_write_frame({\"type\": \"boot_error\", \"traceback\": _repl_tb_mod.format_exc()})\n"
        + "    except Exception:\n"
        + "        pass\n"
    )


def _run_uv(cmd: list[str]) -> None:
    """Invoke `uv ...`; raise RuntimeError with a useful message on failure."""
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError(
            "`uv` not found on PATH. Install uv (https://docs.astral.sh/uv/) "
            "or put it on PATH before constructing PythonReplSession."
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"`{' '.join(cmd)}` failed with exit {e.returncode}\n"
            f"--- stdout ---\n{e.stdout}\n--- stderr ---\n{e.stderr}"
        )


def _cleanup(workspace: str, child_pid_box: list[int | None]) -> None:
    """GC/atexit finalizer — SIGKILLs the (current) child pid and rmtrees the workspace."""
    pid = child_pid_box[0] if child_pid_box else None
    if pid is not None:
        with suppress(ProcessLookupError, PermissionError, OSError):
            os.kill(pid, signal.SIGKILL)
    shutil.rmtree(workspace, ignore_errors=True)


class PythonReplSession:
    """Persistent-kernel REPL backed by a per-session uv venv.

    Cell execution is serial and stateful; the kernel child survives
    user-level errors (syntax, runtime, SystemExit). Hard timeouts and
    crashes trigger SIGKILL + respawn from the cached vars snapshot.

    Parameters
    ----------
    workspace:
        Session working directory. Auto-created if `None`. Always removed on close.
    code_prefix:
        Kernel-prefix code-text run once at boot. Default `PREFIX` sets up
        stdout/stderr capture, RETURN, var collection (for
        crash-restart only), and depth + name-resolution helpers.
    preinstall:
        Packages installed into the per-session `.venv` via `uv pip install`.
    env:
        Extra env vars for the child.
    cell_timeout, sigint_grace:
        On timeout, send SIGINT and wait `sigint_grace` for a graceful traceback;
        otherwise SIGKILL + respawn, rehydrating REPL state from the cached
        vars blob. The blob is collected on every cell for crash recovery
        only — it is NOT surfaced to the caller.
    var_size_max:
        Per-var serialized-size cap for the cached crash-restart blob.
        Oversized values are silently skipped (rehydrate is best-effort).
    """

    def __init__(
        self,
        workspace: str | None,
        code_prefix: str = PREFIX,
        preinstall: tuple[str, ...] | list[str] = DEFAULT_PREINSTALL,
        env: dict[str, str] | None = None,
        cell_timeout: float | None = 300.0,
        sigint_grace: float = 5.0,
        var_size_max: int = 10 * 1024 * 1024,
    ) -> None:
        if not workspace:
            workspace = tempfile.mkdtemp(prefix="plm_session_")

        self.workspace = str(Path(workspace).resolve())
        self._prefix = code_prefix
        self._cell_timeout = cell_timeout
        self._sigint_grace = sigint_grace
        self.var_size_max = var_size_max
        self.execution_count = 0
        self.cached_vars_blob: bytes = b""
        self.last_rehydrate_error: str | None = None

        self._proc: subprocess.Popen[bytes] | None = None
        self._client_sock: socket.socket | None = None
        self._sock_io = None  # type: ignore[assignment]
        self._req_w = None  # type: ignore[assignment]
        self._resp_r = None  # type: ignore[assignment]
        self._child_pid_box: list[int | None] = [None]
        self._io_lock = threading.Lock()

        ws = Path(self.workspace)
        ws.mkdir(parents=True, exist_ok=True)

        # Register the finalizer NOW — before anything that can raise.
        # Otherwise a venv-setup failure would leak the tempdir.
        self._finalizer = weakref.finalize(
            self, _cleanup, self.workspace, self._child_pid_box,
        )

        venv_dir = ws / ".venv"
        venv_python_path = venv_dir / "bin" / "python"
        if not venv_python_path.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)
            _run_uv(["uv", "venv", str(venv_dir)])
        preinstall_list = [p for p in preinstall if p]
        if preinstall_list:
            _run_uv(["uv", "pip", "install", "--python", str(venv_python_path), *preinstall_list])
        self._venv_python = str(venv_python_path)

        self._kernel_path = str(ws / ".repl_kernel.py")
        with open(self._kernel_path, "w", encoding="utf-8") as f:
            f.write(_assemble_kernel_script(self._prefix))

        self._child_env: dict[str, str] = {
            **os.environ,
            **(env or {}),
            "PYTHONPATH": _compute_child_pythonpath(),
            "PYTHONUNBUFFERED": "1",
            "_REPL_VAR_SIZE_MAX": str(self.var_size_max),
        }

        with self._io_lock:
            self._spawn_kernel()

    # ---- context manager --------------------------------------------------

    def __enter__(self) -> "PythonReplSession":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ---- public API -------------------------------------------------------

    def execute_cell(self, code: str) -> dict[str, Any]:
        """Run one cell; return an envelope:
            {type: 'result'|'return', return_obj, stdout, stderr}.

        RETURN(obj) → type='return', return_obj set.
        On hard timeout or child death: SIGKILL + respawn, REPL state
        rehydrated from the cached vars blob; return a 'result' envelope
        whose stderr explains what happened.

        The kernel ships a vars snapshot every cell — cached internally for
        crash-restart rehydrate only. It is NOT included in the returned
        envelope; if the model wants to see a value, the cell must `print()` it.
        """
        self.execution_count += 1

        with self._io_lock:
            try:
                self._write_frame({"type": "code", "code": code})
                envelope = self._read_frame_with_timeout(self._cell_timeout)
            except _CellTimeout:
                try:
                    if self._proc is not None:
                        self._proc.send_signal(signal.SIGINT)
                    envelope = self._read_frame_with_timeout(self._sigint_grace)
                except (_CellTimeout, EOFError, BrokenPipeError, struct.error):
                    self._kill_and_respawn()
                    return {
                        "type": "result", "return_obj": None,
                        "stdout": "",
                        "stderr": f"[child timed out after {self._cell_timeout}s; prior state preserved]\n",
                    }
            except (EOFError, BrokenPipeError, struct.error):
                self._kill_and_respawn()
                return {
                    "type": "result", "return_obj": None,
                    "stdout": "",
                    "stderr": "[child exited without writing payload; prior state preserved]\n",
                }

            if envelope.get("type") == "boot_error":
                self._kill_and_respawn()
                return {
                    "type": "result", "return_obj": None,
                    "stdout": "",
                    "stderr": f"[kernel boot_error]\n{envelope.get('traceback', '')}",
                }

        etype = envelope.get("type") or "result"
        # Cache the vars snapshot for crash-restart rehydrate. Never surfaced.
        self.cached_vars_blob = envelope.get("vars_blob") or b""

        return_obj: Any = None
        stderr_text = envelope.get("stderr") or ""
        # Surface any pending rehydrate error exactly once.
        if self.last_rehydrate_error is not None:
            stderr_text = f"[rehydrate after respawn: {self.last_rehydrate_error}]\n" + stderr_text
            self.last_rehydrate_error = None
        if etype == "return":
            blob = envelope.get("return_blob")
            if blob is None:
                stderr_text += "\n[RETURN: kernel could not serialize return value]\n"
                etype = "result"
            else:
                try:
                    return_obj = dill.loads(blob)
                except Exception as e:
                    stderr_text += f"\n[RETURN: parent failed to decode return value: {e!r}]\n"
                    etype = "result"

        return {
            "type": etype,
            "return_obj": return_obj,
            "stdout": envelope.get("stdout") or "",
            "stderr": stderr_text,
        }

    def close(self) -> None:
        """Idempotent teardown."""
        fin = getattr(self, "_finalizer", None)
        if fin is None or not fin.alive:
            return

        with self._io_lock:
            if self._req_w is not None:
                with suppress(BrokenPipeError, OSError, ValueError):
                    self._write_frame({"type": "shutdown"})

            if self._proc is not None:
                try:
                    self._proc.wait(timeout=_SHUTDOWN_WAIT)
                except subprocess.TimeoutExpired:
                    with suppress(ProcessLookupError, OSError):
                        self._proc.kill()
                    with suppress(subprocess.TimeoutExpired):
                        self._proc.wait(timeout=_REAP_WAIT)

            self._close_pipes()

        fin()

    # ---- internals --------------------------------------------------------

    def _spawn_kernel(self) -> None:
        """Spawn the venv-python kernel and complete the UNIX-socket handshake.

        Path-based AF_UNIX (not pass_fds) so subprocess.Popen stays
        posix_spawn-eligible — `fork()` from a multi-threaded Python process
        can deadlock on glibc's malloc mutex; `posix_spawn` (vfork) avoids it.
        """
        sock_path = f"/tmp/replsock_{uuid.uuid4().hex[:12]}.sock"
        with suppress(FileNotFoundError):
            os.unlink(sock_path)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(sock_path)
            listener.listen(1)

            spawn_env = {**self._child_env, "_REPL_SOCK_PATH": sock_path}
            self._proc = subprocess.Popen(
                [self._venv_python, "-u", self._kernel_path],
                cwd=self.workspace,
                env=spawn_env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )

            listener.settimeout(30.0)
            try:
                client_sock, _ = listener.accept()
            except socket.timeout:
                self._kill_silently()
                raise RuntimeError(
                    f"PLM kernel didn't connect within 30s — the venv at "
                    f"{self._venv_python!r} may be broken"
                )
        finally:
            listener.close()
            with suppress(FileNotFoundError, OSError):
                os.unlink(sock_path)

        self._client_sock = client_sock
        self._sock_io = client_sock.makefile("rwb", buffering=0)
        self._req_w = self._sock_io
        self._resp_r = self._sock_io
        self._child_pid_box[0] = self._proc.pid

        envelope = self._read_frame_with_timeout(30.0)
        etype = envelope.get("type")
        if etype == "boot_error":
            tb = envelope.get("traceback", "<no traceback>")
            self._kill_silently()
            raise RuntimeError(f"PLM kernel boot failed:\n{tb}")
        if etype != "ready":
            self._kill_silently()
            raise RuntimeError(f"PLM kernel handshake: expected 'ready', got {etype!r}")

        if self.cached_vars_blob:
            self._write_frame({"type": "rehydrate", "vars_blob": self.cached_vars_blob})
            envelope = self._read_frame_with_timeout(30.0)
            if envelope.get("type") != "ready":
                self._kill_silently()
                raise RuntimeError(
                    f"PLM kernel rehydrate: expected 'ready', got {envelope.get('type')!r}"
                )
            self.last_rehydrate_error = envelope.get("rehydrate_error")

    def _kill_and_respawn(self) -> None:
        self._kill_silently()
        self._spawn_kernel()

    def _kill_silently(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            with suppress(ProcessLookupError, OSError):
                self._proc.kill()
            with suppress(subprocess.TimeoutExpired):
                self._proc.wait(timeout=_REAP_WAIT)
        self._close_pipes()

    def _close_pipes(self) -> None:
        if self._sock_io is not None:
            with suppress(Exception):
                self._sock_io.close()
            self._sock_io = None
        if self._client_sock is not None:
            with suppress(Exception):
                self._client_sock.close()
            self._client_sock = None
        self._req_w = None
        self._resp_r = None

    def _write_frame(self, obj: dict[str, Any]) -> None:
        if self._req_w is None:
            raise BrokenPipeError("request socket not open")
        body = pickle.dumps(obj)
        self._req_w.write(struct.pack(">I", len(body)))
        self._req_w.write(body)
        self._req_w.flush()

    def _read_frame_with_timeout(self, timeout: float | None) -> dict[str, Any]:
        if self._resp_r is None:
            raise EOFError("response socket not open")

        deadline = None if timeout is None else time.monotonic() + timeout
        resp_fd = self._resp_r.fileno()

        def _read_exact(n: int) -> bytes:
            buf = b""
            while len(buf) < n:
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise _CellTimeout()
                    rlist, _, _ = select.select([resp_fd], [], [], remaining)
                    if not rlist:
                        raise _CellTimeout()
                chunk = self._resp_r.read(n - len(buf))
                if not chunk:
                    raise EOFError("response socket closed mid-frame")
                buf += chunk
            return buf

        hdr = _read_exact(4)
        (n,) = struct.unpack(">I", hdr)
        body = _read_exact(n)
        return pickle.loads(body)
