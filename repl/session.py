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

# The kernel venv is pinned to Python 3.12 — all PLM code targets 3.12 (uv finds an
# installed 3.12 or fetches one). Bump here if the target version ever changes.
DEFAULT_PYTHON: str = "3.12"

_SHUTDOWN_WAIT = 2.0
_REAP_WAIT = 1.0


class _CellTimeout(Exception):
    """Internal — caught by `execute_cell` to drive SIGINT/SIGKILL escalation.

    `partial` is True when the timeout fired AFTER some bytes of the current
    frame were already pulled off the socket (header or partial body). That
    means the response stream is now DESYNCED and cannot be safely re-read —
    the caller must respawn rather than attempt a second read on the same
    socket (which would misread the leftover bytes as a fresh frame)."""

    def __init__(self, partial: bool = False):
        super().__init__()
        self.partial = partial


class _FrameDecodeError(Exception):
    """Internal — a response frame could not be decoded (corrupt length or body,
    typically because a prior mid-frame timeout left the stream misaligned).
    Caught by `execute_cell` -> respawn; never allowed to escape uncaught."""


def _compute_child_pythonpath() -> str:
    """Build PYTHONPATH so the kernel child can import `plm.*`.

    The child runs in a fresh per-session venv that does NOT install `plm` — it is
    live project source, not a wheel. So we hand the child the directory that makes
    `plm` importable in THIS (parent) process, derived from `plm.__path__[0]`.

    This replaces an earlier `Path(__file__).resolve().parents[2]` guess: `resolve()`
    collapses the `plm` symlink, so that guess landed on a dir not containing the
    package, and the child only worked when the parent's PYTHONPATH happened to
    carry the right dir. Deriving from `plm.__path__[0]` (UN-resolved, to preserve
    the symlinked name) is correct regardless of how the parent obtained the
    package (symlink, editable, install). The model backends now live under
    `plm.model_backend` (no AFramework dependency), so only `plm` is needed. The
    inherited PYTHONPATH is still appended as a fallback.
    """
    entries: list[str] = []
    try:
        import plm
        paths = list(getattr(plm, "__path__", []) or [])
        if paths:
            entries.append(str(Path(paths[0]).parent))  # the dir that makes `import plm` work
    except Exception:
        pass
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        entries.append(existing)
    seen: set[str] = set()
    deduped: list[str] = []
    for e in entries:
        if e and e not in seen:
            seen.add(e)
            deduped.append(e)
    return os.pathsep.join(deduped)


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
        self._pending_boot_stderr: str | None = None   # soft extra-policy boot warnings (#26)
        # Bumped on every (re)spawn. A caller streaming incremental state (e.g. the
        # root loop's `plm_messages` delta) reads this to detect a respawn — whose
        # fresh kernel has lost any accumulated state — and resync from scratch.
        self.kernel_epoch: int = 0

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
            _run_uv(["uv", "venv", "--python", DEFAULT_PYTHON, str(venv_dir)])
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

    def execute_cell(
        self,
        code: str,
        seed: dict[str, Any] | None = None,
        plm_messages_delta: list | None = None,
    ) -> dict[str, Any]:
        """Run one cell; return an envelope:
            {type: 'result'|'return', return_obj, stdout, stderr}.

        ``seed`` (optional): a ``{name: object}`` dict bound into the kernel's
        ``__main__`` before this cell runs — a general per-round injection
        channel. Values must be picklable (the frame is pickled to the kernel).

        ``plm_messages_delta`` (optional): messages to APPEND to the kernel's
        ``plm_messages`` accumulator before this cell — an additive, O(delta)
        alternative to re-sending the whole trajectory each round. The root PLM
        loop uses it to stream its own trajectory; after a respawn (``kernel_epoch``
        changes) the caller must resend the full trajectory once, since the fresh
        kernel's accumulator is empty.

        RETURN(obj) → type='return', return_obj set.
        On hard timeout or child death: SIGKILL + respawn, REPL state
        rehydrated from the cached vars blob; return a 'result' envelope
        whose stderr explains what happened.

        The kernel ships a vars snapshot every cell — cached internally for
        crash-restart rehydrate only. It is NOT included in the returned
        envelope; if the model wants to see a value, the cell must `print()` it.
        """
        self.execution_count += 1

        # Snapshot + clear pending diagnostics NOW, before any respawn: the
        # early-return respawn paths below call _kill_and_respawn -> _spawn_kernel,
        # which OVERWRITES these fields with the fresh kernel's values. Capturing
        # first lets us surface the OLD pending once on THIS cell's stderr (on ANY
        # return path), instead of losing it to the respawn. (#10/#26)
        _pre = ""
        if self._pending_boot_stderr is not None:
            _pre += f"[boot: extra-policy install warning]\n{self._pending_boot_stderr}\n"
            self._pending_boot_stderr = None
        if self.last_rehydrate_error is not None:
            _pre += f"[rehydrate after respawn: {self.last_rehydrate_error}]\n"
            self.last_rehydrate_error = None

        with self._io_lock:
            try:
                self._write_frame({"type": "code", "code": code, "seed": seed,
                                   "plm_messages_delta": plm_messages_delta})
                envelope = self._read_frame_with_timeout(self._cell_timeout)
            except _CellTimeout as ct:
                if ct.partial:
                    # Timed out MID-FRAME: the response stream is now desynced
                    # (header/partial body already consumed). A second read would
                    # misread the leftover bytes as a new frame -> garbage that
                    # could even decode wrong. Don't retry on this socket; respawn
                    # directly (prior state is preserved via the cached snapshot).
                    self._kill_and_respawn()
                    return {
                        "type": "result", "return_obj": None,
                        "stdout": "",
                        "stderr": _pre + f"[child timed out after {self._cell_timeout}s; prior state preserved]\n",
                    }
                # Clean frame boundary (nothing of the next frame consumed yet):
                # try a graceful SIGINT so the SESSION can survive — the kernel
                # catches it and writes a clean interrupted-result frame, which we
                # read cleanly. Any timeout / desync / decode failure on the grace
                # read -> respawn.
                try:
                    if self._proc is not None:
                        self._proc.send_signal(signal.SIGINT)
                    envelope = self._read_frame_with_timeout(self._sigint_grace)
                except (_CellTimeout, _FrameDecodeError, EOFError, BrokenPipeError, struct.error):
                    self._kill_and_respawn()
                    return {
                        "type": "result", "return_obj": None,
                        "stdout": "",
                        "stderr": _pre + f"[child timed out after {self._cell_timeout}s; prior state preserved]\n",
                    }
            except (_FrameDecodeError, EOFError, BrokenPipeError, struct.error):
                self._kill_and_respawn()
                return {
                    "type": "result", "return_obj": None,
                    "stdout": "",
                    "stderr": _pre + "[child exited or desynced without a valid payload; prior state preserved]\n",
                }

            if envelope.get("type") == "boot_error":
                self._kill_and_respawn()
                return {
                    "type": "result", "return_obj": None,
                    "stdout": "",
                    "stderr": _pre + f"[kernel boot_error]\n{envelope.get('traceback', '')}",
                }

        etype = envelope.get("type") or "result"
        # Cache the vars snapshot for crash-restart rehydrate. Never surfaced.
        self.cached_vars_blob = envelope.get("vars_blob") or b""

        return_obj: Any = None
        # `_pre` carries any pending boot/rehydrate diagnostics captured at the
        # top (surfaced once, on whatever path we return through).
        stderr_text = _pre + (envelope.get("stderr") or "")
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
                reaped = False
                try:
                    self._proc.wait(timeout=_SHUTDOWN_WAIT)
                    reaped = True
                except subprocess.TimeoutExpired:
                    with suppress(ProcessLookupError, OSError):
                        self._proc.kill()
                    try:
                        self._proc.wait(timeout=_REAP_WAIT)
                        reaped = True
                    except subprocess.TimeoutExpired:
                        pass                              # genuinely stuck — keep the box so
                                                          # the finalizer still attempts a kill
                if reaped:
                    # Child reaped: clear the pid box so the GC/atexit finalizer
                    # never SIGKILLs a now-recycled PID (#R4-6).
                    self._child_pid_box[0] = None

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

        def _handshake_read(stage):
            # A handshake read must NOT let an internal _CellTimeout/
            # _FrameDecodeError sentinel escape across this public boundary, nor
            # leak the just-spawned child. On ANY read failure: kill the child,
            # then raise a clean PLM-process error (#14).
            try:
                return self._read_frame_with_timeout(30.0)
            except (_CellTimeout, _FrameDecodeError, EOFError, BrokenPipeError, struct.error) as e:
                self._kill_silently()
                raise RuntimeError(
                    f"PLM kernel handshake ({stage}) failed: {type(e).__name__}: {e}. "
                    f"The kernel did not produce a valid frame within 30s (a hung/slow "
                    f"boot, a crashed child, or a corrupt handshake)."
                ) from e

        envelope = _handshake_read("ready")
        etype = envelope.get("type")
        if etype == "boot_error":
            tb = envelope.get("traceback", "<no traceback>")
            self._kill_silently()
            raise RuntimeError(f"PLM kernel boot failed:\n{tb}")
        if etype != "ready":
            self._kill_silently()
            raise RuntimeError(f"PLM kernel handshake: expected 'ready', got {etype!r}")
        # Soft EXTRA-policy install warnings captured during boot (a DEFAULT that
        # fails to install is a HARD boot_error above; extras must not brick
        # boot). Surface them once on the next cell instead of losing them to the
        # reset boot buffer (#26).
        self._pending_boot_stderr = (envelope.get("boot_stderr") or "").strip() or None

        if self.cached_vars_blob:
            self._write_frame({"type": "rehydrate", "vars_blob": self.cached_vars_blob})
            envelope = _handshake_read("rehydrate")
            if envelope.get("type") != "ready":
                self._kill_silently()
                raise RuntimeError(
                    f"PLM kernel rehydrate: expected 'ready', got {envelope.get('type')!r}"
                )
            self.last_rehydrate_error = envelope.get("rehydrate_error")

        # A fresh kernel is now live (its in-kernel accumulators start empty);
        # mark a new epoch so incremental-state callers know to resync.
        self.kernel_epoch += 1

    def _kill_and_respawn(self) -> None:
        self._kill_silently()
        self._spawn_kernel()

    def _kill_silently(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            with suppress(ProcessLookupError, OSError):
                self._proc.kill()
            with suppress(subprocess.TimeoutExpired):
                self._proc.wait(timeout=_REAP_WAIT)
        # This proc is no longer our live child (killed/reaped, or already dead);
        # clear the pid box so the finalizer can't SIGKILL a recycled PID. The
        # next _spawn_kernel resets the box to the fresh child's pid (#R4-6).
        self._child_pid_box[0] = None
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

        # timeout=None => deadline=None => fully BLOCKING reads: wait forever.
        # PLM deliberately runs unbounded circuits (a whole sub-agent's worth of
        # rounds can happen inside one cell), so "no timeout" must NEVER
        # synthesize a _CellTimeout — it simply blocks until the frame arrives
        # (or EOF if the child dies). No desync is possible in that mode because
        # we never bail mid-frame.
        deadline = None if timeout is None else time.monotonic() + timeout
        resp_fd = self._resp_r.fileno()
        consumed = 0    # bytes of THIS frame already pulled off the socket

        def _read_exact(n: int) -> bytes:
            nonlocal consumed
            buf = b""
            while len(buf) < n:
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise _CellTimeout(partial=consumed > 0)
                    rlist, _, _ = select.select([resp_fd], [], [], remaining)
                    if not rlist:
                        raise _CellTimeout(partial=consumed > 0)
                chunk = self._resp_r.read(n - len(buf))
                if not chunk:
                    raise EOFError("response socket closed mid-frame")
                buf += chunk
                consumed += len(chunk)
            return buf

        hdr = _read_exact(4)
        (n,) = struct.unpack(">I", hdr)
        body = _read_exact(n)
        try:
            return pickle.loads(body)
        except Exception as e:
            # A corrupt body (e.g. the stream was misaligned by a prior
            # mid-frame timeout) must NOT escape as an arbitrary unpickling
            # error — surface it as a decode error the caller turns into a
            # respawn.
            raise _FrameDecodeError(
                f"frame decode failed: {type(e).__name__}: {e}"
            ) from e
