"""WorkspacePool — a pool of numbered, pre-provisioned kernel venvs that PLM runs reuse.

Each workspace is `<root>/NNNN/` holding a `.venv` with the kernel Python pinned and
`game` (editable) + PLM's kernel deps installed. A PLM run is ALLOCATED a free
workspace (`session_workspace=<ws>`); PLM's own workspace-reuse then reuses that venv
(so a run NEVER installs `game` — it only adds its backend SDK). Run ARTIFACTS live in
`runs/`, not here, so a workspace is disposable:

  * raising the pool size provisions new workspaces,
  * lowering it deletes the highest-numbered FREE workspaces (allocated ones are
    pruned when they're freed).

Status is tracked with marker files so it survives a backend restart:
  `<ws>/.ready`     — venv provisioned OK,    `<ws>/.allocated` — holds the run_id.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_PYTHON = "3.12"
# PLM kernel core deps; `game` (editable) is added explicitly and drags in its own deps
# (python-chess / fastapi / uvicorn). The chosen backend's SDK is added by PLM at run time.
DEFAULT_DEPS = ("dill", "pydantic", "numpy", "requests")


def _run_uv(cmd: List[str], timeout: int = 600) -> None:
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        raise RuntimeError("`uv` not found on PATH. Install uv (https://docs.astral.sh/uv/).")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"`{' '.join(cmd)}` timed out after {timeout}s.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"`{' '.join(cmd)}` failed (exit {e.returncode}):\n{e.stderr}")


class WorkspacePool:
    def __init__(self, root, game_path, *, python: str = DEFAULT_PYTHON,
                 deps=DEFAULT_DEPS) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.game_path = str(Path(game_path).resolve())
        self.python = python
        self.deps = tuple(deps)
        self._lock = threading.Lock()
        self._target = self._count_existing()      # start: keep whatever is already on disk

    # ---- numbering -------------------------------------------------------- #
    def _dir(self, n: int) -> Path:
        return self.root / f"{n:04d}"

    def _numbers(self) -> List[int]:
        return sorted(int(p.name) for p in self.root.iterdir()
                      if p.is_dir() and p.name.isdigit())

    def _count_existing(self) -> int:
        return len(self._numbers())

    # ---- per-workspace status (marker files) ------------------------------ #
    def _ready(self, n: int) -> bool:
        return (self._dir(n) / ".ready").exists()

    def _alloc_file(self, n: int) -> Path:
        return self._dir(n) / ".allocated"

    def _allocated(self, n: int) -> bool:
        return self._alloc_file(n).exists()

    def _run_id(self, n: int) -> Optional[str]:
        f = self._alloc_file(n)
        return f.read_text(encoding="utf-8").strip() if f.exists() else None

    def _status_of(self, n: int) -> str:
        if not self._ready(n):
            return "provisioning"
        return "allocated" if self._allocated(n) else "free"

    def status(self) -> Dict[str, Any]:
        with self._lock:
            workspaces = [
                {"number": n, "path": str(self._dir(n)), "status": self._status_of(n),
                 "run_id": self._run_id(n)}
                for n in self._numbers()
            ]
        return {"target": self._target, "count": len(workspaces), "workspaces": workspaces}

    # ---- provisioning (SLOW — call outside the lock / in the background) --- #
    def provision(self, n: int) -> None:
        """Create + populate workspace `n`'s venv. Idempotent (reuses an existing venv).
        Writes the `.ready` marker on success."""
        ws = self._dir(n)
        ws.mkdir(parents=True, exist_ok=True)
        venv = ws / ".venv"
        py = venv / "bin" / "python"
        if not py.exists():
            shutil.rmtree(venv, ignore_errors=True)
            _run_uv(["uv", "venv", "--python", self.python, str(venv)])
        _run_uv(["uv", "pip", "install", "--python", str(py), "-e", self.game_path, *self.deps])
        (ws / ".ready").write_text("ok", encoding="utf-8")

    def set_target(self, n: int) -> List[int]:
        """Set the desired pool size. SHRINK deletes the highest-numbered FREE workspaces
        immediately (allocated ones are pruned on free). GROW returns the list of numbers
        that still need `provision()` — the caller provisions them (slow) in the background.
        """
        with self._lock:
            self._target = max(0, int(n))
            # shrink: delete FREE workspaces numbered above target, highest first.
            for k in sorted([x for x in self._numbers() if x > self._target], reverse=True):
                if not self._allocated(k):
                    shutil.rmtree(self._dir(k), ignore_errors=True)
            # grow: numbers in 1..target that don't exist yet (caller provisions these).
            to_create = [k for k in range(1, self._target + 1) if not self._dir(k).is_dir()]
            for k in to_create:
                self._dir(k).mkdir(parents=True, exist_ok=True)   # reserve the slot (status=provisioning)
        return to_create

    def ensure_provisioned(self) -> List[int]:
        """Numbers that exist but aren't `.ready` yet (e.g. after set_target or a restart)."""
        with self._lock:
            return [n for n in self._numbers() if not self._ready(n)]

    # ---- allocation ------------------------------------------------------- #
    def allocate(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Reserve the lowest-numbered ready+free workspace for `run_id`; None if none free."""
        with self._lock:
            for n in self._numbers():
                if self._ready(n) and not self._allocated(n):
                    self._alloc_file(n).write_text(str(run_id), encoding="utf-8")
                    return {"number": n, "path": str(self._dir(n))}
        return None

    def free(self, number: int) -> None:
        """Release a workspace. If it's now above the target (a shrink happened while it was
        allocated), delete it instead of returning it to the pool."""
        with self._lock:
            self._alloc_file(number).unlink(missing_ok=True)
            if number > self._target:
                shutil.rmtree(self._dir(number), ignore_errors=True)
