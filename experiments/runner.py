"""RunManager — the orchestration glue: allocate a workspace, launch a PLM run as a
subprocess, track it, free the workspace when it ends, and read back run artifacts.

A run's artifacts live in `runs/<run_id>/` (persistent); the allocated workspace holds
only the kernel venv + cwd and is returned to the pool (or pruned) when the run ends.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .workspaces import WorkspacePool


class RunManager:
    def __init__(self, runs_dir, pool: WorkspacePool, *, dotenv_path: Optional[str] = None) -> None:
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.pool = pool
        self.dotenv_path = dotenv_path
        self._procs: Dict[str, subprocess.Popen] = {}      # run_id -> live process
        self._ws: Dict[str, int] = {}                       # run_id -> allocated workspace number

    # ---- launching -------------------------------------------------------- #
    def create_run(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Allocate a free workspace and launch a run. `spec` carries the per-experiment
        options: metaparam_dir, backend{name,model,base_url}, simulate_config, seed,
        max_turns, return_budget, task, tool_timeout, label."""
        run_id = spec.get("run_id") or ("run-" + uuid.uuid4().hex[:12])
        alloc = self.pool.allocate(run_id)
        if alloc is None:
            raise RuntimeError("no free workspace available — raise the pool size in settings")
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = {
            "run_id": run_id,
            "runs_dir": str(self.runs_dir.resolve()),
            "workspace": alloc["path"],
            "workspace_number": alloc["number"],
            "metaparam_dir": spec["metaparam_dir"],
            "backend": spec["backend"],
            "simulate_config": spec.get("simulate_config"),
            "seed": spec.get("seed", 0),
            "max_turns": spec.get("max_turns", 100),
            "return_budget": spec.get("return_budget", 5),
            "task": spec.get("task"),
            "tool_timeout": spec.get("tool_timeout"),
            "dotenv_path": self.dotenv_path,
            "label": spec.get("label"),
            "created": time.time(),
        }
        (run_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-m", "plm.experiments.run_entry", str(run_dir / "config.json")],
            stdout=(run_dir / "stdout.log").open("w"),
            stderr=(run_dir / "stderr.log").open("w"),
        )
        self._procs[run_id] = proc
        self._ws[run_id] = alloc["number"]
        return {"run_id": run_id, "workspace": alloc["number"]}

    def poll(self) -> None:
        """Reap finished run processes and free (or prune) their workspaces."""
        for run_id, proc in list(self._procs.items()):
            if proc.poll() is not None:
                ws = self._ws.pop(run_id, None)
                if ws is not None:
                    self.pool.free(ws)
                self._procs.pop(run_id, None)

    def stop_run(self, run_id: str) -> bool:
        proc = self._procs.get(run_id)
        if proc and proc.poll() is None:
            proc.terminate()
            return True
        return False

    # ---- reading artifacts ------------------------------------------------ #
    @staticmethod
    def _read_json(p: Path):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    @staticmethod
    def _simulate_rows(d: Path) -> List[Dict[str, Any]]:
        f = d / "simulates.jsonl"
        if not f.exists():
            return []
        rows: List[Dict[str, Any]] = []
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
        return rows

    def list_runs(self) -> List[Dict[str, Any]]:
        self.poll()
        out: List[Dict[str, Any]] = []
        for d in sorted(self.runs_dir.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            run = self._read_json(d / "run.json")
            if run is None:
                continue
            cfg = run.get("config") or {}
            sims = self._simulate_rows(d)
            last = sims[-1].get("summary") if sims else None
            out.append({
                "run_id": d.name,
                "status": run.get("status"),
                "label": cfg.get("label"),
                "backend": cfg.get("backend"),
                "metaparam": Path(cfg.get("metaparam_dir", "")).name,
                "started": run.get("started"),
                "ended": run.get("ended"),
                "num_simulates": len(sims),
                "latest_score": (last or {}).get("score"),
                "latest_elo": (last or {}).get("est_elo"),
                "live": d.name in self._procs,
            })
        return out

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        d = self.runs_dir / run_id
        if not d.is_dir():
            return None
        self.poll()
        return {
            "run": self._read_json(d / "run.json"),
            "result": self._read_json(d / "result.json"),
            "trajectory": self._read_json(d / "trajectory.json"),
            "simulates": self._simulate_rows(d),
            "live": run_id in self._procs,
        }

    def get_game(self, run_id: str, sim_idx: int, game_id) -> Optional[Dict[str, Any]]:
        return self._read_json(self.runs_dir / run_id / "games" / str(sim_idx) / f"{game_id}.json")
