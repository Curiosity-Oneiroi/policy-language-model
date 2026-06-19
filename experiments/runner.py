"""RunManager — the orchestration glue: allocate a workspace, launch a PLM run as a
subprocess, track it, free the workspace when it ends, and read back run artifacts.

A run's artifacts live in `runs/<run_id>/` (persistent); the allocated workspace holds
only the kernel venv + cwd and is returned to the pool (or pruned) when the run ends.
"""

from __future__ import annotations

import json
import shutil
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
        options. Branches on `spec["kind"]` (default "single"):

          "single"       — the EXISTING single PLM run (unchanged): metaparam_dir,
                           backend, simulate_config, seed, max_turns, return_budget,
                           task, tool_timeout, label.
          "optimization" — drives the GEPA recipe loop via `optimize_entry`: recipe,
                           backend, reflection, simulate_config, max_turns,
                           return_budget, generations, max_metric_calls,
                           population_size, label.  The optimizer makes per-candidate
                           workspaces under a workspace ROOT inside the experiment dir,
                           so NO pool slot is consumed."""
        if spec.get("kind") == "optimization":
            return self._create_optimization_run(spec)
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

    def _create_optimization_run(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        run_id = spec.get("run_id") or ("opt-" + uuid.uuid4().hex[:12])
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        # Clean a workspace ROOT for the optimizer (it makes per-candidate workspaces
        # under it). No pool slot is allocated for optimization runs.
        ws_root = run_dir / "workspaces"
        if ws_root.exists():
            shutil.rmtree(ws_root, ignore_errors=True)
        ws_root.mkdir(parents=True, exist_ok=True)
        backend = spec["backend"]
        cfg = {
            "run_id": run_id,
            "runs_dir": str(self.runs_dir.resolve()),
            "kind": "optimization",
            "recipe": spec["recipe"],
            "backend": backend,
            "reflection": spec.get("reflection") or backend,
            "simulate_config": spec.get("simulate_config"),
            "max_turns": spec.get("max_turns", 5),
            "return_budget": spec.get("return_budget", 2),
            "generations": spec.get("generations", 2),
            "max_metric_calls": spec.get("max_metric_calls", 10),
            "population_size": spec.get("population_size", 4),
            "dotenv_path": self.dotenv_path,
            "label": spec.get("label"),
            "created": time.time(),
        }
        (run_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-m", "plm.experiments.optimize_entry", str(run_dir / "config.json")],
            stdout=(run_dir / "stdout.log").open("w"),
            stderr=(run_dir / "stderr.log").open("w"),
        )
        self._procs[run_id] = proc
        return {"run_id": run_id, "kind": "optimization", "recipe": spec["recipe"]}

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
            kind = run.get("kind") or "single"
            if kind == "optimization":
                pop = self._population_rows(d)
                best = max((r.get("scalar") or 0.0) for r in pop) if pop else None
                best_elo = max((r.get("est_elo") for r in pop
                                if isinstance(r.get("est_elo"), (int, float))), default=None)
                out.append({
                    "run_id": d.name,
                    "kind": "optimization",
                    "status": run.get("status"),
                    "label": cfg.get("label"),
                    "backend": cfg.get("backend"),
                    "recipe": cfg.get("recipe"),
                    "started": run.get("started"),
                    "ended": run.get("ended"),
                    "num_candidates": len(pop),
                    "best_scalar": best,
                    "best_elo": best_elo,
                    "live": d.name in self._procs,
                })
                continue
            sims = self._simulate_rows(d)
            last = sims[-1].get("summary") if sims else None
            out.append({
                "run_id": d.name,
                "kind": "single",
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

    @staticmethod
    def _population_rows(d: Path) -> List[Dict[str, Any]]:
        """Parse a normalized `population.jsonl` (one row per evaluated candidate)."""
        f = d / "population.jsonl"
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

    @classmethod
    def _candidates_view(cls, pop: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Derived per-candidate list for the API (a thin projection of population)."""
        return [{
            "candidate_id": r.get("candidate_id"),
            "step": r.get("step"),
            "gen": r.get("gen"),
            "method": r.get("method"),
            "scalar": r.get("scalar"),
            "est_elo": r.get("est_elo"),
            "per_instance": r.get("per_instance"),
            "parent_id": r.get("parent_id"),
            "accepted": r.get("accepted"),
            "on_pareto_front": r.get("on_pareto_front"),
            "run_dir": r.get("run_dir"),
        } for r in pop]

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        d = self.runs_dir / run_id
        if not d.is_dir():
            return None
        self.poll()
        run = self._read_json(d / "run.json")
        if (run or {}).get("kind") == "optimization":
            pop = self._population_rows(d)
            return {
                "kind": "optimization",
                "run": run,
                "population": pop,
                "pareto_front": self._jsonl(d / "pareto_front.jsonl"),
                "optimization_result": self._read_json(d / "optimization_result.json"),
                "candidates": self._candidates_view(pop),
                "live": run_id in self._procs,
            }
        return {
            "run": run,
            "result": self._read_json(d / "result.json"),
            "trajectory": self._read_json(d / "trajectory.json"),
            "simulates": self._simulate_rows(d),
            "live": run_id in self._procs,
        }

    @staticmethod
    def _jsonl(f: Path) -> List[Dict[str, Any]]:
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

    # ---- optimization: per-candidate access ------------------------------- #
    def _candidate_row(self, run_id: str, candidate_id: str) -> Optional[Dict[str, Any]]:
        d = self.runs_dir / run_id
        for r in self._population_rows(d):
            if r.get("candidate_id") == candidate_id:
                return r
        return None

    def get_candidate(self, run_id: str, candidate_id: str) -> Optional[Dict[str, Any]]:
        """Genome text (system_prompt + verifier) + lineage + scores + the
        candidate's PLM trajectory. Paths come from the population row."""
        row = self._candidate_row(run_id, candidate_id)
        if row is None:
            return None
        mp = Path(row.get("metaparam_dir") or "")
        rd = Path(row.get("run_dir") or "")
        sp = mp / "system_prompt.md"
        vf = mp / "verifier.py"
        traj = self._read_json(rd / "trajectory.json") or {}
        return {
            "candidate_id": candidate_id,
            "genome": {
                "system_prompt": sp.read_text(encoding="utf-8") if sp.is_file() else "",
                "verifier": vf.read_text(encoding="utf-8") if vf.is_file() else "",
            },
            "parent_id": row.get("parent_id"),
            "method": row.get("method"),
            "scalar": row.get("scalar"),
            "vector": row.get("vector"),
            "per_instance": row.get("per_instance"),
            "trajectory": traj.get("messages", []),
        }

    def list_candidate_games(self, run_id: str, candidate_id: str) -> Optional[List[Dict[str, Any]]]:
        """Reuse `_simulate_rows` on the candidate's run_dir."""
        row = self._candidate_row(run_id, candidate_id)
        if row is None:
            return None
        return self._simulate_rows(Path(row.get("run_dir") or ""))

    def get_game(self, run_id: str, sim_idx: int, game_id,
                 candidate_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Read one game. When `candidate_id` is given, route to that candidate's
        run_dir; otherwise the single-run experiment dir (unchanged behavior)."""
        if candidate_id is not None:
            row = self._candidate_row(run_id, candidate_id)
            if row is None:
                return None
            base = Path(row.get("run_dir") or "")
        else:
            base = self.runs_dir / run_id
        return self._read_json(base / "games" / str(sim_idx) / f"{game_id}.json")
