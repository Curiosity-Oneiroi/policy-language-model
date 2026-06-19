"""Evaluate one Candidate end-to-end: materialize → run PLM subprocess → score.

Always invokes `python -m plm.experiments.run_entry` in a SUBPROCESS so the
optimizer process never touches PLM's kernel/asyncio/REPL state directly. This
matches how a production sweep would isolate runs and lets the same code work
unchanged when `backend='slate'` is swapped in for the real loop.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .genome import Candidate
from .scorer import score_run


@dataclass
class EvalConfig:
    """Configuration for a single candidate evaluation.

    Concretely these collapse into the JSON config that
    `plm.experiments.run_entry` consumes; the only optimizer-side additions are
    `metaparams_root` (where to materialize candidates) and `runs_root` (where
    to put run_dirs). One eval == one subprocess == one run_dir.
    """

    # Where this evaluation should put materialized metaparam dirs + run_dirs.
    metaparams_root: Path
    runs_root: Path
    workspace_root: Path

    # Backend config — `name` must be a key registered in `experiments.config.BACKENDS`.
    # Defaults to the local vLLM backend (no per-token cost). Override base_url/model
    # to point at your vLLM server (VLLMBackend default is http://0.0.0.0:8005/v1).
    backend_name: str = "VLLMBackend"
    backend_model: Optional[str] = None
    backend_base_url: Optional[str] = None

    # PLM run knobs.
    task: str = "Optimizer eval — drive the metaparam to its terminal RETURN."
    max_turns: int = 5
    return_budget: int = 2
    tool_timeout: Optional[float] = 60.0

    # Optional simulate config (forwarded into run config). Left None unless a
    # run opts into simulate.
    simulate_config: Optional[Dict[str, Any]] = None

    # Subprocess timeout (seconds). Hard kill if PLM hangs.
    subprocess_timeout: float = 300.0

    # Misc. dotenv_path forwarded so the run subprocess loads creds.
    dotenv_path: Optional[str] = None

    # Phase-stage label, just for ledger/log readability (alpha/beta/gamma).
    stage: str = "alpha"

    # Extra args merged into the JSON config (escape hatch for future fields).
    extra_config: Dict[str, Any] = field(default_factory=dict)


def _python_executable() -> str:
    """The Python that ran us — propagating venv to the subprocess. Falls back
    to the literal `python` in PATH if `sys.executable` is empty (rare)."""
    return sys.executable or "python"


def evaluate(candidate: Candidate, eval_cfg: EvalConfig) -> Dict[str, Any]:
    """Materialize the candidate, run PLM as a subprocess, score the run_dir.

    Returns:
        {
          "candidate_id": str,
          "stage":        str,
          "run_dir":      str,
          "metaparam_dir": str,
          "score": {"vector": {...}, "scalar": float, "raw": {...}},
          "subprocess": {"returncode": int, "elapsed_s": float, "tail": str},
        }

    The function NEVER raises on a failed subprocess — a crashed run still
    produces a partial run_dir (the run_entry writes an error.txt + run.json
    with status='error'), and `score_run` is tolerant of that. The optimizer
    treats a low/zero score as the natural penalty for a failed run rather
    than an exception that aborts the whole sweep.
    """
    metaparam_dir = Path(eval_cfg.metaparams_root) / candidate.id
    run_id = f"{eval_cfg.stage}_{candidate.id}"
    run_dir = Path(eval_cfg.runs_root) / run_id
    workspace = Path(eval_cfg.workspace_root) / candidate.id

    # 1. Materialize the candidate metaparam dir (frozen `policies/` + prompt + verifier).
    candidate.materialize(metaparam_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)

    # 2. Compose the run-entry config (mirrors `run_entry._main`'s expected schema).
    cfg: Dict[str, Any] = {
        "run_id": run_id,
        "runs_dir": str(Path(eval_cfg.runs_root).resolve()),
        "workspace": str(workspace.resolve()),
        "backend": {
            "name": eval_cfg.backend_name,
            "model": eval_cfg.backend_model,
            "base_url": eval_cfg.backend_base_url,
        },
        "metaparam_dir": str(metaparam_dir.resolve()),
        "task": eval_cfg.task,
        "max_turns": eval_cfg.max_turns,
        "return_budget": eval_cfg.return_budget,
        "tool_timeout": eval_cfg.tool_timeout,
    }
    if eval_cfg.simulate_config is not None:
        cfg["simulate_config"] = eval_cfg.simulate_config
    if eval_cfg.dotenv_path:
        cfg["dotenv_path"] = eval_cfg.dotenv_path
    cfg.update(eval_cfg.extra_config)

    cfg_path = run_dir / "_optimizer_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # 3. Launch the run subprocess.
    cmd = [_python_executable(), "-m", "plm.experiments.run_entry", str(cfg_path)]
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=eval_cfg.subprocess_timeout,
            env={**os.environ},          # inherit (so PYTHONPATH / SLATE_API_KEY / etc. propagate)
            check=False,
        )
        elapsed = time.time() - started
        rc = proc.returncode
        # Keep only the last few KB of stderr/stdout for the ledger — full logs
        # live in run_dir/error.txt and the run subprocess's own files.
        tail_blob = (proc.stdout[-2000:] if proc.stdout else "") \
                  + (("\n[stderr]\n" + proc.stderr[-2000:]) if proc.stderr else "")
    except subprocess.TimeoutExpired as e:
        elapsed = time.time() - started
        rc = -1
        tail_blob = f"[evaluate] subprocess TimeoutExpired after {eval_cfg.subprocess_timeout}s: {e}"

    # 4. Score the resulting run_dir. `verifier_configured` is exact, not
    #    inferred — we know whether THIS candidate shipped a non-no-op verifier
    #    because we authored it. We treat "any non-empty verifier source" as
    #    configured (the no-op verifier IS configured; it just doesn't act).
    verifier_configured = bool((candidate.verifier or "").strip())
    score = score_run(run_dir, verifier_configured=verifier_configured)

    return {
        "candidate_id": candidate.id,
        "stage": eval_cfg.stage,
        "run_dir": str(run_dir),
        "metaparam_dir": str(metaparam_dir),
        "score": score,
        "subprocess": {
            "returncode": rc,
            "elapsed_s": round(elapsed, 3),
            "tail": tail_blob,
        },
    }
