"""Subprocess entrypoint for ONE PLM experiment run.

    python -m plm.experiments.run_entry <config.json>

Runs in the harness venv (where `plm` + `game` import). Builds a PLM from the config,
runs it on the ALLOCATED workspace, and writes every artifact into `runs/<run_id>/`:

  run.json         — status (starting|running|done|error) + config + timing
  trajectory.json  — the full PLM transcript, rewritten live each round (via on_round)
  result.json      — final answer + metrics
  simulates.jsonl  — appended by the sealed `simulate` (via _PLM_EXPERIMENT_DIR), one row/call
  games/<idx>/*.json — sampled game trajectories (also written by `simulate`)

The run dir lives OUTSIDE the workspace (workspaces are disposable); the kernel inherits
`_PLM_EXPERIMENT_DIR` through os.environ so `simulate` logs into this run dir.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _jsonable(x):
    try:
        json.dumps(x)
        return x
    except Exception:
        return repr(x)


async def _main(cfg: dict) -> None:
    run_dir = Path(cfg["runs_dir"]) / cfg["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "games").mkdir(exist_ok=True)

    # the sealed `simulate` logs here; the PLM kernel inherits this via os.environ
    os.environ["_PLM_EXPERIMENT_DIR"] = str(run_dir.resolve())

    def write_status(status: str, **extra) -> None:
        _atomic_write(run_dir / "run.json",
                      json.dumps({"run_id": cfg["run_id"], "status": status,
                                  "config": cfg, **extra}, default=repr))

    started = time.time()
    write_status("starting", started=started)
    try:
        from plm.plm import PLM, PLMMetaParameters
        from plm.experiments.config import build_backend

        backend = build_backend(
            cfg["backend"]["name"],
            model=cfg["backend"].get("model"),
            base_url=cfg["backend"].get("base_url"),
            dotenv_path=cfg.get("dotenv_path"),
        )
        metaparams = PLMMetaParameters.from_dir(cfg["metaparam_dir"])
        plm = PLM(
            model_backend=backend,
            metaparams=metaparams,
            simulate_config=cfg.get("simulate_config"),
            session_workspace=cfg["workspace"],            # reuse the pool-prepared venv (has game)
            max_turns=cfg.get("max_turns", 100),
            return_budget=cfg.get("return_budget", 5),
            tool_timeout=cfg.get("tool_timeout"),
        )

        def on_round(msgs):
            _atomic_write(run_dir / "trajectory.json",
                          json.dumps({"messages": msgs, "updated": time.time()}, default=repr))

        write_status("running", started=started)
        result = await plm(
            [{"role": "user", "content": cfg.get("task") or "Begin."}],
            on_round=on_round,
        )
        _atomic_write(run_dir / "result.json", json.dumps({
            "answer": _jsonable(result.get("answer")),
            "metrics": _jsonable(result.get("metrics")),
        }, default=repr))
        _atomic_write(run_dir / "trajectory.json",
                      json.dumps({"messages": result.get("messages", []),
                                  "updated": time.time()}, default=repr))
        write_status("done", started=started, ended=time.time())
    except BaseException as e:
        _atomic_write(run_dir / "error.txt", traceback.format_exc())
        write_status("error", started=started, ended=time.time(),
                     error=f"{type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m plm.experiments.run_entry <config.json>", file=sys.stderr)
        sys.exit(2)
    asyncio.run(_main(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))))
