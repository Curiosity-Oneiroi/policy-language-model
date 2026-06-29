"""Subprocess entrypoint for ONE optimization experiment (the GEPA recipe loop).

    python -m plm.experiments.optimize_entry <config.json>

Mirrors `run_entry.py`'s status contract but drives `plm.optimizer.run_experiment`
(a named recipe E1..E10) instead of a single PLM run. The recipe spawns MANY
per-candidate PLM subprocesses (one `run_dir` each, under this experiment dir);
this module normalizes the optimizer's population/genealogy/per-candidate data
into stable artifacts the API/frontend consume WITHOUT ever parsing gepa internals:

  run.json                 — kind="optimization" + status (starting|running|done|error)
                             + config + timing  (SAME contract RunManager polls)
  population.jsonl         — one normalized row per evaluated candidate. Emitted
                             LIVE (preliminary rows, appended as candidate run_dirs
                             complete) by a poller thread, then fully REWRITTEN with
                             finalized genealogy after the optimizer returns.
  pareto_front.jsonl       — one row per generation: {gen, members:[candidate_id...]}
  optimization_result.json — best candidate id + genome + scalar/elo + summary

Experiment-dir layout (everything UNDER runs_dir/<run_id>/):
  metaparams/<cid>/        — each candidate's materialized genome (system_prompt.md
                             + verifier.py + frozen policies/)   [made by the optimizer]
  candidate_runs/<stage>_<cid>/  — each candidate's PLM run_dir (run.json /
                             trajectory.json / simulates.jsonl / games/)  [optimizer]
  workspaces/<cid>/        — per-candidate disposable workspace roots   [optimizer]
  gepa_state/, gepa_ledger.json, gepa_result.json (or island_ledger.json for E10)
                             — raw optimizer output (output_dir=exp_dir)

Sourcing of the normalized fields (see also scorer.py / gepa core/result.py):
  * scalar / vector (6 shaped metrics)  : score_run(candidate run_dir)
  * per_instance {opponent: winrate}    : score_run -> raw["per_instance"]
                                          (fallback: GEPAResult.val_subscores[i])
  * est_elo                             : score_run -> raw["proxy_elo"]["best_est_elo"]
  * genome shas                         : sha1 of metaparams/<cid>/{system_prompt.md,verifier.py}
  * parent_id / gen / accepted / on_pareto_front / method
        E1..E9 : GEPAResult.candidates (discovery order == index), .parents
                 (parent indices), .per_val_instance_best_candidates +
                 .per_objective_best_candidates (the frontier).  method is the
                 recipe's proposer label ("seed" when parentless, "merge" when
                 >1 parent).  gen is genealogy depth from the seed.
        E10    : best-effort from island_ledger.json (best_hash/recipe/round) +
                 the candidate run_dirs (see _reconcile_island).

generations: gepa's budget knob is `max_metric_calls`, NOT a generation count, so
for E1..E9 `generations`/`population_size` are advisory only (recorded in config,
not forwarded — run_gepa has no such kwarg).  For E10 `generations` maps to the
island model's `island_rounds`.
"""
from __future__ import annotations

import hashlib
import json
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---- proposer/method labels per recipe ------------------------------------- #
_PROPOSER_LABEL = {
    "E1": "GEPA-reflective", "E2": "EvoPrompt-GA", "E3": "GEPA+PE2",
    "E4": "PromptBreeder", "E5": "PromptBreeder", "E6": "TextGrad",
    "E7": "ContraPrompt", "E8": "PromptBreeder", "E9": "ContraPrompt",
    "E10": "island",
}


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _sha(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def _read_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# ===========================================================================
# Normalization helpers (shared by the live poller AND the post-hoc reconcile)
# ===========================================================================

def _genome_shas(metaparam_dir: Path) -> Dict[str, str]:
    sp = (metaparam_dir / "system_prompt.md")
    vf = (metaparam_dir / "verifier.py")
    return {
        "system_prompt_sha": _sha(sp.read_text(encoding="utf-8")) if sp.is_file() else "",
        "verifier_sha": _sha(vf.read_text(encoding="utf-8")) if vf.is_file() else "",
    }


def _score_candidate(run_dir: Path) -> Tuple[float, Dict[str, float], Dict[str, float], Optional[float]]:
    """Re-score a candidate run_dir -> (scalar, vector, per_instance, est_elo)."""
    from plm.optimizer.scorer import score_run
    sc = score_run(run_dir)
    raw = sc.get("raw") or {}
    per_instance = raw.get("per_instance") or {}
    est_elo = ((raw.get("proxy_elo") or {}).get("best_est_elo"))
    return float(sc.get("scalar") or 0.0), (sc.get("vector") or {}), per_instance, est_elo


def _row(*, step: int, gen: Optional[int], candidate_id: str, parent_id: Optional[str],
         method: str, accepted: bool, on_pareto_front: bool,
         run_dir: Path, metaparam_dir: Path,
         per_instance_fallback: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    scalar, vector, per_instance, est_elo = _score_candidate(run_dir)
    if not per_instance and per_instance_fallback:
        per_instance = per_instance_fallback
    return {
        "step": step,
        "gen": gen,
        "candidate_id": candidate_id,
        "parent_id": parent_id,
        "method": method,
        "accepted": bool(accepted),
        "scalar": scalar,
        "vector": vector,
        "per_instance": per_instance,
        "est_elo": est_elo,
        "on_pareto_front": bool(on_pareto_front),
        "genome": _genome_shas(metaparam_dir),
        "run_dir": str(run_dir.resolve()),
        "metaparam_dir": str(metaparam_dir.resolve()),
    }


def _candidate_id_of_run_dir(run_dir: Path) -> Optional[str]:
    """Map a candidate run_dir -> its candidate id (basename of its metaparam_dir),
    robust to the `stage_` run_id prefix the optimizer uses."""
    cfg = _read_json(run_dir / "_optimizer_config.json")
    if cfg and cfg.get("metaparam_dir"):
        return Path(cfg["metaparam_dir"]).name
    return None


# ===========================================================================
# LIVE poller — appends preliminary rows as candidate run_dirs appear
# ===========================================================================

class _PopulationPoller(threading.Thread):
    """Tails `candidate_runs/` for newly-completed candidate run_dirs and appends a
    PRELIMINARY normalized row per candidate (gen/parent/method/accepted/pareto are
    placeholders, finalized by the post-hoc reconcile). Gives the UI live updates."""

    def __init__(self, runs_root: Path, metaparams_root: Path, pop_path: Path,
                 seed_cid: Optional[str], poll_s: float = 2.0) -> None:
        super().__init__(daemon=True)
        self.runs_root = runs_root
        self.metaparams_root = metaparams_root
        self.pop_path = pop_path
        self.seed_cid = seed_cid
        self.poll_s = poll_s
        self._stop = threading.Event()
        self._seen: set[str] = set()
        self._step = 0

    def stop(self) -> None:
        self._stop.set()

    def _sweep(self) -> None:
        if not self.runs_root.is_dir():
            return
        for d in sorted(self.runs_root.iterdir()):
            if not d.is_dir() or d.name in self._seen:
                continue
            run_json = d / "run.json"
            if not run_json.is_file():
                continue                                  # not finished writing status yet
            status = (_read_json(run_json) or {}).get("status")
            if status not in ("done", "error"):
                continue                                  # still running — wait for terminal
            cid = _candidate_id_of_run_dir(d)
            if not cid:
                continue
            self._seen.add(d.name)
            mp_dir = self.metaparams_root / cid
            method = "seed" if cid == self.seed_cid else "pending"
            row = _row(step=self._step, gen=None, candidate_id=cid, parent_id=None,
                       method=method, accepted=False, on_pareto_front=False,
                       run_dir=d, metaparam_dir=mp_dir)
            with self.pop_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._step += 1

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                self._sweep()
            except Exception:
                pass
            self._stop.wait(self.poll_s)
        try:
            self._sweep()                                 # final catch-up
        except Exception:
            pass


# ===========================================================================
# POST-HOC reconcile — finalize population.jsonl + pareto + result
# ===========================================================================

def _gen_depths(parents: List[List[Optional[int]]]) -> List[int]:
    """Genealogy depth (gen) for each candidate index via its parent indices."""
    gens = [0] * len(parents)
    # candidates are in discovery order; parents always precede children.
    for i, ps in enumerate(parents):
        real = [p for p in (ps or []) if isinstance(p, int) and 0 <= p < i]
        gens[i] = 0 if not real else 1 + max(gens[p] for p in real)
    return gens


def _frontier_idxs(result: Any) -> set:
    front: set = set()
    pvi = getattr(result, "per_val_instance_best_candidates", None) or {}
    for s in pvi.values():
        front |= set(s)
    pob = getattr(result, "per_objective_best_candidates", None) or {}
    for s in pob.values():
        front |= set(s)
    bi = getattr(result, "best_idx", None)
    if isinstance(bi, int):
        front.add(bi)
    return front


def _reconcile_gepa(result: Any, recipe: str, runs_root: Path, metaparams_root: Path,
                    stage: str) -> List[Dict[str, Any]]:
    """Build finalized rows from a GEPAResult (E1..E9)."""
    from plm.optimizer.gepa_engine import candidate_hash

    candidates = list(getattr(result, "candidates", []) or [])
    parents = list(getattr(result, "parents", []) or [])
    val_sub = list(getattr(result, "val_subscores", []) or [])
    gens = _gen_depths(parents)
    front = _frontier_idxs(result)
    label = _PROPOSER_LABEL.get(recipe, recipe)

    # cid -> run_dir map (robust to stage prefix).
    cid_to_rundir: Dict[str, Path] = {}
    if runs_root.is_dir():
        for d in runs_root.iterdir():
            c = _candidate_id_of_run_dir(d) if d.is_dir() else None
            if c:
                cid_to_rundir[c] = d

    rows: List[Dict[str, Any]] = []
    accepted_cids: set = set()
    for i, cand in enumerate(candidates):
        cid = candidate_hash(cand)
        accepted_cids.add(cid)
        run_dir = cid_to_rundir.get(cid, runs_root / f"{stage}_{cid}")
        mp_dir = metaparams_root / cid
        ps = [p for p in (parents[i] if i < len(parents) else []) if isinstance(p, int)]
        if not ps:
            parent_id, method = None, "seed"
        elif len(ps) > 1:
            parent_id = candidate_hash(candidates[ps[0]])
            method = "merge"
        else:
            parent_id = candidate_hash(candidates[ps[0]])
            method = label
        pi_fb = {str(k): float(v) for k, v in (val_sub[i].items() if i < len(val_sub) and isinstance(val_sub[i], dict) else [])}
        rows.append(_row(step=i, gen=gens[i], candidate_id=cid, parent_id=parent_id,
                         method=method, accepted=True, on_pareto_front=(i in front),
                         run_dir=run_dir, metaparam_dir=mp_dir,
                         per_instance_fallback=pi_fb))

    # Rejected proposals: evaluated candidate run_dirs not kept by gepa.
    step = len(rows)
    for cid, run_dir in sorted(cid_to_rundir.items()):
        if cid in accepted_cids:
            continue
        rows.append(_row(step=step, gen=None, candidate_id=cid, parent_id=None,
                         method="rejected", accepted=False, on_pareto_front=False,
                         run_dir=run_dir, metaparam_dir=metaparams_root / cid))
        step += 1
    return rows


def _reconcile_island(result: dict, runs_root: Path, metaparams_root: Path,
                      exp_dir: Path, stage: str) -> List[Dict[str, Any]]:
    """Best-effort finalized rows for E10 from island_ledger.json + run_dirs."""
    ledger = (result or {}).get("ledger") or []
    # cid -> (recipe, round) for island BEST candidates; seed cids -> ("seed", round)
    best_meta: Dict[str, Tuple[str, int, Optional[str]]] = {}
    for r in ledger:
        if r.get("event") == "migration_in":
            continue
        bh, rid, rnd = r.get("best_hash"), r.get("recipe"), r.get("round", 0)
        sh = r.get("seed_hash")
        if sh and sh not in best_meta:
            best_meta[sh] = ("seed", rnd, None)
        if bh:
            best_meta[bh] = (rid or "island", rnd, sh)
    best_cids = set(getattr_list(result, "best_per_island_hash"))

    cid_to_rundir: Dict[str, Path] = {}
    if runs_root.is_dir():
        for d in runs_root.iterdir():
            c = _candidate_id_of_run_dir(d) if d.is_dir() else None
            if c:
                cid_to_rundir[c] = d

    rows: List[Dict[str, Any]] = []
    for step, (cid, run_dir) in enumerate(sorted(cid_to_rundir.items())):
        meta = best_meta.get(cid)
        if meta:
            method, gen, parent = meta
            accepted = True
            pareto = cid in best_cids or method != "seed"
        else:
            method, gen, parent = "island-eval", None, None
            accepted, pareto = False, False
        rows.append(_row(step=step, gen=gen, candidate_id=cid, parent_id=parent,
                         method=method, accepted=accepted, on_pareto_front=pareto,
                         run_dir=run_dir, metaparam_dir=metaparams_root / cid))
    return rows


def getattr_list(obj, name) -> list:
    v = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
    return [x for x in (v or []) if x]


def _write_pareto(rows: List[Dict[str, Any]], path: Path) -> None:
    """One row per generation: cumulative frontier members up to that gen."""
    pareto = [r for r in rows if r.get("on_pareto_front")]
    gens = sorted({r["gen"] for r in rows if isinstance(r.get("gen"), int)})
    lines: List[str] = []
    for g in gens:
        members = [r["candidate_id"] for r in pareto
                   if isinstance(r.get("gen"), int) and r["gen"] <= g]
        lines.append(json.dumps({"gen": g, "members": members}, ensure_ascii=False))
    if not lines:                                          # no gen info -> single final front
        members = [r["candidate_id"] for r in pareto]
        lines.append(json.dumps({"gen": 0, "members": members}, ensure_ascii=False))
    _atomic_write(path, "\n".join(lines) + "\n")


def _write_result(rows: List[Dict[str, Any]], result: Any, recipe: str, label: Optional[str],
                  metaparams_root: Path, path: Path) -> None:
    accepted = [r for r in rows if r.get("accepted")]
    pool = accepted or rows
    best = max(pool, key=lambda r: r.get("scalar") or 0.0) if pool else None
    genome = {"system_prompt": "", "verifier": ""}
    if best:
        mp = Path(best["metaparam_dir"])
        sp, vf = mp / "system_prompt.md", mp / "verifier.py"
        genome = {
            "system_prompt": sp.read_text(encoding="utf-8") if sp.is_file() else "",
            "verifier": vf.read_text(encoding="utf-8") if vf.is_file() else "",
        }
    total_metric_calls = (result.get("ledger") and sum(
        (r.get("total_metric_calls") or 0) for r in result["ledger"]) if isinstance(result, dict)
        else getattr(result, "total_metric_calls", None))
    out = {
        "best_candidate_id": best["candidate_id"] if best else None,
        "genome": genome,
        "scalar": best["scalar"] if best else None,
        "est_elo": best["est_elo"] if best else None,
        "vector": best["vector"] if best else None,
        "per_instance": best["per_instance"] if best else None,
        "summary": {
            "recipe": recipe,
            "label": label,
            "num_candidates": len([r for r in rows if r.get("accepted")]),
            "num_evaluated": len(rows),
            "total_metric_calls": total_metric_calls,
        },
    }
    _atomic_write(path, json.dumps(out, indent=2, ensure_ascii=False, default=str))


# ===========================================================================
# main
# ===========================================================================

def _main(cfg: dict) -> None:
    exp_dir = Path(cfg["runs_dir"]) / cfg["run_id"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "games").mkdir(exist_ok=True)

    metaparams_root = exp_dir / "metaparams"
    runs_root = exp_dir / "candidate_runs"
    workspace_root = exp_dir / "workspaces"
    for d in (metaparams_root, runs_root, workspace_root):
        d.mkdir(parents=True, exist_ok=True)
    pop_path = exp_dir / "population.jsonl"

    def write_status(status: str, **extra) -> None:
        _atomic_write(exp_dir / "run.json",
                      json.dumps({"run_id": cfg["run_id"], "kind": "optimization",
                                  "status": status, "config": cfg, **extra}, default=repr))

    # SIGTERM (what RunManager.stop_run / the UI "Stop" sends) -> SystemExit, so the
    # try/finally below runs and RETURNS the borrowed pool slot instead of leaking it.
    # (subprocess.run's own cleanup also kills the in-flight run_entry child on this.)
    def _raise_sysexit(*_a):
        raise SystemExit(143)
    try:
        signal.signal(signal.SIGTERM, _raise_sysexit)
    except (ValueError, OSError):
        pass                                            # not main thread / platform lacks it

    started = time.time()
    write_status("starting", started=started)
    poller: Optional[_PopulationPoller] = None
    _pool = None
    _borrowed: set = set()
    try:
        from plm.optimizer import RECIPES, run_experiment, EvalConfig
        from plm.optimizer.gepa_engine import BackendReflectLM, candidate_hash
        from plm.optimizer.experiment import _default_seed_candidate
        from plm.experiments.config import build_backend

        recipe = (cfg.get("recipe") or "E1").strip().upper()
        if recipe not in RECIPES:
            raise ValueError(f"unknown recipe {recipe!r}; known: {sorted(RECIPES)}")

        backend = cfg["backend"]
        reflection = cfg.get("reflection") or backend
        stage = "alpha"                                    # EvalConfig default

        # Borrow kernel venvs from the SHARED workspace pool (the same one single runs use)
        # when the web backend wired its location into the config — so each candidate reuses
        # a pre-built venv instead of creating one, and pool size caps candidate concurrency.
        # Falls back to per-candidate workspaces under workspace_root when no pool is set.
        acquire_ws = release_ws = None
        pool_root = cfg.get("pool_root")
        if pool_root:
            from .workspaces import WorkspacePool
            _pool = WorkspacePool(pool_root, cfg.get("pool_game_path") or pool_root)
            _pool_wait = float(cfg.get("pool_wait_s", 1800.0))

            def acquire_ws(rid, _p=_pool, _w=_pool_wait):
                a = _p.allocate_blocking(rid, timeout=_w)
                if a:
                    _borrowed.add(a["number"])           # track for stop/crash cleanup below
                return a

            def release_ws(num, _p=_pool):
                _borrowed.discard(num)
                _p.free(num)

        eval_cfg = EvalConfig(
            metaparams_root=metaparams_root,
            runs_root=runs_root,
            workspace_root=workspace_root,
            backend_name=backend["name"],
            backend_model=backend.get("model"),
            backend_base_url=backend.get("base_url"),
            simulate_config=cfg.get("simulate_config"),
            max_turns=cfg.get("max_turns", 5),
            return_budget=cfg.get("return_budget", 2),
            dotenv_path=cfg.get("dotenv_path"),
            # Per-CELL timeout: NONE (no cap). evaluate()'s 60s default kills react_llm /
            # natural_llm sub-agent dispatches on the slow 122B (~30 tok/s), forcing the PLM
            # off its delegate-don't-grind strategy. Unbounded cells let dispatches finish;
            # the per-candidate subprocess_timeout below is the only safety ceiling now.
            tool_timeout=cfg.get("tool_timeout_s"),     # None unless overridden -> no cell cap
            # Per-candidate hard ceiling (the run_entry subprocess). Generous, because cells
            # are now unbounded and a real delegate-heavy candidate on the slow model + full
            # maia games runs long; this only stops a wedged candidate holding a pool slot.
            subprocess_timeout=float(cfg.get("eval_timeout_s") or 7200.0),
            # Official Elo: a fixed 150-game eval of the delivered policy after RETURN
            # (tagged final_eval) — the scorer reads that, not the model's own peeks.
            final_eval_games=int(cfg.get("final_eval_games") or 150),
            stage=stage,
            acquire_workspace=acquire_ws,
            release_workspace=release_ws,
        )
        rlm = BackendReflectLM(build_backend(
            reflection["name"], model=reflection.get("model"),
            base_url=reflection.get("base_url"), dotenv_path=cfg.get("dotenv_path"),
        ))

        seed_cid = candidate_hash(_default_seed_candidate())

        # gepa's budget is max_metric_calls; for E10 `generations` -> island_rounds.
        max_metric_calls = int(cfg.get("max_metric_calls", 10))
        generations = int(cfg.get("generations", 2))
        recipe_kwargs: Dict[str, Any] = {}
        if recipe == "E10":
            recipe_kwargs["island_rounds"] = generations

        # Start the live poller BEFORE the loop so the UI updates during the run.
        pop_path.write_text("", encoding="utf-8")
        poller = _PopulationPoller(runs_root, metaparams_root, pop_path, seed_cid)
        poller.start()

        write_status("running", started=started, recipe=recipe)

        result = run_experiment(
            recipe,
            eval_cfg=eval_cfg,
            output_dir=exp_dir,
            reflection_lm=rlm,
            max_metric_calls=max_metric_calls,
            **recipe_kwargs,
        )

        if poller:
            poller.stop()
            poller.join(timeout=10)

        # Post-hoc reconcile -> finalized population/pareto/result.
        if isinstance(result, dict):                       # E10 island summary
            rows = _reconcile_island(result, runs_root, metaparams_root, exp_dir, stage)
        else:                                              # GEPAResult (E1..E9)
            rows = _reconcile_gepa(result, recipe, runs_root, metaparams_root, stage)

        _atomic_write(pop_path, "".join(
            json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        _write_pareto(rows, exp_dir / "pareto_front.jsonl")
        _write_result(rows, result, recipe, cfg.get("label"),
                      metaparams_root, exp_dir / "optimization_result.json")

        write_status("done", started=started, ended=time.time(), recipe=recipe,
                     num_candidates=len([r for r in rows if r.get("accepted")]),
                     num_evaluated=len(rows))
    except BaseException as e:
        if poller:
            poller.stop()
        _atomic_write(exp_dir / "error.txt", traceback.format_exc())
        write_status("error", started=started, ended=time.time(),
                     error=f"{type(e).__name__}: {e}")
        raise
    finally:
        # Backstop: return any pool slots still borrowed by in-flight candidates so a Stop
        # (SIGTERM->SystemExit) or crash mid-candidate never leaks them. evaluate() already
        # releases on its own normal/exception path; freeing twice is harmless (idempotent).
        if _pool is not None and _borrowed:
            for _num in list(_borrowed):
                try:
                    _pool.free(_num)
                except Exception:
                    pass


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m plm.experiments.optimize_entry <config.json>", file=sys.stderr)
        sys.exit(2)
    _main(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
