"""FastAPI control plane for the PLM experiments panel.

Serves the REST API the React app uses + the built frontend. Paths are derived from
the repo (override via env): the metaparam sets under `<repo>/metaparams`, and a DATA
dir (default `<harness>/experiments_data`) holding the workspace POOL + the persistent
`runs/`. The frontend polls the REST endpoints for live updates.

Run it:  uvicorn plm.experiments.app:app --reload --port 8011
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import available_backends
from .metaparams import discover_metaparams
from .runner import RunManager
from .workspaces import WorkspacePool

REPO = Path(__file__).resolve().parents[1]                       # policy-language-model/
HARNESS = REPO.parent                                            # harness/
DATA = Path(os.environ.get("PLM_EXPERIMENTS_DATA", HARNESS / "experiments_data"))
WORKSPACES_DIR = DATA / "workspaces"
RUNS_DIR = DATA / "runs"
METAPARAMS_ROOT = Path(os.environ.get("PLM_METAPARAMS_ROOT", REPO / "metaparams"))
GAME_PATH = Path(os.environ.get("PLM_GAME_PATH", HARNESS / "game"))
DOTENV_PATH = os.environ.get("PLM_DOTENV", str(HARNESS / ".env"))
SETTINGS_FILE = DATA / "settings.json"
WEB_DIST = Path(__file__).resolve().parent / "web" / "dist"

for _d in (DATA, WORKSPACES_DIR, RUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

_dotenv = DOTENV_PATH if Path(DOTENV_PATH).exists() else None
pool = WorkspacePool(WORKSPACES_DIR, GAME_PATH)
runs = RunManager(RUNS_DIR, pool, dotenv_path=_dotenv)
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ws-provision")


def _provision(numbers: List[int]) -> None:
    for n in numbers:
        try:
            pool.provision(n)
        except Exception:
            pass                                                 # status stays "provisioning" on failure


def _load_settings() -> Dict[str, Any]:
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"num_workspaces": 0}


def _save_settings(s: Dict[str, Any]) -> None:
    SETTINGS_FILE.write_text(json.dumps(s), encoding="utf-8")


app = FastAPI(title="PLM Experiments")


@app.on_event("startup")
def _startup() -> None:
    target = int(_load_settings().get("num_workspaces", 0))
    pool.set_target(target)
    todo = pool.ensure_provisioned()
    if todo:
        _executor.submit(_provision, todo)


# ---- settings + pool -------------------------------------------------------- #
class Settings(BaseModel):
    num_workspaces: int


@app.get("/api/settings")
def get_settings() -> Dict[str, Any]:
    return {**_load_settings(), "pool": pool.status()}


@app.post("/api/settings")
def set_settings(s: Settings) -> Dict[str, Any]:
    _save_settings({"num_workspaces": s.num_workspaces})
    todo = pool.set_target(s.num_workspaces)                      # grow/shrink immediately
    if todo:
        _executor.submit(_provision, todo)                       # provision new ones in the background
    return get_settings()


@app.get("/api/workspaces")
def get_workspaces() -> Dict[str, Any]:
    return pool.status()


# ---- discovery -------------------------------------------------------------- #
@app.get("/api/metaparams")
def get_metaparams() -> List[Dict[str, Any]]:
    return discover_metaparams(METAPARAMS_ROOT)


@app.get("/api/backends")
def get_backends() -> List[Dict[str, Any]]:
    return available_backends(_dotenv)


# ---- experiments / runs ----------------------------------------------------- #
class RunSpec(BaseModel):
    kind: str = "single"                                         # "single" | "optimization"
    metaparam: Optional[str] = None                             # set NAME under METAPARAMS_ROOT (single)
    backend: Optional[Dict[str, Any]] = None                    # {name, model, base_url}
    simulate_config: Optional[Dict[str, Any]] = None
    seed: int = 0
    max_turns: int = 100
    return_budget: int = 5
    task: Optional[str] = None
    tool_timeout: Optional[float] = None
    label: Optional[str] = None
    # optimization-only payload: {recipe, backend{...}, reflection{...},
    # simulate_config, max_turns, return_budget, generations, max_metric_calls,
    # population_size}
    optimization: Optional[Dict[str, Any]] = None


# Short blurbs for the 10 recipes (see optimizer/README.md for full provenance).
_RECIPE_BLURBS: Dict[str, Dict[str, str]] = {
    "E1":  {"title": "GEPA-plain",                   "summary": "gepa's default reflective-mutation proposer; instance-Pareto frontier, two-parent merge on."},
    "E2":  {"title": "MO-GA",                         "summary": "EvoPrompt GA mutation as proposer; objective (metric-Pareto) frontier, merge on."},
    "E3":  {"title": "GEPA+PE2",                      "summary": "default proposer with the real PE2 meta-prompt in the template slot; instance frontier, merge on."},
    "E4":  {"title": "GEPA+PE2+PromptBreeder",        "summary": "PromptBreeder mutation framed by the PE2 grammar; instance frontier, merge on."},
    "E5":  {"title": "GEPA+PE2+PromptBreeder+xover",  "summary": "E4 plus crossover via gepa's real two-parent merge op; instance frontier."},
    "E6":  {"title": "GEPA+TextGrad",                 "summary": "TextGrad textual-gradient (backward + step) proposer; instance frontier, merge on."},
    "E7":  {"title": "GEPA+ContraPrompt",             "summary": "ContraPrompt contrastive-pair rule mining proposer; instance frontier, merge off."},
    "E8":  {"title": "MO-GA+PromptBreeder+xover",     "summary": "PromptBreeder (PE2 grammar) proposer; objective frontier, merge on."},
    "E9":  {"title": "Dual-everything",               "summary": "ContraPrompt proposer; hybrid (instance AND objective) frontier, merge on."},
    "E10": {"title": "Island",                        "summary": "three real gepa islands (E4,E5,E7 configs) with elite migration between rounds."},
}


@app.get("/api/recipes")
def get_recipes() -> List[Dict[str, str]]:
    from plm.optimizer import RECIPES
    return [{"id": rid, **_RECIPE_BLURBS.get(rid, {"title": rid, "summary": ""})}
            for rid in sorted(RECIPES, key=lambda x: int(x[1:]))]


@app.get("/api/runs")
def list_runs() -> List[Dict[str, Any]]:
    return runs.list_runs()


@app.post("/api/runs")
def create_run(spec: RunSpec) -> Dict[str, Any]:
    if spec.kind == "optimization":
        from plm.optimizer import RECIPES
        opt = spec.optimization or {}
        recipe = (opt.get("recipe") or "").strip().upper()
        if recipe not in RECIPES:
            raise HTTPException(400, f"unknown recipe {recipe!r}; known: {sorted(RECIPES)}")
        backend = opt.get("backend") or spec.backend
        if not backend:
            raise HTTPException(400, "optimization run requires a backend")
        try:
            return runs.create_run({
                "kind": "optimization",
                "recipe": recipe,
                "backend": backend,
                "reflection": opt.get("reflection"),
                "simulate_config": opt.get("simulate_config", spec.simulate_config),
                "max_turns": opt.get("max_turns", spec.max_turns),
                "return_budget": opt.get("return_budget", spec.return_budget),
                "generations": opt.get("generations", 2),
                "max_metric_calls": opt.get("max_metric_calls", 10),
                "population_size": opt.get("population_size", 4),
                "label": spec.label,
            })
        except RuntimeError as e:
            raise HTTPException(409, str(e))

    if not spec.metaparam:
        raise HTTPException(400, "single run requires a metaparam set name")
    mp_dir = METAPARAMS_ROOT / spec.metaparam
    if not mp_dir.is_dir():
        raise HTTPException(400, f"unknown metaparam set {spec.metaparam!r}")
    try:
        return runs.create_run({
            "metaparam_dir": str(mp_dir.resolve()),
            "backend": spec.backend,
            "simulate_config": spec.simulate_config,
            "seed": spec.seed,
            "max_turns": spec.max_turns,
            "return_budget": spec.return_budget,
            "task": spec.task,
            "tool_timeout": spec.tool_timeout,
            "label": spec.label,
        })
    except RuntimeError as e:                                    # no free workspace
        raise HTTPException(409, str(e))


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    r = runs.get_run(run_id)
    if r is None:
        raise HTTPException(404, "no such run")
    return r


@app.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str) -> Dict[str, Any]:
    return {"stopped": runs.stop_run(run_id)}


@app.get("/api/runs/{run_id}/games/{sim_idx}/{game_id}")
def get_game(run_id: str, sim_idx: int, game_id: str) -> Dict[str, Any]:
    g = runs.get_game(run_id, sim_idx, game_id)
    if g is None:
        raise HTTPException(404, "no such game")
    return g


# ---- optimization: candidate endpoints ------------------------------------- #
@app.get("/api/runs/{run_id}/candidates")
def list_candidates(run_id: str) -> List[Dict[str, Any]]:
    r = runs.get_run(run_id)
    if r is None:
        raise HTTPException(404, "no such run")
    return r.get("candidates") or []


@app.get("/api/runs/{run_id}/candidates/{candidate_id}")
def get_candidate(run_id: str, candidate_id: str) -> Dict[str, Any]:
    c = runs.get_candidate(run_id, candidate_id)
    if c is None:
        raise HTTPException(404, "no such candidate")
    return c


@app.get("/api/runs/{run_id}/candidates/{candidate_id}/games")
def list_candidate_games(run_id: str, candidate_id: str) -> List[Dict[str, Any]]:
    g = runs.list_candidate_games(run_id, candidate_id)
    if g is None:
        raise HTTPException(404, "no such candidate")
    return g


@app.get("/api/runs/{run_id}/candidates/{candidate_id}/games/{sim_idx}/{game_id}")
def get_candidate_game(run_id: str, candidate_id: str, sim_idx: int, game_id: str) -> Dict[str, Any]:
    g = runs.get_game(run_id, sim_idx, game_id, candidate_id=candidate_id)
    if g is None:
        raise HTTPException(404, "no such game")
    return g


# ---- serve the built React app (SPA) --------------------------------------- #
if (WEB_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")


@app.get("/")
def _index():
    idx = WEB_DIST / "index.html"
    if idx.is_file():
        return FileResponse(idx)
    return {"ok": True, "note": "frontend not built yet (run `npm run build` in experiments/web)"}


@app.get("/{path:path}")
def _spa(path: str):
    if path.startswith("api/"):
        raise HTTPException(404, "not found")
    f = WEB_DIST / path
    if f.is_file():
        return FileResponse(f)
    idx = WEB_DIST / "index.html"
    if idx.is_file():
        return FileResponse(idx)
    raise HTTPException(404, "not found")
