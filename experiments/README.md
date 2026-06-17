# PLM Experiments panel

A research/experimentation web app over PLM + the `game` chess env: bulk-provision a
pool of kernel venvs, launch PLM runs against a chosen metaparam set + fixed eval
conditions, and watch each run unfold — live trajectory, the policies it authors,
win-rate / estimated-Elo over `simulate` calls, and replay of sampled games with the
`policyzero` source as it stood at that point.

NOT part of PLM core — it imports `plm` and `game` and orchestrates them.

## Layout

```
experiments/
  config.py       backend selection from .env (build_backend, available_backends)
  workspaces.py   WorkspacePool: numbered venv pool (bulk create/allocate/shrink)
  metaparams.py   discover metaparam sets under metaparams/
  runner.py       RunManager: allocate workspace -> launch run subprocess -> read artifacts
  run_entry.py    subprocess entrypoint: builds PLM, streams runs/<id>/ artifacts
  app.py          FastAPI control plane + serves the built React app
  web/            React + Vite frontend (build -> web/dist)
```

Data lives OUTSIDE the repo by default (override with env vars):

- `PLM_EXPERIMENTS_DATA`  → `<harness>/experiments_data/` (holds `workspaces/` + `runs/`)
- `PLM_METAPARAMS_ROOT`   → `<repo>/metaparams`
- `PLM_GAME_PATH`         → `<harness>/game`
- `PLM_DOTENV`            → `<harness>/.env`  (your API keys: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …)

Workspaces are **disposable** (lowering the pool size deletes free ones); run artifacts in
`runs/<id>/` are **persistent** (trajectory.json, simulates.jsonl, sampled games, result.json).

## Run it

1. Put your API keys in `<harness>/.env`.
2. Build the frontend (already built into `web/dist`; rebuild after edits):
   ```
   cd policy-language-model/experiments/web && npm install && npm run build
   ```
3. Serve (from the harness root, with the harness venv — `plm` + `game` importable):
   ```
   .venv/bin/python -m uvicorn plm.experiments.app:app --port 8011
   ```
   Open http://localhost:8011.

Dev mode (hot-reload UI): `npm run dev` in `web/` (Vite on :5173, proxies `/api` → :8011)
alongside the uvicorn command.

## Flow

1. **Settings → set "Number of workspaces"** (e.g. 4). The pool provisions that many venvs
   in the background — each gets `game` (editable) + PLM deps (~30–60s the first time, fast
   after, since uv caches). Raising adds more; lowering deletes the highest-numbered free ones.
2. **New experiment** → pick a metaparam set (`chess`), a backend + model, the **fixed eval
   conditions** (opponents / clock / per_move_s / max_moves), seed, max_turns. Launch — it
   allocates a free workspace and starts.
3. **Run detail**: Progress (score + est-Elo vs simulate), Trajectory (live), Policies (the
   genealogy + each policy's source), Games (replay a sampled game + the `policyzero` snapshot).
4. **Compare**: overlay win-rate / Elo curves across runs — the tool for tuning the system
   prompt / metaparams (φ).

## What's verified vs. needs your environment

- **Verified offline**: the backend modules (pool lifecycle, metaparam discovery, backend
  listing, run-create error paths), the `simulate` auto-log format + `est_elo`, the API
  surface (TestClient), and the React build.
- **Needs your `.env` key + `uv`/network to validate end-to-end**: a real PLM run
  (`run_entry` → PLM → kernel) and live workspace provisioning. Launch one from the UI to
  confirm against your models. (The chess `system_prompt.md` is still the placeholder — runs
  execute but won't be smart until it's authored.)
