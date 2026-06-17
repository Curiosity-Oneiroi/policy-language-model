"""plm.experiments — a research/experimentation control plane over PLM + the chess env.

NOT part of PLM core: this package orchestrates PLM runs (it imports `plm` and the
`game` chess env) for the experiments web app. It owns

  * a POOL of pre-provisioned kernel venvs (`workspaces/NNNN/.venv`) that PLM runs
    reuse — so a run never installs `game` itself (`WorkspacePool`),
  * discovery of metaparam sets under a root (`discover_metaparams`),
  * backend selection from a `.env` (`available_backends` / `build_backend`),
  * a subprocess run launcher that streams each run's artifacts into `runs/<id>/`
    (persistent — separate from the disposable workspaces).

The FastAPI control plane (`app.py`) + the React frontend (`web/`) sit on top.
"""

from __future__ import annotations
