"""Driver — wire a seed candidate + a recipe + an evaluator into one experiment.

`run_experiment` dispatches through `build_recipe`, which returns a closure over
the REAL `gepa.optimize` loop (or, for E10, the island orchestrator). PLM runs
use the real Slate backend (`eval_cfg`), and `reflection_lm` defaults to a real
`BackendReflectLM` over Slate. Pass an explicit `reflection_lm` to use a
different / cheaper reflection model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .evaluate import EvalConfig
from .gepa_engine import (
    COMPONENT_SYSTEM_PROMPT,
    COMPONENT_VERIFIER,
    BackendReflectLM,
)
from .genome import seed_candidates
from .recipes import build_recipe


def _default_seed_candidate() -> Dict[str, str]:
    """The gepa seed candidate dict {"system_prompt", "verifier"}.

    Uses seed B (chess prompt + delegate-or-measure nudge) — the "minimal coach"
    hypothesis — as the single gepa seed. gepa evolves ONE seed candidate; the
    three `seed_candidates()` are the legacy multi-seed population, of which B is
    the natural starting point (non-trivial verifier, unmodified prompt).
    """
    seeds = seed_candidates()
    seed = next((c for c in seeds if c.id == "seedB_nudge"), seeds[0])
    return {
        COMPONENT_SYSTEM_PROMPT: seed.system_prompt,
        COMPONENT_VERIFIER: seed.verifier,
    }


def run_experiment(
    recipe_id: str,
    *,
    eval_cfg: EvalConfig,
    output_dir: Path,
    reflection_lm: Optional[Callable[[str], str]] = None,
    seed_candidate: Optional[Dict[str, str]] = None,
    max_metric_calls: int = 10,
    **recipe_kwargs: Any,
) -> Any:
    """Run one named experiment ('E1'..'E10') end-to-end on the REAL gepa loop.

    Wires `build_recipe(recipe_id, reflection_lm)` -> a runner closure, then
    invokes it with the seed candidate + eval_cfg + run_dir.

    Args:
      recipe_id:   one of 'E1'..'E10'.
      eval_cfg:    EvalConfig the adapter's evaluator uses (backend, run knobs,
                   metaparam/run roots). Defaults to the real Slate backend.
      output_dir:  run_dir for ledgers/gepa_state.
      reflection_lm: gepa LanguageModel callable. Defaults to a real
                   `BackendReflectLM` over the Slate backend.
      seed_candidate: gepa candidate dict; defaults to seed B.
      max_metric_calls: gepa budget (per island for E10).
      recipe_kwargs: forwarded (e.g. island_rounds/per_island_budget for E10).

    Returns the recipe's result (gepa `GEPAResult`, or the island summary dict
    for E10).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if reflection_lm is not None:
        rlm = reflection_lm
    else:
        # Real reflection LM over the SAME backend the candidate runs use (vLLM by
        # default — local, no per-token cost). base_url/model come from eval_cfg.
        from plm.experiments.config import build_backend
        rlm = BackendReflectLM(build_backend(
            eval_cfg.backend_name,
            model=eval_cfg.backend_model,
            base_url=eval_cfg.backend_base_url,
            dotenv_path=eval_cfg.dotenv_path,
        ))
    seed = seed_candidate if seed_candidate is not None else _default_seed_candidate()

    runner = build_recipe(recipe_id, rlm)
    return runner(
        eval_cfg=eval_cfg,
        run_dir=str(output_dir),
        seed_candidate=seed,
        max_metric_calls=max_metric_calls,
        **recipe_kwargs,
    )
