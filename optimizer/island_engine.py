"""Island-model orchestration over the REAL gepa engine (recipe E10).

The island model (Tanese 1989; standard in evolutionary computation, and the
multi-population variant PromptBreeder itself uses — paper Section 3, "we
distribute the units of evolution... into islands") runs several semi-isolated
populations and periodically MIGRATES elite individuals between them.

E10 runs THREE real gepa optimizations as islands, each configured like a
different single-engine recipe:

    island 0 = E4  (GEPA+PE2+PromptBreeder)
    island 1 = E5  (E4 + crossover via merge)
    island 2 = E7  (GEPA+ContraPrompt)

After each island finishes its per-round budget we MIGRATE: each island's best
candidate (gepa's `best_candidate`) is collected and seeded into the OTHER
islands for the next round (the migrant becomes that island's seed_candidate).
We repeat for `island_rounds` rounds (default 2).

Every island run is a genuine `gepa.optimize` call (via `run_gepa`); nothing
here re-implements the search. The only thing this module adds is the
migration topology + a SHARED ledger (`island_ledger.json`) carrying an
`island`/`recipe` column per row.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .gepa_engine import (
    COMPONENT_SYSTEM_PROMPT,
    COMPONENT_VERIFIER,
    candidate_hash,
    run_gepa,
)
from .pe2_prompt import PE2_REFLECTION_TEMPLATE
from .promptbreeder_proposer import PromptBreederProposer
from .contraprompt_proposer import ContraPromptProposer


# Island configs: (recipe_id, proposer_factory, reflection_prompt_template,
# frontier_type, use_merge). These mirror E4/E5/E7 in recipes.py exactly.
def _island_specs(reflection_lm: Callable[[str], str], recipe_ids: Sequence[str]):
    catalog: Dict[str, Tuple[Optional[Callable[[], Any]], Optional[str], str, bool]] = {
        "E4": (lambda: PromptBreederProposer(reflection_lm, base_template=PE2_REFLECTION_TEMPLATE),
               None, "instance", True),
        "E5": (lambda: PromptBreederProposer(reflection_lm, base_template=PE2_REFLECTION_TEMPLATE),
               None, "instance", True),
        "E7": (lambda: ContraPromptProposer(reflection_lm),
               None, "instance", False),
    }
    specs = []
    for rid in recipe_ids:
        if rid not in catalog:
            raise KeyError(f"island recipe {rid!r} not in {sorted(catalog)}")
        specs.append((rid, *catalog[rid]))
    return specs


def _best_candidate(result: Any) -> Optional[Dict[str, str]]:
    bc = getattr(result, "best_candidate", None)
    if isinstance(bc, dict) and bc:
        return bc
    cands = list(getattr(result, "candidates", []) or [])
    bi = getattr(result, "best_idx", None)
    if cands and isinstance(bi, int) and 0 <= bi < len(cands):
        return cands[bi]
    return cands[0] if cands else None


def run_island(
    *,
    island_recipe_ids: Sequence[str] = ("E4", "E5", "E7"),
    reflection_lm: Callable[[str], str],
    eval_cfg,
    run_dir: str,
    seed_candidate: Dict[str, str],
    island_rounds: int = 2,
    per_island_budget: int = 8,
    seed: int = 0,
    **_ignored,
) -> Dict[str, Any]:
    """Run the E10 island model: `len(island_recipe_ids)` real gepa runs per
    round, with elite migration between rounds.

    Returns a dict summary (also written to `<run_dir>/island_ledger.json`):
        {"rounds": [...], "islands": [...], "best_per_island": [...],
         "ledger": [ {round, island, recipe, best_hash, ...}, ... ]}.
    """
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    specs = _island_specs(reflection_lm, island_recipe_ids)
    n_islands = len(specs)

    # Each island starts from the same seed_candidate; migration overrides it.
    island_seeds: List[Dict[str, str]] = [dict(seed_candidate) for _ in range(n_islands)]
    ledger_rows: List[Dict[str, Any]] = []
    best_per_island: List[Optional[Dict[str, str]]] = [None] * n_islands

    for rnd in range(int(island_rounds)):
        round_bests: List[Optional[Dict[str, str]]] = [None] * n_islands
        for isl, (rid, proposer_factory, template, frontier, use_merge) in enumerate(specs):
            island_run_dir = out / f"round{rnd}" / f"island{isl}_{rid}"
            island_run_dir.mkdir(parents=True, exist_ok=True)
            proposer = proposer_factory() if proposer_factory is not None else None
            result = run_gepa(
                island_seeds[isl],
                eval_cfg=eval_cfg,
                reflection_lm=reflection_lm,
                reflection_prompt_template=template,
                custom_candidate_proposer=proposer,
                use_merge=use_merge,
                frontier_type=frontier,
                max_metric_calls=per_island_budget,
                run_dir=str(island_run_dir),
                label=f"E10 island{isl} {rid} round{rnd}",
                seed=seed + rnd * 100 + isl,
            )
            best = _best_candidate(result)
            round_bests[isl] = best
            best_per_island[isl] = best
            ledger_rows.append({
                "round": rnd,
                "island": isl,
                "recipe": rid,
                "seed_hash": candidate_hash(island_seeds[isl]),
                "best_hash": candidate_hash(best) if best else None,
                "num_candidates": len(getattr(result, "candidates", []) or []),
                "best_idx": getattr(result, "best_idx", None),
                "total_metric_calls": getattr(result, "total_metric_calls", None),
                "system_prompt_len": len(best.get(COMPONENT_SYSTEM_PROMPT, "")) if best else None,
                "verifier_len": len(best.get(COMPONENT_VERIFIER, "")) if best else None,
                "gepa_run_dir": str(island_run_dir / "gepa_state"),
            })

        # MIGRATION: each island's next seed = the best migrant from the PREVIOUS
        # island (ring topology). Falls back to its own best if a neighbor is None.
        if rnd < int(island_rounds) - 1:
            new_seeds: List[Dict[str, str]] = []
            for isl in range(n_islands):
                donor = round_bests[(isl - 1) % n_islands] or round_bests[isl] or island_seeds[isl]
                new_seeds.append(dict(donor))
                ledger_rows.append({
                    "round": rnd,
                    "island": isl,
                    "recipe": specs[isl][0],
                    "event": "migration_in",
                    "migrant_hash": candidate_hash(donor),
                    "from_island": (isl - 1) % n_islands,
                })
            island_seeds = new_seeds

    summary = {
        "island_recipe_ids": list(island_recipe_ids),
        "island_rounds": int(island_rounds),
        "per_island_budget": int(per_island_budget),
        "n_islands": n_islands,
        "best_per_island": best_per_island,
        "best_per_island_hash": [candidate_hash(b) if b else None for b in best_per_island],
        "ledger": ledger_rows,
    }
    (out / "island_ledger.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return summary


__all__ = ["run_island"]
