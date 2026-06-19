"""The 10 named experiments — all driven by the REAL `gepa` library.

Every recipe is a thin closure over `gepa_engine.run_gepa(...)` (or, for E10,
`island_engine.run_island`). The differences between recipes are exactly:

  * which gepa kwargs (`frontier_type`, `use_merge`, `reflection_prompt_template`),
  * which REAL custom proposer (`custom_candidate_proposer`), if any, and
  * for "+PE2" composites, whether the proposer is given `base_template=
    PE2_REFLECTION_TEMPLATE` (PE2's diagnose->edit GRAMMAR framing the real
    method's MUTATION — see Task 2 / pe2_prompt.frame_with_pe2).

There is NO from-memory loop here: `gepa.optimize` drives all 10.

`build_recipe(recipe_id, reflection_lm, **overrides) -> callable`. The returned
callable has signature `(*, eval_cfg, run_dir, **kw) -> GEPAResult` and, when
invoked, runs the real gepa loop. `RECIPES` maps 'E1'..'E10' to these factories.

Mapping (see README.md for full provenance):
  E1  GEPA-plain   : default proposer,  template=None,            instance, merge=on
  E2  MO-GA        : EvoPromptProposer(ga),                       objective, merge=on
  E3  GEPA+PE2     : default proposer,  template=PE2,             instance, merge=on
  E4  GEPA+PE2+PB  : PromptBreeder(base_template=PE2),            instance, merge=on
  E5  E4+crossover : PromptBreeder(base_template=PE2),            instance, merge=on
  E6  GEPA+TextGrad: TextGradProposer,                            instance, merge=on
  E7  GEPA+Contra  : ContraPromptProposer,                        instance, merge=off
  E8  MO-GA+PB+xover: PromptBreeder(base_template=PE2),           objective, merge=on
  E9  Dual-every   : ContraPromptProposer,                        hybrid,    merge=on
  E10 Island       : 3 real gepa islands (E4,E5,E7) + migration (island_engine)
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .gepa_engine import run_gepa
from .pe2_prompt import PE2_REFLECTION_TEMPLATE
from .promptbreeder_proposer import PromptBreederProposer
from .textgrad_proposer import TextGradProposer
from .contraprompt_proposer import ContraPromptProposer
from .evoprompt_ops import EvoPromptProposer
from .island_engine import run_island

RecipeFn = Callable[..., Any]


# ---------------------------------------------------------------------------
# Single-engine recipes — each is `run_gepa(...)` with fixed gepa kwargs and
# (optionally) a real custom proposer constructed from the recipe's reflection_lm.
# `overrides`/`**kw` let the driver bump max_metric_calls / seed / etc.
# ---------------------------------------------------------------------------

def _gepa_recipe(reflection_lm: Callable[[str], str], *, label: str,
                 proposer_factory: Optional[Callable[[Callable[[str], str]], Any]],
                 reflection_prompt_template: Optional[str],
                 frontier_type: str, use_merge: bool) -> RecipeFn:
    """Build a closure that runs one gepa recipe.

    `proposer_factory` builds the REAL custom proposer from reflection_lm (or
    None => gepa's default proposer + reflection_prompt_template).
    """
    def run(*, eval_cfg, run_dir, **kw) -> Any:
        proposer = proposer_factory(reflection_lm) if proposer_factory is not None else None
        return run_gepa(
            kw.pop("seed_candidate"),
            eval_cfg=eval_cfg,
            reflection_lm=reflection_lm,
            reflection_prompt_template=reflection_prompt_template,
            custom_candidate_proposer=proposer,
            use_merge=use_merge,
            frontier_type=frontier_type,
            run_dir=run_dir,
            label=label,
            **kw,
        )
    run.recipe_label = label  # type: ignore[attr-defined]
    return run


def _r_E1(reflection_lm, **overrides) -> RecipeFn:
    """E1 GEPA-plain: gepa's DEFAULT reflective-mutation proposer (no PE2
    template), INSTANCE-Pareto frontier, two-parent merge ON. The reference
    GEPA configuration (Agrawal et al. 2025, arXiv:2507.19457)."""
    return _gepa_recipe(
        reflection_lm, label="E1 GEPA-plain",
        proposer_factory=None, reflection_prompt_template=None,
        frontier_type="instance", use_merge=True,
    )


def _r_E2(reflection_lm, **overrides) -> RecipeFn:
    """E2 MO-GA: EvoPrompt GA mutation as the proposer, OBJECTIVE frontier
    (metric-Pareto over the 7 shaped metrics + opponent_winrate via the
    adapter's objective_scores), merge ON (gepa's real two-parent op stands in
    for GA crossover, our `evoprompt_crossover`). EvoPrompt: Guo et al. 2023,
    arXiv:2309.08532."""
    return _gepa_recipe(
        reflection_lm, label="E2 MO-GA",
        proposer_factory=lambda lm: EvoPromptProposer(lm, mode="ga"),
        reflection_prompt_template=None,
        frontier_type="objective", use_merge=True,
    )


def _r_E3(reflection_lm, **overrides) -> RecipeFn:
    """E3 GEPA+PE2: gepa's DEFAULT proposer but with the REAL PE2 meta-prompt
    as `reflection_prompt_template`. Here PE2 plugs straight into gepa's
    template slot (no custom proposer), so this is the "pure" PE2 composite.
    PE2: Ye et al. 2023, arXiv:2311.05661."""
    return _gepa_recipe(
        reflection_lm, label="E3 GEPA+PE2",
        proposer_factory=None,
        reflection_prompt_template=PE2_REFLECTION_TEMPLATE,
        frontier_type="instance", use_merge=True,
    )


def _r_E4(reflection_lm, **overrides) -> RecipeFn:
    """E4 GEPA+PE2+PromptBreeder.

    Composite realized HONESTLY (see Task 2 / pe2_prompt.frame_with_pe2): a
    custom proposer REPLACES gepa's default proposer, so gepa's
    `reflection_prompt_template` slot is dead for this run. We therefore pass
    the REAL PE2 meta-prompt into PromptBreederProposer as `base_template`; the
    proposer frames its first-order mutation call with PE2's diagnose->edit
    grammar (PE2's <curr_param>=component, <side_info>=PromptBreeder's
    first-order mutation instruction). Both pieces are real: PE2 supplies the
    grammar, PromptBreeder (Fernando et al. 2023, arXiv:2309.16797) supplies
    the mutation operator. INSTANCE frontier, merge ON."""
    return _gepa_recipe(
        reflection_lm, label="E4 GEPA+PE2+PromptBreeder",
        proposer_factory=lambda lm: PromptBreederProposer(lm, base_template=PE2_REFLECTION_TEMPLATE),
        reflection_prompt_template=None,
        frontier_type="instance", use_merge=True,
    )


def _r_E5(reflection_lm, **overrides) -> RecipeFn:
    """E5 = E4 + crossover. Same as E4 (PromptBreeder framed by the real PE2
    grammar), and crossover is realized via gepa's REAL two-parent merge
    operator (`use_merge=True`, already on in E4). gepa's merge IS the
    recombination op our `evoprompt_crossover` mirrors — there is no separate
    "crossover" knob to set; documenting that merge-on == crossover-on here.
    INSTANCE frontier."""
    return _gepa_recipe(
        reflection_lm, label="E5 GEPA+PE2+PromptBreeder+crossover",
        proposer_factory=lambda lm: PromptBreederProposer(lm, base_template=PE2_REFLECTION_TEMPLATE),
        reflection_prompt_template=None,
        frontier_type="instance", use_merge=True,
    )


def _r_E6(reflection_lm, **overrides) -> RecipeFn:
    """E6 GEPA+TextGrad: TextGrad's two-step Textual Gradient Descent as the
    proposer (backward gradient + step update, real TextGrad templates).
    INSTANCE frontier, merge ON. TextGrad: Yuksekgonul et al. 2024,
    arXiv:2406.07496."""
    return _gepa_recipe(
        reflection_lm, label="E6 GEPA+TextGrad",
        proposer_factory=lambda lm: TextGradProposer(lm),
        reflection_prompt_template=None,
        frontier_type="instance", use_merge=True,
    )


def _r_E7(reflection_lm, **overrides) -> RecipeFn:
    """E7 GEPA+ContraPrompt: ContraPrompt's contrastive-pair rule mining +
    input-aware tree merge as the proposer (paper Algorithm 1; offline pairing
    relaxation documented in contraprompt_proposer.py). INSTANCE frontier,
    merge OFF. ContraPrompt: arXiv:2604.17937."""
    return _gepa_recipe(
        reflection_lm, label="E7 GEPA+ContraPrompt",
        proposer_factory=lambda lm: ContraPromptProposer(lm),
        reflection_prompt_template=None,
        frontier_type="instance", use_merge=False,
    )


def _r_E8(reflection_lm, **overrides) -> RecipeFn:
    """E8 MO-GA+PromptBreeder+crossover: PromptBreeder (framed by the real PE2
    grammar, as in E4/E5) as the proposer, OBJECTIVE (metric-Pareto) frontier,
    merge ON (the real two-parent crossover op). Combines the MO-GA selection
    axis (E2) with the PromptBreeder mutation operator."""
    return _gepa_recipe(
        reflection_lm, label="E8 MO-GA+PromptBreeder+crossover",
        proposer_factory=lambda lm: PromptBreederProposer(lm, base_template=PE2_REFLECTION_TEMPLATE),
        reflection_prompt_template=None,
        frontier_type="objective", use_merge=True,
    )


def _r_E9(reflection_lm, **overrides) -> RecipeFn:
    """E9 dual "everything": ContraPromptProposer, HYBRID frontier (gepa's real
    `frontier_type="hybrid"` — instance AND objective Pareto fronts combined),
    merge ON. The maximal single-engine configuration."""
    return _gepa_recipe(
        reflection_lm, label="E9 Dual-everything",
        proposer_factory=lambda lm: ContraPromptProposer(lm),
        reflection_prompt_template=None,
        frontier_type="hybrid", use_merge=True,
    )


def _r_E10(reflection_lm, **overrides) -> RecipeFn:
    """E10 Island model: THREE real gepa runs (E4, E5, E7 configs) as islands,
    with best-candidate migration between island_rounds (default 2). One shared
    ledger with an `island`/`recipe` column. Implemented in island_engine.py."""
    default_rounds = overrides.get("island_rounds", 2)

    def run(*, eval_cfg, run_dir, **kw) -> Any:
        # The driver passes `max_metric_calls`; for the island model that is the
        # PER-ISLAND budget unless an explicit per_island_budget is given.
        per_island = kw.pop("per_island_budget", kw.pop("max_metric_calls", 8))
        return run_island(
            island_recipe_ids=("E4", "E5", "E7"),
            reflection_lm=reflection_lm,
            eval_cfg=eval_cfg,
            run_dir=run_dir,
            island_rounds=kw.pop("island_rounds", default_rounds),
            per_island_budget=per_island,
            **kw,
        )
    run.recipe_label = "E10 Island"  # type: ignore[attr-defined]
    return run


RECIPES: Dict[str, Callable[..., RecipeFn]] = {
    "E1": _r_E1, "E2": _r_E2, "E3": _r_E3, "E4": _r_E4, "E5": _r_E5,
    "E6": _r_E6, "E7": _r_E7, "E8": _r_E8, "E9": _r_E9, "E10": _r_E10,
}


def build_recipe(recipe_id: str, reflection_lm: Callable[[str], str], **overrides) -> RecipeFn:
    """Resolve a recipe id ('E1'..'E10') to a constructed runner closure.

    The returned callable runs the REAL gepa loop when invoked with
    `(*, eval_cfg, run_dir, seed_candidate=..., max_metric_calls=..., ...)`
    (E10 takes `island_rounds`/`per_island_budget` instead of seed_candidate).
    """
    rid = recipe_id.strip().upper()
    if rid not in RECIPES:
        raise KeyError(f"unknown recipe id {recipe_id!r}; known: {sorted(RECIPES)}")
    return RECIPES[rid](reflection_lm, **overrides)


__all__ = ["RECIPES", "build_recipe"]
