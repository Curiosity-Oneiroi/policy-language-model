# plm.optimizer — provenance

This package drives the **real** [`gepa`](https://pypi.org/project/gepa/) library
(installed `gepa==0.1.1`) over the two-component `{"system_prompt", "verifier"}`
genome of a PLM chess metaparam dir. Every one of the 10 named recipes is a thin
closure over `gepa.optimize(...)` (via `gepa_engine.run_gepa`) — there is **no
from-memory optimizer loop**. The recipes differ only in (a) which `gepa` kwargs
they set and (b) which **real, source-grounded** prompt-optimization method acts
as the proposer.

## The substrate (all recipes share these)

| piece | file | role |
|---|---|---|
| `Candidate` / `seed_candidates` | `genome.py` | the `(system_prompt, verifier)` genome; frozen chess `policies/` ride along via `materialize` |
| `EvalConfig` / `evaluate` | `evaluate.py` | one candidate -> one PLM subprocess (`plm.experiments.run_entry`) -> a scored `run_dir` |
| `score_run` / `SCORE_WEIGHTS` | `scorer.py` | 7 shaped metrics + per-opponent win-rate (`raw["per_instance"]`) |
| `PLMChessAdapter` / `run_gepa` | `gepa_engine.py` | the `gepa <-> PLM` adapter + the `gepa.optimize` driver |
| `mock_reflection_lm` | `gepa_engine.py` | the $0, deterministic, protocol-**dispatching** reflection LM (speaks every proposer's protocol) |
| `PE2_REFLECTION_TEMPLATE` / `frame_with_pe2` | `pe2_prompt.py` | the real PE2 meta-prompt + the helper that frames a custom proposer's call with it |

The 6 rated opponent rungs (`OPPONENT_RUNGS`) are the gepa instances
(trainset == valset). On the **mock** backend a run never reaches `simulate`, so
all per-opponent scores are `0.0` — this is the expected $0 plumbing signal: it
proves the loop runs end-to-end, not that candidates discriminate. (gepa
therefore usually does not *accept* a child into the pool on mock; the accepted
$0 signal is that the real proposer produced a **valid proposal** — a child whose
components differ from the seed and whose verifier parses to exactly one
top-level `def verifier`.)

## The 10 recipes

`build_recipe(recipe_id, reflection_lm, **overrides)` returns a runner closure;
`RECIPES` maps `E1`..`E10`.

| recipe | method backing the proposer | real source (repo/paper) | gepa kwargs |
|---|---|---|---|
| **E1** GEPA-plain | gepa's **default** reflective-mutation proposer | GEPA, Agrawal et al. 2025, **arXiv:2507.19457** (the installed `gepa==0.1.1`) | `template=None`, `frontier_type="instance"`, `use_merge=True` |
| **E2** MO-GA | `EvoPromptProposer(mode="ga")` | EvoPrompt, Guo et al. 2023, **arXiv:2309.08532**; templates vendored at `vendor/evoprompt/` | `custom_candidate_proposer=EvoPromptProposer(...,"ga")`, `frontier_type="objective"`, `use_merge=True` |
| **E3** GEPA+PE2 | gepa default proposer + the **real PE2 meta-prompt** in the template slot | PE2, Ye et al. 2023, **arXiv:2311.05661**; text vendored at `vendor/papers/pe2_2311.05661.txt` | `reflection_prompt_template=PE2_REFLECTION_TEMPLATE`, `frontier_type="instance"`, `use_merge=True` |
| **E4** GEPA+PE2+PromptBreeder | `PromptBreederProposer(base_template=PE2)` | PromptBreeder, Fernando et al. 2023, **arXiv:2309.16797**; vendored at `vendor/promptbreeder/` + `vendor/papers/promptbreeder_2309.16797.txt`. PE2 grammar as above. | `custom_candidate_proposer=PromptBreederProposer(...,base_template=PE2)`, `frontier_type="instance"`, `use_merge=True` |
| **E5** E4 + crossover | same as E4 | PromptBreeder + PE2 (as E4); crossover = gepa's real two-parent **merge** op | identical to E4 (`use_merge=True` **is** the crossover; see "composites" below) |
| **E6** GEPA+TextGrad | `TextGradProposer` | TextGrad, Yuksekgonul et al. 2024, **arXiv:2406.07496**; templates vendored at `vendor/textgrad/` | `custom_candidate_proposer=TextGradProposer(...)`, `frontier_type="instance"`, `use_merge=True` |
| **E7** GEPA+ContraPrompt | `ContraPromptProposer` | ContraPrompt, **arXiv:2604.17937** (id as in the vendored paper); text vendored at `vendor/papers/contraprompt_2604.17937.txt` | `custom_candidate_proposer=ContraPromptProposer(...)`, `frontier_type="instance"`, `use_merge=False` |
| **E8** MO-GA+PromptBreeder+crossover | `PromptBreederProposer(base_template=PE2)` | PromptBreeder + PE2 | `custom_candidate_proposer=PromptBreederProposer(...,base_template=PE2)`, `frontier_type="objective"`, `use_merge=True` |
| **E9** dual "everything" | `ContraPromptProposer` | ContraPrompt | `custom_candidate_proposer=ContraPromptProposer(...)`, `frontier_type="hybrid"`, `use_merge=True` |
| **E10** Island | THREE real gepa runs (E4, E5, E7 configs) as islands + elite migration | island/multi-population EC (Tanese 1989; PromptBreeder §3 "islands"); implemented in `island_engine.py` | each island is a full `run_gepa` with that recipe's kwargs; `island_rounds` (default 2), `per_island_budget` |

`frontier_type` values are exactly those `gepa==0.1.1` accepts:
`Literal['instance', 'objective', 'hybrid', 'cartesian']` (verified against the
installed `gepa.api.optimize` signature). `"hybrid"` is real — E9 uses it.

## How the composites are realized on real gepa (honest semantics)

**The key gepa constraint.** `gepa`'s `reflection_prompt_template` configures
*only* its **default** proposer. A `custom_candidate_proposer` **replaces** the
default proposer entirely, so the template slot is dead for any recipe that sets
a custom proposer (confirmed in `gepa/proposer/reflective_mutation/
reflective_mutation.py`: when `custom_candidate_proposer` is set it is called
directly and the template is never consulted). `run_gepa` therefore **forbids**
passing both at once.

- **"+PE2" on a custom proposer (E4, E5, E8).** Because the template slot is
  unavailable, PE2 is wired **into the proposer** instead: each proposer
  optionally takes `base_template`, and when set it frames its own
  `reflection_lm` call through the real PE2 meta-prompt via
  `pe2_prompt.frame_with_pe2` — substituting PE2's `<curr_param>` with the
  component being mutated and PE2's `<side_info>` with the method's own mutation
  instruction. Both pieces are real and present in the rendered prompt: the PE2
  diagnose->edit **grammar** *and* the method's **mutation** (e.g. PromptBreeder's
  `INSTRUCTION MUTANT:` first-order operator). This is verified: a PE2-framed
  PromptBreeder prompt contains the PE2 collaboration preamble *and*
  `INSTRUCTION MUTANT:`, with no unsubstituted placeholders. We do **not** invent
  a multi-method chain — it is exactly "PE2 framing of the one real method".
  (E3, by contrast, uses gepa's **default** proposer, so it can pass PE2 in the
  native template slot — the "pure" PE2 composite.)
- **"+crossover" (E5, E8).** gepa's real two-parent recombination operator is its
  **merge** (`use_merge=True`). E5 = E4 with merge on (already on in E4), so
  "crossover" is realized as gepa's genuine merge op — there is no separate
  crossover knob. Our standalone `evoprompt_crossover` / `evoprompt_de`
  (`evoprompt_ops.py`) are the EvoPrompt two-parent operators exposed for
  completeness; gepa's loop uses its own merge.
- **dual "everything" (E9).** Realized with gepa's real `frontier_type="hybrid"`
  (instance **and** objective Pareto fronts combined) + merge on + the
  ContraPrompt proposer.
- **Island (E10).** `island_engine.run_island` launches three independent
  `run_gepa` calls (E4/E5/E7 configs) per round, then migrates each island's
  `best_candidate` into a neighbour (ring topology) as the next round's seed,
  for `island_rounds` rounds. One shared `island_ledger.json` carries an
  `island`/`recipe` column per row plus `migration_in` events. Each island is a
  genuine gepa optimization; only the migration topology + shared ledger are
  added here.

## What's real vs adapted

- **Real, verbatim from source:** the PE2 meta-prompt (three components,
  `pe2_prompt.py`), the PromptBreeder mutation-prompts / thinking-styles /
  first-order + hyper-mutation operators, the TextGrad backward/step templates
  (loaded from `vendor/textgrad/`), the EvoPrompt GA/DE/mutation templates
  (verbatim from `vendor/evoprompt/`), and the ContraPrompt Algorithm-1 phases
  (paper text `vendor/papers/contraprompt_2604.17937.txt`). The gepa loop,
  Pareto frontiers, merge, and candidate selection are the **installed** `gepa`
  library — not reimplemented.
- **Adapted (documented at each site):**
  - **ContraPrompt offline pairing relaxation** (`contraprompt_proposer.py`
    module docstring): a gepa `ProposalFn` runs **offline** over an
    already-collected `reflective_dataset`, so Algorithm 1's Phase-1 live
    `SolveWithRetries` cannot be run literally. We substitute a same-context
    pairing over the static records (split by parsed outcome score into
    `tau-`/`tau+` buckets), which preserves Phase 2's contrastive primitive
    verbatim in spirit. Phases 3-4 (rule extraction + input-aware TreeMerge) and
    the paper prompts are used as written; Phase-5 Evaluate/checkpoint is owned
    by gepa.
  - **Verifier AST-guard glue** (all proposers): the `verifier` component is
    python and the PLM genome requires exactly one top-level
    `def verifier(messages)`. Each proposer validates the proposed verifier with
    `ast` and **falls back to the parent verifier** on any parse failure or
    wrong def-count. The papers have no "verifier" component — this guard is our
    contract, not theirs.
  - **EvoPrompt single-candidate shaping** (`evoprompt_ops.py`): a gepa
    `ProposalFn` is single-candidate, so `EvoPromptProposer` exposes EvoPrompt's
    **mutation** operator (the operator that fits the shape); the two-parent GA
    crossover and multi-parent DE operators are exposed as standalone functions
    and, in the gepa loop, the two-parent role is played by gepa's `use_merge`.
  - **`mock_reflection_lm` dispatch** (`gepa_engine.py`): the $0 deterministic
    reflection LM inspects protocol markers (`[[CONTRAPROMPT::...]]`,
    `INSTRUCTION MUTANT:`, `<IMPROVED_VARIABLE>`, EvoPrompt `<prompt>` shapes) and
    routes to each proposer's own protocol-correct mock responder. It exists only
    to keep verification at $0; the real loop uses `BackendReflectLM`.
  - **Adapter metric-call accounting** (`PLMChessAdapter`): gepa counts metric
    calls per instance, but the adapter **memoizes** one PLM subprocess per
    distinct candidate, so a 6-instance batch costs ONE subprocess while gepa's
    budget still ticks by instance count.

## $0 verification

All 10 recipes are verified on `backend_name="mock"`, `max_turns=3`, tiny budget
(`max_metric_calls≈10`; E10 `island_rounds=1`, tiny per-island budget). For each
recipe the harness asserts: the **real** `gepa.optimize` drove it (result has
`candidates`/`total_metric_calls`, ledger written), **only** the `mock` backend
was called (every `_optimizer_config.json` names `mock`), and the recipe's real
proposer fired and produced a child differing from the seed whose verifier parses
to exactly one top-level `def verifier`. Result: **10/10 PASS**.
