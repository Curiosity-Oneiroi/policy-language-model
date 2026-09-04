# PLM — research nodes and deferred TODOs

Nodes **R1–R11** are research: the theory of PLM, derived from the policy space rather than from
ReAct's failures. Everything after them is deferred harness work (not bugs; LOW-priority
niceties), each noting why it was deferred and what a correct fix needs.

---

## Where everything is written down

*Index of every working document in `lit_brain_storm`, so anyone picking up part of this work can
tell what has been decided and where. Line counts are as of the last edit to this list.
Third-party trees (`OrSelection/.lake/`, `experiment/repos/`, `experiment/data/`) and the
unrelated `visa_process/` are omitted.*

### Theory — the current front

| doc | lines | what it holds |
|---|---|---|
| `policy-language-model/TODO.md` | 2400+ | **This file.** R1–R11: the policy-space derivation of PLM, coverage(θ), the two-policy decomposition, compositionality as training target, the access hierarchy, and the total-retention harness design. Mirrored verbatim at `experiment/harness/policy-language-model/TODO.md` — **edit both**. |
| `experiment/code/meta_reasoning_definition_revision.md` | 260 | **The definition of meta-reasoning**, in the form intended for the paper. Part I: authoring belongs in the definition. Part II: the repaired sufficiency clause, the assembled definition, the access hierarchy as appendix apparatus, and §12's instruction on what goes in the body vs the appendix. Derived from R9. |

### The paper

| doc | lines | what it holds |
|---|---|---|
| `paper_tmp/main_paper.tex` | 337 | The paper. §1–§3, figures, training and related work are the user's; §4 (Experiments) and the appendices were written here. |
| `paper_tmp/PAPER_LEDGER.md` | 191 | Remaining-work ledger by section; the single source of truth for what is left to write. |
| `paper_tmp/plm_master_dictation.md` | 250 | The user's fullest statement of what a PLM is. Authority for intro ¶4, §3 refinements, metrics, training framing. |
| `paper_tmp/plm_section_notes.md` | 211 | §3 content inventory, pre-structuring. Not a draft. |
| `paper_tmp/plm_section_draft.md` | 39 | Early §3 draft fragment. |
| `overleaf_writing_assistant_instructions.md` | 109 | The incremental Overleaf delta-handoff workflow. |

### The authoring experiment

| doc | lines | what it holds |
|---|---|---|
| `experiment/code/e_program_design.md` | 300 | **DRAFT staged clause register** for E1+E2 (one program, two tiers): ladder L0-L4, frozen architectures, task register with barrier hypotheses, steelman + scaling-curve discipline, cost model (~$135 expected / $200 cap), and the sequential pipeline that feeds E3 its references. Freezes after review. |
| `experiment/code/stages.md` | 168 | **The clause document.** ~60 numbered clauses (S1.x–S4.x, E0–E3) governing design, running, extraction and analysis. Frozen. S1.13b/S1.13c are the disclosure and capability-freeze clauses that R10 touches. |
| `experiment/code/measurement.md` | 164 | Pre-registered protocol: eight metric umbrellas, truncation-log-first attribution, verbatim frozen claim sentences. |
| `experiment/code/open_questions.md` | 53 | Live register — A: awaiting decision, B: blocked, C: consequences, D: resolved. |
| `experiment/code/operations.md` | 46 | Start-to-end run order, Phase 0 onward. |
| `experiment/code/FROZEN_HASHES.txt` | 4 | Freeze anchors. |

### Harness reference

| doc | lines | what it holds |
|---|---|---|
| `policy-language-model/metaparams/systemprompt.md` | 76 | The PLM system prompt. |
| `policy-language-model/constraint/README.md` | 132 | The constraint system. |
| `experiment/harness/policy-language-model-leveled/TODO.md` | 383 | Deferred TODOs for the LEVELED variant. |
| `Benchmark/PLM_DEPTH_1_BASE_LongCoT_Mini/README.md` | 112 | The superseded chess/LongCoT run. |
| `Benchmark/.../runs/20260503_105408/TRAJECTORY_ISSUES.md` | 692 | Trajectory pathologies observed in that run. Empirical, still useful. |

### Literature — PLM line

| doc | lines | what it holds |
|---|---|---|
| `plm_literature.md` | 546 | The PLM-adjacent survey. |
| `references_cache.md` | 152 | **Check before re-fetching anything.** Verified papers, quotes, BibTeX, RLM positioning. |
| `literature/README.md` | 24 | Explains that `literature/` supersedes the scattered root-level notes. |
| `literature/program_learning_landscape.md` | 675 | Program learning / program synthesis landscape. |
| `literature/program_learning_length_generalization_survey.md` | 545 | Length generalisation in program learning. |
| `literature/context_capacity_the_L_wall_in_cot_react.md` | 275 | Context capacity — the L wall. Bears on the salience scope note in the definition doc. |
| `literature/causal_thinking_geometries_in_cot_react.md` | 271 | Causal thinking geometries. |
| `literature/self_reflective_strain_in_cot_react.md` | 373 | Self-reflective strain, canonical version. |
| `literature/exploration_in_cot_react.md` | 143 | Exploration, canonical version. |
| `strain_literature_map.md` | 142 | Map over the strain literature. |
| `exploration_in_cot_react_lit.md` | 143 | Root-level working note superseded by `literature/exploration_in_cot_react.md`. |

### Self-reflective strain — a separate research line

*Its own phenomenon, its own paper. Shares failure-analysis vocabulary with PLM but is not part
of this theory.*

| doc | lines |
|---|---|
| `self_reflective_strain_theory.md` | 167 |
| `self_referential_strain_findings.md` | 143 |
| `phenomenon2_self_reflective_strain.md` | 109 |
| `phenomenon2_examples.md` | 99 |
| `strain_research_macro_cot.md` | 137 |

### Other research lines in this workspace

| doc | lines | status |
|---|---|---|
| `mismanaged_forecasters_hypothesis_mfh.md` | 712 | Separate hypothesis, unrelated to PLM. |
| `trust_clarification_survey.md` | 248 | Tunable-trust workshop paper — 11-paper survey. Separate. |
| `or_knapsack_exact_k_survey.md` | 162 | **Scrapped** (wrong encoding). Kept with the `OrSelection/` Lean environment. |

### Session and memory handoff

| path | what it holds |
|---|---|
| `_handoff/README.md` | Why the Linux move happened and what was trimmed. |
| `_handoff/memory/` | 13 memory files + `MEMORY.md` index. The durable state; matters more than the transcripts. |
| `_handoff/memory_curiosity/` | 4 memory files from `Documents/Curiosity`, where the harness was built. `plm-package-shadowing` and `venv-location` matter operationally. |
| `_handoff/sessions/INDEX.md` | 20 session transcripts, identified. |
| `_handoff/curiosity_workspace/plm_sweep_report.md` | The two-pass verified harness bug sweep (2026-06-14). Only surviving copy. |

### Reading order, for someone starting cold

1. `experiment/code/meta_reasoning_definition_revision.md` — what the theory now says.
2. This file, R1–R10 — how it was derived and what is still open.
3. `paper_tmp/main_paper.tex` + `PAPER_LEDGER.md` — where it is going.
4. `experiment/code/stages.md` + `measurement.md` — what will test it.

**The theory has changed materially and recently.** Anything written before this index that
describes sufficiency as "everything must be retained," or uses the *distance* vocabulary with
labels d0–d3, is superseded. The current terms are **held / handled / latent / lost**, and the
current sufficiency clause turns on *significance*, not possession.

---

## R1 — RESEARCH: prove the policy space is strictly richer than the agentic space

*(Research item, not a deferred bug. Blocks the paper's central claim; listed first.)*

**What must be proven:** that the policy space — arbitrary code composed with raw LLM
capabilities — is strictly larger, and meaningfully larger, than the agentic space
(fixed compositions of LLM calls) and the RLM space (recursive fixed-prompt sub-calls).
Concretely: exhibit a task for which an agentic policy is hard, or does not exist, under
stated constraints, while a policy in the full space exists and solves it.

**Why it matters:** the paper currently asserts that authored computations "do the task
better" (Intro ¶4: *"leaves room for neuro-symbolic programs ... that do the task better
than an LLM doing it autoregressively"*) but never argues it. Every supporting argument in
the draft is **expressivity, and only against RLM sub-calls** ("more expressive than an
RLM's programs", "greater expressive power", "a far greater variety of LLM policies than
simple tool-call delegation allows"). Expressivity gives *room*; it never shows the room
contains anything better. Worse, since agentic compositions live *inside* the policy space,
the superset argument is circular: policies trivially contain delegation, so "policies beat
delegation" is uninformative unless the extra room is shown to matter.

**The collapse threat this closes:** delegation-with-root-trajectory-access is already at
least as good as ReAct by simulation (hand the sub-call the root's context and say
"continue" — same output distribution). If agentic composition then captures everything
useful beyond that, learning to author full policies has diminishing returns and PLM is
over-engineered.

**Why it is decidable:** the unanswerable version asks how the policy space maps onto the
space of thinking problems — malformed, since that problem space is uncharacterised. The
answerable version compares two *constructible* subspaces of the same space, so a **single
witness settles non-containment**.

**Design constraints for the experiment:**
- Same model weights on both sides (the space's shape is parameterised by them).
- Depth-1 throughout; fixed trajectory budget.
- Agentic side represented by a pre-registered set of fixed architectures
  (generator-critic, vote-over-k, iterate-until-stable), each written once and applied
  unchanged to every task.

**What will NOT separate the spaces (design trap):** symbolic power. `react_auto` carries a
Python tool, so any task solvable by "a policy that calls a solver" is solvable by "one
sub-agent that calls a solver". Neuro-symbolic access is already inside the agentic
subspace, so SAT-style witnesses prove nothing here.

**Candidate witness class:** tasks where the *amount or shape* of neural work is a function
of a quantity that must first be computed — branch on a solver's result, spawn k sub-calls
where k derives from data, prune a search by a measured bound, retry only the items a
checker flagged. A fixed architecture cannot adapt its own shape to computed values; an
LLM-decided architecture can, but unreliably, and scaffold-level unreliability is exactly
what Section 2 shows degrades.

### The two things that must be shown (sharpened)

Not one claim but two, and they are close but distinct:

**(a) The space is meaningful.** There exist tasks that one policy does better than another —
so *which* policy you get matters — and consequently the **effort required to find a good
policy is itself meaningful**. If every policy were about as good, authoring would be
pointless regardless of how large the space is.

**(b) The space does not collapse onto the delegation or agentic subsets.** This is the
load-bearing one: **half the paper's argument depends on it.** If the useful part of the space
is just agentic systems, then note that many models are already trained to emit agentic
systems, and PLM is diluted to a repackaging.

> **Claim strength: non-empty for certain; density depends on which setting you are in.**
>
> An earlier formulation asked to show the region beyond agentic is *dense*, and a first
> correction called it *sparse*. Both were too flat — **sparsity is setting-dependent**:
>
> - **The authoring subproblem, with meta-reasoning `θ`.** Here the region beyond agentic does
>   look sparse, and that is fine — it is also where we *want* the gap to be small (R8's radius
>   argument). Note why: authoring collapses many tasks onto the same policy, so the space of
>   *authoring* policies is small relative to the space of tasks it serves.
> - **General tasks, with plain `θ`.** Here we **do not know that it is sparse, and it may not
>   be.** The task space is combinatorially large, and a policy is judged by what it does for a
>   task rather than by the task itself, so counting arguments do not transfer from the
>   authoring case. Sparsity here is an open assumption, not an established fact.
>
> And a consideration that pushes toward *density* in the general setting: once token limits
> and computation cost enter, agentic systems fail more often — they remain probabilistic
> ensembles — so the region where authored policies win should grow, not shrink, under
> realistic constraints.
>
> **What R1 must establish is therefore non-emptiness**, which is sufficient for non-collapse
> and is what the argument needs. Density in the general setting is a live possibility worth
> reporting if the evidence supports it, but it is not the target.

**A second, independent route to the same conclusion** (see R8): the reductio. Any argument
that LLM capabilities are local enough to make simple policies suffice would collapse the space
by its own logic — and the collapse is refuted observationally, since agentic systems exist,
are used in production harnesses, and would not be built if plain delegation matched them.

### The regions, and what distinguishes them

Delineated by what we can actually construct with current `θ`:

- **Delegation** — a single sub-call, essentially a ReAct element wrapped in code.
- **Agentic systems** — fixed compositions with genuine advantages of their own: they behave
  like ensembles, which is why they help in judgmental forecasting and similar settings. This
  region holds real meaning and should not be strawmanned.
- **The remainder** — everything else, which uses the capability in ways the other two do not.

The distinction that matters: delegation and agentic composition both work by **putting a
bubble around the capability** — fixed input, fixed output, autonomous inside. The rest of the
space leverages the raw capability directly, without that encapsulation. That is where the
separation witness must come from, not from symbolic power (see the design trap above).

Note for the experiment: agentic systems are **string-in, string-out** by construction, so
constraints are not used on that side of the comparison.

### Correctness first, cost as a second axis

Argue **correctness** first: some policies are strictly better for some tasks, and the space
does not collapse under that measure. Cost — running cost, time — is then a *second axis*
rather than a competing claim. If the space is dense in correctness, adding a cost axis keeps
it dense, and the practical payoff compounds: searching the space yields not only a correct
policy but the cheapest correct variant.


**Ordering note:** this experiment concerns the *space*, not the behaviour, so it does not
depend on the pending redefinitions of meta-reasoning and assimilation and can be designed
before they settle.

## R2 — RESEARCH: the formal framework (problem space, policy space, and the derivation of meta-reasoning)

*(Research item. R1 is claim (1) below; this node is the frame R1 sits in and the
route from that frame to a definition of meta-reasoning.)*

### Definitions

**Thinking problem space `T`.** Problems requiring test-time compute: the "blank room"
class, where the work is thinking rather than perceiving or acting. The sandbox is the
harness itself (the thing that runs the code), so a string qualifies as a problem when it
can be posed and answered inside it. Restricting to `T` is what licenses an LLM-only
basis: architectures without test-time compute are either dominated on these problems or
usable as ordinary code (a classifier is a function call), so they need not be admitted as
separate basis elements.

**The abstract thinking-LLM capability.** One operation: given a sequence `S1`, produce a
continuation `S2`. Optionally, generation under a rule. Note that in our harness the rule
is *not* applied at decoding — constraints are validated after free generation and retried
— so constrained generation is **derived** (generate + code predicate + retry), not
primitive. Logit-level constraining would be a genuinely distinct primitive; we do not use
it. The abstract basis is therefore a single operation plus code.

**Thinking policy space `P(θ)`.** Everything expressible as code composed with that
capability, under weights `θ`. Written parametrically because the space's *shape* depends
on `θ`, not merely its contents.

### The abstract capability, stated fully (appendix material)

For the abstract case the thinking-LLM capability is a **function**: sequence in, sequence
out, spending test-time compute in between. Two further parts belong *inside* the capability
rather than outside it:

- **Tokeniser and detokeniser.** Programs work with strings; the capability works with tokens.
  The conversion is part of the capability, not a wrapper around it.
- **Controlled generation.** The capability admits code that alters how it decodes — a CFG
  constructed from a requested shape, tokens blocked against it, gates that separate thinking
  tokens from answer tokens. In the abstract case some of this is not fully realisable, but it
  is the right place to put it.

The motivation is explicitly **against strings**: string interfaces are poor for code. A
constraint layer sitting *in* the capability lets the calling program state the shape it wants
and receive that shape, rather than parsing prose. In the real implementation constraints are
derived (validate-and-retry) rather than primitive, which is a consequence of moving from the
abstract case to a working harness, not a design preference.

**Placement.** This belongs in an appendix, not the main paper, and not necessarily in v1 —
it would confuse a first reader. Its purpose is threefold: it justifies why the harness has
the shape it has; it lets us ask whether *other* structures exploit the same compact
neuro-symbolic idea; and it gives the space a definition independent of our implementation.

### Length is a property of the capability, not an artefact of LLMs

This must be stated in the abstract case, not deferred to implementation. The capability
cannot map `S1 → S2` for unbounded sequence length. Past some length attention degrades and
the behaviour stops being the thing we named — **if a thinking capability no longer works
after some length, that capability has broken down.** So `L` is intrinsic: it bounds the
abstract capability itself, and therefore bounds anything built from it, including `P1`.

### Time must parameterise the space, or the relation is vacuous

The number of computations is unbounded, and **any computation permitted to run forever
solves any task** — brute-force search terminates eventually. Without a time bound, "there
exists a policy that solves `t`" is true and meaningless.

So the policy space must be parameterised by time as well: `P_T`, policies whose running time
falls within a stated bound. Notes on doing this:

- The capability's own cost can be treated as a constant or a fixed polynomial in sequence
  length; the classification is not sensitive to which.
- Restrict problems to polynomial time, or polynomial with a stated constant — generous enough
  not to distort the question, strict enough to exclude "wait 500 years."
- **Time changes the shape of the space the way `θ` changes coverage.** Both are parameters,
  and the time parameter has to be introduced in this first section so that it propagates into
  every derived question rather than being bolted on later.


### What θ actually parameterises (corrected)

An earlier framing here wrote `P(θ)`, as though the policy space were indexed by weights.
That is wrong and the correction matters.

**The policy space does not depend on `θ`.** A policy is a program: code with call sites.
The set of such programs is fixed no matter which model sits behind the calls. What `θ`
determines is the *denotation* of those call sites, and therefore which policy solves which
problem. Writing it out:

- `P` — the set of policies. **Fixed. Syntactic.**
- `⟦p⟧_θ` — what policy `p` *does*, once `θ` supplies the behaviour at its call sites.
- `Solve_θ ⊆ P × T` — the pairs `(p, t)` where `⟦p⟧_θ` solves `t`.
- `Coverage(θ) = { t ∈ T : ∃p ∈ P with (p, t) ∈ Solve_θ }`.

So the space is invariant; the **mapping** is what moves. Two consequences worth stating in
Section 3:

1. Claims about the policy space being "bigger" (R1) are about `P` and hold independently of
   `θ` — they are structural. Claims about what can be *solved* are about `Coverage(θ)` and
   are not.
2. The best policy for a given problem changes with `θ`, which is precisely why the choice of
   primitives is not arbitrary (below).

### Why the primitive choice is forced, not chosen

`θ` is not a free variable in this work: **we are not training weights.** `θ` therefore ranges
over off-the-shelf models, and those are trained for two behaviours — chain-of-thought
("just think") and ReAct ("think and act in a line"). That is a very small effective set.

This is what forces the primitive choice. `natural_llm` and `react_auto` are not selected
because they span anything formally — `react_auto` is constructible from `natural_llm` + code
— but because they are the two behaviours available `θ` executes *reliably*:

> **the elementary capabilities are the behaviours `θ` executes reliably; everything else
> must be composed.**

The basis is thus **empirical and temporal**, and it is downstream of the training landscape
rather than of our design. Were tree-of-thought or graph-of-thought training to become
standard, the natural primitives would change with it. A weaker model has a smaller
`Coverage` under an identical `P`.

Open: whether anything stronger can be said relating `Coverage(θ1)` and `Coverage(θ2)` — for
instance whether coverage is monotone in any measurable notion of model quality.

### The chain of claims

**(1) `P(θ) ⊋ agentic space`, for any `θ`.** This is R1. Decidable by witness; see that node
for the design trap (symbolic power does not separate) and the candidate witness class
(control flow depending on computed values).

**(2) Is there always a problem in `T` that no policy in `P(θ)` solves, given the basis has
context length `L`?** Analysis so far suggests the question needs restating, because the
obvious form has a trivial answer: code alone is Turing-complete, so `P(θ)` contains a
solving policy for every computable problem — one that ignores the LLM entirely. Expressive
reach is therefore *not* the binding constraint.

> The real limit is **authorability, not expressiveness**: a problem is out of reach when no
> policy that solves it can be *authored* by a bounded-context thinking process.

If an impossibility result exists, it should be sought there — over what an `L`-bounded
author can conceive — not over what the space can express. Restating (2) in that form is
itself part of the work.

**(2b) Is there a `θ` for which coverage is maximal — and should PLM be built around it?**
*Scoped out of the present work: `θ` is fixed to currently available thinking models, so this
question is not live here. Retained as open theory, and as the principled account executor
selection currently lacks.*
Distinct from (2)'s impossibility framing, and previously lost by being folded into it. The
positive question: does `argmax_θ Coverage(θ)` exist, and can it be characterised?

Two things sharpen it:

- **Separate the two coverages.** `Coverage_exists(θ)` — problems for which *some* solving
  policy exists — is only weakly `θ`-dependent, because `P` contains code-only policies that
  solve every computable problem regardless of `θ`. What varies with `θ` is the
  *judgment-requiring* remainder: problems whose solution needs interpretation, language, or
  open-ended reading that code cannot supply. `Coverage_findable(θ, P1)` — problems for which
  the generator actually *produces* a working policy (R4) — is the one that matters, and is
  strongly `θ`-dependent.
- **The practical shadow.** This is not only a theory question. The experiment fixes a single
  executor model for every sub-call, which *is* a coverage choice made by hand. If a
  principled account of coverage-maximal `θ` existed, executor selection would follow from it
  rather than from convenience, and PLM would be built around that `θ` rather than around
  whatever is cheap.

Open: whether coverage is monotone in any measurable notion of model quality; whether a
`θ` maximal for `Coverage_findable` is the same one maximal for `Coverage_exists` (it need
not be — a model that solves more may still be harder to author around, which is exactly R3's
autonomy/componenthood tension).

**(3) Conditional on (1): existence of solving policies.** Granting that `P(θ)` covers a very
large subset of `T`, can we assert that for each problem in that subset a solving policy
exists, and say what "better" means and by how much? Note (2) partly answers the existence
half for free; the open part is the *quantitative* one — better in correctness, at what
trajectory cost, relative to which alternative. Existence claims here are about a **set** of
policies per problem, not a unique one.

**(4) The policy-generating subspace — the crux.** Since we cannot obtain the right policy
directly, we need a policy that *generates* the policy that solves the task. What can be
said about the subset of `P(θ)` whose members reliably generate task-solving policies? Given
(2)'s reframing, this is where the difficulty actually lives: the solving policy exists, but
must be found by a bounded process.

**(5) ReAct as the approximation, and the derivation of meta-reasoning.** A ReAct trajectory
is one point in that policy-generating subspace. It inherits the bound: every LLM capability
has context length `L`, and ReAct is that capability run as a trajectory, so the *author* is
`L`-bounded even though the policies it writes are not. Training that trajectory to author
well **and** to keep its own length minimal is what we mean by meta-reasoning.

> Meta-reasoning is the behaviour that maximises authorable reach under a fixed trajectory
> budget: it must both (a) conceive computations that do the work, and (b) keep its own
> trajectory sufficient and near-minimal so that conception is not itself the bottleneck.

This derives the definition from the policy-space side rather than from ReAct's failure
modes, which is the framing the paper currently lacks — the existing definition is stated
only relative to what ReAct does badly.

### Ordering

(1) is experimental and independent of the pending redefinitions. (2) needs restating before
it can be attacked. (4) is the crux and is where the training programme gets its target.
(5) is the payoff and belongs in Section 3 once (1) has a result.

## R3 — RESEARCH: the inverse question — training θ for componenthood

*(Research item. R1/R2 take `θ` as given and choose primitives to match it. This node asks
the reverse, and is the natural home of the training programme.)*

**The setup so far runs one way.** `θ` is fixed (we do not train), so the primitives are
chosen to match what available weights execute reliably — chain-of-thought and ReAct. The
harness is designed around the model.

**The inverse:** rather than choosing primitives to fit `θ`, train `θ` to fit the policy
space. Can there be a `θ` that works well **as a component in arbitrary policies**, rather
than one that works well as a whole system?

**Why this is a different training target from anything current.** Models today are trained
to be autonomous: to carry a task end to end as a chain of thought, or to drive a loop as a
ReAct agent. Being a good *component* is a different property — the model is embedded inside
a program it did not write, briefed by an author with its own plan, and its output is
consumed by code rather than by a person. Nothing in current training optimises for that.

**What "works with all policies" could mean — candidate properties to pin down:**
- **Reliability under arbitrary briefing.** Behaves as instructed when the brief comes from a
  program rather than from the distributions it was trained on.
- **Predictability.** Its behaviour can be *estimated in advance* by the authoring model.
  This connects directly to the paper's estimation claim: a policy succeeds when its neural
  parts behave as the author estimated. Training `θ` to be predictable therefore makes
  authoring easier for *every* author — a lever on the whole problem rather than on one
  model's skill.
- **Composability.** Does not degrade when embedded — no drift toward taking over the task,
  no ignoring the contract, no silently changing output shape across calls.
- **Honest failure.** Fails in a way the enclosing program can detect, rather than producing
  a plausible wrong answer. (Under our `None`-on-failure semantics, this is what makes the
  difference between a caught failure and a silently propagated one.)

**A sharper form of the question:** is componenthood in tension with autonomy? A model
trained to be maximally reliable inside someone else's program may be worse at driving its
own trajectory, and vice versa. If they trade off, the interesting object is the trade-off
curve, not either endpoint.

**Co-training the two roles.** In our design the same model can occupy both positions — the
root PLM authoring, and the sub-call executing. So a further question: should the two roles
be trained *together*, so that the author's estimates and the component's behaviour converge?
That is a self-consistency objective, and it is a plausible reading of what "training a
meta-reasoner" actually requires: not just teaching the root to author, but co-adapting the
parts so that authored estimates come true.

### Two concrete items from this thread

**Deferred ablation — does the generated policy depend on which `θ` backs the capability?**
Given a trained PLM, vary the model supplied as the sub-call capability and observe whether
the *policies it authors* change. Current models share training patterns (plan first, reluctant
to ask questions), so a trained `P1` will have learned to author around those quirks — the
question is how tightly. Not runnable now (no trained PLM exists); scheduled for after the
project, and noted here so it is not lost.

**"Can a capability be trained to be a better capability?"** Framed as slightly absurd on its
face — the capability already exists, and pointing at it is what defines it — but the question
is real: take a thinking LLM and train it specifically to be a good *component*, maximising
the coverage obtainable when it sits inside policies. The reassuring part is that current
models are already optimised for thinking, which is what the component role needs; they are
simply not optimised for *being embedded*. That is a narrower gap than it first appears, and
makes the question an experiment rather than a research programme.

**Double search.** Note that R4's authoring problem and this node's training problem together
form a search over two things at once: the policy, and the `θ` that policy will call. A
sharper form of the question: does there exist a `θ` that both works well inside authored
policies *and* is good as the authoring model — or must the two roles be filled by different
`θ`?

### The two-`θ` question, recorded in full

Spelling out the sub-option that follows from the above, since it should not be lost even
though **none of it will be attempted in this work**:

- **`θ_author`** — the weights behind `P1`. The training target here is not coverage in
  general but coverage of the *policy-authoring subspace*: given a task, does `P1` land in the
  region of `P` whose members generate policies that solve it? This is the "train `θ` so that
  the subspace good at creating policies is the one most often hit" question.
- **`θ_capability`** — the weights behind the sub-calls inside authored policies. A separate
  training target: be a good *component* (R3's properties — reliability under arbitrary
  briefing, predictability, composability, honest failure), which is a different objective
  from being a good author.

The open questions this raises, all deferred:

1. Can `θ_author` be trained for authoring-subspace coverage specifically, rather than for
   task coverage in general? These are not obviously the same objective.
2. Can `θ_capability` be trained to raise coverage *as a capability*, independent of who is
   authoring around it?
3. **Must they be different `θ`?** A single model filling both roles is what the current
   harness assumes, but nothing established here requires it — and R3's autonomy/componenthood
   tension is a reason to expect they might pull apart.
4. If they are different, is there value in **co-training** them, so the author's estimates and
   the component's behaviour converge (the self-consistency objective noted above)?

**Status: recorded, not scheduled.** These are named so the design space is documented; the
present work fixes both roles to available thinking models and trains neither.


**Status.** Out of scope for the current paper (no training is performed), but this is the
direction the paper's open training programme points at, and it should be named as such
rather than left implicit.

## R4 — RESEARCH: the two-policy decomposition, and whether ReAct is the right generator

*(Research item. Follows R2's correction that `P` is fixed and `Coverage(θ)` is what moves.)*

### The decomposition

Two policies, not one, and they play different roles:

- **`P2`** — a policy in `P` that solves the given task.
- **`P1`** — a policy in `P` that, *given the task*, generates `P2`.

Both execute under `θ` drawn from the same constrained set (current thinking models: CoT and
ReAct trained; see R2). The distinction matters because the claims attach differently:

- `Coverage(θ)` is the **theoretical shadow** of the whole policy space under `θ` — every
  problem for which *some* `P2` exists. It is large, and it is not what we get.
- What `P1` actually achieves is a **subset** of that: the problems for which `P1`, handed the
  task, in fact produces a working `P2`.

> **The training goal, stated precisely:** bring `P1`'s achieved coverage as close as possible
> to `Coverage(θ)`. The gap between them is exactly what training must close, and it exists
> because the solving policy is *findable in principle* but must be *found by a bounded
> process* (R2, claim 2).

### Why `P1` is a ReAct trajectory, and why that argument is weak

*(Superseded by R8, which gives the real argument — flexibility, applicability, simplicity — and the locality trade-off it costs. Kept here for the record of how the choice was first reached.)*

The route taken was a walk, not a derivation:

1. Start with the simplest generator: a policy that thinks once and emits a policy (natural).
2. ReAct is better, because it carries state — it can author, observe, and revise.
3. Settle there.

That is a plausibility argument, not an optimality one. It was also chosen for a *practical*
reason: `P2` is restricted to CoT/ReAct-shaped `θ`, so choosing `P1` from the same family
minimises how far training has to drift. Convenient, but not a justification.

**Open question, plausibly a paper on its own:** does there exist a `P1` — some other policy
in `P` — whose achieved coverage strictly exceeds that of the ReAct trajectory? If such a
generator exists, *that* is what a PLM should primarily be trained toward, and the current
choice is a local optimum reached by convenience. This question should be posed explicitly
rather than left implicit in the choice.

Note the asymmetry worth preserving: a *golden* `θ` would not need ReAct at all — it could
emit the right `P2` in one shot. ReAct is chosen because we are reasoning about **realistic**
`θ`, where observation and revision buy correctness. The choice is therefore contingent on
model quality, and would change if `θ` improved enough.

### The question, stated formally

Collecting the parameters (time on the policy side, `L` on the trajectory side):

> **Given a problem from the thinking problem space, can `P1` author policies from the
> time-bounded policy space `P_T`, such that the authoring trajectory stays under `L` and each
> authored policy runs within `T`?**

Two notes on what this does and does not assume. It does **not** claim `P1` can reach all of
`P_T` — that is a matter of training and cannot be asserted here. And it says nothing about
pacing: `P1` may author a *set* of policies across rounds; only the total trajectory length is
constrained.

### `L` is the only binding constraint

Of the two parameters, the time bound on `P2` is not really a constraint — polynomial time
(or polynomial with a stated constant) is already generous enough that almost nothing useful
is excluded. **`L` is the real one**, and it binds for a specific reason: the usual escapes do
not work.

- Summarisation loses information (already argued in the paper's ¶2).
- Trajectory manipulation cannot be done safely, because **we cannot predict which earlier
  content the model will need to attend to later.** Some removals are safe; identifying them
  in general is not something any current system can do, so we do not remove.

So the trajectory must simply fit within `L`, and the whole difficulty of `P1` concentrates
there.

### Authoring must be local

A further requirement on `P1`, independent of `L` and prior to it: **authoring a given policy
should take few rounds, not many.** If fifty rounds go into producing one policy, the
arrangement is not worth its cost — ReAct's round structure is not buying anything at that
point, and the advantage claimed for authoring evaporates. The ability to author must be
*internalised* in the model rather than reconstructed by iteration, so that deciding to build a
particular policy is a local act.

**Why locality is the right target, not merely the cheaper one.** The deeper reason is that
**the ability to author is elementary — it is a CoT-level property of the model, not something
ReAct supplies.** ReAct adds rounds to think in; it adds no new capability for conceiving a
computation. So training a model to author is training the *model's own thinking* to arrive at
policies, and the natural consequence is that fewer rounds are needed, because the ability now
sits inside the capability rather than being reconstructed by iteration. Minimising rounds is
therefore not a cost-saving imposed from outside; it is what success at training this ability
*looks like*.

Two readings of a `P1` that burns fifty rounds on one policy, and both matter:

1. **The capability is not internalised.** The model is using rounds as a substitute for an
   ability it should hold directly — iterating toward something it ought to conceive.
2. **`P1` may be the wrong shape.** If the rounds are genuinely needed, that is evidence
   against the trajectory design itself, and the response is to change `P1` rather than to
   spend more of `L` on it.

This is also what makes the ReAct choice for `P1` non-trivial rather than automatic: if
authoring were local enough, fewer rounds would be needed, and the argument for a trajectory
at all would weaken.

### Related open question

**How much does ReAct actually increase coverage**, once the model already knows how to author
`P2` well? If a trained model can emit the right policy in one shot, the rounds add little,
and the case for a trajectory-shaped `P1` rests on how far that is from true. This belongs with
the first experiment.


### The bound that forces meta-reasoning

`P1` is a trajectory, so it inherits the context length `L` of the underlying capability. It
can go on authoring and revising policies only until that budget is consumed. This is not an
incidental limitation — it is the same bound the whole paper is about, arriving at the
generator rather than at the solver.

So the requirement on trained `θ` has two parts, and they are not separable:
- **(a)** it must author well — produce `P2` that works;
- **(b)** it must do so within `L` — keeping its own trajectory short enough that authoring
  does not exhaust the budget before the task is served.

That conjunction is what we mean by meta-reasoning, derived here from the generator's
position rather than from ReAct's failure modes.

## R5 — RESEARCH: repairing "sufficient and minimal" — the enough-information problem

*(Research item. The current definition of meta-reasoning is misleading on both halves; this
node records why and proposes a repair.)*

### Why "minimal" as written is misleading

The definition says the behaviour keeps the trajectory *near-minimal*. Read literally, the
training target becomes "stay under `L`" — but that is trivially achievable by truncation or
tailing, and achieves nothing. Minimality is not the objective.

> **Minimality is instrumental, not terminal.** The target is to keep the trajectory short
> *while doing the task* — brevity purchased at the cost of the task is not the behaviour we
> want, and brevity achieved by dropping the tail is not brevity at all.

The definition needs rewriting so that this is unambiguous, otherwise the obvious degenerate
optimum is exactly what an optimiser would find.

### Why "sufficient" as written is too strong

Sufficiency currently asks for *full* information — that nothing added from off-trajectory
work would significantly increase what the model knows. But full information is neither
achievable nor necessary. What is actually wanted is **enough** information. Defining
"enough" is the open problem.

### Correctness cannot appear in the definition at all

Sharper than "correct and minimal is unsatisfying": **correctness is disqualified from the
definition by the form definitions of this kind take.** Two precedents make the point.

- **Thinking LLMs** are defined as models that *spend more compute thinking* — test-time
  compute. Not "think in a way that turns out correct." Correctness is nowhere in the
  definition, and that is exactly why the definition survived and travels: it names a
  mechanism, applies to any kind of thinking, and composes.
- **RLMs** are defined as *decomposing so that out-of-distribution problems become
  in-distribution*. Not "decomposing correctly." Training toward that definition yields
  correctness, but correctness is the consequence, never the content.

So the definition of meta-reasoning must name **the disposition that produces correctness**,
not correctness itself. "Correct and minimal" fails this test on the first conjunct, and it is
why the definition reads as a score rather than a behaviour.

Corollary: **meta-reasoning is a post-optimisation description, not an optimisation
objective.** We may well *train* on correctness and minimality; what we are defining is the
behaviour that exists once that training has succeeded. Conflating the two is what produced
the current wording.

### "Sufficient" errs by being too wet

The current sufficiency clause — that no information added from off-trajectory work would
increase what the trajectory holds — has the right instinct and the wrong setting.

What it gets right: it argues for **control**, and it points away from dryness. A policy fired
blindly, its answer returned unread and trusted, is not meta-reasoning: it forfeits precisely
the reactive advantage that having a trajectory buys, and gives up the coverage `P1` has over a
one-shot author.

What it gets wrong: as stated it demands the trajectory hold *everything*, which pushes to the
opposite extreme. **Meta-reasoning is wet when it chooses to be, not maximally wet.** The
behaviour we want includes authoring an *assimilation policy* — rather than reading everything a
computation returned, plan what to read and have a policy extract it. That is a decision about
how wet to be at this step, and the definition should permit it rather than legislate one end
of the axis.

So sufficiency is at best a poor formulation and at worst names only a *subset* of what is
wanted, mistaking one point on the dry/wet axis for the behaviour itself.


### Proposed repair: sufficiency as the accuracy of the author's projection

The route out, from the authoring side:

**When a policy is authored, its author also projects what it will return.** That projection
is not a guess about an unknowable future — the author knows the goal, knows the shape it
built, and forms an expectation about what will come back. Assimilation is then designed
*against that projection*: the return is shaped to carry what the author expected to need.

This gives a definition that does not require foresight:

> The trajectory is **sufficient** when the author's projection of what it would need was
> correct. Sufficiency is therefore not an absolute property of information content, but the
> **accuracy of an estimate** — and it is validated by outcome, not by omniscience.

Two things fall out of this immediately:

1. **The failure mode is explained, not just described.** An author who never considers that
   its policy might error will not author assimilation that captures errors — so when one
   occurs, the trajectory is silently insufficient. The information gap is downstream of a
   projection gap. This is exactly the case where a bad estimate produces a confidently wrong
   trajectory.
2. **The circularity is only apparent.** A self-set standard would be vacuous if it were also
   self-graded — but the projection is graded externally, by whether the task succeeds. An
   author with a bad projection meets its own standard and still fails, which is precisely
   what makes the definition bite.

### Consequence: sufficiency and estimation are the same quantity

This collapses two things the paper currently treats separately. The estimation claim
(Intro: *a policy succeeds when it behaves as the PLM estimated it would*) and the
sufficiency requirement turn out to be one property viewed from two sides — the author
estimates what the computation will do and what it will need back, and both authoring quality
and assimilation quality are consequences of that single estimate being right.

If that holds, Section 3 should present them together rather than as separate pillars, and
the training target simplifies accordingly: **train estimation, and both halves follow.**

### Open

- Whether "enough information" admits a definition independent of the author's projection at
  all, or whether projection-relative is the only non-vacuous form.
- How to state minimality so the degenerate optimum (truncation) is excluded by construction
  rather than by intent.
- Whether sufficiency-as-estimation-accuracy can be *measured* separately from task success —
  if not, it is a definition with no independent empirical handle, which is acceptable for
  theory but weak for evaluation.

## R6 — RESEARCH: compositionality as the training target (the invariant formulation)

*(Research item. This is the answer to R4's closing question — what property of `θ` we name
meta-reasoning — and it resolves R5's open item about excluding the degenerate optimum.)*

### The training problem that forces this

We cannot train against the objective directly. The problem space `T` is too large to
enumerate, and `Coverage` is not a trainable target — you cannot present a model with "the set
of problems you should be able to solve." Whatever gets trained has to be a **property that
generalises across problems**, not a fitted mapping over them.

### The property: composition

`P1` does not author one policy. Given a task it authors a **sequence** of policies, running
and assimilating each, continuing until the task is served. The thing that determines how far
that sequence can run is whether each cycle leaves the trajectory in a state from which the
next cycle is possible.

> **The invariant.** After each author → run → assimilate cycle, the trajectory is
> **(a) sufficient** — it carries what is needed to decide the next step — and
> **(b) minimal** — enough budget remains that there *is* a next step.
>
> **Meta-reasoning is the property of `θ` that preserves this invariant across cycles.**

### Why this formulation does work the previous ones could not

**It explains why both halves are required, rather than asserting it.** Sufficiency alone
lets the trajectory continue but exhausts `L`, so the sequence is short. Minimality alone
leaves budget but destroys what the next step needs, so the sequence stops being productive.
Only the conjunction composes — which is exactly the chicken-and-egg relation between the two
halves, now derived rather than posited.

**It excludes the degenerate optimum by construction** (R5's open item). Truncation and
tailing satisfy minimality and violate sufficiency, so they fail at the *next* cycle. The
invariant rules them out without needing "while doing the task" bolted on as an intent — the
failure is mechanical and immediate, one step later.

**It is trainable, because it is local.** The invariant is a per-cycle property, checkable
without reference to the problem space. A model that maintains it on the problems it sees
maintains it on problems it has not seen, because the invariant says nothing about the
problem — this is what buys generalisation, and it is why a compositional property is the
right training target when the objective itself is untrainable.

**It restates reach as a consequence, not a goal.** How far `P1` gets is how many cycles it
can chain; how many cycles it can chain is how well the invariant is preserved. So R4's
"maximise achieved coverage under fixed `L`" and this invariant are the same requirement, the
second being the trainable form of the first.

### Where it sits relative to the paper's existing derivation

Origin matters here and both should be kept. The paper's current definition is derived
**from ReAct's failures** — trajectories grow, degrade, and cannot delete. This node derives
the same behaviour **from the policy space**: `P1` is a ReAct trajectory by choice (R4), it
authors sets of policies because policies are more expressive, and the property that lets it
keep authoring is the invariant above. The two derivations should converge in Section 3
rather than one replacing the other; agreement from independent starting points is itself
evidence the definition is the right one.

### Open

- Stating the invariant precisely enough to be an objective: what "sufficient to decide the
  next step" means operationally, given R5's proposal that sufficiency is the accuracy of the
  author's projection.
- Whether the invariant can be *observed* per cycle during training, or only inferred from
  whether the sequence eventually succeeded — the difference between a dense and a sparse
  signal, and a real determinant of whether this is trainable in practice.
- Whether preserving the invariant is genuinely one property or two that must be traded off,
  in which case the target is a frontier rather than an invariant.

## R7 — RESEARCH: the two trainings, and why "correct and minimal" is still an unsatisfying definition

*(Research item. Records the training decomposition the argument arrives at, and the
dissatisfaction that remains. Open, not settled.)*

### The two trainings, and why they are complementary rather than competing

**Training 1 — author policies that do the task.** This is where the original proposal
(train the model to produce a working policy for a given task) finds its place in the chain:
it is the capability half, and it is prior. Note that this training does not need rounds —
the ability to conceive the right computation sits in the model itself, not in ReAct's loop.
ReAct contributes nothing to *that* particular ability.

**Training 2 — keep the main trajectory minimal.** The behaviour half.

They compound rather than trade off:

- Better authoring means fewer retries and less repair, which **shortens the trajectory** —
  training 1 serves training 2.
- Minimality training raises how many policies can be produced within `L`, and sharpens
  training 1's objective — training 2 serves training 1.
- Minimality training is also what teaches the model to *use* ReAct's theoretical advantage
  rather than merely have it: instead of committing on the first attempt, it learns the
  round-structured behaviour that models trained specifically on ReAct acquire (tool use
  without bifurcating reasoning across rounds).

Combined, the target reads: **solve the task through `P1` by generating a set of policies,
correctly, while keeping the trajectory minimal.**

### Why that is still not a satisfying definition

The dissatisfaction is real and should not be papered over.

**The objective we actually want is not trainable.** "Given any task, work within `L` and be
correct" is the honest statement of the goal — and it cannot be a training objective, because
it is not compositional. We will never have the problem distribution; for any trained model
there will always exist a problem that exceeds `L`. Training against it is training against a
target we cannot sample.

**So the objective was replaced with a compositional one** — minimality — which *is*
trainable. But correctness has the same defect for the same reason, and swapping in a
compositional surrogate does not make the pair a good *definition*.

**And "correct and minimal" describes an outcome, not a property.** It says nothing about
*how* the trajectory must behave to get there — nothing about what the model must do with the
policies it makes, or what it must carry forward. A definition of a behaviour should name the
behaviour, not its score.

**There is also a redundancy.** If training 1 already targets coverage — bringing the
generating model's reach close to the time-bounded policy space — then correctness is already
being pursued there. Asserting it again in the definition of the behaviour adds nothing.

### What this points toward

The gap between "the objective we want" (any task, within `L`, correct) and "the objective we
can train" (compositional, local) is exactly what R6's invariant formulation is trying to
close: a per-cycle property that composes to the global objective without requiring the global
objective to be sampled. Whether that formulation is sufficient — whether preserving the
invariant really does compose to reach — is the open question this node hands to R6.

### Also open, from this thread

- **Think-anywhere vs think-first-then-code.** `P1` need not be ReAct in the narrow sense.
  A think-anywhere variant interleaves reasoning with code and can revise mid-authoring rather
  than planning then emitting. Whether `P1` should be ReAct-shaped at all is R4's open
  question; this is one alternative shape for it.
- Whether the two trainings should be staged (author first, then minimise) or joint, and
  whether a joint objective for both is even expressible.

## R8 — RESEARCH: why ReAct for `P1`, and the goalpost shift to `θ`

*(Research item. R4 records that the ReAct choice rested on a weak walk. This node is the
actual argument, and the trade-off it costs. Supersedes R4's "weak argument" note, which
should be read as pointing here.)*

### The argument, properly

The authoring capability sits in the model's chain of thought, not in ReAct. ReAct supplies no
new ability to conceive a computation — it supplies **surface**. Given that, three properties
justify the choice, and none of them is optimality:

**1. Flexibility.** Whatever policy shape turns out to be best at authoring, a ReAct `P1` can
simply *author that shape and use it*. If some policy `P` is marginally better at authoring
policies, the model in ReAct writes `P` and proceeds. A fixed alternative shape cannot
absorb ReAct this way; ReAct absorbs the alternatives.

**2. Applicability.** ReAct also does inline thinking, adapts by system prompt, and applies far
outside the authoring role. A generator–verifier loop does not have that reach. Since the
harness is meant for more than thinking problems, `P1` doubles as a general engine, and a
specialised shape would constrain it.

**3. Simplicity.** Starting from a simple LLM that can author policies, ReAct is the smallest
step that adds surface. It sits on the Pareto frontier for that reason — not because nothing
beats it on any axis, but because nothing beats it *while staying this simple and this
general*.

### Why ReAct is *not* near-optimal — a reductio, not a concession

The tempting justification — *ReAct suffices because LLM capabilities are local and strong* —
must be rejected, and rejecting it is not a weakness in the argument but a proof.

> If capability-locality justified ReAct for authoring, the same argument would apply to
> **everything**: all work would be local, simple policies would do as well as elaborate ones,
> and **the policy space would collapse** — precisely what R1 must rule out. The justification
> proves too much, so it is unavailable.

And the collapse is refuted independently, by observation rather than by argument:

- Hand a task to a simple ReAct agent that has the capability, let it run, and it exhibits
  ReAct's characteristic failures. If capability were local, a simple policy would have
  sufficed. It does not.
- **Agentic systems exist and are used** — in coding harnesses and elsewhere — which is itself
  the existence proof that they beat plain delegation. Nobody builds them for nothing.

So the space is demonstrably non-local at least up to the agentic region, and ReAct is
therefore not near-optimal. That is the honest position, and the one the rest of the argument
must be built on. (Vanilla ReAct is also not what will be used; see the family point below.)

### We do not start from scratch — `P1` must lie in the ReAct family

A second, practical arm of the applicability argument, and a binding one: **we begin from
current models, and current models are trained toward ReAct.** Whatever `P1` we choose must
therefore sit close to that shape, or training fights the initialisation. The choice is not
"ReAct versus anything" but **which member of the ReAct family** best suits us.

The variant to gravitate toward: a round in which the model does *not* think first and then
emit code, but **interleaves thinking with editing the code it is writing** (think-anywhere).
Rounds still exist and can be larger, but within a round the model engineers the code as it
reasons. This matters specifically for meta-reasoning, where the policy has to be engineered
precisely rather than drafted in one pass.

**Naming caveat.** Once the model can author arbitrary policies, its behaviour differs enough
from ReAct that calling `P1` "ReAct" is a label of convenience. It is a ReAct-*family* shape,
and meta-reasoning is what distinguishes the member we want.

### The radius argument — why "not optimal" and "good enough" are both true

The apparent tension (ReAct is not near-optimal, yet we choose it) resolves once the
*subproblem* and the *`θ`* are held fixed, because the gap between ReAct and the best policy is
not one number:

- **General tasks, plain thinking `θ` (the `P2` setting).** The gap is **large**. This is R1's
  territory, and it must stay large or the paper has no claim.
- **The authoring subproblem, meta-reasoning-trained `θ` (the `P1` setting).** The gap is
  plausibly **small**. Authoring is a narrow downward projection of the policy space, and a
  trained `θ` internalises the structure rather than reconstructing it — so ReAct, with such a
  `θ`, may sit close to the best available authoring policy without ReAct being good in general.

These are different subproblems under different weights, so different radii are not a
contradiction. **The paper's richness claim lives in the first setting; the `P1` choice lives in
the second.** Confusing them is what makes the position look self-undermining when it is not.

Note the mechanism that makes the second gap small: for most policies ReAct authors them
easily; up to the agentic region this is already clear; and with training the higher band
becomes reachable too, because the model can **learn the policies that author policies** and
invoke those rather than authoring from scratch. What it will not do is reach the exactly
optimal authoring policy — and that residual is what we are accepting.

### The goalpost shift, stated plainly

The resolution is not to argue ReAct is the best computation. It is to **move the objective to
`θ`**:

> Fix `P1`'s shape as ReAct-like, and train meta-reasoning **into `θ`**. A model trained this
> way can learn policy structures fundamentally better than ReAct — and author them, in one
> round, as `P2`.

This is legitimate because `θ` is the thing we are free to change: the capability has to be
*some* weights, so choosing which weights is not a dodge. It also relocates the claim honestly
— we are not asserting a computation is optimal, we are asserting a training target.

### Further wiggle room ReAct buys

Three consequences of the ReAct-family choice that are easy to miss and worth recording:

- **Message weaving is available because the trajectory's messages exist.** They can be passed
  down, so `P1` can construct a `react_auto` carrying the root's own context — a sub-agent that
  is, for that call, the PLM again. Nothing in a fixed alternative shape gives this for free.
- **An authored policy can spend compute the trajectory cannot.** `P1` is bounded by `L`, but a
  policy it writes is not bounded the same way. So the ReAct-generated policy space contains
  policies that expand computation beyond what the trajectory could ever host itself — which is
  a distinct advantage from expressiveness.
- **Thinking about policies may be compressible even if thinking is not.** General thinking
  resists compression; thinking *about policies* is a narrower domain and might admit a
  compressed representation, even a different token vocabulary. Recorded as a design
  possibility, deliberately **left open and not adopted for v1.**

### Open: what does search over policies mean?

Whatever authors a policy has to be an LLM or a set of them — there is no polynomial-time code
that writes the code for us. So some arrangement of LLM elements is always responsible for
producing a policy, whether generator-critic, ensemble, or line-by-line construction.

That raises a question this thread reached but did not settle: **how do we characterise
policies that *search* the policy space more thoroughly than ReAct or a one-round author?**
Authoring is a search problem, and defining what a better search looks like — as opposed to a
better single guess — is unresolved. It bears directly on R8's radius argument, since a good
searcher would shrink the ReAct-to-optimal gap by finding rather than by knowing.


### The trade-off this costs: locality

An earlier claim in this thread — that authoring is a *local* capability — was wrong as stated,
and is retracted:

**Policies do not have to be local, and the non-local ones are often better.** A
generator–verifier arrangement is not local: it spans different calls and different rounds of
compute. If authoring really were purely local, then any simple policy would do as well as an
elaborate one, which is plainly false — and especially false for current `θ`, whose peculiar
behaviours a policy must *learn, estimate, and predict* in order to build around them. That
estimation requirement is itself evidence that authoring is not a local act.

So locality is not a fact about authoring. It is a **constraint we choose to accept for `P1`**:

- **Given up:** the non-local authoring structures, which may be more optimal.
- **Bought:** applicability and simplicity, and a justification for the ReAct shape.
- **Not lost:** `P2` remains the *full* policy space. Nothing is constrained to be local there.

### Why the concession is affordable

Because the constraint applies only to `P1`, the interesting structures survive one level down:
**`θ` can internalise them inside `P2`.** For computations that genuinely need rounds and are
poorly served by ReAct, the trained model learns those structures and emits them as authored
policies — where they run without the trajectory paying for them. So the arrangement is:
`P1` local and general; `P2` unrestricted and specialised; and the non-local optimality that
locality gave up reappears as authored structure rather than as trajectory shape.

## R9 — RESEARCH: the definition attempt — "the trajectory is never unaware"

*(Research item. The current best candidate for defining meta-reasoning, arrived at after
R5 ruled correctness out of the definition and diagnosed "sufficient" as too wet.)*

### The candidate

> **Meta-reasoning is the behaviour in which the main trajectory, for every computation it
> commissions, knows what that computation did — either directly or indirectly — and chooses
> which of the two applies.**

The two modes:

- **Directly** — the trajectory reads the return.
- **Indirectly** — the trajectory knows *by design*: it authored an assimilation policy that
  extracts what matters, or attached a constraint guaranteeing the property it cares about, or
  built the computation such that its behaviour was predicted and the prediction held.

The choice between them is itself part of the behaviour. That is what "wet when it chooses to
be" means, made precise.

### The floor: never unaware

The minimum this imposes is not a quantity of information but a prohibition:

> **The trajectory must not be unaware.**

Complete dryness — fire a policy, take the answer, proceed without any idea what happened — is
excluded, because it forfeits exactly the capacity that having a trajectory buys: the ability
to improve across rounds. A `P1` that runs an experiment and returns its output unexamined has
not meta-reasoned; it has degraded itself to a one-shot author with extra steps.

### Why this formulation does better than the previous ones

- **It is not too wet.** Unlike the sufficiency clause, it does not demand the trajectory hold
  everything. Indirect knowledge counts, so authoring an assimilation policy *satisfies* the
  definition rather than being an awkward exception to it.
- **It is not too dry.** Blind firing is excluded by the floor.
- **It contains no correctness term**, satisfying R5's requirement that the definition name a
  disposition rather than a score. Correctness remains the consequence.
- **It is compositional in R6's sense.** "Not unaware" is a per-cycle property, checkable
  without reference to the problem space, so it composes across cycles and is trainable.
- **It connects to R5's repair.** Indirect knowledge *is* the projection: the trajectory knows
  because it estimated what the computation would do and designed the return around that
  estimate. Sufficiency-as-projection-accuracy and awareness-by-design are the same mechanism
  seen from two angles.

### Sufficiency redefined: the information geometry

Treat the information produced by a computation as a graph — nodes are pieces of information,
edges are "knows about". The trajectory holds some nodes. Define distance from the trajectory:

- **Distance 0 — direct.** The content is in the trajectory.
- **Distance 1 — indirect.** The trajectory holds a *reference with a description*: "variable
  `x` holds the experiment data." The content is absent, but the model **can decide about it** —
  read it, apply a policy to it, hand it onward.
- **No distance — lost.** The information was produced and dropped: never stored, never woven,
  or overwritten. There is no path to it at all; it is not in the graph.

> **Sufficiency.** Nothing among the artifacts of the trajectory's off-trajectory work would,
> if added to the main trajectory, **significantly** increase the information available to it.

That is the whole clause, and it is stated without any apparatus. What makes it achievable is a
single behaviour, also stated plainly:

> The trajectory either **holds** the information, or **holds information about the
> information** — it knows the artifact exists, roughly what it is about, and how to reach it.

**Why "holds information about it" is enough, and not a weaker substitute for holding it.**
Knowing where something is *is* knowing what it is, at some resolution. A handle carries its own
description: "`x` holds the experiment data," "that verification message is where the
counterexample came from." The description may be vague, but vagueness is a matter of resolution,
not of category — the trajectory is not surprised by the artifact's existence, and it can decide
about the artifact without reading it: apply a policy to it, hand it onward, or spend a few
tokens to look and sharpen what it knows.

That last cost is what makes the clause hold. Following up on a handle — reading it, or authoring
one small policy to extract the part that was missing — is negligible against the token cost and
context length of the trajectory as a whole. So a trajectory that knows vaguely and a trajectory
that knows precisely are **both meta-reasoning**; the second is better, and that difference is a
gradient optimisation will close, not a boundary that separates the behaviour from its absence.

**Where the clause bites is surprise.** The failure it rules out is the artifact whose *existence*
the trajectory never registered. There, adding the artifact to the trajectory does significantly
increase the information available — the trajectory did not know there was anything to know. That
is the one case sufficiency excludes, and the word **significantly** carries the rest: an artifact
whose restoration would change nothing about what the reasoning requires may be dropped freely,
so no separate inertness clause is needed.

Two clauses doing two different jobs. The first is a **floor**: nothing that matters may fall
out of reach. The second is a **gradient**: among behaviours that clear the floor, the one
holding more of its information at d1 is the better meta-reasoner. This is why the definition
names a region rather than a point.

The second clause is what keeps the condition honest. Demanding that *nothing whatsoever* be
retained is both unachievable and pointless: when a verification sub-call reports its judgment,
that judgment is what the reasoning needs, while the individual steps by which it arrived there
are not. Losing them costs nothing. The requirement is therefore not that everything survive,
but that **whatever is allowed to fall out of reach be inert** — that its restoration would
change nothing about what the reasoning requires.

Note this is the original counterfactual test, aimed correctly for the first time. The old
clause applied it to *all* off-trajectory work, which made it demand everything; applied
instead to the **lost set alone**, it says exactly what was meant. And it does not reintroduce
the relevance problem, because nothing has to be classified in advance: the default is to keep
access, which is cheap, and the test only ever applies to what was dropped.

**There is no distance 2 as a path length, and this is not an accident.** Reachability in code
collapses: if the trajectory can reach the program state, one act of code reaches anything in
it. A dictionary is reachable, so every key inside it is reachable in the same step. Even "the
model does not know variable `x` exists" is not distance 2 — listing the namespace is also one
act. No node can be constructed that genuinely requires two acts, because the first act returns
something the same cell can immediately use.

**But one category does behave like distance 2, and it is about cost rather than steps.**
Consider information recoverable *only by recomputation*: a policy kept the seed and the
parameters but not the result. A path exists — rerun the work — and it costs the full price of
the original computation. That is neither cheaply reachable nor absolutely lost. If distance 2
is meaningful in this space at all, this is what it is, and it suggests the metric is really
about **cost of recovery**, with distance 1 meaning *cheaply* recoverable. Whether this middle
category needs its own treatment, or collapses into "lost" for practical purposes, is open.

Setting that aside, the metric is **effectively binary**: information is cheaply reachable, or
it is gone. And the cut falls where decidability does — reachable information can be decided
about at any later point; lost information can never be, which is what "unaware" means
operationally.

### The hierarchy: distance is cost of reach, not path length

Distance was never about how many steps a path takes — it is about **how expensive it is to
reach the information**. The levels are separated by *what you would be surprised by*:

| Level | State | Surprise | Recovery |
|---|---|---|---|
| **d0 — held** | in the trajectory | none | none needed |
| **d1 — indexed** | in state, existence and location known | content only | one operation |
| **d2 — latent** | in state, existence unknown | existence *and* location | search |
| **d3 — reproducible** | not in state; inputs retained | everything | rerun the computation |
| **lost** | not present, not reproducible | — | impossible |

**d1 versus d2 is the useful cut.** At d1 you may be surprised by *what* the information says,
but never by *whether it is there*. At d2 you would not know to look — a sub-agent created a
variable nobody accounted for. Good authoring is largely the work of pulling things from d2 up
to d1: naming what will exist, so that later nothing has to be discovered.

**Why the hierarchy bottoms out at d2 inside the state.** Search is exhaustive. Once you are
enumerating the namespace you find everything, at any depth — there is no "harder to discover
than discoverable." So no level deeper than d2 can exist within reachable state, and d3 is
qualitatively different because it lies outside the state altogether.

**A sharp constraint on d3.** Recomputation recovers information only when the computation is
**deterministic**. Rerunning a sub-LLM does not recover its reasoning; it generates *different*
reasoning. So for every neural component, dropped is **lost, never reproducible** — which is
the strongest argument for message weaving there is, and the reason the neural parts of a policy
deserve more retention care than the symbolic ones.

### Why d1 is the equilibrium, not merely a middle

The cost structure decides where the behaviour should sit:

| | carrying cost | recovery cost |
|---|---|---|
| d0 | high — tokens, paid always, whether needed or not | zero |
| **d1** | **near-zero — a name and a line of storage** | **near-zero — one operation** |
| d2 | zero | moderate — search, paid on demand |
| d3 | zero | full price of the original computation |

Storing something and knowing its name costs almost nothing against a context window; the
saving against holding the content inline is enormous. So:

> **Sufficiency alone would push everything to d0 (hoard it). Minimality alone would push
> everything to nothing. d1 is where both are satisfied simultaneously.**

The "dance between sufficiency and minimality" is therefore not an unresolved tension with an
arbitrary resting point — it has a **structural answer**, and the answer is d1. The behaviour we
want maximises information at d1, keeps d0 for what is actively in use, tolerates d2 where
anticipating everything would have cost more than searching later, and lets nothing important
reach d3 or beyond.

### Near-sufficiency, and why exact sufficiency is the wrong target

Nothing is exactly sufficient or exactly minimal, and pursuing exact sufficiency is actively
bad meta-reasoning:

- **Anticipating everything is intractable.** To guarantee every produced value sits at d1, the
  author must foresee every value the computation will create. For non-trivial programs that
  costs more authoring tokens than the search it saves.
- **Hoarding is a failure mode, not a virtue.** A behaviour that maximises sufficiency alone —
  never letting go of anything — is a *bad* meta-reasoner, because it spends the budget that
  minimality was protecting.

**How this squares with the floor.** Exact sufficiency would put *everything* at d0 or d1.
Near-sufficiency admits some residual d2 — but as a **priced trade rather than an oversight**:
the author judged that anticipating a particular value would cost more than searching for it
later, and accepted the risk knowingly. That is different in kind from d2 arising because the
author never considered the value at all. The floor still bites; what near-sufficiency relaxes
is the demand that the floor be reached by anticipation alone.

So the target is **near-sufficiency**: most information at d1, the rest cheaply findable, and
whatever is truly dropped being inert. If at test time the model discovers something sat at d2
that it would have preferred at d1, that is acceptable — it costs a search, not the task.

### Exhaustiveness of the hierarchy (proposition for the theory appendix)

> **Proposition.** For any piece of information produced by a computation, exactly one of
> d0, d1, d2, d3, or *lost* applies.

*Proof sketch.* Classify by a decision tree over three questions.

1. **Is it in the trajectory?** If yes → **d0**.
2. **Otherwise, is it reachable by code?** Take *state* to mean everything code can touch —
   REPL variables, woven message lists, files, network resources.
   - Reachable and its location known → **d1**.
   - Reachable but location unknown → **d2**, recoverable by search, and search is exhaustive,
     so no deeper level exists inside the state.
3. **Otherwise it is outside the state.** Are its inputs retained *and* the computation
   deterministic?
   - Yes → **d3**, recoverable by recomputation at the original price.
   - No → **lost**.

The cases are mutually exclusive and jointly exhaustive by construction. ∎

**d3 is lost in practice, and should be presented that way.** Recomputation is not recovery: it
pays the full original price a second time, and for any stochastic component it does not return
the same information at all — rerunning a sub-LLM generates new reasoning rather than restoring
the old. So d3 earns its place in the *proof* (it is what makes the tree exhaustive) but not in
the *behavioural account*, where it behaves as loss. The operative hierarchy for the definition
is therefore three-valued: **held**, **handled**, **latent** — with everything past that being
loss.

**Scope note: the hierarchy classifies reachability, not salience.** A reader will raise
lost-in-the-middle — information sitting in the trajectory but not attended to, held by location
yet functionally unavailable. Three reasons it is not a live threat *here*, which is why the
scope restriction costs nothing:

1. **Minimality is already the training target.** The behaviour being trained keeps the
   trajectory near-minimal for the work done, so it does not reach the lengths at which
   attention degrades. ReAct accumulates because every step of every computation lands in the
   trajectory; a PLM trajectory carries handles to computations whose bulk lives in the state.
   The comparison that matters is trajectory length *per unit of work done*, and that is the
   axis PLM is optimising.
2. **The state is a repair mechanism.** Even where a long trajectory does blur, the computation
   and its outputs are still in the state, so the trajectory can refresh itself by reading them
   back. Attention failure is recoverable here in a way it is not for a pure-context method.
3. **It is a property of the context, not of the information's location.** Salience belongs to
   the analysis of why long contexts degrade. Folding it into a reachability hierarchy would
   conflate two independent failure modes and make both harder to reason about.

Stated this way the restriction is a scoping decision with an argument behind it, not an
unexplained exclusion.

**Corollary (already noted, worth stating formally).** d3 is unavailable to stochastic
components. Rerunning a sub-LLM generates new reasoning rather than recovering the old, so for
every neural part of a policy the tree collapses to d0/d1/d2/lost, with no reproducible tier.

### Presentation: the definition first, the apparatus after

The order is fixed, and it is the order the reader needs:

1. **Meta-reasoning is defined in plain text, with no notation at all** — authors computations,
   holds or holds information about what they produce, at minimal trajectory cost. A reader who
   stops there has the idea intact.
2. **The access hierarchy is then introduced separately**, as apparatus, with the proposition
   above as its warrant. It explains *why* the definition is stated the way it is, and it gives
   the experiment something to measure, but the definition does not depend on it.

**On the notation itself.** "Distance" carried a false suggestion of path length, which is why
that reading had to be argued away at length. The levels are about *cost and certainty of
access*, so the names should say so. Use words in the text and reserve symbols for tables:

| | name | the trajectory's relation to the artifact |
|---|---|---|
| a0 | **held** | it is in the trajectory |
| a1 | **handled** | it knows the artifact exists, roughly what it is, and how to reach it |
| a2 | **latent** | it is reachable, but the trajectory does not know it exists |
| — | **lost** | not reachable, or reachable only by paying its full price again |

Sufficiency is then simply: **held or handled, and whatever is neither is insignificant.** The
gradient is: prefer **handled** to **held**, since a handle costs almost nothing to carry and
almost nothing to cash in.

**Why the theory must be assembled up front, unlike RLM.** RLM produced results first and grew
its theory afterwards. That route is unavailable here: the work will be presented to those
authors, and without results in hand the base theory has to stand on its own from the start.
The hierarchy, the policy-space derivation (R2–R9), and the questions they generate are that
base, and they belong in the appendix in assembled form rather than as accumulated notes.


### The distances, concretely (what they are in the harness)

The abstract graph has a direct reading in the actual substrate:

- **Distance 0** — returned or printed into the trajectory.
- **Distance 1** — anything reachable by writing code against persistent REPL state: stored
  variables, woven message lists, and **the code itself**, since the code is the knowledge of
  what exists and where. Note this is coarser than it first appears: a dictionary is distance 1
  and so is every key inside it, because reaching any of them is one act of code. The
  *program* is the distance-1 layer.
- **No distance** — information **produced but not retained anywhere reachable**: an
  intermediate never stored, a sub-agent's reasoning never woven, state a computation
  overwrote. Not far away: *absent*.

> **Loss is not a fact of nature; it is a design failure.** The policy produced information and
> dropped it. Nothing forced it to, and one stored variable would have prevented it.

### Loss *is* the information barrier

This identifies the second barrier of delegation precisely. Work runs elsewhere and its traces
do not survive — that is the fall out of the graph entirely, and it is what makes delegation go
dry even when the authoring was good. **Message weaving is the mechanism that prevents it**, keeping a
sub-agent's trajectory at distance 1 rather than letting it evaporate; storing intermediates
rather than printing summaries does the same for data.

So the barrier is not something PLM must overcome by cleverness. It is something an authored
policy can close **by construction**, which is a concrete statement of what authoring buys.

### Sufficiency restated: nothing is lost

The old clause ("no information added from off-trajectory work would increase what the model
knows") can now be stated directly and checkably:

> **Nothing that would matter has been lost.** For everything the computations produced, the
> trajectory either holds it or holds information about it; whatever falls outside that would
> not, if restored, significantly increase what the reasoning requires.

Whether the trajectory holds something, or holds a handle to it, is a fact about the program and
can be read off. Only the *significance* of what was dropped needs judgement, and it needs it
only for what was dropped.

**Latent artifacts fail this clause.** An artifact that is reachable but unanticipated is not
lost — the state contains it — yet it is not accounted for either, because nothing in the
trajectory points at it. It is its own defect, and it is the one the experiment is built to
observe.

### Relevance dissolves, because distance 1 is cheap

An earlier worry — must *every* node be within distance 1, when some information is plainly
irrelevant? — resolves in favour of universal access, and for a practical reason rather than a
principled one:

**You do not have to classify relevance in advance. You only have to keep access, and access is
nearly free.** The details of a failed experiment cost nothing sitting in a variable, occupy no
trajectory tokens, and may matter three steps later. Deciding relevance up front is both
expensive and error-prone; keeping distance 1 is neither.

This also removes the circularity R5 was trying to escape: sufficiency no longer needs a notion
of decision-relevant information, because it demands *access to everything* rather than
*retention of what matters*.

### Quality within the boundary

Access is the hard boundary; how *well* a distance-1 reference supports the decision is a soft
measure inside it. Two returns from the same experiment:

- **A:** "here is the experiment data."
- **B:** "here is the experiment data; experiment two showed X, you may want to look there."

Both are distance 1. B is the better meta-reasoner, because its meta-information supports the
decision and A's barely does — while neither pays the cost of ingesting the content.

The resulting picture: **the trajectory is a catalogue, not a library.** It does not hold the
books; it holds entries good enough to choose which to open.

### The warrant: indirect knowledge requires checks

This resolves the open question about telling *knows indirectly* from *wrongly believes it
knows*. The difference is not in the belief, it is in what backs it:

> **Assimilation is not only extraction. It is the provision of the guarantee that lets the
> trajectory proceed without looking.**

So a policy must author its own checks — validation, a verifying policy, whatever gives warrant
— even when the author is confident it will work. A policy that merely believes, with nothing
backing the belief, is a **bad meta-reasoner even when it happens to be right**. Checks are the
observable difference, and they are already counted by the artifact-shape metrics.

### Meta-reasoning is a region, not a point

The two conditions are not the same kind of thing, and the definition should say so:

- **Sufficiency / awareness — a hard boundary.** Distance ≤ 1. Either satisfied or not.
- **Minimality — a soft direction.** Not a threshold but a gradient: among behaviours inside
  the boundary, less trajectory is better meta-reasoning.

So meta-reasoning names a **spectrum of admissible behaviours**, bounded on one side and
pressured on the other, not a single optimum. The behaviour dances between them, and where it
sits is a choice made per step.

### Why this is the hypothesis: the two barriers of delegation

Delegation fails for two separable reasons, and sufficiency is aimed at the second:

1. **The barrier of authoring/estimation** — can the right computation be conceived, with the
   checks that make its result trustworthy? (R1, R4.)
2. **The barrier of information** — even when the computation is authored well, it runs as a
   separate process and its work does not come back. The trajectory goes dry again, not through
   bad authoring but through disconnection.

A model that delegates well is still capped by the second barrier. **If every piece of
information is at distance ≤ 1, both barriers fall**: the first because checks provide warrant,
the second because access is guaranteed by construction. That is the hypothesis about why
correctness and minimality can be jointly achieved rather than traded off — and, being a
hypothesis about `P1`'s failure modes, it is falsifiable: trained behaviour may not display it.

### The definition, assembled

Meta-reasoning is the behaviour of a trajectory that:

1. **authors** computations for the task, from the time-bounded policy space, with coverage as
   large as it can reach (R4);
2. **loses nothing that would matter** — for every artifact those computations produce, the
   trajectory either holds it or holds information about it, and whatever falls outside that
   would not, if restored, significantly increase what the reasoning requires — with as much as
   possible held as a *handle* rather than carried in full, and what is kept backed by checks;
   and
3. does so at **minimal trajectory cost**, minimality being the gradient rather than the
   boundary.

### Open

- **Resolved by the warrant clause:** telling *knows indirectly* from *believes it knows* is
  answered by whether checks were authored, not by inspecting the belief. The calibration join
  remains the measurement of whether the projection behind those checks was accurate.
- **Resolved by the region framing:** awareness and minimality are not one property and need
  not be — one is a boundary, the other a gradient. R6's "invariant or frontier" question
  resolves to *both*: a hard invariant on access, a frontier within it.
- **Still open — distance in practice.** The graph is clean in the abstract; measuring it on a
  real trajectory is not. What counts as a "description good enough to decide on" admits
  degrees, and the 1-versus-2 boundary may be sharp in theory and fuzzy in transcripts. A
  scoring rule is needed before this can be evaluated rather than argued.
- **Resolved — universal access, not selective retention.** Distance ≤ 1 holds for every node,
  because keeping access is nearly free while classifying relevance up front is expensive and
  error-prone. This is what removes R5's relevance circularity.
- **Still open — is deliberate loss ever *beneficial*?** Retention is not always free: some
  intermediates are large, and keeping them may force a worse-structured computation. If
  dropping information is ever the right trade, the condition needs a stated exception rather
  than being absolute.
- **Note — the condition is a low bar, and that is the point.** Since reachability collapses,
  satisfying it requires only that a policy *store what it produces*. It is easy. Policies
  nonetheless violate it routinely — printing summaries and discarding the rest, dispatching
  sub-agents without weaving — which is exactly why stating it as a condition has teeth.
- **Still open — "chooses which applies."** Whether the direct/indirect choice needs its own
  criterion, or is adequately disciplined by the token cost of the direct mode.

## R10 — HARNESS: total retention (nothing falls out of the state), and why it is not a theory change

*(Harness item, not a theory item. Same category as message weaving: a capability of the shell
that lets a theoretical point be demonstrated without being part of the theory. Designed here in
full; **disabled for the authoring experiment**.)*

### The correction to the first reading

The first assessment of this proposal claimed a cost: that if the harness retains everything, the
retention failure can no longer occur, so the experiment loses the ability to observe it. **That
was wrong about what the experiment observes.**

The observable is not *held versus lost*. It is *handled versus latent* — did the trajectory plan
for the artifact, or did it have to go and find it. Total retention does not touch that:

- The harness stores the artifact, but it **never announces it**. Nothing says "`x` is here, the
  verification message is there." The store exists; its contents are not advertised.
- So a policy whose author anticipated the artifact still shows a **handle** — a name, a
  structured return, a place the trajectory points at.
- And a policy whose author did not still leaves the artifact **latent** — present, reachable,
  unaccounted for, found only if the model goes looking.

The measurement survives intact. What changes is that the third outcome, the artifact being gone
entirely, stops occurring — and that outcome was never the interesting one.

### Why removing it strengthens the argument

Without total retention a policy has to be **pre-emptively defensive**: store everything it might
conceivably want, because anything unstored is unrecoverable. That is not the behaviour to
reward. Pre-emptive storage costs thinking tokens to decide on and code tokens to write, and a
well-trained model that correctly judges it will not need an artifact should not be penalised for
omitting the code that would have saved it.

With total retention that judgement stops being a gamble. The model writes handles for what it
plans to use; everything else is quietly kept and can be investigated later if a computation
turns out to have revealed something its author did not anticipate — a verification message
carrying a counterexample nobody asked for, say. Nothing is forfeited by not having guessed.

This is the honest version of the claim: **PLM does not require an author to be omniscient about
its own computations.** The mechanism is free, so it does not raise the cost of good behaviour,
and it does not substitute for good behaviour either.

### Harness or theory

Harness. The definition does not change by a word: sufficiency is still *held or handled, and
whatever is neither is insignificant*. Total retention removes one way of falling out, the
accidental one, and leaves untouched the distinction the definition is built on.

Message weaving is the precedent — a shell capability, not part of the theory, and also the thing
that lets us demonstrate something ReAct-style methods structurally cannot do. Total retention is
the same shape.

### Where the loss actually is

One fact about the current harness narrows the problem sharply.

**Cell-level bindings already persist.** `repl/exec_ns.py` runs every cell with **one dict used
as both globals and locals**, deliberately, so a `def` in a cell can see its siblings. A
consequence is that everything a cell binds at top level is a global and survives the cell. There
is nothing to fix at the cell level, and no prompt instruction along the lines of "never delete
your globals" is needed — the shell does not drop them in the first place.

**The loss is inside policy bodies.** A `@policy` function's locals are ordinary frame locals:
they vanish when the call returns, and everything the policy computed on the way to its return
value goes with them. That is the whole source of accidental loss, and it is exactly what
`@policy` already owns — which is what makes the decorator the right place, as suspected.

### The mechanism

Two halves, and the second is the one that makes the first worth anything.

#### Half 1 — record every call, at the proxy

`_FunctionPolicy.__call__` (`policy/proxy.py:329`) is the single boundary every policy call
crosses, whatever the caller — sealed defaults, PLM-authored policies, top-level cells. It already
opens a `_policy_call` context for the depth cap, so the call id, the parent call id and the depth
all come from machinery that exists. Around that boundary it records **arguments in, and the
return value or exception out**.

This alone answers one of the two loss cases: a policy's *return* is captured whether or not
anyone bound it to a name.

#### Half 2 — record every intermediate, by naming it

The remaining case is the real one. `locals()` only ever contains **bound names**, so

    @policy
    def check(x):
        return verify(transform(x))

retains `x` and the return value, and loses whatever `transform(x)` produced — the very artifact
most likely to hold something its author did not anticipate.

The fix is a source transform into **A-normal form**: every non-trivial subexpression is hoisted
into its own assignment before use.

    def check(x):
        _plmt_3_23 = transform(x)
        _plmt_3_11 = verify(_plmt_3_23)
        return _plmt_3_11

Now `locals()` at exit contains every intermediate the body computed. Nothing has to be
anticipated, and no storage code has to be written.

**Temps are keyed by source position**, `_plmt_{line}_{col}`, not by a counter. That makes them
stable across calls and across edits to unrelated lines, and it makes a record self-describing:
an entry *is* "the value of the expression at line 3, column 23," which a model reading the store
can line up against the policy source it can already print. Position keys are what make the store
searchable rather than a bag of anonymous values.

#### What gets hoisted: calls, and only calls

Full A-normal form hoists every subexpression. That is more than is wanted. A bare literal or a
name reference produces nothing new — writing `7` on a line means nothing, and storing it as an
intermediate is noise that dilutes the store.

**The rule is: hoist `Call` nodes.** A call is the only construct that *does work*, and the result
of doing work is the only thing that can carry information its author did not anticipate. That
covers exactly the cases worth covering:

| construct | retained | how |
|---|---|---|
| a bound local, reassigned | final value only | frame local |
| an argument, including a mutated one | the object, one version | frame local |
| an inline call result, never bound | the value | position-keyed temp |
| the return expression | the value | hoisted temp *and* recorded at the proxy |
| a bare literal or name | nothing | deliberately not hoisted |

Narrowing to calls also makes the transform cheaper and shrinks its blast radius — fewer rewritten
nodes, fewer chances to perturb evaluation.

#### When the record is taken: at runtime, at exit, once

The record is taken **inside the running frame**, in the `finally`, at the moment the call leaves
it — not afterwards from the outside, which would be impossible, since the frame and its locals no
longer exist by then.

Taking it once, at exit, is what produces the behaviour that was wanted:

- **A local reassigned during the call yields one entry, not a history.** `locals()` holds
  bindings, so the record has the final value of each name. No version chain, no diff to read.
- **An argument mutated in place yields one entry too.** The record holds the object, and the
  object reflects every mutation that happened to it. There is no "before" copy to disagree with
  the "after" — the two versions the naive design would have produced never exist.
- **The exception path is covered**, because a `finally` runs on the way out either way. A policy
  that raised leaves behind everything it had computed up to the raise, which is when that state
  is most worth having.

The cost is that intra-call history is not recoverable: if a name held three values in turn, the
record shows the third. That is the right trade. Keeping history would mean snapshotting on every
assignment — unbounded in memory, and it would turn a readable record into a log a model has to
reconstruct state from.

#### Where the transform goes

Five sites compile policy source today, and all five must route through one helper or the
behaviour diverges between a freshly-decorated policy and an edited one:

| site | path |
|---|---|
| `policy/decorator.py:167` | fresh install, function |
| `policy/decorator.py:181` | fresh install, class |
| `policy/proxy.py:444` | `_rewrite` — function edit |
| `policy/proxy.py:657` | `_rewrite_class_policy` — class edit |
| `policy/registry.py:439` | restore after respawn |

Two of them already compile an `ast` tree rather than a string, so the transform slots in without
restructuring. Class policies reach their methods through `_iter_funcs`, the same walk
`_warn_if_captures_locals` and `_reject_async` already use, so methods, staticmethods,
classmethods and properties are covered by construction.

**The source the model sees is unaffected.** `_p_source` on the proxy holds the verbatim original
and is what `read_policy`, `getsource` and the `<policy-{name}>` linecache slot serve; only
`_inner` runs transformed code. Hoisted statements carry the line number of the expression they
came from, so tracebacks still point at the original line.

#### What the transform must not touch

Semantics are preserved exactly, which means the hoist is refused wherever hoisting would change
evaluation:

- **Short-circuit forms** — `and`, `or`, conditional expressions. Hoisting the right operand out
  of `a and b` would evaluate `b` unconditionally.
- **Comprehension elements and conditions, and lambda bodies** — they are deferred or
  per-element; hoisting lifts them out of the scope that gives them meaning.
- **`yield` and `yield from`** — hoisting across a suspension point reorders the generator.

Everything else hoists left-to-right, which is Python's own evaluation order, so the transform is
order-preserving by construction.

### The line on scope, and why it is the right line

Only `@policy` bodies are instrumented. A plain nested `def` inside a policy, and the frame a
comprehension creates, keep ordinary Python semantics and lose their locals on exit.

That is deliberate, for two reasons.

**The language must not change.** A model writing PLM policies is writing Python, not a dialect in
which every scope secretly persists. Instrumenting arbitrary frames would make `locals()`,
memory behaviour and object lifetime stop matching what the model knows about the language, and
that is a far larger tax than the loss it prevents.

**The incentive points the right way.** Work worth retaining should be a policy. Making `@policy`
the boundary at which retention begins means the decorator earns its keep for a second reason
beyond editability and depth-counting: it is the unit of work whose internals survive. A model
that wants an intermediate kept has an obvious, cheap and already-idiomatic move — make that step
a policy — rather than a defensive-storage move.

Control flow needs no special handling, since `for`, `while`, `with` and `if` do not create
scopes in Python; their bindings are already frame locals and are captured.

**Loop iterations are the author's business.** A temp inside a loop is rebound each pass, so the
record holds the final iteration's value. Accumulating across iterations is an ordinary
programming decision — append to a list — not something the harness should do silently, since
doing it silently would be unbounded. This is the one place where retention stops, and it stops
at a line every programmer already knows.

**Deletion stays meaningful.** `del x` removes `x` from `locals()` and therefore from the record,
which is exactly the theory's position: to delete deliberately, the model had to know the value
existed, so the information was handled before it was dropped.

### Two cases the design has to answer explicitly

#### Nesting: a policy called from inside another policy

Policies calling policies is supported and depth-capped, so this is not a corner case.

**Each call gets its own record, in its own policy's buffer, and the tree links them.** If `A`
calls `B`:

- `B`'s intermediates are recorded under **`B`**, in a record whose `parent_call_id` is `A`'s
  call id.
- `A`'s record holds **`B`'s return value**, because the call `B(...)` is a hoisted expression and
  therefore has a position-keyed temp of its own in `A`'s frame.
- Nothing is duplicated. `A` gets the result, `B` gets the workings, and the link between them is
  the parent pointer.

This is the property that makes the store worth having rather than merely large. A model asking
"the answer came back wrong — what did the step that produced it actually see?" walks down the
tree from the call it knows about to the call it does not, instead of scanning a flat log and
guessing which entries belong together. The structure of the store is the structure of the
computation.

Depth is already bounded by the policy-call cap, so the tree cannot run away.

#### Defaults inside a policy

Two different things are called "default" here and they behave differently. Both are worth stating
because the second is where the design still has a real gap.

**A sealed default policy called from inside an authored policy** — `natural_llm`, `react_auto`.
These are excluded from the source transform: they are immutable by design and they are harness
code, not authored code. They get **half 1** — arguments and return recorded at the proxy — plus
two things the code already does, which together close most of the gap an earlier draft of this
node claimed was open.

**Messages are already retained, by reference.** `react_auto`'s `_norm`
(`policy/defaults/react_auto.py:260`) returns *the caller's list itself* when a list is passed —
this is message weaving, and it is pass-by-reference on purpose. The sub-agent only ever appends
turns; it never mutates the caller's existing dicts. So:

- A list the calling policy built is a frame local (or, if built inline, a position-keyed temp),
  and it is therefore in the call record.
- Because `react_auto` appends *to that same object*, the record's entry shows the **woven
  conversation after the call**, including every round the sub-agent took.
- The sub-agent's reasoning is thus retained without any new mechanism, and retained under a name
  the author chose — which is *handled*, not latent.

The one condition is that a list be passed. A `str` or a single `dict` gives `_norm` nothing to
weave into and it builds a fresh list, which is discarded. That is a real cliff, and it is a
property of the harness as it stands rather than of retention: passing a string to `react_auto`
forfeits the trajectory.

**The sub-agent's REPL state is one dict, and retaining it is easy.** `ns`
(`policy/defaults/react_auto.py:400`) is built **once, before the round loop** at line 444, and is
used as both globals and locals for every round. It therefore accumulates the sub-agent's entire
working state across the whole call: every variable it bound, every intermediate it kept.

Since `react_auto` is harness code, this needs no AST transform at all — the sealed policy hands
`ns` to the store on its way out. Strip `__builtins__`, `RETURN`, and the injected grants
(`args`/`kwargs`/`objects`, which are the caller's own objects and are already retained on the
caller's side), and what is left is exactly what the sub-agent *created*. That is small, and it is
the interesting part.

**Nothing remains uncaptured.** An earlier draft of this node reserved a residue — "reasoning
that reached neither the messages nor the namespace." There is no such residue, because a ReAct
sub-agent reasons *by emitting text*: that is the whole mechanism. `react_auto.py:454–461` appends
each assistant turn with its `reasoning` field alongside its content, so the model's thinking is
in `msgs` by construction, not merely often. Messages plus namespace is therefore not "most of"
the sub-agent's interior — it is all of it.

`_subcall_log` continues to record the *shape* of each sub-call — budget granted, rounds used,
truncation, clamping, outcome — which is orthogonal and stays as it is.

**A default argument value in a policy's signature** — `def p(x, y=make_thing())`. This is
ordinary Python and needs nothing special:

- The default object is evaluated once at definition time and lives on the function object's
  `__defaults__`, so it persists for as long as the policy does. It cannot be lost.
- When the caller does not override it, it is bound as a frame local like any parameter, so it
  appears in the call record anyway.
- The usual mutable-default sharing across calls is unchanged; retention neither causes nor cures
  it.

### The store

Per policy, a bounded ring buffer of call records; per record:

    call_id, parent_call_id, policy, depth, elapsed
    args, kwargs
    locals at exit          # bound names + position-keyed temps
    result                  # return value, or the exception raised

`parent_call_id` makes the store a **tree that mirrors the computation**, not a flat log, so a
model looking for "what did the verification step actually see" can descend rather than scan. It
holds live references in-process, so there is no serialisation cost during a run.

**Where the records live.** The store is central, keyed by policy name, because the
`parent_call_id` tree has to span policies and a per-object store could not. But the ergonomic
handle is the policy object itself: `_FunctionPolicy` carries a `__dict__` (it is in `__slots__`
for `functools.update_wrapper`), so an accessor hangs off the proxy and filters the central store
to that policy. A model that has `check` by name can ask `check` about its own calls without
knowing the store's layout. Function-policy edits keep proxy identity, so records survive
`edit_policy`; a kind change (function ↔ class) does not, and starts a fresh buffer.

Reachable from the REPL like any other global, through a single accessor. **Not itemised**: the
system prompt states that run state is retained and how to reach it; it never enumerates what is
in there. The store's existence is *handled*; any particular value inside it stays *latent* until
the model looks. That is the whole basis of the measurement, and it survives precisely because
announcing the store is not the same as announcing its contents.

**Sealed default policies are excluded from the transform.** They are immutable by design
(`_p_immutable`, enforced in `_FunctionPolicy.__setattr__`) and they are harness code rather than
authored code; their sub-call artifacts already go to `_subcall_log`. They still get half 1 —
args and return recorded at the proxy — since that needs no rewriting.

### Deferred harness tasks (record now, build at release)

*Not to be implemented during the authoring experiment. The harness ships with the paper — arXiv
or the NeurIPS workshop — and these land then. Recorded here so the design is complete even
though the code is not.*

**H1 — `messages` is a list of dicts, and nothing else.** `react_auto._norm`
(`policy/defaults/react_auto.py:260`) currently accepts a `str`, a single `dict`, or any iterable,
and builds a fresh list for each. Only the list case is woven; the others silently forfeit the
sub-trajectory, because the list the sub-agent appended to is discarded at return. That is a cliff
with no warning on it, and no author benefits from the convenience that creates it.

The fix is to narrow the contract rather than patch around it: **`messages` must be a list of
message dicts**, and anything else is a `TypeError` at the call. `natural_llm` takes the same
change, for the same reason and so the two primitives do not diverge. A one-line convenience
(`[{"role": "user", "content": s}]`) is a trivial thing to ask an author to write and it makes the
weaving contract visible in the code they wrote.

This removes the string cliff at its root, which is better than disclosing it in the task card.

**H2 — an access path from a policy to the sub-call it just made.** The problem is real and the
current design does not answer it. `react_auto` is a *single sealed object* shared by every caller,
so an accessor hanging off the policy — `react_auto.calls` — returns every call made anywhere in
the run. An author who wants the namespace of *the call on line 7 of the policy they are writing*
has no way to name it. The return value is the sub-agent's answer, not a handle to the call.

The resolution follows from a small amendment to the record lifecycle, which is worth making
anyway:

> **Records are created at call entry, not at exit.** Only `locals()` is filled at exit. A child
> call appends itself to its parent's record when it completes.

With that, a record exists and is reachable *while its call is still running*, which gives the
natural access path:

    this_call()                  # the record of the currently-executing policy call
    this_call().subcalls[-1]     # the sub-call that just returned
    this_call().subcalls[-1].ns  # its namespace

No handle has to be threaded, no signature changes, and disambiguation is by position in the
caller's own record rather than by searching a shared log. It also answers the timing question
directly: the record lives *during* the run; the locals snapshot is the only part taken at the
end.

### Open

- **Serialisation.** The store holds live references in-process, which is free; the snapshot/dill
  path is where unpicklable values bite. Retention there has to be best-effort, and a value that
  cannot be snapshotted is lost across a crash-restart — reintroducing loss through the back
  door, narrowly.
- **Whether retaining the sub-agent's `ns` makes weaving ambient.** Messages are already woven
  by reference when a list is passed, so nothing changes there. But handing `ns` to the store on
  every `react_auto` return gives the caller the sub-agent's namespace whether or not it asked —
  and weaving is one of the items separating FULL from LEVELED, so making the sub-agent's interior
  automatically available might blur that contrast. The narrower option is to retain `ns` only in
  FULL, where weaving is already on the table.
- **The string cliff, until H1 lands.** Until `messages` is narrowed to a list, passing a `str`
  or a single `dict` forfeits the sub-trajectory. For the experiment, which runs the harness as
  released, this is disclosed in the task card as a harness semantic (S1.13b) so an author can
  avoid it deliberately.
- **Memory bound.** Ring-buffer depth is a free parameter — too shallow and a long-running policy
  loses its early calls, too deep and a loop over a large corpus retains everything. Interacts
  with A-normal form: hoisting keeps a reference to every intermediate alive until the call
  returns, so a policy that builds large objects holds more than it used to.
- **Transform overhead, unmeasured.** ANF adds a store and a load per hoisted expression. Policy
  runtime is dominated by sub-LLM latency, so the effect should be invisible, but that is an
  assumption until it is timed on a compute-heavy reference policy.

### Status

**Designed, not enabled.** Off for the authoring experiment: this study measures authoring
behaviour on the harness as released, and switching on a capability mid-design would change what
the reference policies are references for. Recorded as a clause so the choice is on the record
rather than implicit.

## R11 — RESEARCH: the experimental program, and the gaps the theory left unposed

*(Opened after R1–R10 were assembled. Two jobs: state the three experiments the theory now
demands, and record the questions the earlier nodes never explicitly posed.)*

### The three experiments

The theory implies three claims, and only the third has a design.

**E1 — the quality gradient is pervasive, not a long-context artifact.** The weak version of
this claim — "for some task, some policies do markedly better than others" — is already
established externally: RLM reports a median 130% gap over CodeAct-with-sub-calls, two systems
that both have code and both have sub-calls, differing mainly in whether the long prompt sits in
the context window or as a variable in the REPL. A gradient exists. Claiming only that would be
claiming something already shown.

**The real claim is about where the gradient lives.** RLM demonstrated it at one point in task
space: the regime where **context length is the binding constraint**, and where the shape
decision that matters is where the data is put. Nothing in that result speaks to tasks whose
bottleneck is something else. So the open question, and E1's actual subject, is:

> Is the policy-quality spread large **across task families with different binding
> constraints**, or does it appear only where context is the bottleneck?

This is the difference between PLM being a paradigm and PLM being a long-context technique. If
the spread only shows up where context binds, RLM already owns that ground and PLM is a
repackaging. If large spreads appear in families where context is comfortably sufficient — where
what binds is combinatorial search, or verification precision, or aggregation, or ambiguity in
the task statement — then the space is **pervasively differentiated**, which is a claim about the
space itself rather than about one corner of it.

Design consequence: **task families must be selected by what makes them hard, not by domain**,
and at least one family must be one where context length is plainly *not* the constraint. That
family is what answers the "RLM already showed this" objection, and it should be identified as
serving that purpose in the pre-registration rather than after the fact.

Pre-registered null (partial answer to Q2): if spread collapses in the non-context-bound
families, that is a real and reportable finding — PLM is a long-context method — and it must be
stated as such rather than absorbed.

Like the weak version, this does **not** depend on any model's ability to author. It is a
property of the space, measured by running policies *we* write.

**E2 — the gradient has a barrier, not just a slope.** Sharper and harder than E1. Is there a
*cutoff* — a class of policies that clears a task and a class that cannot, rather than a smooth
spread of scores? Quality here means doing the task properly and fully, so the question is
whether some tasks admit policies that succeed while the rest of the space plateaus below them.
A slope says "better authoring helps." A barrier says "below this shape the task is not done at
all," which is a much stronger statement and a different kind of object.

**E3 — the non-agentic part of the space is where the barrier lives, and models cannot author
into it.** Much of the policy space *is* agentic systems, and delegation is agentic. The
separation has to be shown for policies that are **not** agentic compositions — the region that
uses the raw capability without putting a bubble around it (R1). The authoring experiment
already designed (`experiment/code/stages.md`) is the second half of this: models given the raw
harness fail to reach that region.

**Why E1 and E2 are the strategically valuable ones.** Neither depends on the subject models'
authoring competence, so neither can be spoiled by a frontier release. They are entirely ours to
run, which makes them harder (nothing to lean on) and easier (nothing to wait for). And E2 in
particular is unwalked ground: work on programs, LLMs, and neuro-symbolic harnesses is almost
uniformly agentic systems under another name, so a demonstrated barrier in the non-agentic
region is a result with no incumbent.

### External position: what RLM has actually trained, as of 2026-08

*Checked against the primary sources rather than the abstract. Matters because this work will be
presented to those authors.*

**What exists.** Blog (2025-10, inference only, no training) -> arXiv 2512.24601 (post-trains
RLM-Qwen3-8B) -> RL work (2026-05, 4B models, parent and child RLMs under one shared policy) ->
**"Language model harnesses are compositional generalizers" (2026-07-20)**, the most recent and
the most relevant.

**What the RL actually trains.** Not tool-calls in the JSON sense: the action space is **code in
a REPL**, with `rlm_query()` for recursive child calls and `FINAL()` to answer. Reward is
rubric-based LLM judging on root rollouts only; children **inherit the parent's advantage**
(weighted 1/k) so no per-child reward is needed. Extended GRPO with a KL penalty, formulated
recursively for arbitrary depth. A 4B model reaches 0.6 average rubric score against Claude
Sonnet's 0.607 at 7s/query versus 60s+.

**The harness post's claim, which is close to R6.** That compositional generalization should live
in the **harness architecture rather than the network**: *"The primary job of the harness should
be to carry a higher-level inductive bias that can reduce unfamiliar and complex problems to
compositions of simpler ones for the underlying neural network."* Trained on short tasks with
Qwen3-30B frozen, generalizing to tasks **8-32x longer at roughly 10x the eval lift** over
training the transformer directly, and transferring across domains.

**The mechanism they credit is our access hierarchy.** Two things make each call "locally
in-distribution" and create *task equivalence classes*: **context offloading** (domain
information passes as symbolic variables instead of appearing in the main context) and
**programmatic sub-agent calling** (intermediate results stored in code variables instead of
appended to context). That is information at **handled** rather than **held**, and they now have
two independent empirical results for it -- the 130% inference gap and the generalization lift.
Strong external support for R9's equilibrium claim, and it is not ours to have to establish.

**What this costs us -- narrower than a first pass suggested.** An earlier reading of this node
called R6 "substantially anticipated." Re-reading R6 against the harness post, that is too
strong, and the distinction matters for how §on-training gets written.

*Anticipated:* the **framing**. That the thing worth training is a structural property which
generalises, rather than the objective itself; and that this property lives in the harness rather
than the weights. Zhang has published that with experiments -- 8-32x length generalisation,
cross-domain strategy transfer, base transformer flat on eval while its train reward rises. Any
draft presenting *that* as novel will be flagged, and it should instead be cited as support.

*Not anticipated:* the **content**. The two senses of "compositional" are different objects.

| | Zhang | R6 |
|---|---|---|
| what composes | **sub-problems** -- a novel task is solved by recombining familiar ones | **cycles** -- author, run, assimilate, repeat |
| the property | each call is *locally in-distribution*; the harness induces equivalence classes so the root sees token-for-token similar prompts | after each cycle the trajectory is **sufficient and minimal**, so a next cycle is possible |
| what it is about | the **distribution of the root's inputs** | the **information and budget state of the trajectory** |
| what is learned | which decomposition strategy to apply within one fixed harness | the property of `θ` that lets authoring continue indefinitely |

R6's invariant is closer to a **liveness condition** than to compositional generalisation in the
classical sense: it says the sequence can keep going, not that novel problems decompose into
familiar parts. Neither claim implies the other. A model could preserve R6's invariant while
never producing an in-distribution sub-call, and could satisfy LID while exhausting `L` after two
cycles.

**Positioning, therefore:** cite the harness post as evidence for the shared premise, and keep
the invariant as R6's own contribution. Do not present "structure generalises better than
weights" as ours; do present sufficiency-and-minimality-per-cycle as ours, and note that it is a
different property serving a different failure mode.

**What remains ours, precisely.**

1. **Authoring versus execution -- conceded in their own limitations.** Their harness is one
   fixed, hand-designed architecture and the model learns to operate it well. Stated limitation:
   *"Strategy discovery at scale hasn't been achieved; current focus is execution rather than
   discovering novel decomposition approaches."* Also: models still need *"elaborate strategy
   prompts"* because they lack native understanding. Discovery is exactly PLM's objective, and
   its absence is on the record.
2. **A space, not a point.** RLM has one harness. Their own criterion -- *"a good harness is a
   harness that reduces unfamiliar problems to familiar ones and reduces complex problems to
   simple ones"* -- presupposes harnesses of differing quality without characterising the space
   they range over. E1 and E2 are that characterisation, and their framing invites it rather
   than blocking it.
3. **Coverage.** Training one strategy for one harness is not coverage of the policy space.
   Nothing published addresses it.

**Consequence for positioning.** Trainability is settled and citable; claim it as background,
never as contribution. The live differentiators are **authoring/discovery**, **the space and its
gradient**, and **coverage**.

**Update 2026-08-24: Speculative Programmatic Tool Calling (sPTC).** New post plus an
open-source repo (`alexzhang13/spec-ptc`), one day after this assessment was written. A
**shadow REPL** \u2014 a deepcopied fork \u2014 speculatively executes tool calls from partially generated
code as tokens stream; the real REPL then returns cached results instead of blocking. Speculation
fires only on literals and variables computed by pure functions; I/O, non-determinism and unsafe
dependencies are blocked. Reported gain: **1\u20131.2x** on OOLONG with Qwen3-30B.

**Assessment: orthogonal, and it does not move any boundary.** It is *"exclusively a latency
optimization that doesn't alter harness expressiveness, capabilities, or interfaces"* \u2014 tool
calls are unchanged, only their timing shifts. The string contract is untouched, nothing
reifies a trajectory or a sub-agent deck, and the model neither authors nor reasons about the
mechanism. So the contract boundary (A1c) stands exactly where it was, and the reviewer's
instruction to scope the mapping to "the RLM design as published" still holds \u2014 sPTC changes
the harness's speed, not its region.

**Two things it does tell us.**

1. **The cadence is faster than quarterly.** July 20 harness post \u2192 August 24 sPTC is about five
   weeks. Revise the earlier "one substantial release per quarter" estimate; the time pressure
   on publishing is real.
2. **The effort is going into systems, not discovery.** JIT overlap tricks and serving-load
   engineering are the stated future work \u2014 *"the real value in the long run will come from more
   clever JIT compilation tricks."* That is orthogonal to the authoring question their own
   limitations section flags as unachieved, so **the gap this programme targets stays open**, and
   the evidence that it is being deliberately left open gets stronger with each systems release.

**One small consequence for our design:** wall-clock cost comparisons are now
implementation-dependent in a way that will keep moving. Our cost axes are recorded beside
correctness and never blended into a score (A4b), and no barrier verdict rests on timing, so
nothing needs changing \u2014 but wall-time numbers should be reported as observed rather than as a
property of either paradigm.

### Proximity assessment: is RLM heading into PLM's territory?

**Direction of travel: yes, unmistakably.** Feb 2026 "Language Models will be Scaffolds" argues
models are *underutilised* and the field should move from raw capability to scaffold-based
systems. July 2026 closes with: *"the capacity for compositional generalization looks like it has
to largely live in what today we refer to as a harness, but which in the future may blur quite
seriously with what we consider the fundamental architecture of our frontier AI systems."* That
is one step short of "the model writes the harness."

**Entry: no, and he has explicitly declined the move.** The clearest statement in the July post:

> *"It is easy to walk away from these powerful early results thinking that we should all be
> tinkering with harness designs or imposing our problem-specific intuitions around overly
> structured programmatic strategies such as MapReduce or dynamic programming. But make no
> mistake, doing that we will inevitably run afoul of the bitter lesson."*

**This is the most important sentence for our positioning, and it cuts both ways.**

*In our favour.* He has named the trap and left the exit unused. Hand-designing harnesses
violates the bitter lesson; his answer is to fix one general harness and train inside it. The
third option -- **the model authors the harness per problem** -- is PLM, and it is the
bitter-lesson-*compliant* answer to his own warning: no human intuition is imposed, because the
structure is learned and emitted rather than designed. That framing should be explicit in the
paper, because it converts his objection into our motivation.

*Against us.* A reviewer can read our fixed primitives, our sealed briefings and above all our
**48 hand-authored reference policies** as exactly the tinkering he warns against. Pre-empt it:
the references are *measuring instruments*, not the method -- they establish what the space
contains so that authoring can be scored against it. The method is that the model authors.

**The gap is admitted from two directions.** As a limitation: *"strategy discovery at scale
hasn't been achieved; current focus is execution rather than discovering novel decomposition
approaches."* And as a hedge on supervision: *"our intuition is that at scale, no supervision is
necessary, but for the sake of sample efficient learning, some form of supervision or hint is
helpful"* -- they use an explicit "nudge to decompose" variant, which is a human telling the
model which shape to use. Discovery is the unclaimed ground, twice over.

**An opening worth noting.** The appendix concedes that characterising whether trajectories are
*in-distribution* is *"very hard to formalize"* and needs *"a more rigorous treatment of this
topic."* Their central construct is admitted to lack theory. Ours is a definition with a
proposition behind it. That is a contribution to offer rather than a rivalry to press, and it is
the natural opening for the CSAIL conversation.

**Distinguish LID from held/handled -- they are adjacent, not identical.** Locally
in-distribution is about the *root's prompt distribution staying familiar* so training
generalises; the access hierarchy is about *what information the trajectory can reach and at what
cost*. Both are served by context offloading, which is why the same mechanism supports both, but
they are different claims and neither subsumes the other. Conflating them would be an overclaim,
and separating them cleanly is what lets us cite his result as evidence without appearing to
restate his idea.

**Cadence.** Oct 2025 inference -> Dec 2025 paper with 8B post-training -> May 2026 RL at 4B ->
July 2026 harness generalisation at 30B. Roughly one substantial release per quarter, with an RL
stack, a benchmark suite and CSAIL compute already in place. Time is not on our side, and the
defensible ground is the part he has declined rather than the part he has not reached.

### Execution versus discovery, and the bitter-lesson defense (sharpened)

**The terms, precisely.** In the RL-for-RLMs work, a *strategy* is the computational shape the
root writes for a task: chunk-and-map over the document, grep-then-read, hierarchical summarise,
or one big sub-call. *Execution* is operating a given shape well -- their models receive the
shape through "elaborate strategy prompts" and an explicit "nudge to decompose" variant, and RL
teaches reliable implementation: chunk sizes, child prompts, aggregation, stopping. Learning
*within* a shape. *Discovery* is inventing the shape when nobody hinted it. Learning *over*
shapes. "At scale" is their hedge, in two senses: at training-compute scale ("our intuition is
that at scale, no supervision is necessary" -- an unrun experiment) and at task-diversity scale
(beyond benchmark families where a human can pre-seed the right shape).

Two consequences. Execution/discovery *is* the authoring distinction, independently arrived at.
And the hedge is PLM's testable null: E3 measures discovery-without-training in frontier models
(their "elaborate strategy prompts" complaint is already evidence for E3's hypothesis that the
capability is absent), while R6/R7's training program is discovery-with-training. PLM
operationalises Zhang's own open question.

**The trilemma the bitter-lesson quote creates.** Three ways to relate humans to harness
structure:

1. **Hand-design structure per problem** -- MapReduce here, DP there. He rejects this, correctly.
2. **Freeze one general harness; train execution inside it.** His choice.
3. **Learn a generator over structures.** Unclaimed. This is PLM.

His warning targets (1) and cannot reach (3), because (3) removes the human from structure
entirely -- the shape is emitted by a trained model, per problem, with nothing imposed. Of the
three, (3) is the *most* Sutton-compliant: search and learning applied to structure itself.
And (2) is not innocent of priors: a frozen harness is a permanent structural intuition,
defended only by the bet that it is general enough -- a bet his own limitation ("strategy
discovery at scale hasn't been achieved") reports as not yet paying out.

**Where the bitter lesson could genuinely bite PLM, and the answer.** If the primitive set,
depth-1, or the harness contract were themselves smuggled strategies, the objection would
transfer. The answer is basis-versus-strategy: MapReduce is a *point* in the policy space; the
primitives are a *coordinate system* for it -- forced rather than chosen (R2: think-answer and
think-code-act are the elementary modes of a reasoning LLM), and expressively complete, so they
constrain learnability at most, never reachability. The bitter lesson objects to fixing points;
PLM fixes none. (The 48 reference policies stay measuring instruments -- the proximity
subsection above already covers that reading.)

**"RLM ignores ReAct's main problems" -- made precise, because expressivity will not carry it.**
RLM's REPL is Turing-complete with recursive sub-calls; on expressivity the two spaces nearly
coincide, and R1's design-trap note applies -- arguing space-versus-space by expressive power
loses. The real separators:

1. **What is reified as data.** RLM makes the *input prompt* an environment variable, and
   intermediates live in code variables. The root's own trajectory -- its rounds, its reasoning
   -- stays an append-only LM context it cannot touch. PLM reifies the trajectory itself:
   `plm_messages` is the root's history as a manipulable object, editable, delegable. RLM
   offloads the input; PLM manages the trajectory.
2. **The trained target.** Task reward with rubric judges, LID emerging as the explanation,
   versus the R6 invariant: per-cycle sufficiency and minimality of the trajectory's
   information state. One trains success; the other trains the property that keeps long
   sequences of authoring alive.
3. **Amortisation.** Policies are named, persistent, editable, reusable across the run;
   RLM children are ephemeral calls. Long-horizon work repeats subtasks; authored structure
   amortises, ephemeral delegation does not.
4. **The scaling axis.** All six of the harness post's benchmarks scale *input length*
   (64k to 2M tokens); none scale *horizon*. The three-days-versus-ten-minutes regime is the
   horizon axis, and LID does not address it: familiar calls do not stop the root accumulating
   rounds it can never restructure.

**Consequence for E1's task selection.** The cleanest family answering "RLM already showed
this" is not merely non-context-bound but **horizon-bound**: modest context, many dependent
steps. On that family RLM's mechanism has nothing to say, and PLM's whole apparatus -- weaving,
assimilation, the invariant -- is the thing under test.

### Turing-completeness is the vacuous comparison; the budgeted one separates (R2 applied)

A loose sentence used above -- "on expressivity the two spaces nearly coincide" -- is the
*unbounded* statement, and R2 already rules it inadmissible: **time must parameterise the space,
or the relation is vacuous.** Both REPLs are Turing-complete; so is a Turing machine with a
pencil. The comparison that means anything is between the *budgeted* reachable sets, and there
the limitation of the RLM space has three distinct characters, which should never be blurred
into one claim:

**1. Structural absence (no budget crosses it).** In RLM, two objects are simply not data:

- **The root's own trajectory.** It lives inside the LM context, not in the environment. A
  policy whose *input* is the root's history -- the floor one-liner, deck editing before
  delegation, verification over one's own reasoning -- does not exist in RLM's space at any
  budget. The nearest approximation is the root manually re-typing its history into a variable:
  token cost proportional to trajectory length, paid per use, lossy, and unreliable at length.
  PLM has the object for free (`plm_messages`, deep-copied per cell).
- **Child trajectories.** `rlm_query()` returns the child's *answer*; the child's rounds and
  reasoning are discarded by the interface. In the access hierarchy's terms they are **lost by
  construction** -- and by R9's irreversibility result, unrecoverable, since re-running a
  stochastic child produces different reasoning. PLM's weaving keeps the same objects at
  *handled*. So the hierarchy is not only our measurement lens; it marks a structural boundary
  of the rival space.

**2. Budgeted separation (grows with horizon).** Amortisation, restated as R2 demands: a named
policy is authored once and thereafter invoked at O(1) trajectory cost; an ephemeral child's
shape is re-emitted, or carried in the root's growing context, per use. Identical unbounded
expressivity, polynomially separated trajectory cost in the number of reuses -- which is the
horizon parameter again. The separation vanishes on one-shot short tasks and widens on exactly
the regime PLM targets.

**3. Coverage, not space (his hedge, in our vocabulary).** "At scale, no supervision is
necessary" is not a claim about the RLM *space* at all; it is a claim about **Coverage(θ)** --
that a scale-trained θ, inside one fixed harness, will come to emit the shapes that matter.
The space is fixed and syntactic; what θ reaches is the moving part (R2). Mapped this way, his
program and ours stop being rivals over one question and become answers to two different ones:
E2 interrogates the *space* (is there a barrier?), his bet interrogates *Coverage under scale*
(can a trained θ cross it unsupervised?). Neither result decides the other; they compose, and
E2's tasks are the meeting point.

**Consequence for E2's task families (third selection criterion).** The barrier, if it exists,
should be sought where the three limitations bind: **horizon-bound** tasks (limitation 2),
**trajectory-referential** tasks -- ones whose good policies take the trajectory or sub-agent
decks as input (limitation 1) -- and **reuse-heavy** tasks (limitation 2 again). A barrier found
there is simultaneously a positive E2 and a demonstration that the separating region is exactly
the part of the space RLM cannot enter cheaply, which is the strongest form of the result.

**Honest boundary of the claim.** For one-shot tasks of modest horizon whose winning shapes
reference neither the trajectory nor reuse, budgeted RLM and the budgeted PLM slice roughly
coincide, and no separation should be claimed there. E1's pervasiveness sweep must include such
families and report the coincidence where it appears -- that is what makes the separated
families credible.

### E2 witness classes, ranked by deflectability (design rule)

An E2 **witness** is the exhibit that establishes the barrier for one task: the triple
*(task, crossing policy, plateau evidence)* -- a task, an authored policy that clears it, and
the measured plateau of everything below it (the pre-registered fixed architectures, naive
delegation, and the scripted baselines) on the same held-out set. "Witness" in R1's sense: a
single exhibit settles the existence question.

Witnesses are pre-registered in two classes, by what their crossing policy *needs*:

**Class 1 -- shape witnesses (carry the E2 claim).** The crossing policy is built from plain
code and sub-LLM calls only: variables, loops, solver calls, branching on computed values,
k sub-calls where k derives from data, retry-only-flagged-items. Every ingredient exists in any
code-plus-LLM-calls REPL, including RLM's. Example shape: parse the instance symbolically; get
a bound from a solver; decompose into k subproblems *where k comes from the solver's output*;
one constrained sub-call per subproblem; symbolic check; retry only failures; assemble.

The point of choosing witnesses here is deliberate and strategic: **Class 1 lives in the region
where the budgeted spaces overlap.** So when the rival system does not cross the barrier, no
appeal to harness features is available -- the shape was expressible in his REPL all along, and
the only remaining explanation is that his system's trained behaviour never authors it. In R2
vocabulary: Class 1 separates **Coverage**, not contract. It converts his own admission
("current focus is execution rather than discovering novel decomposition approaches") from a
limitation statement into a measured result. That is the trap structure, and it is why the
headline claim rests here.

To be clear about ownership: expressible-in-principle in a REPL is not the same as being an RLM
policy. The policy space is syntactic (R2); systems differ in which points their behaviour
reaches. Class-1 witnesses are PLM-authored points in the shared region that rival behaviour
demonstrably does not reach.

**Correction (2026-08-24, author's ruling).** Weaving must not be classed as a harness
feature. It is **space-definitional**: the policy space is built from *raw* elementary
capabilities -- no pre-baked briefings, no fixed system prompts -- and rawness means the message
deck is an authored object. Weaving is what "raw" implies, not something the shell added. The
genuinely harness-specific machinery is the **constraint system**, which is used but never
load-bearing for a barrier claim. Consequently the classes below are **presentation guidance
only** -- they rank how a witness is argued, and do not constrain what E1/E2 author. Since E1/E2
policies are authored by us directly (not by a subject LLM), Class-1-style witnesses are
producible by construction; see `experiment/code/e_program_design.md`.

**Class 2 -- witnesses whose crossing policy uses weaving or persistent-policy structure.**
Argued from the space definition above, not defended as features. If the "we could add weaving
tomorrow" deflection is still raised, the precise answers remain:

- Adding the features moves the rival's **harness contract** toward PLM's. It does not make it
  PLM, because PLM is not a feature list -- it is the policy space, the authoring behaviour, and
  the training target; the harness merely encapsulates them (and is deliberately not the
  spotlight of the paper).
- **Access is not behaviour.** A harness with weaving available still needs a model that weaves
  well -- chooses what to hand over, edits the deck, assimilates what comes back. R9 already
  records that policies violate the retention condition routinely *even when access is free*.
  So a Class-2 witness paired with the authored behaviour still separates: the feature alone
  would not have produced the result.

The full space is used without restriction when *finding* barriers; the classes govern what each
witness is **claimed as**, never what may be authored.

### What rests on E2, and what does not (partially closes Q2)

**The dependency, stated honestly.** The *space* claim leans on E2. If E2 is positive -- a
barrier exists, and policies outside what execution-training inside a fixed harness can reach
clear tasks the rest of the space cannot -- then the richer representation is worth learning,
expressivity gains meaning, and discovery is worth training even though it is hard. The paper's
space pillar stands or falls with that result, and pretending otherwise would be spin.

**But the paper has two pillars, and only one is on E2.**

- **Pillar A -- the space.** E1 (pervasive gradient) and E2 (barrier). Carries: the policy-space
  derivation, non-collapse, the case for discovery training.
- **Pillar B -- meta-reasoning as trajectory management.** Weaving, assimilation, minimality,
  the R6 invariant, the horizon axis. Derived from ReAct's failure analysis, independently of
  the space argument, and untouched by any RLM result: his benchmarks scale input, not horizon,
  and his training never targets the trajectory's information state.

**Answer to Q2 for E2 (pre-registered interpretation).** A negative E2 does not kill PLM; it
re-scopes it. If no barrier appears in the tested families, the space's useful part collapses
toward what execution-training reaches, Zhang's postponement of discovery is vindicated, and
that is reported plainly. What survives is Pillar B whole: the trajectory-economy problem is
real regardless of whether a shape barrier exists, and the meta-reasoning apparatus is its
treatment. Negative-E2 PLM is "the trajectory-management paradigm" rather than "the
richer-space paradigm" -- smaller, still standing. This interpretation is fixed now, before any
data, which is what Q2 demanded.

**What the training claim is, and is not.** The paper does not claim the training is easy or
that it has been done. It claims the *target* is identified (the invariant), the *space*
justifies attempting it (conditional on E1/E2), and the instruments to measure progress exist
(the reference corpus, the metric umbrellas). Theory plus measurement, offered before results --
stated as such.

### The collaboration stance (for the CSAIL approach)

**The perception risk.** He deliberately postponed discovery ("current focus is execution");
someone arriving with PLM can be read as "going for the thing we said was premature." The
defusing frame, which is also simply true:

- **Upstream, not parallel.** E1/E2/E3 are not a competing training run; they are the
  measurements his agenda needs before it spends compute on discovery. Is there anything out
  there to discover (E1/E2)? Do frontier models already have it latent (E3 -- his own
  "elaborate strategy prompts" complaint says no)? Whichever way the results fall, they feed
  his decision, not just ours.
- **His hedge is currently untestable, and we make it testable.** "At scale, no supervision is
  necessary" is an empirical claim with no instrument behind it, because no benchmark exists on
  which it could fail: any current failure can be blamed on the task not needing discovery. E2
  supplies the discriminating instances. Negative E2: discovery was never needed, his
  postponement is vindicated, and we measured that. Positive E2: the hedge sharpens into a
  concrete test -- does a scaled, unsupervised RLM author the barrier-crossing shapes on E2's
  tasks? If it does, he is vindicated *inside our framework* (the space mattered; authoring
  emerged). If it does not, discovery needs a trained target, which is the invariant. Either
  outcome is stated in our terms. A falsifiable disagreement is the best footing available with
  a serious researcher -- frame the disagreement as an experiment, never as a debate.
- **The theory offer.** His appendix concedes LID is "very hard to formalize." The access
  hierarchy and the invariant are formal apparatus adjacent to his empirics. Offering theory to
  a group whose experiments outran their formalism is a welcome shape of collaboration, not an
  encroachment.

**One overread to drop from our own rhetoric.** His bitter-lesson quote bans a *methodology*
(humans imposing structural priors), not a *direction* (structure learning). It cannot be
paraphrased as "he told people looking at anything else is worthless," and we should not argue
against that version -- it is weaker than what he said, and attacking it would look like
strawmanning to the one reader who knows exactly what he meant.

### The floor argument (new — was never posed in R1)

R1 argues the space is non-empty at the *top*: somewhere in it sits a policy better than any
agentic composition. Nobody argued the **bottom**, and the bottom may be the stronger claim.

Take the worst reasonable PLM: it thinks about neither correctness nor computation shape, and
simply delegates with the same information ReAct would have had. Its authoring cost is minimal
because it authored almost nothing. What it gets back is a **work artifact**, and it assimilates
that rather than producing it.

The claim: **assimilating an artifact is cheaper than producing it.** Consuming a result costs
less trajectory than deriving it. So even this degenerate PLM matches ReAct's answer quality —
by the simulation argument already in R1, delegation with root-trajectory access reproduces
ReAct's output distribution — while paying less in trajectory. It gets the work for free.

If that holds, the conclusion is much stronger than non-emptiness:

> **The floor of the policy space already dominates ReAct.** Not "the space contains something
> better" but "the space is better everywhere," with meta-reasoning determining *how much*
> better rather than *whether*.

Consequences worth tracking:

- **What it does not establish.** That the margin is large. A floor that beats ReAct by
  epsilon justifies nothing. So the floor argument makes the interesting question **magnitude**,
  which is exactly what E1 and E2 measure: is the space rich enough to justify a paradigm and
  the cost of training into it?
- **The condition is satisfied by construction, not assumed.** An earlier draft of this node
  hedged the floor claim on the sub-call receiving the root trajectory, treating root-access
  delegation as an assumption that might fail. It is not an assumption. **Message weaving makes
  it a one-liner.** `plm_messages` is a public REPL global holding the root's own trajectory,
  rebound as a fresh **deep copy** every root cell (`repl/kernel.py:403-427`), and the copy is
  deliberate: a cell may append, clear, reassign, or edit individual turns without corrupting
  the accumulator the parent extends. So

      react_auto(plm_messages + [{"role": "user", "content": "continue"}], max_turns=k)

  hands a sub-agent the identical context the root holds at that round. The simulation is
  **constructive** — the harness supplies the materials — rather than existential.

  Two things follow. First, the floor claim carries no condition to state alongside it. Second,
  the PLM can hand over an **edited** deck, which is already outside anything ReAct can do to
  its own context: it can drop a misleading turn, sharpen an instruction, or splice in a result
  before delegating. So the relation is not merely "at least as good" but strictly richer at the
  floor.

- **The same fact appears in R1 as a threat.** R1 lists delegation-with-root-access as a
  *collapse threat*: if it already matches ReAct, and agentic composition captures everything
  beyond, PLM is over-engineered. Here the identical fact is the *floor*. Both readings are
  correct and they are not in tension, but the paper must not state one without the other, or a
  reader will find the seam. The resolution is R1's own: simulation establishes the floor, and
  what defeats the collapse threat is the region above it, which is E1/E2's subject.

- **Honest limits, none of which touch the argument.** The sub-agent is a separate call, so it
  reproduces the output *distribution*, not the sample. Depth-1 stops the sub-agent recursing
  further. And total token spend is not reduced — the sub-call is paid for; what falls is **root
  trajectory** cost, which is the resource PLM holds scarce. The deep copy also costs memory on
  long trajectories.

- **This is a supporting beam, not the thesis.** The floor argument is not the main argument of
  PLM and should not be presented as one. PLM's claim is the richer space and meta-reasoning as
  the training target. The floor's job is narrower: it removes "maybe PLM is just worse than
  ReAct at the low end" as a line of attack, so the whole dispute moves to magnitude, where
  E1 and E2 answer.
- **What it does to meta-reasoning.** Meta-reasoning stops being what makes PLM viable and
  becomes what makes it *worth training* — the difference between the floor and the ceiling.
  Minimality and assimilation are the levers on that difference. This reframing is compatible
  with R9's definition and should be reflected when §3 is rewritten.

### Questions the earlier nodes never explicitly posed

Checked R1–R10 for stated uncertainty. R5, R6, R8, R9 and R10 carry explicit open registers;
R3, R4 and R7 carry open items under other headings; **R1 and R2 carry none.** They are the two
most load-bearing nodes in the arc, and they record no uncertainty at all. The following were
assumed rather than asked.

**Q1 — Is the quality relation a gradient or a barrier?** R1(a) says only "there exist tasks
that one policy does better than another." Whether that difference is continuous or admits a
cutoff was never distinguished, and the two support very different papers. Now E2.

**Q2 — What does a negative result mean?** R1 says a single witness settles non-containment.
It never says what a *failed search* for a witness establishes — nothing, weak evidence of
collapse, or a bound on where separation can live. For a pre-registered design this has to be
answered before the run, not after, or a null result becomes unpublishable and unfalsifiable at
the same time.

**Q3 — Is the system-prompt slice stated as a restriction?** The policy space as measured is a
*slice*: the system prompts of the sub-calls are themselves part of the program, so the true
space is far larger than anything that can be sampled. Every result is therefore relative to the
slice in which those prompts are authored the way we author them. R2 fixes what `θ`
parameterises but does not, as written, foreground this as a scoping assumption. A reviewer will
find it; better to state it first.

**Q4 — How does the sample of policies get chosen without begging the question?** E1 and E2
measure a spread over policies *we* write. If we write two bad ones and one good one, we have
produced the gradient rather than found it. The sampling procedure needs to be pre-registered
and adversarial — including policies written to be strong under a different theory of the task —
or the result is an artifact of the author's beliefs. This is the single biggest threat to E1
and E2, and it has no answer yet.

**Q5 — Does better search close the authoring gap?** R8 opened "how do we characterise policies
that *search* the policy space more thoroughly than ReAct" and left it. It was abstract then. It
is load-bearing now: E3's claim that models cannot author into the non-agentic region invites
the reply *"not with that scaffold — try search."* Until there is a story about what a good
policy-search looks like, E3 is a claim about one authoring shape rather than about models.

**Q6 — Is the floor argument above actually sound?** Posed here for the first time; see the
failure condition already noted.

### Exhaustiveness of the E-program (confirmed, with two free claim-extractions)

The question: is {E1, E2, E3} exhaustive of what can be shown *now* -- excluding meta-reasoning
behaviour (needs a trained model) and training (needs training runs)?

**The syllogism the three results form is the complete pre-training argument:**

1. **E1/E2: the space is worth searching.** Value exists in it, pervasively (E1), and for some
   tasks only there (E2).
2. **E3: current frontier models cannot reach it unaided**, under maximal elicitation.
3. **Therefore training the behaviour is motivated, and the theory names the target** (the
   invariant, R6) and the definition (R9).

Nothing in the motivation-for-PLM argument requires a fourth experiment. The audit of every
testable-now claim in R1-R11 and the definition doc turns up only two items outside the three
experiments, and both are **claim-extractions from arms that already exist**, not new runs:

- **The floor corollary (A7 in the design doc).** The floor argument -- weave-delegation
  matches the bare agent's output distribution while the root trajectory stays near-constant --
  is currently a priori. It becomes empirical for free: compare L1's held-out quality against
  the bare-executor baseline, and compare root-trajectory token cost. Parity + cost gap =
  the floor measured. No new arm needed.
- **Capability profiles as a by-product.** The paper argues each primitive has a behavioural
  profile (reliable regime, degrading regime). A3's budget-scaling curves *are* the degradation
  profiles of `react_auto` under growing rounds -- the flat region is the profile's ceiling.
  Report them as such; the "capability behaviour" section gets its evidence without a dedicated
  study.

**The one honest residual, correctly deferred: Q5.** A skeptic can say E3 shows only that
*one authoring shape* (single-pass, ReAct-shaped P1) fails -- a subject wrapped in a
policy-search scaffold might close the gap. That is the only genuine candidate for a fourth
experiment, it is already recorded as Q5/R8-open, and it belongs in limitations and future work
for v1, not in the program. Everything else a skeptic can raise (under-prompting, budget,
harness withholding) is already answered inside E3's design (maximal elicitation, S1.13b,
cost disclosure).

**Also excluded correctly:** Coverage(theta) claims (about trained models), the R3/R7 training
questions, and Section 2's failure analysis (carried by literature plus the existing trajectory
evidence, not by new experiments).

### What this changes about the built experiment

`stages.md` and `measurement.md` design **E3 only**. E1 and E2 have no design, no clauses, and
no metrics. They are not modifications of the authoring experiment — they are two new
experiments that happen to reuse its infrastructure.

The reuse is substantial and is the reason to run all three together:

| asset | built for | also serves |
|---|---|---|
| task cards, dev/held-out splits | E3 | E1, E2 |
| grading harness (accuracy/F1 on held-out) | E3 | E1, E2 |
| the 48 reference policies | E3 | E1, E2 — they *are* a sample of policy space |
| scripted baselines (inline, naive-delegate) | difficulty labels | E1's floor, and the floor argument above |

What is missing is **density**. Four references plus two baselines per task is roughly six points
— enough to show a spread exists (E1), too thin to locate a cutoff (E2). E2 needs a deliberately
denser sample per task, spanning from clearly-inadequate to clearly-sufficient shapes, and it
needs tasks chosen because a barrier is *expected* rather than tasks chosen for authoring
difficulty.

### Ordering

E1 and E2 do not depend on the meta-reasoning definition settling, on subject-model access, or
on the Gemini key. E3 depends on all three. So the program runs E1 → E2 → E3, and Phase 0 work
already underway is the shared prerequisite for all of it.

## R12 — HARNESS (future): think-anywhere rounds, and why sPTC layers onto them

*(Design note, not scope for the current paper. Recorded so it is not lost.)*

**The variant.** Rounds as currently defined alternate think-then-code at round granularity.
The **think-anywhere** variant lengthens a round so the model can think *between* writing pieces
of code, and edit code it has already written within the same round. `operations.md` already
records the ruling that this is **invariant for the authoring experiment** \u2014 K-round authoring
interleaves thought and code at round granularity, so nothing in E3 depends on which of the two
is used. That ruling stands; this node is about the harness's future, not the experiment.

**Why sPTC (2026-08-24) composes with it unusually well.** Speculative programmatic tool calling
executes tool calls from *partially generated* code in a shadow REPL while tokens are still
streaming. Think-anywhere rounds produce exactly the structure that feeds on: long rounds
containing **code buffers separated by stretches of thinking**. Each buffer is a speculation
target that can run and cache while the model is still thinking about what comes next, and the
edit-in-round capability means a buffer superseded by an edit simply discards its cached result
\u2014 the shadow/real split already tolerates that, since the real REPL never executes speculated
state. The two ideas were arrived at independently and the fit is structural rather than
coincidental.

**Not to be mentioned in the paper.** sPTC is an optimisation technique, not an idea, and this
paper is about the policy space and the behaviour trained over it. Citing a latency trick would
dilute the contribution and invite the comparison on the wrong axis. Kept here for the harness.

## R13 — HARNESS (future): the primitives, renamed and de-bundled

*(Author, 2026-08-25. The rename is DONE across all three harnesses; the de-bundling is
recorded for the PLM harness's future and is out of scope for the E-program, which runs on
capabilities as released.)*

### The rename: `natural_llm` → `llm`

The capability is the reasoning LLM itself: **think, then answer**, where the answer is a
string or, under a constraint, a constraint-shaped generation. That transfers directly from
the abstract capability the policy space is built over (R2): sequence-to-sequence generation
with tokenisation internal to the capability, and constraint-directed generation covering the
"algorithms on generation" half. Since it is *the* capability, it carries the plain name.
`react_auto` keeps its name — it is an LLM used in the shape these models are additionally
trained for.

### The de-bundling question: autonomy is not elementary

`react_auto` currently bundles three things: (1) the elementary **round** — one think+code step
over a message state; (2) an **autonomy loop** — run rounds until done; (3) a **termination
protocol** — the RETURN obligation, final-round warnings, retry slack. Only (1) is elementary.

**The concrete failure this causes.** A policy cannot build `react_verifier_llm` out of
`react_auto`. The natural construction — run one round, let a verifier edit the message list,
feed it back, repeat — is blocked because every 1-round call injects the termination pressure:
the sub-agent is told to RETURN, behaves like an agent finishing, and the state that comes back
is shaped by an obligation the composition never wanted. The bundled primitive is *too
autonomous to compose*.

**Direction: capability = the round; autonomy = an authored policy.** A step-mode primitive
(one or k rounds over a given message state + namespace, NO return obligation, state handed
back to the caller) makes the current `react_auto` just a *default policy composed from it* —
loop + termination protocol — and `react_verifier_llm` another such composition. This also
strengthens the elementarity story the paper tells: the primitive set stays elementary, and
autonomy becomes visible as what it is — authored structure.

**Scope.** FULL and the future PLM harness: redesign candidate. Agentic and LEVELED: keep the
autonomous form — there the sub-agent *is* the bubble, full autonomy is the point, and the
round-driving capability would be beyond the string contract anyway (it hands back trajectory
state — which is exactly why it belongs above the boundary, with weaving). For the E-program,
L4's process-verification needs are already served by `react_verifier_llm` in FULL.

**Round-7 addendum (reviewer).** When R13 is written up, note explicitly that the step-mode
primitive *hands back trajectory state*, which places it **above the contract boundary** with
weaving — consistent with the A1c taxonomy rather than an exception to it. And the paper's
wording guard: `react_auto` is never called "elementary" unqualified; the frozen phrase is
"minimal working repertoire", citing R13.

### R13 RESOLVED (2026-08-25, final same-day revision): `react` + `react_auto` — functional, no state object

Two same-day author corrections shaped the final design. First, a session OBJECT
(`a.step()`) was rejected: capabilities in this framework are FUNCTIONS (R2), and an
object with behaviour imports a second type discipline into a space whose story is
functions composed by code. Second, the replacement's `task`-plus-returned-state surface
was rejected too: **the caller already holds the state — the woven message list.** A
returned state value was double bookkeeping.

**The final surface (FULL only):**

    msgs = [{"role": "user", "content": "Analyse the table in `df`."}]
    react(msgs, rounds=1, system=..., kwargs={"df": df})   # base capability -> None
    msgs.append({"role": "user", "content": "[verifier] recheck step 2."})
    react(msgs, rounds=2)                                  # same session: ns persists
    ans = react_auto(msgs, max_turns=4)                    # termination protocol -> value|None
    ans = react_auto("whole task", max_turns=8)            # fresh = react_auto behaviour
    ans = react_auto(msgs, max_turns=0)                    # COLLECT a spontaneous RETURN

**One capability, two forms.** `react` is the trained interleaving — think+code rounds
over the weave, `rounds` REQUIRED, no termination pressure, returns None. `react_auto`
is the autonomy protocol over the same machinery (budget briefing, final-round pressure,
RETURN obligation) — behaviourally `react_auto`, plus the ability to continue a stepped
session. The paper's wording: the repertoire is {`llm`, `react`}, with `react_auto` the
closed form; never "three capabilities".

**Mechanics.** The message LIST is the session handle: the sub-agent's private namespace
persists across calls keyed to the list's identity (strong-ref registry, entry dropped
when a finish call completes; bounded; id-reuse guarded by an identity check). `react`
takes a list ONLY — a str would create state the caller could never reach again;
`react_auto` accepts str/dict/list like `react_auto` (the bubble needs no handle).
Spontaneous RETURN during `react` is RECORDED, not returned — `react` stays None-typed
and the value is collected via `react_auto(msgs, max_turns=0)`. A constraint violation
on a mid-step RETURN feeds back into the weave and the session continues. Stepping a
concluded session raises. Construction kwargs on an existing session raise. Under
`parallel()`, branches copy message lists, so sessions are branch-isolated by
construction. `react_auto` is untouched (an alias decision waits for the E3 cards).

**Status.** Implemented and blessed in FULL (`react`, `react_auto` in the blessed-caller
set, three harnesses byte-identical); syntax and diff-surface tests green; the
END-TO-END acid test is rewritten for this surface and QUEUED — WSL egress died
mid-session (VPN blocking WSL TLS), so v3's kernel run is pending network return. The
object version passed the same acid test on identical internals earlier the same day.

**Round-10 conditions (reviewer; verdict SOUND, all landed).** (1) Budget disclosure holds in
`react()` too — the sub-agent is told its granted rounds per call (`STEP_ROUNDS_NOTE`);
disclosure is hygiene, pressure is protocol. (2) The protocol ships as inspectable constants
(`STEP_BRIEFING`, `AUTO_BUDGET_BRIEFING`, `AUTO_FINAL_ROUND`) so an author can reproduce
`react_auto`'s exact protocol over raw `react()` verbatim — closing the one inexpressible case.
(3) The sub-call log carries a four-way outcome — `concluded` / `still-open` / `collected` /
`truncated` — so a collect returning None is never mistaken for starvation at extraction.
(4) Eviction leaves a tombstone and ANY touch of an evicted session **raises
`SessionEvicted`**, never silently reopens; id collisions raise too; every fresh session open
is logged (`react.open`), so branch-copied lists under `parallel()` are visible. (5)
Kernel-respawn session loss joined the A5f(a) harness-failure taxonomy as item 5. (6) Wording
guard, frozen: **`react_auto` is never called a capability** — it is the closed autonomous
form, itself a composition over `react`; writing otherwise would contradict R13's own
autonomy-is-authored theory. (7) The E2E acid test **gates any policy authoring on this
surface** — D1's pass on identical internals is evidence, not a pass.

**v3.2 (author, same day): the rough edge closed — RETURN is protocol-only.** The author asked
the right question: if a raw `react(msgs, rounds=5)` sees a RETURN at round two, "what are we
supposed to do? It's not react_auto." v3.1's answer (record and stop early) was WRONG — early
termination is protocol behaviour smuggled back into the capability, the exact leak R13 exists
to remove. Final semantics: **RETURN does not exist in raw `react`.** rounds=k runs k rounds,
full stop; the step-mode `RETURN` binding is a sealed stub whose call produces an in-weave
redirect ("not active in this working stretch — record your result; you will be asked to
finish later"), so a return-trained model self-corrects into state — which IS raw ReAct
behaviour. RETURN, the obligation, and `constraint` all arrive only with `react_auto`'s
protocol (the stub is swapped for the real sealed RETURN when the protocol begins, and
`constraint` is legitimately settable at finish time even on a continued session). Downstream
simplifications, all shadows of the same leak: spontaneous conclusion is impossible, so the
collect channel and the `max_turns=0` special case are DELETED (`react_auto` requires
`max_turns >= 1`), and a finished session is tombstoned so a later `react()` on its list raises
rather than silently reopening.

**v3.3 (author, same day): the stub itself was a leak.** The v3.2 redirect — "not active in
this working stretch; you will be asked to finish later" — COACHES: it hands the capability
protocol foreknowledge. A base capability does not explain the protocol. Final semantics:
**RETURN is simply undefined in raw `react`** — no stub, no redirect; a model that calls it
gets the raw consequence of any undefined name, a NameError in its tool output, and the step
briefing shrinks to pure capability text ("you work in rounds; variables persist"). The name
RETURN — bound, briefed, and demanded — arrives only with `react_auto`. Disposition logging
keeps counting attempts via the NameError signature (`react.return_attempt`, outcome
`name_error`). Re-verified end-to-end after the change: all six deterministic acid checks
pass, including "step briefing carries no RETURN instruction".

## C23 — auto consistency (satisfiability) check on constraint construction

**What:** make `Constraint.field(...)` construction auto-run a fast satisfiability check and
raise a new `ConstraintConsistencyError` when the constraint is *provably* unsatisfiable —
instead of silently building a constraint that nothing can ever pass (e.g. one field given both
`int_ge=5` and `int_le=3`, or `str_length` with min > max).

**Why deferred (complex, sizable, its own change):** general satisfiability is
**undecidable** here — arbitrary Python `predicate=` / `coercer=` bodies are
opaque, so the algo can never reason about them. It must therefore be **sound**
(only ever raise when *provably* empty — NEVER a false positive on a predicate it
can't see) and confine itself to the **declarative** kwargs of the field. That's a
real, self-contained algorithm + a `ConstraintConsistencyError` type + auto-wiring
into the field builder + a focused test matrix — too big to fold into a LOW-cluster pass.

**Scope when built (sound, declarative-only):**
- Numeric: `int_ge=5` with `int_le=3`, `int_gt=5` with `int_lt=5`, an empty `*_range`.
- Length: `str_length` / `list_length` with min > max or negatives.
- Membership: `not_one_of` covering every value a required `kind`/`Literal` allows; an empty
  `Literal[...]`; a `Literal` value that can't match a required `str_pattern`.
- Pattern: `not_pattern` equal to a required `str_pattern`.
- Predicate / coercer kwargs contribute "unknown" → never trigger a raise.

**How:** inspect the field's declarative kwargs at build time and raise if their intersection is
provably empty. Run automatically inside the field builder. Confirmed scope with user (sound /
declarative-only; predicates stay unchecked).

## R6-4 — `_tool_python()` flat tool shape is rejected by Chat-Completions backends

**What:** `_tool_python()` (`plm_tools.py`) emits the flat Responses-API tool
shape `{"type":"function","name":...,"parameters":...}` (no nested `"function"`
wrapper). `BaseModelBackend._sanitize_and_cache_tools` only rewraps tools that
already have a nested `"function"` key, so the flat shape passes through unchanged
to `chat.completions.create(tools=...)`, which the Chat-Completions API rejects
(missing `tools[0].function`). Affects every Chat-Completions path: VLLMBackend,
SlateBackend, and `OpenAIBackend` with `use_responses_api=False` (the default for
non-reasoning models). The OpenAI **Responses API** path accepts the flat shape,
so it works there.

**Why deferred:** confirmed with user — PLM runs **reasoning models throughout,
including sub-LLMs**, which auto-select the Responses API (`use_responses_api`
auto-detects True for o-series/gpt-5), where the flat tool shape is correct. The
Chat-Completions code is KEPT but is not on the active path, so this is a known
low-priority gap (user: "don't worry too much about it").

**If picked up:** make the wire-call paths wire-format-aware — normalize a
function tool to the nested `{"type":"function","function":{...}}` form for
Chat-Completions backends (in `_sanitize_and_cache_tools`, recognizing BOTH
shapes), and flatten nested→flat in the OpenAI Responses path. Add a per-backend
tool-shape round-trip test.

**Related symptoms (round-2 sweep — same root cause, deferred together):** the flat/nested
duality also surfaces as `AnthropicBackend._convert_tools_to_anthropic` hard-indexing
`tool["function"]["name"]` → KeyError on the first call (Anthropic is listed in
`_BACKEND_DEPS` but PLM ~never uses it); `_get_tools_signature` hashing only names (not
params) and giving flat tools an empty signature; `_sanitize_and_cache_tools` skipping
name-sanitization for flat tools; and `AnthropicBackend.generate` never sanitizing. ALL are
LATENT in the active wiring (one fixed flat `python` tool, openai/vllm) and collapse into the
single shape-aware-accessor fix above — a `_tool_def_fields(tool)` helper read from BOTH the
signature, the sanitizer, and the Anthropic converter. Not worth doing until Anthropic or
multi/varying tools are actually used.

## D6 / S-F6 / S-F7 — kernel IPC trust model (pickle over a Unix socket)

**What:** the parent `pickle.loads` every frame the kernel sends and `dill.loads`
every `RETURN` value (`repl/session.py`). Deserializing attacker-controlled bytes
is arbitrary code execution (via `__reduce__`), so a 3rd party that became the
socket peer could run code IN THE PARENT (the trusted meta-reasoner), not just in
the sandboxed cell.

**Status (research stage — known + accepted):** the design is
`cell-trust == parent-trust` — the kernel runs PLM's OWN cell code, and pickle is
simply how the two exchange Python objects. The only real exposure is a 3rd party
hijacking the channel, and that is now mitigated cheaply: the handshake socket
lives in a private **0700 control dir** (no longer a world-reachable `/tmp` path),
so another user can't connect to it (**D6 — done**). Same-user is not a trust
boundary.

**Before production (optional defense, intentionally NOT implemented now):** add
`SO_PEERCRED` verification on `accept()` (reject a peer whose uid != ours) AND
restrict deserialization with an allowlist `find_class` `Unpickler` — or move
frame framing to JSON, keeping dill only for `return_blob`/`vars_blob`/`seed`. The
allowlist is awkward for arbitrary RETURN values, so this stays deferred until PLM
leaves the single-trusted-user research setting.

## D5 — crash-restart fidelity: constraint objects (DONE) + remaining rare gaps

**Constraint OBJECTS now SURVIVE crash-restart** (was a gap): a `Constraint.field(...)` and a
structural `class X(Constraint)` stored in a variable are snapshotted as a picklable RECIPE and
rebuilt on rehydrate (`constraint/snapshot.py` — `to_recipe`/`from_recipe`/`is_constraint_class`),
mirroring how a class POLICY survives by its `_p_source`. A field replays its construction;
structural classes reconstruct from `model_fields` + the validator decorators (fields, defaults,
Field bounds, and model/field validators all preserved; arbitrary METHODS on a constraint class are
not — rare). The recipe path is REQUIRED because no Constraint class cross-process dill-pickles (a
dynamic factory class recurses; structural pydantic classes pickle by-reference and would otherwise
sink the whole snapshot). Verified end-to-end
(`test_int_crash_restart_constraints_survive`).

**Remaining (still deferred — rare):**
  - live INSTANCES of a callable `@policy class` — the class SOURCE is replayed, but an
    instance a cell built and holds is gone (its depth-cap `__call__` wrap closes over a
    ContextVar, so dill can't pickle the instance). (WF Corr#17)

**Status:** acceptable for research. Contract for the remaining case: after a respawn
(detectable via `kernel_epoch`), rebuild any held callable-class instances.

**If picked up:** for callable class instances, snapshot a construction recipe (class +
init args) rather than the unpicklable instance.

## S-F9 / S-F13 — kernel↔parent IPC robustness (work today; low)

**What:** **S-F9** — a SIGINT arriving in the kernel mid frame-write could desync the
protocol (the outer except writes a second frame atop the partial first → the parent reads
a garbage length → respawn, losing the cell result). **S-F13** — `_write_frame` doesn't
loop on a SocketIO short-write (relies on blocking-AF_UNIX behavior; an EINTR mid-send could
short-write → framing corruption).

**Why deferred:** both work today on blocking Linux AF_UNIX (the full lifecycle/respawn
test suite is green); the fixes touch the IPC hot path and are riskier than the value.

**If picked up:** S-F9 — `pthread_sigmask(SIG_BLOCK)` around the kernel's frame write (or a
flag-setting SIGINT handler). S-F13 — switch the writer to a `BufferedWriter` (buffered) with
a single `flush()` after writing the header+body, both parent (`_write_frame`) and kernel.

## Corr#4 — fresh `@policy class` re-execs its body (double side-effects)

**What:** installing a fresh `@policy class` re-execs the class SOURCE under `<policy-{name}>`
so its methods' code objects live in the stable linecache slot, not the disposable
`<cell-N>`. Side effect: a class with TOP-LEVEL side effects in its body runs them twice
(the cell already ran the body once).

**Why deferred (documented constraint instead):** the re-exec is REQUIRED for
getsource/traceback fidelity (and rehydrate already replays the source); reusing the cell's
class object would push tracebacks/getsource onto the disposable `<cell-N>` slot. So the
contract is: **keep policy class BODIES side-effect-free** (define methods/attrs only; put
side effects in `__init__`/methods). `def` policies are unaffected (re-execing a `def` is
side-effect-free). Comment lives in `policy/decorator.py`.

## P-F3 / P-F5 — policy-core micro-gaps (latent / undetectable)

**P-F3 (RESOLVED by NP3-1):** a class policy's immutability used to be bypassable via
`cls._p_immutable = False` (class policies are raw types with no `__setattr__` guard like the
function-policy proxy has). This became reachable once metaparams shipped sealed CLASS extras.
FIXED: the seal mutation gates AND `registry._is_default` now anchor to the canonical seal record
by OBJECT IDENTITY (`registry._is_sealed_obj`, membership of `_SEALED_POLICIES`), so flipping the
writable flag — or renaming `__name__` — can't un-seal, for BOTH function and class policies. The
per-object flag is now a fast-path cache, not the authority; no metaclass needed.

**P-F5 (undetectable at decorate time):** `@policy` doesn't note when it shadows a
pre-existing non-policy global of the same name — but the `def`/`class` statement overwrites
that global BEFORE `@policy` runs, so there is nothing left to compare against. **If picked
up:** detect at the cell-audit level (pre-exec), not in the decorator.

<!-- P-F7 removed: NOT a bug. A direct `_PLM_POLICIES[name]=x` (bypassing @policy) being
cleaned up by `_sync` is the registry being correctly STRICT — @policy/_install_policy_source
are the only install surfaces (they bind both the registry and __main__). Documented in the
`_sync` docstring; nothing to fix. -->

## DOC-NC3-3 — document that constraints do NOT coerce

**What to document (README + Constraint docstrings):** `is_instance_of` — and the constraint
type checks generally — are a STRICT `isinstance` check with **NO coercion**. A value must
already be the exact type: `is_instance_of=float` rejects an `int` (no `5`→`5.0`),
`is_instance_of=<Model>` rejects a `dict` (no build-from-dict), `is_instance_of=int` rejects a
`bool`, and a subclass instance is returned UNCHANGED (never flattened to its base). Callers must
not expect pydantic-style coercion — pass the value already in the target type.

**Why a TODO (not code):** the behavior is correct + tested as of NC3-3 (`_strict_base` uses
`pydantic_core.is_instance_schema`); only the user-facing docs need the explicit "no coercion"
statement so the contract is discoverable without reading `constraint/base.py`.

## DOC-NR3-1 — document the cell-crash contract: RE-RUN, don't trust crash survivors

**What to document (README + kernel/REPL contract):** if a REPL cell crashes or times out
mid-run, the kernel is respawned from the LAST SNAPSHOT (taken at the end of the previous
*successful* cell), and the result carries `executed: False` + a "RE-RUN this cell" note. The
contract for PLM/the model is: **re-run the cell.** Do NOT reason about "what state the crashed
cell left behind" — the in-flight cell's partial effects are gone; only the pre-crash snapshot
survives. State is always restored to the last clean boundary, never a partial mid-cell state.

**Why a TODO (not code):** the behavior is implemented + tested as of NR3-1 (`_respawn_result`
sets `executed: False` on every respawn path with a RE-RUN note); only the docs need to spell out
the re-run contract so PLM doesn't try to inspect crash survivors or expect partial state.

## LIMIT-NC3-2 — RESOLVED: super() in a constraint is rejected at definition

**Resolution (chosen over making it dill-survivable):** a `super()`-calling method on a Constraint is
now REJECTED at class definition with a clear error — `_ConstraintMeta.__init__` scans the class's
own methods and refuses any whose own bytecode loads `super` (`'super' in co_names`). That catches
BOTH zero-arg `super()` and explicit `super(C, self)`, and — unlike a `'__class__' in co_freevars`
check — does NOT false-positive on a method whose NESTED helper uses super (F5). The rarer residual
`__class__`-cell cases the narrower check intentionally allows (a nested-helper super, a bare
`__class__` reference) are handled defensively by `from_recipe`'s cell-rebind, which is therefore
NOT dead code (F3). A Constraint is a FLAT
structural validator (fields + @field/@model_validator + predicates), NOT an OOP inheritance
hierarchy; `super()` implies inheriting + delegating to a parent constraint, which the model doesn't
support (and which couldn't survive crash-restart anyway — the method's `__class__` cell can't
dill-pickle). So the crash-restart-drop case is now UNREACHABLE: you can't define such a constraint.

**Notes:** the scan runs in `__init__` (NOT `__new__`) — pydantic resolves forward refs by walking
frames in its metaclass `__new__`, so an extra `__new__` would break nested/forward-ref constraints;
`__init__` runs after that. It gates on `FunctionType` before touching `.__code__`, so a pathological
class member (raising `__getattr__`) can't break definition (NC-8). Field-only INHERITANCE
(`class B(A)` with no super() methods) is NOT rejected — pydantic flattens inherited fields into
`model_fields`, which the recipe already handles. The NC3-2 `from_recipe` cell-rebind is kept as a
cheap defensive backstop for a method monkey-patched onto a class AFTER definition (bypasses the
metaclass), but the primary guarantee is now the definition-time reject.

## LIMIT-NR3-5 — Guard C+ restores modules by reference (attr-mutation not undone)

**What:** `_REPL_INJECTED_CANON` (repl/prefix.py) restores kernel-internal modules a cell may have
REBOUND (`_dill = None`) — by reference. That undoes a rebind but NOT an attr MUTATION
(`_dill.dumps = lambda: b""`): the module object is identity-unchanged, so the poisoned attr
persists into the next cell's snapshot/frame-I/O.

**Why deferred (LOW, graceful):** the snapshot/frame-I/O paths are wrapped in try/except and degrade
to "no snapshot this round" rather than corrupting state, and the trust model is cell-trust ==
parent-trust (see D6/S-F6), so a cell deliberately poisoning a module attr is within the accepted
boundary. A full fix is broad: capture the bound methods kernel internals actually use
(`_dill_dumps = dill.dumps`, ...) at boot and call THOSE, so a later `module.attr =` can't reach
them. Deferred as disproportionate to a LOW by-inspection gap; the comment at the CANON now states
the real behavior.

**If picked up:** boot-capture the kernel-internal module methods (dill/pickle/struct/io/linecache)
as `_repl_`-prefixed bound references and use them throughout the snapshot + frame-I/O helpers.

## NEST-IDENTITY — a nested struct is rebuilt as a DISTINCT class across crash-restart (instance-value validate path only)

**What:** when a struct `Outer` references another top-level struct `Inner` (field `inner: Inner`) and
BOTH are separate snapshot variables, crash-restart rebuilds each from its OWN recipe — so the rehydrated
`Outer` references an `Inner` class that is a DIFFERENT object than the rehydrated top-level `Inner`. The
dominant idiom (validating a dict/JSON — `Outer.validate({"inner": {...}})`) works: pydantic builds the
inner from the dict. Only passing a PRE-BUILT instance of the separately-rehydrated `Inner`
(`Outer.validate({"inner": Inner(...)})`) is rejected ("not strictly an instance of Inner").

**Why deferred (rare + case-specific):** needs inner + outer BOTH as top-level vars, a crash-restart
between define and use, AND a call passing a CONSTRUCTED instance (not a dict). A held struct INSTANCE
var doesn't survive restart anyway, so the trigger only arises if the model freshly builds an instance
post-crash and nests it. Same non-dedup root as C25; `describe()`/`json_schema()` (schema-based) are unaffected.

**If picked up:** dedup nested recipes across the snapshot so a shared `Inner` rebuilds to ONE class (a
recipe-id registry rehydrated once), mirroring how policies dedup by name.

## OVERSIZE-VAR — an oversized snapshot var is dropped with NO rehydrate note

**What:** in `_repl_collect_vars` (`repl/prefix.py`) a var whose dill-serialized form exceeds
`_REPL_VAR_SIZE_MAX` (default 10 MiB) is `continue`d WITHOUT appending to `_var_dropped` — unlike the
adjacent unpicklable branch — so it vanishes across respawn silently and a later cell hits a surprise
`NameError`.

**Why deferred (case-specific + documented best-effort):** normal PLM meta-reasoning never binds a single
>10 MiB picklable module-scope global (that needs a deliberately materialized large structure); the size
cap is intentional (`session.py:191`), and before `_var_dropped` existed ALL drops were silent.

**If picked up:** append the name to `_var_dropped` in the oversized branch too (ideally noting "too
large"), so the rehydrate note names it like the unpicklable case.

## D5-MUTUAL — mutually-recursive structural constraints drop cleanly on crash-restart (self-recursion SURVIVES)

**What:** a SELF-recursive `class Node(Constraint)` now survives crash-restart (the `_SELF_REF` recipe
marker + `model_rebuild()`), and a MUTUAL cycle (`A` ↔ `B`) no longer POISONS the snapshot (its recipe
is clean — no live class embedded), but it may not fully REBUILD (the partner class isn't defined yet
when the first is rebuilt). In that case the per-recipe replay guard drops just that constraint with a
`[rehydrate]` note — the rest of the session restores intact.

**Why deferred (rare + accepted):** mutual structural recursion across two Constraint classes is an
uncommon authoring shape, and the general per-object-resilient rehydrate makes it safe (drop + note,
never catastrophic). Self-recursion — the common case — survives fully.

**If picked up:** capture the whole mutual-recursion GROUP in `to_recipe` and co-rebuild both classes
with forward refs + a single `model_rebuild()` pass over the group.

## Deferred sweep findings (2026-06-17 reconciliation) — LOW / by-design / off-path

A two-pass sweep (full enumeration in the workspace: `../plm_sweep_report.md`) was reconciled against
this `main` after the type=/algebra-drop landing. The items below are confirmed-present but
intentionally deferred (low impact, off the active path, by-design tightness, or a documented
contract). Fixed in the same pass (not listed here): SD1 (a timeout is now labeled "timed out", no
spurious RE-RUN), SD4 (a slow-dying kernel is reaped by a daemon — no zombie), R3 (`executed` is
always on the cell envelope), SD7 (idle-read SIGINT comment corrected). CD3/CD5 are subsumed by
**C23** (satisfiability) above.

**constraint (`constraint/base.py`, `registry.py`)**
- **C4** — `_interpret_kwargs` never asserts it consumed all kwargs at the end; it relies entirely on
  the `_KNOWN_KWARGS` front gate. A user can't trigger it (the gate rejects typos), but a maintainer
  who adds a name to `_KNOWN_KWARGS` and forgets its handler ships a SILENT no-op (the kwarg is
  accepted and ignored, no error). Fix: `if kwargs: raise RuntimeError(f"unhandled kwargs {sorted(kwargs)}")`
  at the end of `_interpret_kwargs`, so a half-added kwarg fails loudly in tests.
- **CD4** — `sorted=`/`monotonic=` on a `set`/`frozenset` type validate the set's arbitrary hash
  order (meaningless). Fix: reject these kwargs on unordered types.
- **C5** — `list_contains` and `str_contains` render the identical describe phrase "contains X" for
  element vs substring. Fix: distinguish the wording.
- **C6** — `subset_of`/`superset_of` give a generic "could not evaluate" message on a non-iterable
  and emit no schema hint.
- **C7** — single `predicate=` describe wording ("1 predicate(s) checked") is a separate bucket from
  the plural untagged count; cosmetic.
- **C9** — a struct-typed field is built-from-dict (lenient pydantic), diverging from
  `is_instance_of`'s strict isinstance. Arguably intended pydantic semantics.
- **CD9** — `kind="semver"` regex accepts leading zeros (`01.2.3`) and rejects build metadata
  (`1.2.3+build`). Fix: tighten the regex to the SemVer spec.

**repl (`repl/session.py`, `prefix.py`, `kernel.py`)**
- **R1** — `cached_vars_blob` is refreshed on EVERY completed cell incl. one that errored, so a
  half-mutated-then-errored cell becomes the crash-restart baseline. Fix: only refresh on a clean
  cell. (Note: SD1's fix now distinguishes a graceful timeout — state preserved live — from a
  crash/respawn; DOC-NR3-1's "re-run on timeout" applies to the hard-timeout/crash path only.)
- **R2** — the kernel keeps the trajectory in a HIDDEN accumulator `_repl_plm_messages` (source of
  truth; the parent's deltas `.extend()` it) and reseeds the PUBLIC `plm_messages` each cell as a deep
  copy of it. The public list is leak-proof, but the accumulator is NOT: it's `_repl_`-prefixed
  (excluded from the snapshot AND the Guard C+ canon) and unsealed, and the kernel reads it as a bare
  `__main__` global. A cell doing `_repl_plm_messages = []` / `.clear()` wipes the trajectory for the
  rest of the session (every later cell deep-copies the corrupted accumulator). LOW (needs a write to
  an obvious kernel-internal name; cell-trust==parent-trust). Fix: add `_repl_plm_messages` to the
  Guard C+ canon, or move the accumulator into the `_kernel_state` side-module out of cell reach.
- **SD3** — the rehydrate handshake reuses the 30s cold-boot deadline despite unbounded
  re-exec/recipe work, so a large session can false-fail and lose ALL state. Fix: a separate,
  work-scaled rehydrate budget.
- **R4** — a mutable extra policy can pre-empt a sealed-extra of the same NAME (installed first),
  leaving it mutable; reachable only via raw `_PLM_*_POLICIES` env vars (the `PLMMetaParameters` API
  rejects cross-bucket names). Fix: install sealed extras first / refuse the collision loudly.
- **R5** — a kernel boot hang is resolvable only by the 30s handshake timeout, never SIGINT
  (kernel-wide SIG_IGN). Accepted consequence.
- **R6** — `_REPL_INJECTED_CANON` only covers callables/modules, so a rebind of a non-callable
  injected singleton (e.g. the constraint `REGISTRY`) isn't Guard-C+-reverted.
- **SD5** — kernel pre-exec setup (buffer reset, seed bind, Guard A audit + imports) is billed
  against `cell_timeout`, so a timeout during setup is reported as if the cell ran.
- **SD6** — `_read_exact` can busy-spin on a large frame trickling in near the deadline (unbuffered
  chunk reads under select). Latent (real frames are tiny); BufferedReader is the S-F13 direction.

**policy (`policy/...`, `policy/defaults/...`)**
- **P2** — `_PolicyStore.popitem`/`clear` silently skip default policies with no `[policy]` note
  (unlike `pop`/`del`). Fix: emit the note.
- **P5** — rename/`duplicate_policy` refuse a `new_name` colliding with ANY `__main__` var (incl. a
  transient one), not just policies/kernel-internals. Over-broad.
- **P7** — `_install_policy_source` re-execs `@policy class` bodies (bootstrap + duplicate) — third
  surface of **Corr#4** (keep class bodies side-effect-free).
- **P8** — `base_verifier` silently no-ops a broken proposer (swallowed). Fix: surface a
  verifier-channel note.
- **P9** — `natural_llm` gives the model BOTH the JSON schema and an NL contract from `describe()`;
  if `describe()` is empty or raises, it proceeds with the schema ONLY and emits NO warning, so the
  NL steering vanishes silently (worse outputs / more retries, no signal). Fix: log a one-liner when
  the produced steering is empty. (Rare — a structural constraint's `describe()` isn't normally empty.)
- **PD2** — a sealed policy's dunder metadata (`__name__`/`__doc__`/`update_wrapper`) is still
  writable, so introspection can lie even though behavior is locked. Narrow the claim or freeze.
- **PD3** — `_PolicyStore.setdefault` (a conditional WRITE — inserts `key` if absent) checks the
  writable-context flag but bypasses the seal/default-protection (`_blocked`) that every other mutator
  (`__setitem__`/`pop`/`popitem`/`clear`/`update`/`__ior__`) routes through. Benign today (used only in
  sanctioned paths, only creates new keys) but the one write surface with a different gate. Fix: route
  `setdefault` through `__setitem__` so there is a single guarded write path.
- **PD4** — react bodies `raise ConstraintViolation` via a `__main__` global only present if the
  constraint import succeeded; on a degraded kernel that path raises `NameError`. Fix: import it.
- **PD5** — `_SyncBackend.generate` uses `asyncio.run` per call with no guard/doc against a granted
  callable already inside an event loop.
- **P4** — (borderline; user sees no real problem) `_compute_insert` reports empty content as
  NO_MATCH "empty content", conflating no-op with bad-arg + a redundant double-validate. Likely a
  non-issue; tracked for completeness.

**model_backend (off the active OpenAI-Responses path)**
- **M2** — Chat-Completions forces `temperature=1` for reasoning models (gpt-5 /chat 400). **M5** —
  Anthropic `json.loads(args)` crashes on one bad prior tool_call. **M6** — Anthropic `tc["id"]`
  KeyError on an id-less call. **M3/M4/M7/M8** — token-floor / reasoning-effort-echo /
  usage-floor-order / join-vs-`output_text` inconsistencies. All latent: PLM runs reasoning models on
  the Responses API and ~never uses Anthropic. Fix if/when those backends go active (M5/M6 are
  one-line guards).

**core (`plm.py`, `_react_helper.py`, `plm_tools.py`)**
- **CO1** — `chat_template_kwargs={'enable_thinking':True}` is dropped by the OpenAI allowlists
  (vLLM-only) → no-op on the primary path.
- **CO2** — the last-round reminder fires `return_budget` rounds early ("final round" while several
  remain).
- **CO3** — `_tool_python` flat Responses tool shape (by-design on the active path; see **R6-4**).
- **CO4** — the internal last-round reminder text leaks into the kernel `plm_messages` view (benign).
- **CO5** — `_format_repl_output` would raise on a truthy non-string section (unreachable today).
- **CO6** — `metrics.model_calls`/`tool_calls` are lists, but the docstring implies scalar counts.
- **CO7** — `main_turns` is tracked in two places, reconciled only at return.
- **CO8** — `return_budget` is not a reserved RETURN window — just extra rounds appended to the loop.
- **CO9** — no-progress rounds (text-only / malformed args) silently burn budget with no early-abort
  or distinct failure signal.
