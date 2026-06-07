# PLM — deferred TODOs

Items intentionally deferred (not bugs; LOW-priority niceties). Each notes why
it was deferred and what a correct fix needs.

## C25 — memoize composite constraint classes

**What:** Every `A & B` / `A | B` / `A ^ B` / `~A` calls
`_make_composite_wrapper` (`constraint/algebra.py`), which builds a brand-new
dynamic class via `_ConstraintMeta` each time. So in a loop that re-evaluates
`A & B` every round you churn a fresh class object per round, and
`(A & B) is (A & B)` is False (identity/perf nicety, not a correctness hole —
the rebuilt class validates identically).

**Why deferred (not the trivial version):** the obvious fix — a module-level
dict keyed by `(label, tuple(children))` (constraint classes are hashable by
identity, so the key is hashable) — holds **strong refs to the children and the
composite forever**. For the churn case that motivates the fix (composites built
from ever-distinct transient children) that dict grows unboundedly; for the
common case (same `A`/`B` reused each round) the key is identical so there's
exactly one entry and *no churn to fix anyway*. A correct fix therefore needs a
**bounded or weak** cache (LRU, or weak-valued with care around class-identity
keys), which is more than a trivial change for a LOW-priority nicety.

**If picked up:** add the cache in `_make_composite_wrapper` keyed by
`(suffix, label, children_tuple)`; use a bounded/weak structure; add a test that
`(A & B) is (A & B)` and that the cache doesn't retain transient children.

## C23 — auto consistency (satisfiability) check on constraint construction

**What:** make `Constraint.field(...)` and composite construction (`&`/`|`/`^`/`~`)
auto-run a fast satisfiability check and raise a new `ConstraintConsistencyError`
when the constraint is *provably* unsatisfiable — instead of silently building a
constraint that nothing can ever pass.

**Why deferred (complex, sizable, its own change):** general satisfiability is
**undecidable** here — arbitrary Python `predicate=` / `coercer=` bodies are
opaque, so the algo can never reason about them. It must therefore be **sound**
(only ever raise when *provably* empty — NEVER a false positive on a predicate it
can't see) and confine itself to the **declarative** parts of the constraint.
That's a real, self-contained algorithm + a `ConstraintConsistencyError` type +
auto-wiring into `of()`/algebra + a focused test matrix — too big to fold into a
LOW-cluster pass.

**Scope when built (sound, structural-only):**
- Numeric: `int_ge=5 & int_le=3`, `gt=5 & lt=5`, empty `*_range`.
- Length: `str_length` / `list_length` with min > max or negatives.
- Membership: `one_of ∩ not_one_of` empties the allowed set; empty `one_of`; a
  `one_of` value that can't match a required `str_pattern`.
- Type: AND of incompatible `instance_of` / `kind` base types.
- NOT/AND: `~A & A`, `not_pattern == str_pattern`.
- Predicate / coercer nodes contribute "unknown" → never trigger a raise.

**How:** walk the constraint tree; at AND nodes intersect the declarative domains
of children and raise if the intersection is provably empty; OR/XOR raise only if
*every* branch is independently empty. Run automatically inside `of()` and at
composite construction. Confirmed scope with user (sound / structural-only;
predicates stay unchecked).

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
