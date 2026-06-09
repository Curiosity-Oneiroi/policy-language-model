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

**Constraint OBJECTS now SURVIVE crash-restart** (was a gap): a `Constraint.field(...)`, a
composite (`& | ^ ~`), and a structural `class X(Constraint)` stored in a variable are
snapshotted as a picklable RECIPE and rebuilt on rehydrate (`constraint/snapshot.py` —
`to_recipe`/`from_recipe`/`is_constraint_class`), mirroring how a class POLICY survives by
its `_p_source`. Field/composite replay their construction; structural classes reconstruct
from `model_fields` + the validator decorators (fields, defaults, Field bounds, and model/
field validators all preserved; arbitrary METHODS on a constraint class are not — rare).
The recipe path is REQUIRED because no Constraint class cross-process dill-pickles (dynamic
factory/composite classes recurse; structural pydantic classes pickle by-reference and
would otherwise sink the whole snapshot). Verified end-to-end
(`test_int_crash_restart_constraints_survive`).

**Remaining (still deferred — rare):**
  - constraint `kind=`s registered AT RUNTIME into the shared `REGISTRY` (a cell doing
    `REGISTRY["x"] = pred`, NOT the built-in kinds) — the REGISTRY is rebuilt fresh on
    respawn, so a runtime-added kind is lost. A field recipe that *uses* such a kind then
    fails to replay (surfaced via `rehydrate_error`). (WF Corr#18)
  - live INSTANCES of a callable `@policy class` — the class SOURCE is replayed, but an
    instance a cell built and holds is gone (its depth-cap `__call__` wrap closes over a
    ContextVar, so dill can't pickle the instance). (WF Corr#17)

**Status:** acceptable for research. Contract for the two remaining cases: after a respawn
(detectable via `kernel_epoch`), re-register any runtime constraint kinds and rebuild any
held callable-class instances.

**If picked up:** snapshot a replay recipe for runtime `REGISTRY` kinds (the registration
args) + re-apply on rehydrate; for callable class instances, snapshot a construction
recipe (class + init args) rather than the unpicklable instance.

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

**Resolution (chosen over making it dill-survivable):** a `super()`-using method on a Constraint is
now REJECTED at class definition with a clear error — `_ConstraintMeta.__init__` scans the class's
own methods and refuses any that carry a bare-`super()` `__class__` free-var. A Constraint is a FLAT
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
