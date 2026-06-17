# PLM — deferred TODOs

Items intentionally deferred (not bugs; LOW-priority niceties). Each notes why
it was deferred and what a correct fix needs.

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
