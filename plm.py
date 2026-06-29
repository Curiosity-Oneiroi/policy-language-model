from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type

from plm.model_backend import ModelBackend
from plm._react_helper import (
    _build_last_round_reminder,
    _format_repl_output,
    _strip_code_fences,
    _track_model_call,
    _validate_return_against_constraint,
    clone_messages,
    public_trajectory,
)
from plm.constraint import Constraint
from plm.metaparams import get_system_prompt
from plm.plm_tools import _tool_python
from plm.repl import PythonReplSession, PREFIX, DEFAULT_PREINSTALL

_PLM_ROOT_DEPTH: int = 2


_INTERNAL_KWARGS: frozenset[str] = frozenset({
    "cumulative_prompt_tokens",
    "original_cumulative_prompt_tokens",
})


_BACKEND_DEPS: dict[str, tuple[str, ...]] = {
    "SlateBackend":     ("httpx",),
    "OpenAIBackend":    ("openai",),
    "AnthropicBackend": ("anthropic",),
    "VLLMBackend":      ("openai",),
}


class PLMTaskFailure(RuntimeError):
    """Raised when PLM exhausts max_turns + return_budget without a successful
    RETURN(obj) call."""
    pass


def _policy_def_name(source: str, where: str) -> str:
    """A policy's NAME is intrinsic to its SOURCE — the single top-level @policy def/class name —
    never a filename or dict key (those are only transport labels). Parse `source` and return that
    name, raising a clear error unless it is exactly one top-level def/class."""
    import ast
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise ValueError(f"PLMMetaParameters: {where}: source is not valid Python ({e})")
    defs = [n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    if len(defs) != 1:
        raise ValueError(
            f"PLMMetaParameters: {where}: a policy is exactly ONE top-level def/class "
            f"(found {len(defs)}); the policy name comes from that def/class, not the filename/key"
        )
    return defs[0].name


@dataclass
class PLMMetaParameters:
    system_prompt: str
    extra_policies: dict[str, str] | None = None         # mutable + duplicable (normal)
    sealed_policies: dict[str, str] | None = None        # immutable + un-duplicable (sealed, NOT blessed)
    verifier: str | None = None

    def __post_init__(self) -> None:
        if not (self.system_prompt or "").strip():       # NP3-8: empty/whitespace system prompt is
            import warnings                               # almost always a mistake (e.g. an empty
            warnings.warn(                                # system_prompt.md) — PLM would run with NO
                "PLMMetaParameters: system_prompt is empty/whitespace-only — PLM will run with no "
                "system instructions; provide a non-empty system prompt.", stacklevel=2)
        for _attr in ("extra_policies", "sealed_policies"):
            _v = getattr(self, _attr)
            if _v is not None and not (
                isinstance(_v, dict)
                and all(isinstance(k, str) and isinstance(s, str) for k, s in _v.items())
            ):
                raise TypeError(
                    f"PLMMetaParameters.{_attr} must be a dict[str, str] "
                    f"(policy name -> @policy source) or None"
                )
        # Re-key both buckets by the REAL policy name (the @policy def/class name parsed from each
        # source), NEVER the dict key / file stem the caller or from_dir supplied. A policy's name
        # is intrinsic to its def/class; a filename or param key is only a transport label, and
        # trusting it silently mis-registers / mis-seals (this is the API-side half of NP3-4; the
        # kernel seals the install delta as the matching backstop). Also catches two same-name files.
        for _attr in ("extra_policies", "sealed_policies"):
            _bucket = getattr(self, _attr)
            if not _bucket:
                continue
            _canon: dict[str, str] = {}
            for _k, _src in _bucket.items():
                _name = _policy_def_name(_src, f"{_attr}[{_k!r}]")
                if _name in _canon:
                    raise ValueError(
                        f"PLMMetaParameters.{_attr}: two policies both define {_name!r} "
                        f"(the name comes from the def/class, so they collide)"
                    )
                _canon[_name] = _src
            setattr(self, _attr, _canon)
        # A name in BOTH buckets is ambiguous (mutable vs sealed) — reject (REAL names now).
        if self.extra_policies and self.sealed_policies:
            _dup = sorted(set(self.extra_policies) & set(self.sealed_policies))
            if _dup:
                raise ValueError(
                    f"PLMMetaParameters: policy name(s) {_dup} appear in BOTH extra_policies "
                    f"(mutable) and sealed_policies (immutable) — put each in exactly one"
                )
        # Validate verifier source (if any): must parse and define EXACTLY ONE
        # top-level `def verifier(messages)` (FunctionDef or AsyncFunctionDef).
        # Other top-level statements (helper defs, imports, assignments) are
        # ALLOWED so the verifier can ship helpers, and the function keeps any
        # cross-round state via the kernel namespace (it runs kernel-resident
        # with full ambient access). This is a STRUCTURAL gate only — we do
        # NOT exec/compile here; the kernel boot install (prefix.py) compiles the
        # source in the policy space (full ambient namespace) once per run.
        # Failure to parse OR a missing/duplicated/renamed top-level `verifier`
        # is a metaparam bug, not an evolved runtime bug, so this validation is
        # FAIL-LOUD (unlike the runtime call, which is fail-soft). The verifier
        # is SEALED + HIDDEN in the kernel (not in the policy registry), so it can
        # mutate the trajectory and use the full ambient namespace (react_llm /
        # natural_llm / parallel / policy edit APIs / policyzero) but can never
        # abort the run or edit a sealed policy (the universal kernel rules).
        if self.verifier is not None:
            if not isinstance(self.verifier, str):
                raise TypeError(
                    f"PLMMetaParameters.verifier must be a str (Python source) or None, "
                    f"got {type(self.verifier).__name__}"
                )
            import ast as _ast
            try:
                _vtree = _ast.parse(self.verifier)
            except SyntaxError as _e:
                raise ValueError(
                    f"PLMMetaParameters.verifier: source is not valid Python ({_e})"
                ) from _e
            # STRICT: the verifier source is EXACTLY ONE top-level node — a sync
            # `def verifier(messages)`. NO other top-level statements (helpers,
            # imports, constants, classes, async). Any helper functions, imports,
            # or constants the verifier needs live INSIDE the function body. This
            # keeps the verifier a single self-contained unit (trivial to validate,
            # mutate, and serialize) AND guarantees nothing leaks into the kernel
            # namespace (nested helpers are locals — invisible to policies/snapshot).
            _vbody = _vtree.body
            if len(_vbody) != 1 or not (
                isinstance(_vbody[0], _ast.FunctionDef) and _vbody[0].name == "verifier"
            ):
                _async_hint = ""
                if any(isinstance(n, _ast.AsyncFunctionDef) and n.name == "verifier"
                       for n in _vbody):
                    _async_hint = (" (`async def verifier` is NOT allowed — the kernel "
                                   "calls it synchronously)")
                raise ValueError(
                    "PLMMetaParameters.verifier: source must be EXACTLY ONE top-level "
                    "`def verifier(messages)` function and NOTHING else"
                    + _async_hint
                    + ". Put any helper functions, imports, and constants INSIDE the "
                    f"function body. (Got {len(_vbody)} top-level statement(s).)"
                )

    @classmethod
    def from_dir(cls, path: Any) -> "PLMMetaParameters":
        """Load a metaparam FOLDER:

            <path>/
              system_prompt.md            # (or .txt) — the system prompt (required)
              policies/
                sealed/  <name>.py        # immutable + un-duplicable (NOT blessed)
                mutable/ <name>.py        # mutable + duplicable (normal)

        Each `.py` is one `@policy def/class <name>(...)`; the policy NAME comes from that
        def/class — the filename is only a label (a file `auth.py` defining `authenticate`
        registers `authenticate`). Files starting with `_` are skipped as helpers. Either
        policies subfolder may be absent. Point PLM at one such folder per call — multiple
        folders under a common root act as a library of selectable metaparams.
        """
        import pathlib
        base = pathlib.Path(path)
        if not base.is_dir():
            raise FileNotFoundError(f"PLMMetaParameters.from_dir: not a directory: {base}")
        sp = next((base / f for f in ("system_prompt.md", "system_prompt.txt") if (base / f).is_file()), None)
        if sp is None:
            raise FileNotFoundError(
                f"PLMMetaParameters.from_dir: missing system_prompt.md (or .txt) in {base}"
            )

        def _load(sub: str) -> dict[str, str]:
            out: dict[str, str] = {}
            pdir = base / "policies" / sub
            if pdir.is_dir():
                for f in sorted(pdir.glob("*.py")):
                    if not f.name.startswith("_"):       # skip helper/private files
                        out[f.stem] = f.read_text(encoding="utf-8")
            return out

        # Optional EVOLVED trajectory verifier lives at the metaparam TOP LEVEL
        # (NOT under policies/) as a fixed filename. Absent => None (legacy
        # metaparam dirs continue to load unchanged). The `_`-prefix skip rule
        # used for policy helper files does not apply here — `verifier.py` is a
        # fixed name, not a glob.
        _vp = base / "verifier.py"
        _verifier_src = _vp.read_text(encoding="utf-8") if _vp.is_file() else None

        return cls(
            system_prompt=sp.read_text(encoding="utf-8"),
            extra_policies=_load("mutable") or None,
            sealed_policies=_load("sealed") or None,
            verifier=_verifier_src,
        )


class PLM:
    """PLM meta-reasoner
    """

    def __init__(
        self,
        *,
        model_backend: ModelBackend,
        metaparams: PLMMetaParameters,
        max_turns: int = 100,
        return_budget: int = 5,
        pip_install_packages: str = "",
        tool_timeout: Optional[float] = None,
        session_workspace: Optional[str] = None,
        simulate_config: Optional[Dict[str, Any]] = None,
        final_probe: Optional[str] = None,
    ):
        self.model_backend = model_backend
        self.metaparams = metaparams
        self.max_turns = max_turns
        self.return_budget = return_budget
        self.pip_install_packages = pip_install_packages
        self.tool_timeout = tool_timeout
        self.session_workspace = session_workspace
        # OPTIONAL harness code run as ONE kernel cell AFTER a successful RETURN and
        # BEFORE the kernel is torn down — used by the optimizer to evaluate the delivered
        # policy under a fixed, larger sample (the "official" Elo). Best-effort; a probe
        # failure never affects the run's answer. None -> no probe (single runs etc.).
        self.final_probe = final_probe
        # FIXED evaluation conditions for the `simulate` default (opponents, clock /
        # per_move_s / game_clock_s, max_moves, evaluate / eval_depth / reference_engine,
        # on_illegal / max_illegal_retries, max_workers). Passed to the kernel as
        # `_PLM_SIMULATE_CONFIG` so the MODEL cannot pick its own opponents or relax the
        # clock — it improves the policy, not the test. None -> simulate's built-in defaults.
        self.simulate_config = simulate_config

    async def __call__(
        self,
        messages: List[Dict[str, Any]],
        *,
        constraint: Optional[Type[Constraint]] = None,
        on_round: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
        **generate_kwargs: Any,
    ) -> Dict[str, Any]:
        """Drive the React loop until RETURN or budget exhaustion.

        Returns:
            {
                "answer":   EXACTLY the value PLM passed to RETURN(...) — the
                            Constraint-validated/coerced value when a constraint was
                            set, else whatever object the model returned. (PLM and
                            its sub-policies always finalize via RETURN, so the
                            terminal assistant text is never the answer.)
                "messages": final trajectory (system prompt + all rounds),
                "metrics":  {"main_turns", "model_calls", "tool_calls", "total_tokens"}
            }
        """
        
        messages = list(messages)
        messages.insert(0, {"role": "system", "content": self.metaparams.system_prompt})

        repl_env = {"AGENT_DEPTH": str(_PLM_ROOT_DEPTH)}
        if self.session_workspace:
            repl_env["SESSION_WORKSPACE"] = self.session_workspace

        mb = self.model_backend
        spec: dict[str, Any] = {
            "model_backend_class_name": type(mb).__name__,
            "model": getattr(mb, "model", None),
        }

        # `use_responses_api` round-trips an EXPLICIT API-style override to the
        # kernel sub-LLM. OpenAIBackend.from_spec already reads the key; without it
        # here, an explicit `use_responses_api=` set against the model-name
        # auto-detection silently flips in-kernel. The hasattr guard no-ops it for
        # the other backends (they lack the attr).
        for attr in ("api_key", "base_url", "max_context_length", "use_responses_api"):
            if hasattr(mb, attr):
                spec[attr] = getattr(mb, attr)
        repl_env["_PLM_BACKEND_SPEC"] = json.dumps(spec)

        if getattr(self.metaparams, "extra_policies", None):
            repl_env["_PLM_EXTRA_POLICIES"] = json.dumps(self.metaparams.extra_policies)
        if getattr(self.metaparams, "sealed_policies", None):
            repl_env["_PLM_SEALED_EXTRA_POLICIES"] = json.dumps(self.metaparams.sealed_policies)

        if getattr(self.metaparams, "verifier", None):
            repl_env["_PLM_VERIFIER_SOURCE"] = self.metaparams.verifier

        # FIXED simulate eval conditions -> the sealed `simulate` default reads these from
        # the env; the model cannot choose its own opponents / relax the clock.
        if self.simulate_config:
            repl_env["_PLM_SIMULATE_CONFIG"] = json.dumps(self.simulate_config)

        # Append the chosen backend's runtime SDK dep so the kernel's per-session
        # uv venv can `import plm.model_backend.<x>` and call it successfully.
        backend_deps = _BACKEND_DEPS.get(type(mb).__name__, ())
        extra_packages = (
            tuple(self.pip_install_packages.split()) if self.pip_install_packages.strip() else ()
        ) + backend_deps

        repl = PythonReplSession(
            workspace=self.session_workspace,
            env=repl_env,
            cell_timeout=self.tool_timeout,
            code_prefix=PREFIX,
            preinstall=DEFAULT_PREINSTALL + extra_packages,
        )

        tools = [_tool_python()]

        # Reserved-kwarg guard (mirrors react_llm / react_verifier_llm / natural_llm): the per-round
        # call passes `messages` and `tools` explicitly, then splats **clean_generate_kwargs — so a
        # caller-supplied `tools=`/`messages=` in generate_kwargs would collide into a cryptic
        # `TypeError: got multiple values for keyword argument`. Reject it up front with a clear message.
        _reserved_gk = {"messages", "tools"} & set(generate_kwargs)
        if _reserved_gk:
            raise ValueError(
                f"generate_kwargs may not contain {sorted(_reserved_gk)} — PLM passes `messages` and "
                f"`tools` itself; put backend params (temperature, max_tokens, ...) here")

        clean_generate_kwargs = {
            k: v for k, v in generate_kwargs.items() if k not in _INTERNAL_KWARGS
        }
        clean_generate_kwargs.setdefault(
            "chat_template_kwargs",
            generate_kwargs.get("chat_template_kwargs", {"enable_thinking": True}),
        )

        plm_ctx: Dict[str, Any] = {"main_turns": 0}
        token_state: Dict[str, Any] = {
            "cumulative_prompt_tokens": generate_kwargs.get("cumulative_prompt_tokens", 0) or 0,
        }
        metrics: Dict[str, Any] = {
            "model_calls": [],
            "tool_calls": [],
            "total_tokens": 0,
            "main_turns": 0,
            # Verifier-hook counters: ALWAYS present (0 when no verifier is configured)
            # so downstream consumers can rely on the keys existing regardless of
            # whether the metaparam dir shipped a verifier.py.
            "verifier_calls": 0,
            "verifier_errors": 0,
        }

        # The (optional) evolved trajectory verifier is NO LONGER compiled in the
        # PARENT process. It is shipped to the kernel via `_PLM_VERIFIER_SOURCE`
        # (see repl_env above) and compiled there at boot into a PRIVATE, HIDDEN,
        # SEALED object (`_repl_verifier_obj`), with the FULL ambient namespace
        # (react_llm / natural_llm / parallel / policy edit APIs / policyzero). The
        # parent dispatches each round via `_repl_run_verifier(...)` IN THE KERNEL
        # (see `_run_verifier_in_kernel` below). `verifier=None` => the env var is
        # unset => the kernel installs nothing => dispatch is a no-op (legacy
        # metaparams behave EXACTLY as before).
        _verifier_enabled = bool(getattr(self.metaparams, "verifier", None))

        # --- kernel dispatch round-trip ------------------------------------
        # Sentinel-delimited stdout is the kernel->parent channel for the mutated
        # trajectory (the only structured non-terminating channel `execute_cell`
        # exposes — `return_blob` would TERMINATE the run via RETURN). We:
        #   1. SEED the current full trajectory into the kernel's `plm_messages`
        #      accumulator (the seed channel REPLACES it — see kernel.py), so the
        #      verifier sees an up-to-date, writable list.
        #   2. Run a tiny dispatch cell that calls `_repl_run_verifier(plm_messages)`
        #      and prints `<<<PLM_VERIFIER>>>{json}<<<END>>>` carrying the (possibly
        #      mutated) messages + an error flag.
        #   3. Parse that line; on success REPLACE the parent `messages` in place.
        # CRASH-SAFE: if the cell times out / the kernel dies, `execute_cell`
        # SIGKILLs + respawns and returns an envelope with NO sentinel — we detect
        # the missing sentinel, count it as a verifier error, and KEEP the parent's
        # current `messages` untouched (the run proceeds). Because we always re-seed
        # the full trajectory immediately BEFORE dispatch, a respawn's empty
        # accumulator is irrelevant. After dispatch we RESET seeded_count/epoch so
        # the next normal cell re-seeds the full (post-verifier) trajectory once.
        _V_SENTINEL_OPEN = "<<<PLM_VERIFIER_RESULT>>>"
        _V_SENTINEL_CLOSE = "<<<PLM_VERIFIER_END>>>"

        async def _run_verifier_in_kernel() -> None:
            """Dispatch the kernel-resident verifier on the current trajectory,
            FAIL-SOFT. EVERY dispatch increments `verifier_calls`; any failure
            (kernel error, respawn, missing/garbled sentinel, verifier exception
            captured in-kernel) increments `verifier_errors` and is logged into
            `metrics['verifier_error_log']` but NEVER aborts the run — the verifier
            is evolved code under optimization, so a buggy generation must not kill
            the parent run. On success, the parent `messages` list is replaced
            in-place with the kernel-returned (possibly mutated) trajectory."""
            nonlocal force_full_reseed
            if not _verifier_enabled:
                return
            metrics["verifier_calls"] += 1

            def _log_err(kind: str, detail: str) -> None:
                metrics["verifier_errors"] += 1
                metrics.setdefault("verifier_error_log", []).append({
                    "round": _round, "error_type": kind, "error": detail[:500],
                })

            dispatch_code = (
                "import json as _vj\n"
                "_v_msgs, _v_err = _repl_run_verifier(plm_messages)\n"
                # NO default=str: a verifier that weaves a NON-JSON-native value
                # must RAISE here (in-kernel), which the kernel's per-cell
                # try/except catches -> stdout carries NO sentinel -> the parent
                # detects the missing sentinel, counts a verifier error, and KEEPS
                # its clean `messages` (never adopts a lossily-coerced trajectory).
                "print('" + _V_SENTINEL_OPEN + "' + _vj.dumps("
                "{'messages': _v_msgs, 'error': _v_err}) + '"
                + _V_SENTINEL_CLOSE + "')\n"
            )
            try:
                # Seed the full current trajectory (REPLACES the accumulator) and
                # run the dispatch cell. No delta — the seed is the source of truth.
                exec_result = await asyncio.to_thread(
                    repl.execute_cell, dispatch_code, {"plm_messages": list(messages)}, None
                )
            except BaseException as _e:                      # hard parent-side failure
                _log_err(type(_e).__name__, str(_e))
                return
            finally:
                # A verifier just ran: it mutated a COPY of the trajectory (and may
                # have PRUNED or ADDED turns), so the kernel accumulator is now stale.
                # Force the NEXT normal cell to REPLACE the accumulator with the full
                # post-verifier trajectory (the seed channel). An append-delta cannot
                # represent a prune and would DOUBLE the in-kernel `plm_messages` view.
                force_full_reseed = True

            stdout = (exec_result or {}).get("stdout") or ""
            # OPEN: first match; CLOSE: LAST match (rfind). Echoed trajectory
            # content could itself contain the sentinel strings; using rfind for
            # the close marker means embedded content can't prematurely truncate
            # the JSON payload.
            i = stdout.find(_V_SENTINEL_OPEN)
            j = stdout.rfind(_V_SENTINEL_CLOSE)
            if i < 0 or j < 0 or j < i:
                # No sentinel: crash/respawn/garbled. Keep parent messages as-is.
                _log_err("VerifierNoResult",
                         "verifier dispatch returned no result sentinel (kernel may have "
                         "respawned); stderr=" + ((exec_result or {}).get("stderr") or "")[:300])
                return
            payload = stdout[i + len(_V_SENTINEL_OPEN):j]
            try:
                parsed = json.loads(payload)
                new_msgs = parsed.get("messages")
                v_err = parsed.get("error")
            except Exception as _e:
                _log_err("VerifierResultDecode", str(_e))
                return
            if v_err:                                        # verifier raised inside the kernel
                _log_err("VerifierRaised", str(v_err))
                # The kernel returns the messages unchanged on a verifier exception,
                # so adopting them is a no-op; still adopt to stay consistent.
            if isinstance(new_msgs, list):
                messages[:] = new_msgs                       # adopt the (possibly mutated) trajectory
            else:
                _log_err("VerifierBadMessages",
                         f"verifier result 'messages' was {type(new_msgs).__name__}, not a list")

        constraint_is_set = constraint is not None
        _DONE = object()
        terminal_answer: Any = _DONE
        reasoning: Optional[str] = None

        total_rounds_budget = self.max_turns + self.return_budget

        seeded_count = 0
        seeded_epoch: Optional[int] = None
        # After a verifier round, the next normal cell must REPLACE the kernel
        # `plm_messages` accumulator with the full post-verifier trajectory (seed
        # channel) rather than append a delta — see `_run_verifier_in_kernel`.
        force_full_reseed = False

        # Optional observability hook: called with the running transcript each round (and once
        # at the end). A harness uses it to stream the live trajectory; failures never break the run.
        def _emit(_msgs: List[Dict[str, Any]]) -> None:
            if on_round is not None:
                try:
                    on_round(list(_msgs))
                except Exception:
                    pass

        # ---- React loop ------------------------------------------------------
        try:
            for _round in range(total_rounds_budget):
                
                if _round > 0:
                    await _run_verifier_in_kernel()
                _emit(messages)                  # live trajectory: the running transcript so far

                if _round == self.max_turns - 1:
                    messages.append(_build_last_round_reminder(constraint))

                msgs = clone_messages(messages)

                call_start = time.time()
                cpt = token_state.get("cumulative_prompt_tokens", 0) or 0
                
                response = await self.model_backend.generate(
                    messages=msgs,
                    tools=tools,
                    cumulative_prompt_tokens=cpt if cpt > 0 else None,
                    **clean_generate_kwargs,
                )
                if not isinstance(response, dict):        # non-conformant backend (None / list / ...):
                    response = {}                          # degrade to an empty turn, never a raw crash.
                # ONE normalization at the top of the turn makes EVERYTHING below — _track_model_call,
                # .get("content"/"reasoning"/"tool_calls") — safe by construction, so a malformed
                # backend response can't escape PLM.__call__ as an AttributeError. Mirrors react_llm.

                _track_model_call(response, call_start, metrics, token_state)

                content = response.get("content", "") or ""
                reasoning = response.get("reasoning")
                plm_ctx["main_turns"] = plm_ctx.get("main_turns", 0) + 1

                tool_calls = response.get("tool_calls") or []
                if isinstance(tool_calls, dict):                  # a single call returned bare -> wrap
                    tool_calls = [tool_calls]
                elif not isinstance(tool_calls, list):            # any other malformed shape -> none
                    tool_calls = []

                messages.append({
                    "role": "assistant",
                    "content": content,
                    "reasoning": reasoning,
                    # Store ONLY the executed+answered call, and ONLY if it's a dict — a non-dict
                    # entry would corrupt the next round's history sanitizer. Mirrors react_llm.
                    "tool_calls": ([tool_calls[0]] if tool_calls and isinstance(tool_calls[0], dict) else None),
                    "_timestamp": time.time(),
                })

                if not tool_calls:
                    # Text-only round (the model emitted no python call). This
                    # never terminates the task — only a successful RETURN does.
                    # Whatever it said, just take another round.
                    continue

                tool_call = tool_calls[0]
                # PLM emitted parallel tool_calls; the root loop runs ONLY the first. Surface the
                # nudge in PLM's OWN tool result (its message stream) so PLM self-corrects to one
                # python call per turn — mirrors react_llm's note to its sub-LLM: each model's
                # coaching stays in its own stream, never leaks across levels.
                _parallel_note = (
                    "[plm] you emitted " + str(len(tool_calls)) + " tool_calls; only the FIRST ran "
                    "— send ONE `python` call per turn.\n\n") if len(tool_calls) > 1 else ""

                if not isinstance(tool_call, dict):
                    # Malformed first tool_call from a non-conformant backend: the assistant turn
                    # stored tool_calls=None for it, so reply as a USER turn (NOT an orphan `tool`).
                    # Mirrors react_llm's hardening; the root loop's outer try has no except,
                    # so without this a bare-subscript below would crash PLM.__call__ raw.
                    messages.append({"role": "user",
                                     "content": _parallel_note + f"[error] malformed tool_call "
                                                f"({type(tool_call).__name__}); emit a single `python` tool call"})
                    metrics["tool_calls"].append({
                        "tool_name": None, "tool_status": "malformed_tool_call",
                        "tool_result": f"malformed tool_call ({type(tool_call).__name__})"})
                    continue
                _fn = tool_call.get("function")
                _fn = _fn if isinstance(_fn, dict) else {}        # `function: null`/malformed -> {} (mirror react)
                tname = _fn.get("name")

                raw_args = _fn.get("arguments")
                if isinstance(raw_args, dict):
                    targs = raw_args
                elif isinstance(raw_args, str):
                    try:
                        targs = json.loads(raw_args)
                    except Exception:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.get("id") or "",
                            "content": _parallel_note + f"error: invalid JSON arguments for tool {tname!r}",
                        })
                        metrics["tool_calls"].append({
                            "tool_name": tname, "tool_status": "code_arg_invalid",
                            "tool_result": f"error: invalid JSON arguments for tool {tname!r}",
                        })
                        continue
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id") or "",
                        "content": _parallel_note + f"error: invalid arguments for tool {tname!r}",
                    })
                    metrics["tool_calls"].append({
                        "tool_name": tname, "tool_status": "code_arg_invalid",
                        "tool_result": f"error: invalid arguments for tool {tname!r}",
                    })
                    continue
                

                if tname != "python":
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id") or "",
                        "content": _parallel_note + f"error: tool {tname!r} not supported (only `python`)",
                    })
                    metrics["tool_calls"].append({
                        "tool_name": tname, "tool_status": "code_arg_invalid",
                        "tool_result": f"error: tool {tname!r} not supported (only `python`)",
                    })
                    continue

                if not isinstance(targs, dict):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id") or "",
                        "content": _parallel_note + f"error: arguments for tool {tname!r} must be a JSON object",
                    })
                    metrics["tool_calls"].append({
                        "tool_name": tname, "tool_status": "code_arg_invalid",
                        "tool_result": f"error: arguments for tool {tname!r} must be a JSON object",
                    })
                    continue

                raw_code = targs.get("code", "")
                if not isinstance(raw_code, str):
                    # non-string `code` (null/number/object): clear error + retry, NOT a raw
                    # AttributeError from `_strip_code_fences`
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id") or "",
                        "content": _parallel_note + "error: `code` must be a string of Python source",
                    })
                    metrics["tool_calls"].append({
                        "tool_name": tname, "tool_status": "code_arg_invalid",
                        "tool_result": "error: `code` must be a string of Python source",
                    })
                    continue
                try:
                    code = _strip_code_fences(raw_code)
                except SyntaxError as e:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id") or "",
                        "content": _parallel_note + f"stderr:\n{e}",
                    })
                    metrics["tool_calls"].append({
                        "tool_name": tname, "tool_status": "code_arg_invalid",
                        "tool_result": str(e)[:500],
                    })
                    continue

                if repl.kernel_epoch != seeded_epoch:
                    # Kernel (re)spawned: accumulator is empty → full reseed via delta.
                    seeded_count = 0
                    seeded_epoch = repl.kernel_epoch
                    force_full_reseed = False
                if force_full_reseed:
                    # A verifier ran last round; REPLACE the accumulator with the full
                    # post-verifier trajectory (seed channel) so the in-kernel
                    # `plm_messages` view EXACTLY matches the parent's authoritative
                    # `messages` — no doubling, and prunes/edits are reflected.
                    _seed = {"plm_messages": public_trajectory(messages)}
                    _delta = None
                    force_full_reseed = False
                else:
                    _seed = None
                    _delta = public_trajectory(messages[seeded_count:])
                seeded_count = len(messages)
                seeded_epoch = repl.kernel_epoch

                exec_result = await asyncio.to_thread(
                    repl.execute_cell, code, _seed, _delta
                )
                etype = exec_result.get("type", "result")

                if etype == "return":
                    return_obj = exec_result.get("return_obj")
                    ok, parsed_or_err = _validate_return_against_constraint(
                        return_obj, constraint,
                    )
                    if not ok:
                        # Constraint validation failed → surface the error and
                        # let the model retry in the next round.
                        augmented = dict(exec_result)
                        augmented["stderr"] = (
                            (exec_result.get("stderr") or "").rstrip()
                            + "\n\n" + parsed_or_err + "\n"
                        )
                        out = _parallel_note + _format_repl_output(augmented)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.get("id") or "",
                            "content": out,
                        })
                        metrics["tool_calls"].append({
                            "tool_name": tname, "tool_status": "return_invalid",
                            "tool_result": out[:500],
                        })
                    else:
                        terminal_answer = parsed_or_err
                        out = _format_repl_output(exec_result)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.get("id") or "",
                            "content": out + "\n[PLM] RETURN accepted; task terminating.",
                        })
                        metrics["tool_calls"].append({
                            "tool_name": tname, "tool_status": "return_valid",
                            "tool_result": out[:500],
                        })
                else:
                    out = _parallel_note + _format_repl_output(exec_result)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id") or "",
                        "content": out,
                    })
                    metrics["tool_calls"].append({
                        "tool_name": tname, "tool_status": "completed",
                        "tool_result": out[:500],
                    })

                if terminal_answer is not _DONE:
                    break
            else:
                raise PLMTaskFailure(
                    f"PLM exhausted max_turns ({self.max_turns}) + "
                    f"return_budget ({self.return_budget}) without calling "
                    f"RETURN successfully"
                    + (" (an output constraint was configured)" if constraint_is_set else "")
                    + "."
                )

            # RETURN succeeded (the loop broke, not the max-turns `else`). Run the optional
            # final-eval probe as ONE last kernel cell — a fixed, larger evaluation of the
            # delivered policy whose logged row the scorer reads as the OFFICIAL Elo — while
            # the policy is still live, BEFORE the kernel is torn down below. Strictly
            # best-effort: a probe error must never change the answer already returned.
            if self.final_probe:
                try:
                    await asyncio.to_thread(repl.execute_cell, self.final_probe, None, None)
                except Exception:
                    pass
        finally:
            try:
                repl.close()
            except Exception:
                pass

        metrics["main_turns"] = plm_ctx.get("main_turns", 0)

        _emit(messages)                      # final transcript (after the loop completes)
        return {
            "answer": terminal_answer,     # exactly the value PLM passed to RETURN(...)
            "messages": messages,
            "metrics": metrics,
        }


# ---- Quick demo (run as script: python -m plm.plm or `python plm/plm.py`) ----

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run a one-shot PLM task.")
    parser.add_argument("--backend", choices=["openai", "vllm", "slate"], default="openai")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--prompt", required=True, help="The user prompt for PLM to solve")
    args = parser.parse_args()

    from plm.model_backend.openai_backend import OpenAIBackend
    from plm.model_backend.vllm_backend import VLLMBackend
    from plm.model_backend.slate_backend import SlateBackend
    backend_cls = {"openai": OpenAIBackend, "vllm": VLLMBackend, "slate": SlateBackend}[args.backend]
    backend = backend_cls(model=args.model, base_url=args.base_url, api_key=args.api_key)
    metaparams = PLMMetaParameters(system_prompt=get_system_prompt())
    plm = PLM(model_backend=backend, metaparams=metaparams, max_turns=args.max_turns)

    async def _main():
        result = await plm([{"role": "user", "content": args.prompt}])
        print("---- ANSWER ----")
        print(result["answer"])
        print("\n---- METRICS ----")
        print(json.dumps(result["metrics"], indent=2, default=str))

    asyncio.run(_main())
