@policy
def react_llm_verifier(messages, *, args=(), kwargs=None, objects=None,
                       constraint=None, max_turns=8, return_budget=5, depth=None,
                       generate_kwargs=None, verifier=None):
    """ReAct sub-agent (model + python tool, same REPL) + a per-round VERIFIER
    hook — the trajectory control axis.

    Identical to `react_llm` except for an optional `verifier` callable. After
    every NON-terminal round (text-only or tool), once that round's messages
    are complete, the verifier is called with the LIVE `msgs` list and mutates
    it in place; the (possibly edited) trajectory then drives the next
    `backend.generate`. A successful `RETURN(value)` short-circuits and the
    verifier does NOT run on the terminal round. `verifier=None` ⇒ this policy
    behaves exactly like `react_llm`.

    Terminates ONLY when the model calls `RETURN(value)` inside its python
    tool. Text-only assistant turns just append-and-continue (and still trigger
    the verifier). The sub-LLM's ambient capabilities are intentionally minimal
    — `__builtins__` and `RETURN`. Everything else (sub-LLM calls, policy
    creation, access to other policies, user globals) is a DELIBERATE grant
    from PLM via `args` / `kwargs` / `objects`.

    Quick examples:

        # Simple — model writes code, calls RETURN:
        answer = react_llm_verifier("Compute 7! using python")
        # -> 5040  (whatever the model passed to RETURN)

        # Attach a per-round verifier (the trajectory control axis). The
        # verifier sees the live messages after each non-terminal round and
        # mutates them in place. PREFER a verifier POLICY (forkable/editable):
        react_llm_verifier("solve X step by step", verifier=base_verifier)
        # or fork + specialize it:
        #   duplicate_policy("base_verifier", "my_verifier")
        #   my_verifier._edit(...)            # tailor the prompts / gate / checks
        #   react_llm_verifier("solve X", verifier=my_verifier)

        # Pass named data; the model uses each kwarg key as a DIRECT
        # local name (no `kwargs["a"]` subscripting needed):
        result = react_llm_verifier(
            "Add the two numbers and RETURN the sum",
            kwargs={"a": 3, "b": 5},
        )
        # Model writes: `RETURN(a + b)`  -> 8

        # Grant policy-authoring (the @policy decorator) by name:
        from plm.policy import policy
        result = react_llm_verifier(
            "Author a doubler policy named `doubler` and use it on input.",
            kwargs={"policy": policy, "input": 7},
        )
        # Model writes:
        #   @policy
        #   def doubler(x): return x * 2
        #   RETURN(doubler(input))   -> 14

        # Grant nested-LLM access:
        result = react_llm_verifier(
            "Decompose into a sub-question and call natural_llm on it.",
            kwargs={"natural_llm": natural_llm},
        )
        # Model writes: `RETURN(natural_llm("the sub-question"))`

        # Constraint-validated final answer (validated INSIDE _exec on
        # the model's RETURN value; a ConstraintViolation prints into
        # the tool result and the loop continues):
        class Answer(Constraint):
            value: int
        result = react_llm_verifier("Compute fact(7)", constraint=Answer)
        # Model writes: `RETURN({"value": 5040})` -> Answer(value=5040)

        # Voluntarily lower the depth budget for this sub-task:
        react_llm_verifier(msg, depth=1)

    Parameters:

        messages: str / dict / list of message dicts. Initial conversation.

        args=()   : tuple of positional inputs. Model accesses as `args[i]`.

        kwargs=None (resolved to {}): dict of named inputs. **Each
            key/value pair is ALSO bound DIRECTLY in the model's exec
            namespace as a local name** — so `kwargs={"x": 7}` lets the
            model write `print(x)` (no subscripting). The full `kwargs`
            dict is ALSO accessible as a local `kwargs` for programmatic
            use. **This is the recommended way to grant the sub-LLM any
            named capability** (the @policy decorator, a passed-through
            LLM, a custom helper) because the model writes the granted
            name directly.

            Validation (bulletproof, at call time):
              * Non-string keys -> TypeError.
              * `"RETURN"` / `"__builtins__"` -> ValueError (would shadow
                the ambient termination primitive / builtins).
              * `"args"` / `"kwargs"` / `"objects"` are NOT rejected, but
                the meta-channel value wins silently — user data is still
                reachable via `kwargs[<the meta name>]`.
              * Identifier-invalid keys ("foo-bar", Python keywords like
                "if") -> UserWarning. Still reachable via `kwargs[<key>]`.

        objects=None (resolved to []): positional list of references
            accessed by index (`objects[i]`). Use when order matters
            more than naming; otherwise prefer `kwargs`.

        constraint=None: optional Constraint CLASS, validated INSIDE
            `_exec` on the model's `RETURN(value)`. On
            ConstraintViolation the violation's traceback prints into
            the tool result (via stderr) — the model sees it next turn
            and can retry. UNLIKE natural_llm, factory/predicate
            Constraints ARE supported here, because the model can run
            python and the validator runs against the actual value.

        max_turns=8, return_budget=5: these are SUMMED into a single flat
            per-round budget — the loop runs `max_turns + return_budget`
            iterations and EVERY round (text-only, tool, non-python tool, or a
            post-constraint-failure RETURN retry) consumes exactly one slot.
            The split is a sizing convention, not two separate phases:
            `max_turns` is the round budget you communicate to the model as the
            work it should plan for, and `return_budget` is the extra slack
            reserved for finalizing / RETURN-retries on top of that. Both are
            coerced to a non-negative int (None/non-int -> default, negative ->
            0); if their sum is 0 the loop makes no model call.

        depth=None: voluntary LLM-recursion-depth cap. None inherits the
            current scope; an int clamps to min(depth, current) — PLM may
            voluntarily LOWER the budget, never raise it above the sealed ceiling.

        generate_kwargs=None (resolved to {}): backend kwargs (temperature,
            top_p, response_format, ...) forwarded to each
            `backend.generate(...)`.

        verifier=None: optional callable `verifier(messages) -> None`,
            invoked after EACH non-terminal round with the LIVE `msgs`
            list. It MUTATES `msgs` in place (inject a correction / doubt,
            edit a turn, ...); its return value is IGNORED. A successful
            RETURN short-circuits first, so the verifier never runs on the
            terminal round. `None` ⇒ behaves exactly like `react_llm`.
            PREFER a verifier POLICY (e.g. `base_verifier` or a duplicate)
            over a bare function — a policy is source-editable via the M
            layer, forkable with `duplicate_policy`, and reusable. Whether
            the verifier GATES itself (a should_run check) or verifies every
            round is the verifier's OWN choice; gating is a good pattern (see
            `base_verifier`) but optional. The verifier runs in trusted
            policy scope (NOT the sub-LLM's restricted exec), wrapped in its
            own `descend()` so the depth guard caps any circuit it spawns at
            depth-1 (root=2). Because it runs one level below and its circuits
            are depth-1, a verifier needs depth >= 2: calling WITH a verifier at
            depth=1 (or any remaining < 2) raises LLMDepthExceeded UP FRONT,
            before any backend call. Exceptions the verifier raises propagate
            (fail-loud).

    Returns:

        Whatever the model passed to `RETURN(value)`, validated by
        `constraint` if set (returns the validated Constraint instance
        / raw value). NEVER silently returns the model's last text.

    Raises:

        - ValueError: kwargs contains a reserved key (`"RETURN"` or
          `"__builtins__"`). No backend call is made.
        - TypeError: a kwargs key is not a string. No backend call.
        - RuntimeError: budget exhausted without a successful RETURN
          and `constraint is None`. Message names "model never called
          RETURN(value)".
        - ConstraintViolation: budget exhausted; final RETURN(s) all
          failed validation. The model has seen each failure in the
          conversation history.
        - LLMDepthExceeded: the depth budget is at 0 before a generate, OR a
          `verifier` is set with fewer than 2 levels remaining (e.g. depth=1) —
          the latter is raised UP FRONT, before any backend call.
        - UserWarning (emitted, not raised): kwargs has identifier-invalid
          or Python-keyword keys.
        - Anything the `verifier` raises propagates out (fail-loud; the
          verifier is trusted PLM code).

    The verifier hook (trajectory control axis):

        `react_llm` exposes the input (messages) and output (constraint)
        axes; `react_llm_verifier` adds the TRAJECTORY axis. After each
        non-terminal round the verifier reads/edits the running `msgs`, and
        the (possibly changed) trajectory drives the next generate. This is
        how PLM steers a sub-agent mid-run: inject doubt, force an
        independent re-derivation, prune a branch. The verifier call is
        wrapped in `descend()`, so the existing depth guard accounts it one
        level below this policy — any react_llm/natural_llm circuit it builds
        is capped at depth-1 (root=2) and respects the global budget. See
        `base_verifier` for the reference template.

    What is ambient inside the sub-LLM's exec scope?

        ONLY two names by default:
        - `__builtins__` (print, len, range, dict, list, ...).
        - `RETURN(value)` (the sole termination primitive — the model
          can't end the loop without it).

        Everything else — `policy`, `list_policies`, `get_policy`,
        `read_policy`, `_PLM_POLICIES`, `parallel`, `natural_llm`,
        `react_llm`, user-cell globals — is BLOCKED. The sub-LLM cannot
        "discover" the available policies; PLM is the only thing that
        knows the namespace and chooses what to plumb through.

    Internals:

        - Per round: `backend.generate` -> if the model emitted a python
          tool call, parse the code, run it inside `with descend():` so
          any nested LLM call inherits a decremented depth. `_exec` runs
          the code in the sub-agent's OWN repl namespace `ns` — a single
          dict (globals == locals, like PLM's kernel `__main__`) preseeded
          with args/kwargs/objects + each kwarg key as a direct name, and
          isolated from kernel main.
        - After each NON-terminal round (text-only or tool), the verifier
          (if any) runs inside its own `with descend():` on the live `msgs`.
        - Containment: `tmp = 5` lands in `ns` (contained). `@policy def
          helper(): ...` (when policy is granted via kwargs) goes
          through the decorator's normal path; the decorator re-execs
          the body under kernel main, so the resulting policy has full
          main-globals access and is registered globally — as if PLM
          had authored it at cell-level.
        - Termination: only on `RETURN(value)`. A text-only assistant
          turn is appended to history and the loop continues. Budget
          exhausted -> raise (not silent fallback).
        - Constraint validation happens INSIDE `_exec`'s `_REPLReturn`
          catch — on failure the violation prints to stderr through
          the same path as any other exception, ending up in the tool
          result for the model to see.
    """
    import sys, json, keyword, warnings
    from plm.plm_tools import _tool_python
    from plm._react_helper import _strip_code_fences
    from plm.policy.defaults._llm_infra import (
        _make_backend, descend, llm_call, check_depth_or_raise, _remaining, LLMDepthExceeded)

    # Resolve None-sentinel mutable defaults (no shared-default footgun).
    kwargs = {} if kwargs is None else kwargs
    objects = [] if objects is None else objects
    generate_kwargs = {} if generate_kwargs is None else generate_kwargs

    # `generate_kwargs` is splat into `backend.generate(msgs, tools=tools, **generate_kwargs)`,
    # which already passes `messages` (positionally) and `tools` — a colliding key would raise a
    # cryptic `TypeError: got multiple values for keyword argument`. Reject it up front with a
    # clear message (model params like temperature/max_tokens go here; tool grants via kwargs).
    _reserved_gk = {"messages", "tools"} & set(generate_kwargs)
    if _reserved_gk:
        raise ValueError(
            f"generate_kwargs may not contain {sorted(_reserved_gk)} — react_llm passes "
            f"`messages` and `tools` itself; put backend params (temperature, max_tokens, ...) here")

    # ---- kwargs validation (bulletproof) ---------------------------------
    # `kwargs` is splat into the model's exec namespace below so each key
    # becomes a direct local name (ergonomic for the model — no
    # `kwargs["foo"]` subscripting). To keep this safe we validate up front:
    #
    #   * All keys must be strings (Python's `**dict` splat requires strings).
    #     Non-string keys -> TypeError (cannot proceed).
    #   * `"RETURN"` and `"__builtins__"` are reserved — splatting them would
    #     shadow the ambient termination primitive / builtins in the exec
    #     locals, breaking model termination or print/len/etc. -> ValueError.
    #   * Meta-channel collisions (`"args"`, `"kwargs"`, `"objects"`) are NOT
    #     rejected. The splat happens FIRST and the meta-channel values are
    #     assigned AFTER, so the meta-channel always wins. PLM-supplied
    #     `kwargs={"args": data}` keeps `ns["args"]` as the empty tuple from
    #     the `args=` param; the user's data is reachable via `kwargs["args"]`.
    #     This is silent precedence by design.
    #   * Identifier-invalid keys (`"foo-bar"`, Python keywords like `"if"`)
    #     splat into `ns` but Python's name resolution can't reach them by
    #     name. Emit a UserWarning so PLM authors notice; the value remains
    #     accessible via `kwargs["foo-bar"]`.
    _RESERVED_KWARG_NAMES = frozenset({"RETURN", "__builtins__"})
    for _k in kwargs:
        if not isinstance(_k, str):
            raise TypeError(
                f"react_llm_verifier: kwargs keys must be strings, got "
                f"{type(_k).__name__} ({_k!r})."
            )
    _bad_reserved = _RESERVED_KWARG_NAMES & set(kwargs)
    if _bad_reserved:
        raise ValueError(
            f"react_llm_verifier: kwargs keys {sorted(_bad_reserved)} are reserved — "
            f"they would shadow the ambient termination primitive (`RETURN`) "
            f"or `__builtins__` in the model's exec scope. Rename them."
        )
    _bad_idents = [
        k for k in kwargs
        if not (k.isidentifier() and not keyword.iskeyword(k))
    ]
    if _bad_idents:
        warnings.warn(
            f"react_llm_verifier: kwargs keys {sorted(_bad_idents)} are not valid "
            f"Python identifiers (or are Python keywords); they will only "
            f"be reachable via `kwargs[<key>]`, not as direct local names.",
            UserWarning,
            stacklevel=2,
        )

    # ---- nested helpers (NOT policies; local to this call) ----

    def _norm(m):
        # DEEP-copy the message dicts so this policy never mutates the CALLER's own
        # dicts — react_llm appends new turns, and a verifier mutates `msgs` in
        # place; either way the caller's `messages` must stay untouched.
        import copy
        if isinstance(m, str):
            return [{"role": "user", "content": m}]
        try:
            _items = [m] if isinstance(m, dict) else list(m)   # TypeError here = not a valid shape
        except TypeError:
            raise TypeError(
                "react_llm_verifier: messages must be a str, a message dict, or an iterable of "
                f"message dicts; got {type(m).__name__}")
        try:
            return [copy.deepcopy(x) for x in _items]
        except Exception as _norm_copy_exc:                     # a message holds an un-copyable value
            raise TypeError(                                    # (a lock/handle/etc.) -> clear, not generic
                "react_llm_verifier: a message could not be copied ("
                + type(_norm_copy_exc).__name__ + ": " + str(_norm_copy_exc)
                + "); messages must be plain JSON-like dicts") from _norm_copy_exc

    def _exec(code, ns, round_no):
        """Run the model's code in the sub-agent's OWN repl namespace `ns`.

        `ns` is a SINGLE dict used as BOTH globals and locals — it IS the
        sub-agent's private "repl globals", modeled on how PLM's kernel runs
        cells in `__main__`. It holds ONLY `__builtins__`, `RETURN`, and what
        PLM injected (kwargs splatted as direct names, plus args/kwargs/
        objects), so the sub-LLM has NO ambient access to the kernel `__main__`
        globals — the `@policy` decorator, `list_policies`, `_PLM_POLICIES`,
        `parallel`, other policies, or the root agent's variables. Any such
        capability must be GRANTED by PLM via the args/kwargs/objects channels.

        Why ONE namespace (not split globals/locals): a `def`/`class` captures
        only its module GLOBALS as `__globals__`. With a split `exec(code,
        globals, locals)`, a helper the model writes would see neither the
        injected names nor sibling helpers (those live only in `locals`), so
        `def f(): return x + 1` would NameError on a kwarg `x`, and recursion
        would break. One dict means everything the model is given OR defines
        behaves like a global — visible at top level, inside any `def`/`class`,
        and across rounds — exactly like PLM's own REPL.

        When PLM grants the `@policy` decorator via `kwargs={"policy": policy}`,
        the splat binds `policy` as a name in `ns`, so the model can write
        `@policy def helper(): ...` naturally — the kernel's decorator
        recognizes the `ast.Name("policy")` when stripping it during the
        re-exec under kernel main, and the resulting policy is registered
        globally as if PLM had written it at cell-level. (PEP-614 `@objects[0]`
        / `@kwargs["policy"]` does NOT work because the strip targets
        `ast.Name`, not `ast.Subscript`; use the named grant.)

        On `_REPLReturn` (model called `RETURN(value)`): validate against
        the closed-over `constraint`. Success -> `term="return"` with the
        validated value. Failure -> surface the violation (via the shared
        constraint formatter) through the captured-stderr path, leave
        `term=None` so the outer loop continues. Any OTHER cell exception
        (e.g. a `NameError` from a stray name) is captured by `exec_ns` into
        the tool result the same way.
        """
        # Delegate the CORE run (namespace exec + captured stdout/stderr + traceback
        # + linecache slot) to `exec_ns` — a repl-injected global (ambient, no import,
        # like the rest); layer ONLY the react-specific RETURN handling here, via
        # `on_return`. Behavior is identical to the old inline runner. The slot keys
        # off the MONOTONIC round counter, not len(ns): the model can define-then-del
        # names so len(ns) repeats across rounds, colliding the linecache slot and
        # overwriting an earlier round's source while a live function's co_filename
        # still points there (#9; mirrors the kernel's monotonic <cell-N>).

        def _on_return(proposed):
            if constraint is None:
                return "return", proposed
            try:
                return "return", constraint.validate(proposed)   # closed-over from outer scope
            except Exception as ce:
                # ConstraintViolation OR any other validator error (a buggy/edge predicate,
                # a wrong-typed value): surface it through the model's normal stderr capture
                # — it lands in the tool result, the model self-corrects next turn, and
                # `term` stays None so the loop continues within budget (never crashes the
                # react loop). Use the SHARED formatter so react's diagnostic matches
                # natural_llm's clear "RETURN value failed ... Required ... Fix the object"
                # form rather than a hand-rolled describe()+traceback.
                from plm._react_helper import (
                    _format_constraint_error, _format_unexpected_validate_error)
                if type(ce).__name__ == "ConstraintViolation":
                    sys.stderr.write(_format_constraint_error(constraint, ce) + "\n")
                else:
                    sys.stderr.write(_format_unexpected_validate_error(constraint, ce) + "\n")
                return None, None

        return exec_ns(code, ns, slot=f"react-{id(ns)}-{round_no}", on_return=_on_return)

    # ---- main ReAct loop ----

    msgs = _norm(messages)
    # ---- the sub-agent's OWN repl namespace (`ns`) ----
    # ONE dict, used as BOTH globals and locals in `_exec`, so it behaves like
    # the sub-agent's private "repl globals" — modeled on how PLM's kernel runs
    # cells in `__main__`. Everything the model is given OR defines is visible
    # at top level, INSIDE any `def`/`class` it writes, and across rounds. (A
    # split globals/locals would hide injected names + sibling helpers from a
    # `def` body — see `_exec`.)
    #
    # It is SEALED from the kernel `__main__`: the sub-LLM has NO ambient access
    # to the @policy decorator, `list_policies`, `get_policy`, `_PLM_POLICIES`,
    # `parallel`, `natural_llm`, `react_llm`, or the root agent's variables. Its
    # world is only its own repl. Every other capability must be a DELIBERATE
    # grant from PLM via the args/kwargs/objects channels, e.g.:
    #     react_llm_verifier(msg, kwargs={"policy": policy})            # grant policy-authoring
    #     react_llm_verifier(msg, kwargs={"natural_llm": natural_llm})  # grant a sub-LLM
    #     react_llm_verifier(msg, kwargs={"call": some_user_policy})    # grant a specific tool
    # so the sub-LLM's capability surface is entirely engineered by PLM.
    #
    # Seeding order: builtins + RETURN (base), then kwargs splat as direct names,
    # then the three meta-channels (which win over any colliding kwarg key).
    # kwargs are validated upstream to exclude `RETURN`/builtins, so those base
    # names are never shadowed.
    # SEALED — a containment boundary, NOT just hygiene: `__builtins__` is a CURATED
    # dict (no __import__/eval/exec/compile/open) and `RETURN` is rebuilt over a minimal
    # globals dict, so neither `__import__(...)` nor `RETURN.__globals__` reaches the
    # kernel `__main__` / `_PLM_POLICIES` / the depth + blessed-caller gates. RETURN
    # BEHAVES IDENTICALLY for the model — it raises `_REPLReturn(value)`, matched by name
    # by `exec_ns`/`_exec`. (The deliberate `().__class__...__subclasses__()` walk stays
    # reachable — see exec_ns; true isolation is a subprocess concern.)
    from plm.repl.exec_ns import safe_builtins, make_sealed_return
    _safe_b = safe_builtins()
    ns: dict = {
        "__builtins__": _safe_b,                        # curated: print/len/range/dict/...; no import/eval/exec/open
        "RETURN":       make_sealed_return(_safe_b),     # termination primitive; __globals__ does NOT leak __main__
        **kwargs,                                       # each kwarg as a direct name
        "args":    args,                                # meta-channels override any collisions
        "kwargs":  kwargs,                              # full dict still available programmatically
        "objects": objects,
    }
    tools = [_tool_python()]                            # python by default

    # `max_turns`/`return_budget` are model-controlled; coerce None/non-int/
    # negative -> a sane non-negative int (no raw TypeError / no surprise
    # negative range — see #18/#22). They sum into one flat per-round budget.
    from plm._react_helper import _coerce_budget
    max_turns = _coerce_budget(max_turns, 8)
    return_budget = _coerce_budget(return_budget, 5)
    
    with llm_call(depth):                               # voluntary lowering (no-op if depth=None)
        check_depth_or_raise()                          # depth gate FIRST: a clear LLMDepthExceeded, not a backend-spec error
        if verifier is not None and _remaining() < 2:
            # A verifier runs ONE level below (its own descend()) and its checks are
            # depth-1 circuits, so it needs >= 2 levels of remaining budget. Fail UP
            # FRONT with a clear error rather than crashing mid-loop when the verifier
            # hook's descend() hits 0. (D2 — depth=1 + verifier is unsupported.)
            raise LLMDepthExceeded(
                f"react_llm_verifier: a verifier needs depth >= 2 (it runs one level "
                f"below and composes depth-1 circuits), but only {_remaining()} level(s) "
                f"remain. Call without a verifier, or raise the depth / AGENT_DEPTH.")
        backend = _make_backend()
        for _round in range(max_turns + return_budget):
            check_depth_or_raise()                      # policy owns the depth gate
            resp = backend.generate(msgs, tools=tools, **generate_kwargs)
            if not isinstance(resp, dict):                  # malformed backend response -> empty turn
                resp = {}
            content = resp.get("content") or ""
            reasoning = resp.get("reasoning") or ""
            tool_calls = resp.get("tool_calls") or []
            if isinstance(tool_calls, dict):                # a single call returned bare -> wrap it
                tool_calls = [tool_calls]
            elif not isinstance(tool_calls, list):          # any other malformed shape -> none
                tool_calls = []
            msgs.append({"role": "assistant", "content": content,
                         "reasoning": reasoning,
                         # Record ONLY the executed+answered call (tool_calls[0]), and ONLY if
                         # it is a dict: a non-dict first element (from a non-conformant provider)
                         # stored here would crash the NEXT round's backend history sanitizer;
                         # the R-F1 guard below still answers it + retries. (storing extra calls
                         # would also leave their ids unanswered.)
                         "tool_calls": ([tool_calls[0]] if tool_calls and isinstance(tool_calls[0], dict) else None)})

            if tool_calls:
                # --- model emitted a tool call: verify it's `python`, then parse/exec/RETURN ---
                tc = tool_calls[0]
                if not isinstance(tc, dict):
                    # Malformed tool_call (str/int/None): the assistant turn stored tool_calls=None
                    # for it (M4), so there is NO tool_call to answer — a `tool` reply would be a
                    # wire-format ORPHAN (tool message with no matching tool_call) and the NEXT
                    # request 400s. Send the error as a USER turn instead, then fall through to the
                    # verifier hook (this round is non-terminal).
                    msgs.append({"role": "user",
                                 "content": f"[error] malformed tool_call ({type(tc).__name__}); "
                                            f"emit a single `python` tool call"})
                else:
                    _fn = tc.get("function")
                    _fn = _fn if isinstance(_fn, dict) else {}   # `function: null` / malformed -> {}
                    tname = _fn.get("name")
                    if tname != "python":
                        # react_llm_verifier only offers the python tool. A backend that calls
                        # anything else gets a clear error back (its id IS answered, so history
                        # stays well-formed) — but DON'T `continue`: this is a non-terminal
                        # round, so it must fall through to the verifier hook below. Mirrors the
                        # root loop's guard (plm.py) rather than silently exec'ing "".
                        msgs.append({"role": "tool", "tool_call_id": tc.get("id") or "",
                                     "content": f"error: tool {tname!r} not supported (only `python`)"})
                    else:
                        args_raw = _fn.get("arguments") or "{}"
                        try:
                            targs = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                            if not isinstance(targs, dict):
                                targs = {}
                        except Exception:
                            # Any parse failure -> {} (not just JSONDecodeError): a huge
                            # integer literal raises ValueError (int_max_str_digits) and
                            # deeply-nested JSON raises RecursionError, neither a
                            # JSONDecodeError.
                            targs = {}
                        code = targs.get("code", "")
                        if not isinstance(code, str):
                            # non-string `code`: clear error + fall through
                            msgs.append({"role": "tool", "tool_call_id": tc.get("id") or "",
                                         "content": "error: `code` must be a string of Python source"})
                        else:
                            out, term, val = "(no output)", None, None  # defensive init
                            # `_exec` (= exec_ns) CAPTURES the cell's stdout AND any error together
                            # and returns them in `out`: a print-then-error cell surfaces BOTH (stdout
                            # THEN traceback), never just the error, and exec_ns NEVER raises for the
                            # cell's own code. So this try/except is an ORCHESTRATION safety net
                            # (fence-strip / `descend` / the `_exec` call), NOT for cell errors —
                            # cell stdout is never lost.
                            try:
                                code = _strip_code_fences(code)
                                with descend():                 # decrement for the act phase ONLY
                                    out, term, val = _exec(code, ns, _round)
                            except SyntaxError as e:
                                out = f"stderr:\n{e}"
                            except Exception as e:              # any other failure -> stderr back to the model
                                out = f"stderr:\n{type(e).__name__}: {e}"

                            if len(tool_calls) > 1:
                                # The SUB-LLM emitted parallel tool_calls; only the first ran. Surface
                                # the nudge in the SUB-LLM's OWN tool result (its message stream) so it
                                # self-corrects to one call per turn — NOT to PLM's stderr.
                                out = ("[react_llm_verifier] you emitted " + str(len(tool_calls))
                                       + " tool_calls; only the FIRST ran — send ONE `python` call "
                                       "per turn.\n\n" + out)
                            msgs.append({"role": "tool", "tool_call_id": tc.get("id") or "", "content": out})
                            if term == "return":                # RETURN(value) succeeded (constraint passed
                                return val                      # or there was no constraint) — done; NO verifier.
            # else: text-only turn — the assistant message is already appended; the
            # model sees its own text-turn next round. Either way this round did NOT
            # terminate, so the verifier runs below.

            # --- per-round VERIFIER hook (the trajectory control axis) ---
            # Runs after EVERY non-terminal round (text-only or tool), once this
            # round's messages are complete, BEFORE the next generate. A successful
            # RETURN above already returned, so the verifier never runs on the
            # terminal round. Wrapped in `descend()` so the depth guard accounts the
            # verifier ONE level below react_llm_verifier — any react_llm/natural_llm
            # circuit it builds is then capped at depth-1 (root=2) and respects the
            # global budget. The verifier mutates `msgs` in place; its return value
            # is ignored. Exceptions propagate (fail-loud; the verifier is trusted).
            if verifier is not None:
                with descend():
                    verifier(msgs)

        # --- budget exhausted: model never produced an accepted RETURN(value) ---
        if constraint is None:
            raise RuntimeError(
                "react_llm_verifier: budget exhausted; model never called RETURN(value). "
                "react_llm_verifier terminates only on RETURN(value); text-only assistant "
                "turns are not finalizations.")
        
        raise ConstraintViolation(
            "react_llm_verifier: budget exhausted; model never produced a RETURN(value) "
            "that satisfied the constraint. Each rejected attempt's violation "
            "was returned to the model via the tool result; see the conversation "
            "history for details.")
