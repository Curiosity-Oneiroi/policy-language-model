@policy
def react_auto(messages, namespace=None, *, max_turns, constraint=None,
               return_budget=5, generate_kwargs=None, model=None):
    """ReAct sub-agent: model + python tool, looped in the SAME REPL.

    `max_turns` is REQUIRED: a sub-agent is a raw capability, and a raw
    capability arrives with no round budget any more than it arrives with a
    system prompt. Deciding how many rounds the work needs is part of briefing
    it. BOTH round budgets are capped (`_MAX_TURNS_CAP`, `_MAX_RETURN_BUDGET`)
    to bound cost — the loop runs their sum, so capping only one would leave
    the bound evadable by a single keyword.

    Terminates ONLY when the model calls `RETURN(value)` inside its
    python tool. Text-only assistant turns just append-and-continue.
    The sub-LLM's ambient capabilities are intentionally minimal —
    `__builtins__` and `RETURN`. Everything else (sub-LLM calls, policy
    creation, access to other policies, user globals) is a DELIBERATE
    grant from PLM, all through ONE channel: a key in `namespace`.

    Quick examples:

        # Simple — model writes code, calls RETURN:
        answer = react_auto("Compute 7! using python")
        # -> 5040  (whatever the model passed to RETURN)

        # Grant named data: every `namespace` key is a DIRECT local
        # name for the model (no subscripting):
        result = react_auto("Add the two numbers and RETURN the sum",
                            {"a": 3, "b": 5}, max_turns=4)
        # Model writes: `RETURN(a + b)`  -> 8
        # ...and the dict is STILL YOURS afterwards: anything the model
        # defined is in it.

        # Grant policy-authoring (the @policy decorator) by name:
        from plm.policy import policy
        result = react_auto(
            "Author a doubler policy named `doubler` and use it on input.",
            {"policy": policy, "input": 7}, max_turns=4)
        # Model writes:
        #   @policy
        #   def doubler(x): return x * 2
        #   RETURN(doubler(input))   -> 14

        # Grant nested-LLM access:
        result = react_auto(
            "Decompose into a sub-question and call llm on it.",
            {"llm": llm}, max_turns=4)
        # Model writes: `RETURN(llm("the sub-question"))`

        # Constraint-validated final answer (validated INSIDE _exec on
        # the model's RETURN value; a ConstraintViolation prints into
        # the tool result and the loop continues):
        class Answer(Constraint):
            value: Constraint.field(type=int)
        result = react_auto("Compute fact(7)", constraint=Answer, max_turns=6)
        # Model writes: `RETURN({"value": 5040})` -> Answer(value=5040)

        # Finish a session `react` has been stepping (max_turns=0 grants no
        # working rounds -- just "conclude now, on the work already done"):
        ans = react_auto(msgs, ns, max_turns=0)

    Parameters:

        messages: str / dict / list of message dicts. Initial conversation.
            MESSAGE WEAVING — if you pass a LIST, react_auto works on THAT SAME
            list by reference: the assistant/tool turns it runs are appended to
            YOUR list, so after the call you (or another policy you hand the list
            to) can read the full woven conversation. Your original message dicts
            are never mutated — only new turns are added. (A str / single dict has
            no caller list to weave into, so a fresh one is built.)

        namespace=None: the sub-agent's namespace, as a dict YOU own — the
            SECOND woven container, alongside `messages`. Whatever is in it is
            a variable the sub-agent can use BY NAME; whatever the sub-agent
            defines lands in it and is STILL THERE when the call returns. That
            is how a stepped `react` session continues: hand back the same
            dict. Pass None for a pure bubble — a private namespace that dies
            with the call (exactly the old react_llm behaviour).

                ns = {"df": df}                 # grant, by name
                react(msgs, ns, rounds=2)       # the agent works
                ns["surviving"]                 # read what it built
                ns.pop("bad_guess", None)       # evict one variable
                ns.clear()                      # end the session

            Keys must be valid Python identifiers — they ARE the model's
            variable names. `RETURN` is refused: termination is the protocol's
            to bind, and it is bound only for the duration of this call.
            `__builtins__` is seeded on first use; its presence is what marks
            the dict as an open session.

        constraint=None: optional Constraint CLASS, validated INSIDE
            `_exec` on the model's `RETURN(value)`. On
            ConstraintViolation the violation's traceback prints into
            the tool result (via stderr) — the model sees it next turn
            and can retry. UNLIKE llm, factory/predicate
            Constraints ARE supported here, because the model can run
            python and the validator runs against the actual value.

        max_turns (REQUIRED), return_budget=5: SUMMED into a single flat
            per-round budget — the loop runs `max_turns + return_budget`
            iterations and EVERY round (text-only, tool, non-python tool, or a
            post-constraint-failure RETURN retry) consumes exactly one slot.
            The split is a sizing convention, not two separate phases:
            `max_turns` is the round budget you communicate to the model as the
            work it should plan for, and `return_budget` is the extra slack
            reserved for finalizing / RETURN-retries on top of that. Both are
            coerced to a non-negative int (None/non-int -> default, negative ->
            0); if their sum is 0 the loop makes no model call.


        generate_kwargs=None (resolved to {}): backend kwargs (temperature,
            top_p, response_format, ...) forwarded to each
            `backend.generate(...)`.

        Trajectory grants need no flag either: `namespace["messages"] =
            messages` hands the sub-agent its own LIVE list (it can then
            rewrite its past — deliberate, and the author's call);
            `namespace["trace"] = copy.deepcopy(messages)` hands it a frozen
            snapshot. For harness-mediated trajectory control there is
            react_verifier_llm.

    Returns:

        Whatever the model passed to `RETURN(value)`, validated by
        `constraint` if set (returns the validated Constraint instance
        / raw value). NEVER silently returns the model's last text.

    Raises:

        - ValueError: a `namespace` key is reserved (`RETURN`) or is not a
          valid Python identifier. No backend call is made.
        - TypeError: `messages` / `namespace` are the wrong type. No backend call.
        - LLMDepthExceeded: the depth budget is at 0 before a generate.

        Budget exhaustion without an accepted RETURN is NOT an exception: it
        returns None (S2.7b), and the true cause is recorded in the sub-call
        log rather than inferred from the value.

    What is ambient inside the sub-LLM's exec scope?

        ONLY what you put in `namespace`, plus two names:
        - `__builtins__` (print, len, range, dict, list, ...).
        - `RETURN(value)` (the sole termination primitive — the model
          can't end the loop without it). Bound for THIS call only, and
          removed again on the way out: in raw `react` the name does not
          exist, because termination is protocol, not capability.

        Everything else — `policy`, `list_policies`, `get_policy`,
        `read_policy`, `_PLM_POLICIES`, `parallel`, `llm`,
        `react_auto`, user-cell globals — is BLOCKED. The sub-LLM cannot
        "discover" the available policies; PLM is the only thing that
        knows the namespace and chooses what to plumb through.

    Internals:

        - Per round: `backend.generate` -> if the model emitted a python
          tool call, parse the code, run it inside `with descend():` so
          any nested LLM call inherits a decremented depth. `_exec` runs
          the code in the sub-agent's OWN repl namespace — a single dict
          (globals == locals, like PLM's kernel `__main__`), which IS the
          caller's `namespace`, and is isolated from kernel main.
        - Containment: `tmp = 5` lands in `namespace` (yours). `@policy def
          helper(): ...` (when `policy` is granted via `namespace`) goes
          through the decorator's normal path; the decorator re-execs
          the body under kernel main, so the resulting policy has full
          main-globals access and is registered globally — as if PLM
          had authored it at cell-level.
        - Termination: only on `RETURN(value)`. A text-only assistant
          turn is appended to history and the loop continues. Budget
          exhausted -> None, with the true cause in the sub-call log.
        - Constraint validation happens INSIDE `_exec`'s `_REPLReturn`
          catch — on failure the violation prints to stderr through
          the same path as any other exception, ending up in the tool
          result for the model to see.
    """
    import sys, json, keyword
    from plm.plm_tools import _tool_python
    from plm._react_helper import _strip_code_fences
    from plm.policy.defaults._llm_infra import (
        _make_backend, descend, llm_call, check_depth_or_raise)

    # Resolve None-sentinel mutable defaults (no shared-default footgun).
    generate_kwargs = {} if generate_kwargs is None else generate_kwargs

    # `generate_kwargs` is splat into `backend.generate(msgs, tools=tools, **generate_kwargs)`,
    # which already passes `messages` (positionally) and `tools` — a colliding key would raise a
    # cryptic `TypeError: got multiple values for keyword argument`. Reject it up front with a
    # clear message (model params like temperature/max_tokens go here; grants go in namespace).
    _reserved_gk = {"messages", "tools"} & set(generate_kwargs)
    if _reserved_gk:
        raise ValueError(
            f"generate_kwargs may not contain {sorted(_reserved_gk)} — react_auto passes "
            f"`messages` and `tools` itself; put backend params (temperature, max_tokens, ...) here")

    # ---- namespace validation ------------------------------------------------
    # `namespace` IS the sub-agent's globals dict, so every key is a name the model
    # can write. Keys must therefore be real identifiers — an unreachable key
    # would be a silent grant. `RETURN` is refused because this call binds it.
    if namespace is None:
        namespace = {}                      # a private bubble: dies with the call
    if isinstance(namespace, bool):
        raise TypeError("react_auto: namespace must be the state DICT itself "
                        "(or None for a private bubble) — `namespace: bool` "
                        "(all-or-none) is the AGENTIC harness's convention; "
                        "FULL weaves the dict (A1c).")
    if not isinstance(namespace, dict):
        raise TypeError(f"react_auto: namespace must be a dict (it IS the "
                        f"sub-agent's namespace), got {type(namespace).__name__}")
    for _k in namespace:
        if not isinstance(_k, str):
            raise TypeError(f"react_auto: namespace keys must be strings, got "
                            f"{type(_k).__name__} ({_k!r}).")
        if _k == "RETURN":
            raise ValueError("react_auto: 'RETURN' is reserved — the protocol "
                             "binds it for the duration of this call.")
        if _k != "__builtins__" and not (_k.isidentifier()
                                         and not keyword.iskeyword(_k)):
            raise ValueError(f"react_auto: namespace key {_k!r} is not a valid "
                             f"Python identifier; namespace keys ARE the model's "
                             f"variable names, so it could never be reached.")

    # ---- nested helpers (NOT policies; local to this call) ----

    def _norm(m):
        # PASS-BY-REFERENCE (message weaving): when the caller passes a LIST, react_auto works on
        # THAT SAME list — the turns it appends (and the namespace system-note it inserts) become
        # visible to the CALLER, so the woven conversation can be handed to another policy (e.g. a
        # verifier) for inspection. react_auto only APPENDS/inserts new turns (it never mutates the
        # caller's existing dicts) and backends read messages without mutating them, so the caller's
        # ORIGINAL dicts stay intact — weaving only ADDS turns. A str / single dict / other iterable
        # has no caller list to weave into, so a fresh list is built (only a passed LIST is woven).
        if isinstance(m, str):
            return [{"role": "user", "content": m}]
        if isinstance(m, dict):
            return [m]                                          # single dict -> wrap (by reference)
        if isinstance(m, list):
            return m                                            # the caller's list, BY REFERENCE
        try:
            return list(m)                                      # other iterable -> materialize (nothing to weave)
        except TypeError:
            raise TypeError(
                "react_auto: messages must be a str, a message dict, or an iterable of "
                f"message dicts; got {type(m).__name__}")

    def _exec(code, ns, round_no):
        """Run the model's code in the sub-agent's OWN repl namespace `ns`.

        `ns` is a SINGLE dict used as BOTH globals and locals — it IS the
        sub-agent's private "repl globals", modeled on how PLM's kernel runs
        cells in `__main__`. It IS the caller's `namespace` dict — holding
        `__builtins__`, `RETURN` (this call only), and whatever the caller put
        there — so the sub-LLM has NO ambient access to the kernel `__main__`
        globals: the `@policy` decorator, `list_policies`, `_PLM_POLICIES`,
        `parallel`, other policies, or the root agent's variables. Any such
        capability must be GRANTED by PLM as a key in `namespace`.

        Why ONE namespace (not split globals/locals): a `def`/`class` captures
        only its module GLOBALS as `__globals__`. With a split `exec(code,
        globals, locals)`, a helper the model writes would see neither the
        injected names nor sibling helpers (those live only in `locals`), so
        `def f(): return x + 1` would NameError on a kwarg `x`, and recursion
        would break. One dict means everything the model is given OR defines
        behaves like a global — visible at top level, inside any `def`/`class`,
        and across rounds — exactly like PLM's own REPL.

        When PLM grants the `@policy` decorator via `namespace["policy"]`,
        the key IS the name `policy` in the model's globals, so it can write
        `@policy def helper(): ...` naturally — the kernel's decorator
        recognizes the `ast.Name("policy")` when stripping it during the
        re-exec under kernel main, and the resulting policy is registered
        globally as if PLM had written it at cell-level. (A PEP-614
        subscript form does NOT work because the strip targets `ast.Name`,
        not `ast.Subscript`; grant it under the name `policy`.)

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
                # llm's clear "RETURN value failed ... Required ... Fix the object"
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

    # depth-1 is the experiment's fixed condition (S2.3), kernel-enforced via
    # `_PLM_ROOT_DEPTH = 1` -> AGENT_DEPTH. The old `depth=` parameter only ever
    # permitted VOLUNTARY LOWERING, which can only reduce a policy's capability,
    # so no policy would use it and exposing it implied depth was an authoring
    # choice. Removed; the scope is simply inherited.
    depth = None

    msgs = _norm(messages)
    # ---- continuation ------------------------------------------------------
    # There is no registry and no session object: `namespace` IS the session. A dict
    # that already carries `__builtins__` has been stepped by `react`, so this
    # call CONTINUES it — same variables, same weave. Nothing to look up,
    # nothing to evict, nothing to tombstone.
    from plm.policy.defaults._react_protocol import (AUTO_BUDGET_BRIEFING,
                                                     AUTO_FINAL_ROUND)
    continued = "__builtins__" in namespace

    # ---- the sub-agent's OWN repl namespace ----
    # ONE dict, used as BOTH globals and locals in `_exec`, so it behaves like
    # the sub-agent's private "repl globals" — modeled on how PLM's kernel runs
    # cells in `__main__`. Everything the model is given OR defines is visible
    # at top level, INSIDE any `def`/`class` it writes, and across rounds. (A
    # split globals/locals would hide injected names + sibling helpers from a
    # `def` body — see `_exec`.)
    #
    # It is SEALED from the kernel `__main__`: the sub-LLM has NO ambient access
    # to the @policy decorator, `list_policies`, `get_policy`, `_PLM_POLICIES`,
    # `parallel`, `llm`, `react_auto`, or the root agent's variables. Its
    # world is only its own repl. Every other capability is a DELIBERATE grant:
    # a key the caller put in `namespace`, e.g.
    #     react_auto(msg, {"policy": policy}, max_turns=4)    # policy-authoring
    #     react_auto(msg, {"llm": llm}, max_turns=4)          # a sub-LLM
    #     react_auto(msg, {"call": some_policy}, max_turns=4) # one specific tool
    # so the sub-LLM's capability surface is entirely engineered by PLM.
    # Keys are validated upstream (identifiers only, RETURN refused), so the
    # base names are never shadowed.
    # SEALED — a containment boundary, NOT just hygiene: `__builtins__` is a CURATED
    # dict (no __import__/eval/exec/compile/open) and `RETURN` is rebuilt over a minimal
    # globals dict, so neither `__import__(...)` nor `RETURN.__globals__` reaches the
    # kernel `__main__` / `_PLM_POLICIES` / the depth + blessed-caller gates. RETURN
    # BEHAVES IDENTICALLY for the model — it raises `_REPLReturn(value)`, matched by name
    # by `exec_ns`/`_exec`. (The deliberate `().__class__...__subclasses__()` walk stays
    # reachable — see exec_ns; true isolation is a subprocess concern.)
    from plm.repl.exec_ns import safe_builtins, make_sealed_return
    # The namespace IS the caller's dict. Seed the seal once (its presence is
    # the open-session marker); bind RETURN for this call only.
    ns: dict = namespace
    if not continued:
        ns["__builtins__"] = safe_builtins()
    ns["RETURN"] = make_sealed_return(ns["__builtins__"])
    tools = [_tool_python()]                            # python by default

    # `max_turns`/`return_budget` are model-controlled; coerce None/non-int/
    # negative -> a sane non-negative int (no raw TypeError / no surprise
    # negative range — see #18/#22). They sum into one flat per-round budget.
    from plm._react_helper import _coerce_budget
    # Safety cap: the ONE artificial bound in the experiment. A raw capability
    # carries no round budget, so the author must supply one (max_turns is
    # required); the cap exists only to bound cost, is disclosed in the task
    # card, and is enforced here rather than silently.
    _MAX_TURNS_CAP = 20
    _requested_turns = max_turns
    _was_clamped = False
    max_turns = _coerce_budget(max_turns, 8)
    if max_turns > _MAX_TURNS_CAP:
        _was_clamped = True
        # VISIBLE clamp, never silent: grading an artifact the author did not
        # write would be invalid, and the clamp itself is a measurement
        # (requesting past the cap is budget misestimation).
        import sys as _sys
        print(f"[react_auto] max_turns={_requested_turns} exceeds the safety cap "
              f"({_MAX_TURNS_CAP}); clamped to {_MAX_TURNS_CAP}.", file=_sys.stderr)
        max_turns = _MAX_TURNS_CAP
    _MAX_RETURN_BUDGET = 10          # return_budget is a ROUND budget too: the loop
                                     # runs max_turns + return_budget, so leaving it
                                     # uncapped let one keyword evade the safety cap
                                     # entirely (and the cost bound with it).
    _requested_rb = return_budget
    return_budget = _coerce_budget(return_budget, 5)
    if return_budget > _MAX_RETURN_BUDGET:
        import sys as _sys2
        print(f"[react_auto] return_budget={_requested_rb} exceeds the cap "
              f"({_MAX_RETURN_BUDGET}); clamped.", file=_sys2.stderr)
        return_budget = _MAX_RETURN_BUDGET

    # The protocol's own briefing: a sub-agent that must RETURN before its
    # rounds run out is told how many it has. (Disclosure belongs to the
    # protocol, not to the capability — raw `react`, having nothing to race,
    # says nothing. Authors may add briefing but cannot make a sub-agent that
    # is under a deadline blind to it.)
    #
    # max_turns=0 is the ELICITATION ADAPTER: no working rounds, just finish
    # now with whatever the session already has. It is the natural end of a
    # stepped session — `react` explores, `react_auto(msgs, max_turns=0)` asks
    # for the answer — so there is no budget to announce, only the demand,
    # which the final-round pressure below delivers on round 0.
    if max_turns:
        # ALWAYS APPENDED, never inserted at the top (author ruling): the
        # deadline is "from here on", and that reads identically whether this
        # is a fresh task or a continuation of work already in the list. One
        # path, no flag — the briefing simply goes after whatever was passed.
        msgs.append({"role": "system",
                     "content": AUTO_BUDGET_BRIEFING.format(max_turns=max_turns)})


    try:
        with llm_call(depth):                               # voluntary lowering (no-op if depth=None)
            check_depth_or_raise()                          # depth gate FIRST: a clear LLMDepthExceeded, not a backend-spec error
            backend = _make_backend(model=model)
            for _round in range(max_turns + return_budget):
                check_depth_or_raise()                      # policy owns the depth gate
                if _round == max(max_turns - 1, 0):     # max_turns=0 -> pressure at once
                    msgs.append({"role": "user", "content": AUTO_FINAL_ROUND})
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

                if not tool_calls:
                    # Text-only turn: append assistant message above and continue.
                    # react_auto only terminates on RETURN(value) inside the python
                    # tool. The model sees its own text-turn in history next round.
                    continue

                # --- model emitted a tool call: verify it's `python`, parse, exec, handle RETURN ---
                tc = tool_calls[0]
                # The SUB-LLM emitted parallel tool_calls; react processes ONLY the first. Surface the
                # nudge in the SUB-LLM's OWN tool result (its message stream) so it self-corrects to one
                # call per turn — NOT to PLM's stderr (sub-agent business, not the meta-reasoner's).
                # Mirror plm.py: compute it ONCE and prepend to EVERY branch below (malformed / non-python
                # / bad-code / exec), so it is NOT lost on a branch that `continue`s before the exec (B2).
                _parallel_note = (
                    "[react_auto] you emitted " + str(len(tool_calls)) + " tool_calls; only the FIRST ran "
                    "— send ONE `python` call per turn.\n\n") if len(tool_calls) > 1 else ""
                if not isinstance(tc, dict):
                    # Malformed tool_call (str/int/None): the assistant turn stored tool_calls=None for it
                    # (M4), so there is NO tool_call to answer — a `tool` reply would be a wire-format
                    # ORPHAN (a tool message with no matching tool_call) and the NEXT request 400s. Send
                    # the error as a USER turn instead; the model sees it and retries.
                    msgs.append({"role": "user",
                                 "content": _parallel_note + f"[error] malformed tool_call ({type(tc).__name__}); "
                                            f"emit a single `python` tool call"})
                    continue
                _fn = tc.get("function")
                _fn = _fn if isinstance(_fn, dict) else {}      # `function: null` / malformed -> {}
                tname = _fn.get("name")
                if tname != "python":
                    # react_auto only offers the python tool. A backend that calls
                    # anything else gets a clear error back (its id IS answered, so
                    # history stays well-formed) and another round — mirroring the
                    # root loop's guard (plm.py) rather than silently exec'ing "".
                    msgs.append({"role": "tool", "tool_call_id": tc.get("id") or "",
                                 "content": _parallel_note + f"error: tool {tname!r} not supported (only `python`)"})
                    continue
                args_raw = _fn.get("arguments") or "{}"
                try:
                    targs = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    if not isinstance(targs, dict):
                        targs = {}
                except Exception:
                    # Any parse failure -> {} (not just JSONDecodeError): a huge integer
                    # literal raises ValueError (int_max_str_digits) and deeply-nested
                    # JSON raises RecursionError, neither a JSONDecodeError.
                    targs = {}
                code = targs.get("code", "")
                if not isinstance(code, str):
                    # `code` present but not a string (null/number/object): clear error +
                    # retry, not a confusing AttributeError from fence-stripping.
                    msgs.append({"role": "tool", "tool_call_id": tc.get("id") or "",
                                 "content": _parallel_note + "error: `code` must be a string of Python source"})
                    continue

                out, term, val = "(no output)", None, None  # defensive init
                # `_exec` (= exec_ns) CAPTURES the cell's stdout AND any error together and
                # returns them in `out`: a print-then-error cell surfaces BOTH (stdout THEN
                # traceback), never just the error, and exec_ns NEVER raises for the cell's
                # own code. So this try/except is an ORCHESTRATION safety net (fence-strip /
                # `descend` / the `_exec` call), NOT for cell errors — cell stdout is never lost.
                try:
                    code = _strip_code_fences(code)
                    with descend():                         # decrement for the act phase ONLY
                        out, term, val = _exec(code, ns, _round)
                except SyntaxError as e:
                    out = f"stderr:\n{e}"
                except Exception as e:                      # any other failure -> stderr back to the model
                    out = f"stderr:\n{type(e).__name__}: {e}"

                msgs.append({"role": "tool", "tool_call_id": tc.get("id") or "",
                             "content": _parallel_note + out})
                if term == "return":                        # RETURN(value) succeeded (constraint passed
                    from plm.policy.defaults._subcall_log import record as _log_subcall
                    _log_subcall(primitive="react_auto", granted=max_turns + return_budget,
                                 used=_round + 1, truncated=False, clamped=_was_clamped,
                                 requested=_requested_turns, opened=not continued,
                                 outcome="ok")
                    return val

            # --- budget exhausted: model never produced an accepted RETURN(value) ---
            # Failure is a VALUE, not an exception (S2.7b). The caller receives None
            # whether the sub-agent ran out of rounds or never satisfied its
            # constraint; the two causes are deliberately indistinguishable at the
            # policy level and that ambiguity is disclosed in the task card. The
            # harness records the true cause here, so failure attribution reads this
            # log rather than inferring anything from the returned value.
            from plm.policy.defaults._subcall_log import record as _log_subcall
            _log_subcall(primitive="react_auto", granted=max_turns + return_budget,
                         used=max_turns + return_budget, truncated=True,
                         clamped=_was_clamped, requested=_requested_turns,
                         opened=not continued,
                         outcome="truncated" if constraint is None else "constraint_exhausted")
            return None
    finally:
        # The protocol's name never outlives its call — on the RETURN path,
        # on exhaustion, and on ANY exception (depth, backend, constraint).
        # Otherwise a later raw `react` on this namespace would find RETURN
        # defined, which is precisely what the capability must not have.
        ns.pop("RETURN", None)
