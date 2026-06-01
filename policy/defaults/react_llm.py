@policy
def react_llm(messages, *, args=(), kwargs=None, objects=None,
              constraint=None, max_turns=8, return_budget=5, depth=None,
              generate_kwargs=None):
    """ReAct sub-agent: model + python tool, looped in the SAME REPL.

    Terminates ONLY when the model calls `RETURN(value)` inside its
    python tool. Text-only assistant turns just append-and-continue.
    The sub-LLM's ambient capabilities are intentionally minimal —
    `__builtins__` and `RETURN`. Everything else (sub-LLM calls, policy
    creation, access to other policies, user globals) is a DELIBERATE
    grant from PLM via `args` / `kwargs` / `objects`.

    Quick examples:

        # Simple — model writes code, calls RETURN:
        answer = react_llm("Compute 7! using python")
        # -> 5040  (whatever the model passed to RETURN)

        # Pass named data; the model uses each kwarg key as a DIRECT
        # local name (no `kwargs["a"]` subscripting needed):
        result = react_llm(
            "Add the two numbers and RETURN the sum",
            kwargs={"a": 3, "b": 5},
        )
        # Model writes: `RETURN(a + b)`  -> 8

        # Grant policy-authoring (the @policy decorator) by name:
        from plm.policy import policy
        result = react_llm(
            "Author a doubler policy named `doubler` and use it on input.",
            kwargs={"policy": policy, "input": 7},
        )
        # Model writes:
        #   @policy
        #   def doubler(x): return x * 2
        #   RETURN(doubler(input))   -> 14

        # Grant nested-LLM access:
        result = react_llm(
            "Decompose into a sub-question and call natural_llm on it.",
            kwargs={"natural_llm": natural_llm},
        )
        # Model writes: `RETURN(natural_llm("the sub-question"))`

        # Constraint-validated final answer (validated INSIDE _exec on
        # the model's RETURN value; a ConstraintViolation prints into
        # the tool result and the loop continues):
        class Answer(Constraint):
            value: int
        result = react_llm("Compute fact(7)", constraint=Answer)
        # Model writes: `RETURN({"value": 5040})` -> Answer(value=5040)

        # Voluntarily lower the depth budget for this sub-task:
        react_llm(msg, depth=1)

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

        max_turns=8: budget for tool-using rounds (loop iterations
            where the model emitted a tool call).

        return_budget=5: additional rounds for RETURN-retries after
            constraint failures. The actual loop bound is
            `max_turns + return_budget`.

        depth=None: voluntary LLM-recursion-depth cap. Same semantics
            as natural_llm (None inherits scope; int clamps to
            min(depth, current); cannot raise above the sealed ceiling).

        generate_kwargs=None (resolved to {}): backend kwargs (temperature,
            top_p, response_format, ...) forwarded to each
            `backend.generate(...)`.

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
        - LLMDepthExceeded: the depth budget is at 0 before a generate.
        - UserWarning (emitted, not raised): kwargs has identifier-invalid
          or Python-keyword keys.

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
          with `globals=exec_globals` (the restricted dict) and
          `locals=ns` (the per-call dict preseeded with args/kwargs/
          objects + each kwarg key as a direct local name).
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
    import sys, json, io, linecache, traceback, keyword, warnings
    from contextlib import redirect_stdout, redirect_stderr
    from plm.plm_tools import _tool_python
    from plm._react_helper import _strip_code_fences
    from plm.policy.defaults._llm_infra import (
        _make_backend, descend, llm_call, check_depth_or_raise)
    from plm.constraint import ConstraintViolation

    # Resolve None-sentinel mutable defaults (no shared-default footgun).
    kwargs = {} if kwargs is None else kwargs
    objects = [] if objects is None else objects
    generate_kwargs = {} if generate_kwargs is None else generate_kwargs

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
                f"react_llm: kwargs keys must be strings, got "
                f"{type(_k).__name__} ({_k!r})."
            )
    _bad_reserved = _RESERVED_KWARG_NAMES & set(kwargs)
    if _bad_reserved:
        raise ValueError(
            f"react_llm: kwargs keys {sorted(_bad_reserved)} are reserved — "
            f"they would shadow the ambient termination primitive (`RETURN`) "
            f"or `__builtins__` in the model's exec scope. Rename them."
        )
    _bad_idents = [
        k for k in kwargs
        if not (k.isidentifier() and not keyword.iskeyword(k))
    ]
    if _bad_idents:
        warnings.warn(
            f"react_llm: kwargs keys {sorted(_bad_idents)} are not valid "
            f"Python identifiers (or are Python keywords); they will only "
            f"be reachable via `kwargs[<key>]`, not as direct local names.",
            UserWarning,
            stacklevel=2,
        )

    # ---- nested helpers (NOT policies; local to this call) ----

    def _norm(m):
        if isinstance(m, str):
            return [{"role": "user", "content": m}]
        if isinstance(m, dict):
            return [m]
        return list(m)

    def _exec(code, exec_globals, ns):
        """Run the model's code in a RESTRICTED execution scope.
        `globals=exec_globals` exposes ONLY `__builtins__` and `RETURN` —
        the sub-LLM has no ambient access to the `@policy` decorator,
        `list_policies`, `_PLM_POLICIES`, `parallel`, other policies, or
        user-cell globals. Sub-LLM calls, policy creation, and any other
        non-builtin capability must be granted by PLM via the
        args/kwargs/objects channels (preseeded into `ns`). `locals=ns`
        so bare assignments (`tmp=5`) stay in `ns` (contained) AND the
        data-channel names `args`/`kwargs`/`objects` plus any `def
        helper(): ...` defined here all live alongside in `ns` across
        rounds. When PLM grants the `@policy` decorator via
        `kwargs={"policy": policy}`, the splat binds `policy` directly
        as a local name in `ns`, so the model can write `@policy def
        helper(): ...` naturally — the kernel's `@policy` decorator
        recognizes the local `ast.Name("policy")` when stripping it
        during the re-exec under kernel main, and the resulting policy
        is registered globally as if PLM had written it at cell-level.
        (PEP-614 `@objects[0]` / `@kwargs["policy"]` decorator syntax
        does NOT work because the strip targets `ast.Name`, not
        `ast.Subscript`; the recommended idiom is therefore the named
        `kwargs={"policy": policy}` grant.)

        On `_REPLReturn` (model called `RETURN(value)`): validate against
        the closed-over `constraint`. Success -> `term="return"` with the
        validated value. Failure (ConstraintViolation) -> print the
        violation through the same stderr path as any other exception,
        leave `term=None` so the outer loop continues. Any other
        exception (including `NameError` from a stray `RETURN_LLM(...)`
        — not defined in v1 PREFIX) prints its traceback to `buf_e`.
        """
        fn = f"<react-{id(ns)}-{len(ns)}>"
        linecache.cache[fn] = (len(code), None, code.splitlines(True), fn)
        buf_o, buf_e = io.StringIO(), io.StringIO()
        term = val = None
        with redirect_stdout(buf_o), redirect_stderr(buf_e):
            try:
                exec(compile(code, fn, "exec"), exec_globals, ns)
            except BaseException as e:
                if type(e).__name__ == "_REPLReturn":   # name-based (like the kernel loop)
                    proposed = getattr(e, "value", None)
                    if constraint is None:
                        term, val = "return", proposed
                    else:
                        try:
                            val = constraint.validate(proposed)   # closed-over from outer scope
                            term = "return"
                        except ConstraintViolation as ce:
                            # Route the violation through the model's normal stderr/traceback
                            # surface — it lands in the tool result, model sees it next turn.
                            # `term` stays None -> outer loop continues within budget.
                            sys.stderr.write(constraint.describe() + "\n")
                            traceback.print_exception(type(ce), ce, ce.__traceback__)
                else:
                    traceback.print_exc()               # any other exception (incl. NameError on a
                                                         # stray RETURN_LLM(...) — not defined in v1)
                                                         # -> captured into buf_e for the tool result.
        return (buf_o.getvalue() + buf_e.getvalue()) or "(no output)", term, val

    # ---- main ReAct loop ----

    msgs = _norm(messages)
    # ---- per-call exec locals (`ns`) ----
    # Splat `kwargs` FIRST so each key/value becomes a direct local name in
    # the model's exec scope (e.g. `kwargs={"policy": policy, "x": 7}` lets
    # the model write `@policy def foo(): ...` and `print(x)` directly).
    # Then assign the three meta-channels AFTER — they win over any kwarg
    # key with the same name (silent precedence; see kwargs-validation
    # docstring above).
    ns: dict = {
        **kwargs,                                       # bind each kwarg as a direct local name
        "args":    args,                                # meta-channels override any collisions
        "kwargs":  kwargs,                              # full dict still available for programmatic access
        "objects": objects,
    }
    main_g = sys.modules["__main__"].__dict__           # kernel main — used here ONLY to plumb
                                                         # RETURN into exec_globals; never passed to exec.
    # ---- RESTRICTED exec globals for the model's code ----
    # The sub-LLM has NO ambient access to the kernel namespace. It sees
    # ONLY two names by default:
    #   __builtins__ — so print/len/range/dict/list/etc. work.
    #   RETURN       — the sole termination primitive (without it the loop
    #                  can never end; this is a contract, not a capability).
    # NO ambient `policy` (the @policy decorator), no `list_policies`,
    # `get_policy`, `read_policy`, `rewrite_policy`, `_PLM_POLICIES`,
    # `parallel`, `natural_llm`, `react_llm`, or user-cell globals.
    #
    # Every other capability — sub-LLM calls, depth control, policy
    # creation, access to other policies — must be a DELIBERATE grant
    # from PLM via the args/kwargs/objects channels (since `kwargs` is
    # splat as direct local names, grants read naturally):
    #     react_llm(msg, kwargs={"policy": policy})       # grant policy-authoring
    #     react_llm(msg, kwargs={"natural_llm": natural_llm})  # grant sub-LLM
    #     react_llm(msg, kwargs={"call": some_user_policy})    # grant a specific tool
    # PLM then instructs the model how to use them in the message prompt.
    # This makes the sub-LLM's capability surface entirely engineered by
    # PLM — the sub-LLM cannot "discover" what other policies exist or
    # chain itself recursively without PLM's explicit say-so.
    _builtins_mod = sys.modules["builtins"]
    exec_globals = {
        "__builtins__": _builtins_mod,
        "RETURN":       main_g["RETURN"],               # for termination
    }
    tools = [_tool_python()]                            # python by default

    with llm_call(depth):                               # voluntary lowering (no-op if depth=None)
        backend = _make_backend()
        for _round in range(max_turns + return_budget):
            check_depth_or_raise()                      # policy owns the depth gate
            resp = backend.generate(msgs, tools=tools, **generate_kwargs)
            content = resp.get("content") or ""
            tool_calls = resp.get("tool_calls") or []
            msgs.append({"role": "assistant", "content": content,
                         "reasoning": resp.get("reasoning"),
                         "tool_calls": tool_calls or None})

            if not tool_calls:
                # Text-only turn: append assistant message above and continue.
                # react_llm only terminates on RETURN(value) inside the python
                # tool. The model sees its own text-turn in history next round.
                continue

            # --- model emitted a python tool call: parse, exec, handle RETURN, append result ---
            tc = tool_calls[0]
            args_raw = tc.get("function", {}).get("arguments") or "{}"
            try:
                targs = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                if not isinstance(targs, dict):
                    targs = {}
            except json.JSONDecodeError:
                targs = {}
            code = targs.get("code", "")

            out, term, val = "(no output)", None, None  # defensive init
            try:
                code = _strip_code_fences(code)
                with descend():                         # decrement for the act phase ONLY
                    out, term, val = _exec(code, exec_globals, ns)
            except SyntaxError as e:
                out = f"stderr:\n{e}"
            except Exception as e:                      # any other failure -> stderr back to the model
                out = f"stderr:\n{type(e).__name__}: {e}"

            msgs.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": out})
            if term == "return":                        # RETURN(value) succeeded (constraint passed
                return val                              # or there was no constraint) — done.

        # --- budget exhausted: model never produced an accepted RETURN(value) ---
        if constraint is None:
            raise RuntimeError(
                "react_llm: budget exhausted; model never called RETURN(value). "
                "react_llm terminates only on RETURN(value); text-only assistant "
                "turns are not finalizations.")
        raise ConstraintViolation(
            "react_llm: budget exhausted; model never produced a RETURN(value) "
            "that satisfied the constraint. Each rejected attempt's violation "
            "was returned to the model via the tool result; see the conversation "
            "history for details.")
