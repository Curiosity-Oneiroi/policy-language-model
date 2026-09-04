@policy
def react(messages, namespace, *, rounds, generate_kwargs=None, model=None):
    """The raw ReAct capability: interleaved think+code, N rounds, no exit.

    Run EXACTLY `rounds` think+code rounds and stop, still open. Returns None,
    because everything it produced is already in the two containers YOU own:

        messages : list  — the trajectory (the weave)
        namespace    : dict  — the sub-agent's namespace

    Both are yours by reference, and both persist between calls. That is the
    whole session mechanism: there is no registry, no handle, no session
    object. To continue, pass the same two back. To fork, copy them. To evict
    a variable, `namespace.pop(name)`. To end it, `namespace.clear()`.

        ns   = {"df": df}                             # grant, by name
        msgs = [{"role": "system", "content": "Be careful."},
                {"role": "user",   "content": "Analyse `df`."}]
        react(msgs, ns, rounds=1)                     # one round
        ns["subtotal"]                                # read what it built
        msgs.append({"role": "user", "content": "[verifier] step 2 is wrong"})
        react(msgs, ns, rounds=2)                     # same words, same namespace
        ans = react_auto(msgs, ns, max_turns=0)       # "finish now" -> value

    This is `react_auto` with the TERMINATION PROTOCOL removed, and nothing
    else changed. `max_turns`, `return_budget` and `constraint` are all about
    RETURNING, so they are protocol and live only there.

    Design rulings (author):
      * TWO woven containers, both plain data the caller already owns. No
        state object is returned or required: uniformity with `llm` and
        `react_auto`, and the reason nothing here needs preserving,
        tombstoning, or evicting.
      * `rounds` is REQUIRED, exactly as `max_turns` is on `react_auto`: a raw
        capability arrives with no budget.
      * LIST + DICT only. `react_auto` still takes a bare string (and a namespace
        of None) because the bubble needs no handles.
      * RETURN does not EXIST here — not the obligation, not a coaching stub,
        not even the name. A model that calls it gets the raw consequence of
        any undefined name: a NameError in its tool output, which is counted
        (`react.return_attempt`). Termination is protocol.
      * INJECTS NOTHING. Not a budget, not a round count, not a mechanism
        briefing — no text of any kind enters the weave. `react_auto` inserts
        only its TERMINATION PROTOCOL (it must: its sub-agent races a
        deadline), and the capability has no protocol to insert. The python
        tool's own description already tells the model it has a tool; that a
        namespace persists is something the models do; and anything else the
        author wants said, the author says, in their own system message. A
        harness that narrates the mechanism is teaching, and teaching is
        authoring.
      * NO `system=` convenience: under weaving the author puts their system
        message in the list, like every other primitive here.
      * NO `expose_messages=` flag. Showing the sub-agent its own trajectory
        is a GRANT like any other, so it goes through the one channel:
        `namespace["messages"] = messages` hands over the live list (the agent
        can then rewrite its own past — deliberate, and yours to decide), while
        `namespace["trace"] = copy.deepcopy(messages)` hands over a frozen
        snapshot. The old flag was neither: it rebuilt a deep copy before every
        code run, so "live" meant REFRESHED PER ROUND. Stepping gives you that
        for free, and under your control:

            for _ in range(k):
                namespace["trace"] = copy.deepcopy(messages)
                react(messages, namespace, rounds=1)

        Same behaviour, three lines, no knob — and you choose the cadence, and
        what to show (last 3 rounds, reasoning stripped, ...). `react_auto`
        cannot do this, because it runs to completion with no round boundary
        to refresh at; that is the difference between a protocol and a
        capability, not a missing feature.

    Boundary (A1c): stepping hands the caller a live trajectory it can edit
    between rounds AND a namespace it can read, write, and prune — beyond the
    string contract on both axes — so this policy exists in FULL only.
    """
    import json, keyword, sys
    from plm._react_helper import _coerce_budget, _strip_code_fences
    from plm.plm_tools import _tool_python
    from plm.repl.exec_ns import exec_ns, safe_builtins
    from plm.policy.defaults._llm_infra import (_make_backend, llm_call,
                                                check_depth_or_raise, descend)
    from plm.policy.defaults._subcall_log import record as _log

    # The same safety cap react_auto applies: the ONE artificial bound in the
    # experiment, disclosed in the task card and enforced visibly.
    _MAX_ROUNDS_CAP = 20

    if not isinstance(messages, list):
        raise TypeError("react: messages must be a LIST of message dicts — it "
                        "is one of your two handles on the session (weaving). "
                        "For the hand-it-a-string bubble, use react_auto.")
    if isinstance(namespace, bool):
        raise TypeError("react: namespace must be the state DICT itself — "
                        "`namespace: bool` (all-or-none) is the AGENTIC "
                        "harness's convention; FULL weaves the dict (A1c).")
    if not isinstance(namespace, dict):
        raise TypeError(f"react: namespace must be a dict — it IS the sub-agent's "
                        f"namespace, and your handle on it; got "
                        f"{type(namespace).__name__}. Start with {{}}.")
    for _k in namespace:
        if not isinstance(_k, str):
            raise TypeError(f"react: namespace keys must be strings, got "
                            f"{type(_k).__name__} ({_k!r}).")
        if _k == "RETURN":
            raise ValueError("react: 'RETURN' is reserved — and in the raw "
                             "capability it does not exist at all.")
        if _k != "__builtins__" and not (_k.isidentifier()
                                         and not keyword.iskeyword(_k)):
            raise ValueError(f"react: namespace key {_k!r} is not a valid Python "
                             f"identifier; namespace keys ARE the model's variable "
                             f"names, so it could never be reached.")
    generate_kwargs = {} if generate_kwargs is None else generate_kwargs
    if {"messages", "tools"} & set(generate_kwargs):
        raise ValueError("react: generate_kwargs may not contain messages/"
                         "tools — react passes those itself")

    def _turn(backend, tools):
        """One generate: append the assistant turn, stash any python tool call.
        Returns the (id, code, note) to run, or None."""
        resp = backend.generate(messages, tools=tools, **generate_kwargs)
        if not isinstance(resp, dict):                  # malformed -> empty turn
            resp = {}
        tool_calls = resp.get("tool_calls") or []
        if isinstance(tool_calls, dict):                # a single call, bare
            tool_calls = [tool_calls]
        elif not isinstance(tool_calls, list):
            tool_calls = []
        messages.append({"role": "assistant",
                         "content": resp.get("content") or "",
                         "reasoning": resp.get("reasoning") or "",
                         # only the executed+answered call, and only if a dict:
                         # a non-dict stored here would crash the next round's
                         # backend history sanitizer.
                         "tool_calls": ([tool_calls[0]] if tool_calls and
                                        isinstance(tool_calls[0], dict) else None)})
        if not tool_calls:                              # text-only turn: it stands
            return None
        # Parallel calls: only the FIRST runs. The nudge goes into the SUB-agent's
        # own stream (its business, not the meta-reasoner's) and onto every branch.
        note = (f"[react] you emitted {len(tool_calls)} tool_calls; only the "
                f"FIRST ran — send ONE `python` call per turn.\n\n"
                ) if len(tool_calls) > 1 else ""
        tc = tool_calls[0]
        if not isinstance(tc, dict):
            # No tool_call was recorded above, so a `tool` reply would be a wire
            # ORPHAN and the next request would 400. Report it as a user turn.
            messages.append({"role": "user", "content":
                             note + "[error] malformed tool_call; emit a "
                                    "single `python` tool call"})
            return None
        fn = tc.get("function")
        fn = fn if isinstance(fn, dict) else {}         # `function: null` -> {}
        tcid = tc.get("id") or ""
        if fn.get("name") != "python":
            messages.append({"role": "tool", "tool_call_id": tcid,
                             "content": note + f"error: tool {fn.get('name')!r} "
                                               f"not supported (only `python`)"})
            return None
        raw = fn.get("arguments") or "{}"
        try:
            targs = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(targs, dict):
                targs = {}
        except Exception:           # huge ints / deep nesting raise non-JSON errors
            targs = {}
        code = targs.get("code", "")
        if not isinstance(code, str):
            messages.append({"role": "tool", "tool_call_id": tcid,
                             "content": note + "error: `code` must be a string "
                                               "of Python source"})
            return None
        return (tcid, code, note)

    def _run(pending, round_no):
        """Execute the stashed cell. The caller wraps this in `descend()`."""
        tcid, code, note = pending
        out = "(no output)"
        try:
            code = _strip_code_fences(code)
            out, _term, _val = exec_ns(code, namespace,
                                       slot=f"react-{id(messages)}-{round_no}")
        except SyntaxError as e:
            out = f"stderr:\n{e}"
        except Exception as e:      # orchestration safety net only: a cell's own
            out = f"stderr:\n{type(e).__name__}: {e}"     # errors are captured
        messages.append({"role": "tool", "tool_call_id": tcid,   # inside exec_ns
                         "content": note + out})
        # Disposition data: a model reaching for RETURN inside the raw capability
        # is worth counting. With no RETURN bound the raw consequence is a
        # NameError whose text is stable. This logger never raises.
        if "name 'RETURN' is not defined" in out:
            _log(primitive="react.return_attempt", granted=0, used=0,
                 truncated=False, clamped=False, outcome="name_error")

    # Seed the seal on first touch. This is the ONLY thing a first call does
    # differently, and it is invisible plumbing: no text enters the weave.
    # `opened` is LOGGED (reviewer round 12, A3) rather than reconstructed later
    # from the dict's contents — attribution consults logs, never values.
    opened = "__builtins__" not in namespace
    if opened:
        # SEALED: a curated builtins (no __import__/eval/exec/open), and NO
        # RETURN — in the capability the name simply does not exist.
        namespace["__builtins__"] = safe_builtins()

    rounds = _coerce_budget(rounds, 1)
    if rounds > _MAX_ROUNDS_CAP:
        # VISIBLE clamp, never silent: grading a trajectory the author did not
        # write would be invalid.
        print(f"[react] rounds={rounds} exceeds the safety cap "
              f"({_MAX_ROUNDS_CAP}); clamped.", file=sys.stderr)
        rounds = _MAX_ROUNDS_CAP
    tools = [_tool_python()]
    with llm_call(None):
        check_depth_or_raise()      # depth gate first: a clear LLMDepthExceeded
        backend = _make_backend(model=model)
        for _round in range(rounds):
            check_depth_or_raise()
            pending = _turn(backend, tools)
            if pending is not None:
                with descend():     # decrement for the act phase ONLY
                    _run(pending, _round)
    # A step call always ends still-open: with no RETURN, nothing could conclude
    # it. (`ok` / `truncated` live on react_auto.)
    _log(primitive="react", granted=rounds, used=rounds, opened=opened,
         model=model, truncated=False, clamped=False, outcome="still-open")
    return None
