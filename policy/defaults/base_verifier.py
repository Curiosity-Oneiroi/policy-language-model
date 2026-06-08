@policy
def base_verifier(messages):
    """BASE / reference verifier — the trajectory CONTROL LAYER.

    A verifier is, fundamentally, just a function that takes the running
    `messages` list and MUTATES IT IN PLACE. It can be plain Python, an LLM, or
    a mix — that choice is yours. `react_llm_verifier` calls a verifier after
    every non-terminal round (each call wrapped in its own `descend()`), and the
    (possibly edited) trajectory drives the next model turn. This is the THIRD
    control axis over a sub-agent — input (messages), output (Constraint), and
    TRAJECTORY (this). PLM engineers a verifier to CONTROL the sub-agent by FREELY
    manipulating the `messages` list — it is a plain Python list, so ANY operation is
    fair game: APPEND a turn, INSERT one between existing turns, REWRITE/edit a
    message, DELETE a turn, reorder, or replace the whole trajectory. With that you
    inject doubt, force an INDEPENDENT re-derivation, prune a bad branch, splice in a
    correction, or steer the next round — adding a turn is just the simplest case.

    THIS is the base/reference, and it is MUTABLE + DUPLICABLE (unlike
    natural_llm / react_llm / react_llm_verifier, which are sealed). Recommended:
        duplicate_policy("base_verifier", "my_verifier") or my_verifier = base_verifier.duplicate()
        my_verifier._edit(...)            # specialize prompts / gate / checks
        react_llm_verifier(task, verifier=my_verifier)
    Editing the base in place is allowed but discouraged (it is the shared base).

    Contract:
        base_verifier(messages) -> None   # mutate `messages` in place; return ignored

    An optional guide: The `verifier` ROLE — a PRIVATE trajectory channel. Messages you add as
    `{"role": "verifier", "content": ...}` are INVISIBLE to the sub-LLM's model
    (the backend strips them before every generate) but you can read them back from
    `messages`. Use them for your OWN records / markers (e.g. "already verified this
    error") without polluting the model's conversation with fake `user` turns — it's
    what the gate below keys off. To STEER the model, manipulate the model-VISIBLE
    turns — `messages` is a plain list you fully control: append OR INSERT a
    `user`/`system` turn anywhere, REWRITE or DELETE a message, reorder, or replace
    the list. Any list operation is fair game (adding a turn is just the simplest).

    `_verify` below shows the simplest control (a NOTE steer); `_verify_propose_approve`
    shows the rich end — a self-checked edit where one circuit PROPOSES code and a
    second APPROVES it before the verifier runs it. Between them they sketch the range
    of what a verifier can do to a trajectory.

    A verifier — and each inside it — IS A POLICY THAT REASONS AND MANIPULATES THE TRAJECTORY.
    PLM is a meta-reasoner whose whole job is to AUTHOR POLICIES; a verification policy is simply a
    different kind, so author it as one: Policy that composes depth-1 LLM circuits
    (or no LLM at all). Its depth is already pinned to 1 here, so nothing nests a
    further sub-LLM beyond a single circuit — but it is otherwise a full policy.
    There is NO one general verifier for everything; you TAILOR it to the task and
    almost always rewrite the example below (prefer several specific `_verify_*`).

    Shape (a template — adapt freely):
      * `_should_run(messages)` — a GATE: should we verify THIS round? A good
        verifier does NOT verify every round (it is *called* every round, so that
        is wasteful). Gating is ONE good pattern, not mandatory — you may delete
        it and verify every round. Keep any gate LIGHT.
      * one or more `_verify_*(messages)` — the actual verification(s). EACH IS
        ITSELF A VERIFICATION POLICY: ordinary Python that may compose any
        combination of depth-1 `react_llm` / `natural_llm` / 'react_llm_verifier'
        circuits AND/OR plain code — or no LLM at all — exactly like any policy.
        The DEPTH-1 limit is on the NESTING of each LLM call, NOT on how many you make or how much
        code you run: one check can fan out several depth-1 circuits, run Python
        between them, and combine their results. Each has its own docstring and a
        system + user prompt you CUSTOMIZE to the behavior checked. Define several
        `_verify_*` and run more than one.

    What to verify is OPEN — the prompts decide: arithmetic, a numeric result, a
    hypothesis/claim, an assumption, a tool result, output format... anything.

    Depth (HARD LIMIT, for now): each LLM CALL must stay DEPTH-1 — use only
    `natural_llm` or `react_llm` or `react_llm_verifier`, and pass `depth=1`. This caps the NESTING of a
    call, not the verification: you may still make as many depth-1 calls and run
    as much Python as you like — the verification is a full policy. Via
    `react_llm_verifier` the `descend()` wrap already caps you at depth-1 (root=2);
    `depth=1` keeps it true when `base_verifier` is called standalone. You cannot
    reach the raw model primitives here in any case: the blessed-caller gate in
    `_llm_infra` (`_make_backend` / `descend` / `llm_call` each check the caller's
    code object against `_BLESSED_CALLERS`) refuses any caller that isn't a
    sanctioned LLM default. A mutable policy — this base, or any duplicate — is
    NOT blessed, so importing the helper succeeds but the CALL raises RuntimeError.
    Reach the model only through `natural_llm` / `react_llm` (which are blessed and
    reach `_make_backend` on your behalf).
    """
    import copy   # used to hand a verification circuit a non-aliased trajectory

    # ===== GATE: should we verify THIS round? (one good pattern; optional) =====
    def _should_run(messages):
        """Return True to run verification this round, False to no-op.

        react_llm_verifier calls the verifier EVERY non-terminal round, so gate
        it — verifying every round is wasteful.

        Like `_verify_*`, the gate is ITSELF A POLICY: it can be plain code, an
        LLM, a react_llm, or any combination — there is NO restriction on the
        form (don't read the examples below as the only options). The ONLY
        requirement is that it be VERY LIGHT, because it runs every round: keep
        any LLM use to a very restricted number of calls / rounds (e.g. depth=1
        with 1-2 rounds, or a single yes/no `natural_llm`). Examples, not limits:
          * code-based: scan the trajectory for a trigger — an error in the latest
            tool output, a suspicious value, a phase boundary, a task marker.
          * a light LLM / react_llm: a single `natural_llm(..., depth=1)` yes/no,
            or a short `react_llm(..., depth=1, max_turns=1)` inspecting the
            trajectory. Pass the messages via `kwargs={"trajectory": messages}`
            (the circuit reads the local `trajectory`) or fold a summary into the
            prompt; combine code + LLM however you like.
        Always returning True (no gate) is also valid.

        This default also avoids RE-FIRING on a STALE error: it scans newest-first
        and stops at whichever it meets first — a fresh error tool output (-> run)
        or a prior `verifier`-role note (-> this error was already handled, skip). So
        one error triggers exactly one verification, not one per following round.
        """
        for m in reversed(messages):
            role, c = m.get("role"), (m.get("content") or "")
            if role == "verifier":
                return False                       # already verified since the last signal
            if role == "tool":
                # Anchor on a REAL error section, not the bare word "Error": react_llm
                # prefixes a failed tool's output with "stderr:" and a captured traceback
                # contains "Traceback". Dropping "Error" avoids firing on benign prose like
                # "Error rate: 0.05" or "no errors found".
                return ("stderr:" in c) or ("Traceback" in c)
        return False

    # ===== VERIFICATION(S): each is its own function; you can run several =====
    # Each `_verify_*` is itself a VERIFICATION POLICY: it may combine several
    # depth-1 react_llm / natural_llm / react_llm_verifier calls/circuits with arbitrary Python — or use no LLM
    # at all — just like any policy. depth-1 caps each LLM call's nesting, not the
    # orchestration. The system + user prompts below are GENERIC placeholders and
    # are the KNOB you turn: a verifier "system instruction" can be many things —
    # rewrite both prompts for the behavior you want checked (arithmetic, a
    # number, a hypothesis/claim, an assumption, output format, ...). Define more
    # `_verify_*` functions for more checks and add them to the `_checks` tuple.
    def _verify(messages):
        """ONE verification over the trajectory; mutate `messages`. GENERIC EXAMPLE — rewrite it.

        This is a VERIFICATION POLICY, authored the way PLM (a meta-reasoner)
        authors any policy — Python composing a depth-1 LLM circuit here. Its
        depth is already pinned to 1: you cannot spawn a further sub-LLM beyond
        this circuit. Otherwise it is a full policy.

        There is NO single general verifier for everything — you almost always
        REWRITE this. Prefer SEVERAL functions each tailored to a specific check
        (`_verify_arithmetic`, `_verify_hypothesis`, `_verify_format`,
        `_verify_invariant`, ...) and run the ones the task needs; a broad
        smoke-check like the generic example below is fine too. Keep the prompts
        unbiased toward any one task — customize them for the behavior YOU check.

        The example: a depth-1 react_llm with a python tool that reads a COPY of
        the trajectory (`trajectory` kwarg), checks it looks sound, and RETURNs a
        short note ('' if fine). The note is then (a) surfaced as a model-visible
        turn to STEER the next round, and (b) recorded on the private `verifier`
        channel so the gate fires once per error (see the role note in the top
        docstring).
        """
        system = (
            "You are a verifier. Look over the agent's trajectory and check it is "
            "going soundly — watch for mistakes, unjustified leaps, or anything "
            "that does not add up; verify independently where you can (you have a "
            "python tool). <customize: spell out exactly what THIS verifier checks. If "
            "your check hands the circuit a python tool, also describe what that tool can "
            "do — the REPL helpers in scope and how to use them — since react_llm's "
            "sub-agent only knows what you tell it.>"
        )
        instruction = (
            "The trajectory is in `trajectory`. Inspect it, flag anything wrong or "
            "worth double-checking, and RETURN a short note (empty string '' if it "
            "looks fine). <customize for your specific check>."
        )
        note = react_llm(
            [{"role": "system", "content": system},
             {"role": "user", "content": instruction}],
            # Hand the circuit a COPY: react_llm binds kwargs by reference into the
            # sub-LLM's scope, so passing `messages` directly would let the
            # verification sub-LLM mutate the AGENT's real trajectory.
            kwargs={"trajectory": copy.deepcopy(messages)},
            max_turns=3, depth=1,                # DEPTH-1 circuit; bounded rounds
        )
        if note:
            # STEER the sub-LLM: a turn it SEES (its view) — inject doubt / force a
            # recheck next round. The model has no `verifier` role, so a nudge it can
            # READ must ride a visible role (user/system).
            messages.append({"role": "user", "content": "[verifier] " + str(note)})
            # RECORD on the verifier's OWN channel: role `verifier` is INVISIBLE to the
            # model (the backend strips it) but the verifier reads it back, so the gate
            # fires once per error — not every round. This is the marker `_should_run`
            # keys off (no more `[verifier]`-string-matching a fake user turn).
            messages.append({"role": "verifier", "content": str(note)})

    def _verify_propose_approve(messages):
        """RICHER capability + a SAFETY GATE — a PROPOSE -> APPROVE trajectory edit.

        Beyond a plain note: one circuit PROPOSES an edit as PYTHON code (its RETURN
        shaped by a Constraint to a compilable string that touches `trajectory`); a
        SECOND, INDEPENDENT circuit REVIEWS that exact code and RETURNs True/False (a
        bool Constraint); ONLY on True does the verifier `exec` the code against the
        LIVE trajectory. Two circuits — one writes the change, one approves it — and
        the VERIFIER (trusted scope), not the sub-LLMs, applies it. The edit can
        append / insert / rewrite / delete / reorder — any list op. This is the
        pattern for self-checked trajectory surgery; adapt the prompts + constraints.
        """

        def _compiles(s):                            # the proposed edit must be valid python
            compile(s, "<verifier-edit>", "exec")
        EditCode = Constraint.field(
            is_instance_of=str, predicate=_compiles,
            description="python that edits the `trajectory` list in place "
                        "(append/insert/rewrite/delete/reorder); '' for no edit",
        )
        Approval = Constraint.field(is_instance_of=bool)

        # react_llm #1 — PROPOSE the edit as code (reads a COPY; does NOT apply it).
        # A proposer that can't produce a valid edit (or never RETURNs) just means
        # "no change this round" — catch the constraint failure rather than crashing
        # the agent over an auxiliary control circuit.
        try:
            code = react_llm(
                [{"role": "system", "content":
                  "You PROPOSE a repair to the agent's trajectory. `trajectory` is the list of "
                  "message dicts. RETURN a short PYTHON snippet (a string) that edits "
                  "`trajectory` IN PLACE — append/insert a turn, rewrite/delete a message, "
                  "reorder — or RETURN '' if no edit is warranted. Do NOT apply anything "
                  "yourself; an independent reviewer decides whether to run your code."},
                 {"role": "user", "content": "Propose the edit as a python string, or ''."}],
                kwargs={"trajectory": copy.deepcopy(messages)},
                constraint=EditCode, max_turns=3, depth=1,
            )
        except ConstraintViolation:
            return                                   # no valid proposal -> no change
        if not code:
            return
        # react_llm #2 — APPROVE or reject that exact code (an independent check).
        try:
            approved = react_llm(
                [{"role": "system", "content":
                  "You are an INDEPENDENT reviewer. The python in `code` will be run against the "
                  "live `trajectory` if you approve. Judge whether it is a SAFE, correct repair. "
                  "RETURN True to apply it, False to reject."},
                 {"role": "user", "content": "Proposed edit:\n" + str(code)}],
                kwargs={"trajectory": copy.deepcopy(messages), "code": str(code)},
                constraint=Approval, max_turns=2, depth=1,
            )
        except ConstraintViolation:
            return                                   # reviewer gave no valid verdict -> don't apply
        if approved is True:
            # APPROVED: the VERIFIER (trusted scope) applies the edit to the LIVE
            # trajectory via `exec_ns` — a REPL global already in scope (injected by the
            # kernel like `react_llm` / `parallel`; no import needed) — so the
            # proposing/approving sub-LLMs never touched the real list. The restricted
            # builtins (list surgery only — no import/open/eval) raise the bar against the
            # obvious/accidental reach, but they are NOT a true sandbox (the deliberate
            # `().__class__.__subclasses__()` walk remains — the same residual the
            # react_llm ns has; see exec_ns). The REAL gate is the two independent LLM
            # reviewers above (propose -> approve); this is PLM's OWN code, self-approved.
            _safe = {"len": len, "range": range, "list": list, "dict": dict,
                     "str": str, "enumerate": enumerate, "sorted": sorted}
            out, _term, _val = exec_ns(str(code), {"__builtins__": _safe, "trajectory": messages},
                                       slot=f"verifier-edit-{len(messages)}")
            # exec_ns CAPTURES (never raises) a failing edit into `out`; check it so we do
            # NOT record "applied" for an edit that actually errored.
            if "Traceback (most recent call last)" in out:
                messages.append({"role": "verifier",
                                 "content": "approved trajectory edit FAILED:\n" + out})
            else:
                messages.append({"role": "verifier", "content": "applied approved trajectory edit"})

    # ===== main flow: gate, then run each verification in turn =====
    if not _should_run(messages):
        return                                   # no-op this round; trajectory unchanged
    # Two demonstrations of the range (add your own `_verify_*` and list them here):
    #   _verify                 — the simplest: a NOTE steer + verifier-channel marker.
    #   _verify_propose_approve — a self-checked trajectory EDIT (propose -> approve -> apply).
    _checks = (_verify, _verify_propose_approve)
    for _check in _checks:
        try:
            _check(messages)
        except Exception:
            # A verifier is an AUXILIARY control circuit — a check that fails (budget
            # exhausted, a backend error, a bug in a custom check) must NOT abort the
            # OUTER agent. Swallow and move on; the trajectory is just left un-edited.
            pass
    # Record a `verifier`-channel marker UNCONDITIONALLY (the gate keys off it), so a
    # round where no check produced an edit/note still can't re-fire on the SAME stale
    # error next round. Skip if a check already left one as the latest message.
    if messages and messages[-1].get("role") != "verifier":
        messages.append({"role": "verifier", "content": "verified (no change)"})
