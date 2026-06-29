@policy
def natural_llm(messages, *, constraint=None, depth=None, return_budget=5, generate_kwargs=None):
    """Single-shot LLM call — one generate, optionally constraint-validated.

    No tools, no act phase. Just `messages -> backend.generate -> answer`.
    When a `constraint` is set, the constraint's JSON schema is sent to the
    backend as `response_format` so the model knows what to produce, and
    the answer is validated against the constraint with retry-on-failure.

    Quick examples:

        # Free-form text:
        answer = natural_llm("What is 2 + 2?")
        # -> "4"  (whatever the model said, as a str)

        # Constraint-validated structured output (recommended pattern):
        class Sum(Constraint):
            value: Constraint.field(type=int)
        result = natural_llm("Compute 2+2", constraint=Sum)
        # -> Sum(value=4)   (a validated Constraint instance)
        # access the structured value: result.value -> 4

        # Several fields/rules at once -> author ONE constraint with all of them:
        class Person(Constraint):
            name: Constraint.field(type=str)
            age:  Constraint.field(type=int)
        person = natural_llm("Make up a person", constraint=Person)
        # -> a Person instance satisfying the schema

        # Retry budget (defaults to 5):
        result = natural_llm(msgs, constraint=Sum, return_budget=3)

        # Backend kwargs (temperature, top_p, etc.):
        result = natural_llm(msgs, generate_kwargs={"temperature": 0.0})

    Parameters:

        messages: a str (treated as one user turn) / a dict (a single
            message) / a list of message dicts. The conversation context.

        constraint=None: optional Constraint CLASS (not instance). When set,
            the model is steered by the constraint's JSON schema (the SHAPE)
            plus its describe() text (the rules, in words), then the answer is
            validated with retry. Predicates / coercers / @model_validators are
            fine — the model is told about them via describe() and can comply.
            The ONE requirement: every user predicate/coercer/@model_validator
            must carry a `@constraint("...")` description (built-ins
            auto-describe). A constraint with an UNDESCRIBED check is REJECTED
            upfront with TypeError — natural_llm can't steer toward a rule it
            can't state in words. For checks better validated by running the
            model's code, use `react_llm(..., constraint=C)`.


        return_budget=5: how many retries on ConstraintViolation. Total
            generates = 1 + return_budget when constraint is set.

        generate_kwargs=None (resolved to {}): explicit dict of
            backend-specific kwargs (temperature, top_p, response_format,
            ...). With a constraint set, natural_llm hard-sets
            `response_format = {"type": "json_schema", "json_schema":
            {"name": "answer", "schema": constraint.json_schema()}}` so
            the model knows what JSON to produce.

    Returns:

        - With no constraint: the model's `content` as a str.
        - With a constraint: the validated Constraint instance. The
          structured value is on the instance — e.g. `result.value` for
          `class Sum(Constraint): value: Constraint.field(type=int)`, or `result.name` /
          `result.age` for a multi-field structural class.

    Raises:

        - TypeError: `constraint` carries an UNDESCRIBED user predicate /
          coercer / @model_validator (natural_llm can't state it to the model).
        - ConstraintViolation: `return_budget` exhausted; the most recent
          validation error propagates.
        - LLMDepthExceeded: the LLM-recursion-depth budget is at 0
          (a sub-LLM trying to call further when ancestors used it up).

    Internals:

        Single-shot: no act phase, no `descend()`. `check_depth_or_raise()`
        runs before each generate (the depth-gate hard ceiling). On
        ConstraintViolation, the failed answer is kept in history and a
        user-turn is appended carrying `constraint.describe()` + the
        violation error — giving the model both the schema and the
        specific failure. Up to `return_budget` retries; budget exhaustion
        re-raises the last violation.
    """
    import json
    from plm.policy.defaults._llm_infra import _make_backend, llm_call, check_depth_or_raise
    from plm.constraint.base import _all_checks_described, _has_generatable_structure
    generate_kwargs = {} if generate_kwargs is None else generate_kwargs

    # reserved-key guard (mirrors react_llm). natural_llm passes `messages` itself, and — when a
    # constraint is set — hard-sets `response_format` from the constraint's schema. A caller-supplied
    # one would be silently dropped, so reject it loudly instead.
    _reserved_gk = ({"messages", "tools"} | ({"response_format"} if constraint is not None else set())) \
        & set(generate_kwargs)
    if _reserved_gk:
        raise ValueError(
            f"natural_llm: generate_kwargs may not contain {sorted(_reserved_gk)} — natural_llm "
            f"supplies messages itself and (with a constraint) hard-sets response_format from the "
            f"constraint's schema. Drop those keys."
        )

    def _norm(m):
        if isinstance(m, str):
            return [{"role": "user", "content": m}]
        if isinstance(m, dict):
            return [m]
        if isinstance(m, list):                            # pass-by-reference (message weaving):
            return m                                       # work on the caller's OWN list
        try:
            return list(m)
        except TypeError:                                  # None / int / ... -> clear error
            raise TypeError(
                "natural_llm: messages must be a str, a message dict, or an iterable of "
                f"message dicts; got {type(m).__name__}")

    def _content(resp):                                    # non-dict backend response -> ""
        return ((resp.get("content") if isinstance(resp, dict) else None) or "")

    msgs = _norm(messages)
    with llm_call(depth):
        check_depth_or_raise()                             # depth gate FIRST: a clear LLMDepthExceeded, not a backend-spec error
        backend = _make_backend()
        if constraint is None:
            check_depth_or_raise()                         # policy owns the depth gate
            return _content(backend.generate(msgs, **generate_kwargs))

        # Shape guard: `constraint` must actually be a Constraint (a struct class or a
        # Constraint.field(...)). Otherwise json_schema()/validate() below would raise a raw
        # AttributeError; fail fast with a clear message instead.
        if not callable(getattr(constraint, "json_schema", None)):
            raise TypeError(
                "natural_llm: `constraint` must be a Constraint class or Constraint.field(...) "
                f"(it needs json_schema()/validate()); got {constraint!r}"
            )

        # COMMUNICABILITY GATE: natural_llm steers a single-shot model with the schema (the SHAPE)
        # + describe() (the rules, IN WORDS). A check it cannot state in words is invisible to the
        # model — it can only fail it. So reject up front iff ANY user predicate / coercer /
        # @model_validator anywhere in it (struct fields + type= generic elements) has
        # NO description. Built-in refinements (int_range/str_pattern/...) auto-describe, so they
        # pass. A DESCRIBED predicate/coercer/validator is
        # fine — the model is told about it and can comply (and validate+retry gates it). This is a
        # natural_llm-only rule; the general constraint mechanism does NOT require descriptions.
        if not _all_checks_described(constraint):
            raise TypeError(
                f"natural_llm: constraint {getattr(constraint, '__name__', repr(constraint))!r} "
                f"has a predicate/coercer/@model_validator with NO description — natural_llm can "
                f"only steer the model toward checks it can state in words, so an undescribed check "
                f"is one the model can only fail. Tag each user predicate/coercer/validator with "
                f"@constraint('...'), or use `react_llm(..., constraint=C)` (it runs the model's "
                f"code and lets it self-correct on the violation)."
            )

        from plm._react_helper import _coerce_budget, _safe_describe, _strip_md_fence
        # json_schema() is AIRTIGHT (never raises post-reframe). Hard-set it as response_format ONLY
        # when it pins a concrete shape (type/enum/properties/anyOf) — then the constraint defines
        # the output and ALWAYS wins over a caller-passed response_format. For a STRUCTURELESS
        # constraint (e.g. a bare described predicate on `type=object`) there is no shape to force,
        # so fall back to describe-only steering (no response_format) — the model generates freely
        # and validate+retry gates it.
        schema = constraint.json_schema()
        gk = dict(generate_kwargs)
        if _has_generatable_structure(schema):
            gk["response_format"] = {"type": "json_schema",
                                     "json_schema": {"name": "answer", "schema": schema}}
        # Feed the FULL contract upfront (describe() — every check in words) as a leading system
        # note, so the model is steered toward bounds/predicates/cross-field rules from the FIRST
        # attempt, not only after a failed retry. Built fresh per call; NOT woven into the caller's
        # message list.
        _contract = _safe_describe(constraint)
        _steer = ([{"role": "system",
                    "content": "Produce output that satisfies ALL of this contract:\n" + _contract}]
                  if _contract else [])
        last_err = None
        # `return_budget` is model-controlled; coerce None/non-int/negative -> a
        # sane non-negative int (no raw TypeError at the range bound; see #18).
        rb = _coerce_budget(return_budget, 5)
        for _ in range(1 + rb):
            check_depth_or_raise()
            content = _content(backend.generate(_steer + msgs, **gk))
            try:
                # Strip a structural ```json ... ``` fence first. response_format is
                # hard-set to json_schema (so OpenAI/vLLM structured outputs return
                # clean JSON and this is a no-op), but a caller-overridden
                # response_format or a non-strict endpoint can still fence the
                # reply — without this, a fenced-but-correct answer would fail
                # json.loads, validate as a raw string, and burn the whole retry
                # budget. _strip_md_fence is structural-only, so it never
                # touches unfenced JSON.
                value = json.loads(_strip_md_fence(content))
            except Exception:
                value = content
            try:
                return constraint.validate(value)
            except Exception as e:                         # ConstraintViolation OR any other
                # validator error (a struct's @model_validator raising a raw
                # KeyError/TypeError that pydantic does NOT wrap). Treat ANY
                # validation failure as a retry — surface the last one if the
                # budget is exhausted — instead of letting it abort the call.
                # STRUCTURELESS rescue: for a describe-only constraint (no response_format) the
                # model emits a BARE value, and json.loads above can coerce a CORRECT answer away
                # from a string ("1221" -> int 1221, "true" -> True, "null" -> None) so it fails
                # here though the raw text validates. Before counting a failure, retry the raw
                # string whenever json.loads changed it — else a model that answers correctly
                # every round burns the entire budget on a parsing artifact and PLM mis-reads it
                # as "the sub-LLM couldn't satisfy the constraint".
                if value != content:
                    try:
                        return constraint.validate(content)
                    except Exception:
                        pass
                last_err = e
                # M6: append IN PLACE (extend, not `msgs = msgs + [...]`) so the failed answer +
                # violation weave back into the caller's trajectory by reference, matching react_llm.
                # The contract (describe()) is already fed UPFRONT via `_steer`, so the retry turn
                # carries ONLY the violation — no redundant re-describe.
                msgs.extend([
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": "Your previous answer failed validation: " + str(e)},
                ])
        # Budget exhausted: ALWAYS surface a clear budget-exhaustion
        # ConstraintViolation to the CALLER (PLM / a policy / a function),
        # chaining the last error so its detail is preserved. Constraint.validate()
        # now always raises ConstraintViolation, so there is no raw error to
        # special-case — we add the exhaustion context uniformly.
        raise ConstraintViolation(
            f"natural_llm: budget exhausted ({1 + rb} attempts); last validation "
            f"error {type(last_err).__name__}: {last_err}"
        ) from last_err
