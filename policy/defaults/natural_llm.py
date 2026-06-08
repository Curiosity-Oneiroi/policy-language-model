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
            value: int
        result = natural_llm("Compute 2+2", constraint=Sum)
        # -> Sum(value=4)   (a validated Constraint instance)
        # access the structured value: result.value -> 4

        # Composite constraints (& | ^ ~) work — schema is the merged
        # allOf/anyOf/oneOf/not form:
        class HasName(Constraint):
            name: str
        class HasAge(Constraint):
            age: int
        person = natural_llm("Make up a person", constraint=HasName & HasAge)
        # -> a Constraint instance satisfying both schemas

        # Retry budget (defaults to 5):
        result = natural_llm(msgs, constraint=Sum, return_budget=3)

        # Backend kwargs (temperature, top_p, etc.):
        result = natural_llm(msgs, generate_kwargs={"temperature": 0.0})

    Parameters:

        messages: a str (treated as one user turn) / a dict (a single
            message) / a list of message dicts. The conversation context.

        constraint=None: optional Constraint CLASS (not instance). When
            set, MUST be JSON-schema-expressible — a structural
            Pydantic-class Constraint or a composite of such via `& | ^ ~`.
            Factory/predicate Constraints (built via `Constraint.field(...)`
            carrying Python AfterValidator predicates) are REJECTED upfront
            with TypeError because the model has no way to read Python
            predicates from a schema — retrying would be pure waste.
            For predicate validation use `react_llm(..., constraint=C)`,
            which runs the model's python code and can re-validate.


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
          `class Sum(Constraint): value: int`, or `result.name` /
          `result.age` for a multi-field structural class.

    Raises:

        - TypeError: `constraint` is a factory/predicate Constraint
          (not JSON-schema-expressible).
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
    from plm.constraint.base import _contains_factory
    generate_kwargs = {} if generate_kwargs is None else generate_kwargs

    def _norm(m):
        if isinstance(m, str):
            return [{"role": "user", "content": m}]
        if isinstance(m, dict):
            return [m]
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

        # Shape guard: `constraint` must actually be a Constraint (class or
        # composite). Otherwise json_schema()/validate() below would raise a raw
        # AttributeError; fail fast with a clear message instead.
        if not callable(getattr(constraint, "json_schema", None)):
            raise TypeError(
                "natural_llm: `constraint` must be a Constraint class or composite "
                f"(it needs json_schema()/validate()); got {constraint!r}"
            )

        # Constraint shape check: must be schema-expressible. Factory constraints
        # carry predicate/coercer callables the model can't read, so retry is
        # pointless. `_contains_factory` walks composites (`&`/`|`/`^`/`~`) too, so
        # a factory HIDDEN inside a composite (e.g. StructA & Constraint.field(...))
        # is rejected as well — not just a directly-factory constraint. A purely
        # structural constraint, or a composite of structural ones, passes.
        if _contains_factory(constraint):
            raise TypeError(
                f"natural_llm: constraint {getattr(constraint, '__name__', repr(constraint))!r} "
                f"carries predicate/AfterValidator callables (built via "
                f"Constraint.field(...), possibly inside a `&`/`|`/`^`/`~` composite) "
                f"that the language model has no way to read from a JSON schema, "
                f"so single-shot generation would retry to exhaustion. natural_llm "
                f"requires a structural Constraint (Pydantic-class) or a composite "
                f"of purely structural ones, whose json_schema() is a complete spec "
                f"the model can follow. For predicate validation use "
                f"`react_llm(..., constraint=C)` — the ReAct loop lets the model see "
                f"the violation in the tool result and self-correct."
            )

        # HARD-SET the schema on response_format — the constraint defines the output,
        # so it ALWAYS wins (we do NOT let a caller-passed response_format override it).
        # NO try/except — if json_schema() fails on a non-factory constraint, that's a
        # real bug we want surfaced (the LLM has nothing to converge on without the
        # schema, so silently retrying would be busy-work).
        schema = constraint.json_schema()
        gk = dict(generate_kwargs)
        gk["response_format"] = {"type": "json_schema",
                                 "json_schema": {"name": "answer", "schema": schema}}

        from plm._react_helper import _coerce_budget, _safe_describe, _strip_md_fence
        last_err = None
        # `return_budget` is model-controlled; coerce None/non-int/negative -> a
        # sane non-negative int (no raw TypeError at the range bound; see #18).
        rb = _coerce_budget(return_budget, 5)
        for _ in range(1 + rb):
            check_depth_or_raise()
            content = _content(backend.generate(msgs, **gk))
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
                last_err = e
                msgs = msgs + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": _safe_describe(constraint)
                     + "\n\nYour previous answer failed validation: " + str(e)},
                ]
        # Budget exhausted: ALWAYS surface a clear budget-exhaustion
        # ConstraintViolation to the CALLER (PLM / a policy / a function),
        # chaining the last error so its detail is preserved. Constraint.validate()
        # now always raises ConstraintViolation, so there is no raw error to
        # special-case — we add the exhaustion context uniformly.
        raise ConstraintViolation(
            f"natural_llm: budget exhausted ({1 + rb} attempts); last validation "
            f"error {type(last_err).__name__}: {last_err}"
        ) from last_err
