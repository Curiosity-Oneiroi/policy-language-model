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
        result = natural_llm("Compute 2+2 as JSON {value: int}", constraint=Sum)
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

        # Voluntarily lower the LLM-recursion-depth budget for this sub-task:
        result = natural_llm(msgs, depth=1)

        # Backend kwargs (temperature, top_p, etc.):
        result = natural_llm(msgs, generate_kwargs={"temperature": 0.0})

    Parameters:

        messages: a str (treated as one user turn) / a dict (a single
            message) / a list of message dicts. The conversation context.

        constraint=None: optional Constraint CLASS (not instance). When
            set, MUST be JSON-schema-expressible — a structural
            Pydantic-class Constraint or a composite of such via `& | ^ ~`.
            Factory/predicate Constraints (built via `Constraint.of(...)`
            carrying Python AfterValidator predicates) are REJECTED upfront
            with TypeError because the model has no way to read Python
            predicates from a schema — retrying would be pure waste.
            For predicate validation use `react_llm(..., constraint=C)`,
            which runs the model's python code and can re-validate.

        depth=None: voluntary LLM-recursion-depth cap. None = inherit
            current scope. An integer clamps the budget to
            min(depth, current) for this call. CAN ONLY LOWER, never raise
            above the sealed ceiling (depth=999 at root=2 is silently
            clamped to 2).

        return_budget=5: how many retries on ConstraintViolation. Total
            generates = 1 + return_budget when constraint is set.

        generate_kwargs=None (resolved to {}): explicit dict of
            backend-specific kwargs (temperature, top_p, response_format,
            ...). With a constraint set, natural_llm hard-sets
            `response_format = {"type": "json_schema", "json_schema":
            {"name": "answer", "schema": constraint.json_schema()}}` so
            the model knows what JSON to produce. Pass an explicit
            response_format in `generate_kwargs` to override (setdefault
            preserves caller intent).

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
    generate_kwargs = {} if generate_kwargs is None else generate_kwargs

    def _norm(m):
        if isinstance(m, str):
            return [{"role": "user", "content": m}]
        if isinstance(m, dict):
            return [m]
        return list(m)

    msgs = _norm(messages)
    with llm_call(depth):
        backend = _make_backend()
        if constraint is None:
            check_depth_or_raise()                         # policy owns the depth gate
            return (backend.generate(msgs, **generate_kwargs).get("content") or "")

        # Constraint shape check: must be schema-expressible. Factory constraints
        # carry predicate callables that the model can't read, so retry is
        # pointless. Composites (allOf/anyOf/oneOf/not) are accepted — they're
        # already merged into a single JSON schema by `json_schema()`.
        if getattr(constraint, "_constraint_is_factory", False):
            raise TypeError(
                f"natural_llm: constraint {getattr(constraint, '__name__', repr(constraint))!r} "
                f"is not just a JSON schema, it's a little complex — it carries "
                f"predicate/AfterValidator callables (built via Constraint.of(...)) "
                f"that the language model has no way to read from a schema. "
                f"natural_llm requires a structural Constraint (Pydantic-class) "
                f"or a composite of such (`&` / `|` / `^` / `~`), whose "
                f"json_schema() is a complete spec the model can follow. "
                f"For predicate validation, use `react_llm(..., constraint=C)` "
                f"— the ReAct loop lets the model see the violation in the tool "
                f"result and self-correct."
            )

        # HARD-SET the schema on response_format. NO try/except — if json_schema()
        # fails on a non-factory constraint, that's a real bug we want surfaced
        # (the LLM has nothing to converge on without the schema, so silently
        # retrying would be busy-work).
        schema = constraint.json_schema()
        gk = dict(generate_kwargs)
        gk.setdefault("response_format",
                      {"type": "json_schema",
                       "json_schema": {"name": "answer", "schema": schema}})

        from plm.constraint import ConstraintViolation
        last_err = None
        for _ in range(1 + max(0, return_budget)):
            check_depth_or_raise()
            content = backend.generate(msgs, **gk).get("content") or ""
            try:
                value = json.loads(content)
            except Exception:
                value = content
            try:
                return constraint.validate(value)
            except ConstraintViolation as e:
                last_err = e
                msgs = msgs + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": constraint.describe()
                     + "\n\nYour previous answer failed validation: " + str(e)},
                ]
        raise last_err                                     # budget exhausted -> surface the last violation
