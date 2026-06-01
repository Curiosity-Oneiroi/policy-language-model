"""36-test suite for plm.constraint — covers structural + factory authoring,
all four algebra operators (& | ~ ^), JSON Schema export, context-passing,
the @constraint decorator, exactly_one_of, the registry, and the struct-only
auto-merge."""

from __future__ import annotations

from typing import Literal, Optional

import pytest
from pydantic import model_validator

from plm.constraint import (
    Constraint,
    ConstraintViolation,
    Field,
    REGISTRY,
    constraint,
)


# =============================================================================
# 1 — structural subclass validates a struct
# =============================================================================


def test_subclass_validates_struct():
    class MyAnswer(Constraint):
        score: int = Field(ge=0, le=100)
        label: str

    ok = MyAnswer.validate({"score": 80, "label": "yes"})
    assert ok.score == 80 and ok.label == "yes"

    with pytest.raises(ConstraintViolation):
        MyAnswer.validate({"score": 200, "label": "x"})


# =============================================================================
# 2 — Constraint.of(int_range) returns a scalar
# =============================================================================


def test_of_int_range_returns_scalar():
    c = Constraint.of(int_range=(0, 9))
    assert c.validate(5) == 5
    with pytest.raises(ConstraintViolation):
        c.validate(10)
    with pytest.raises(ConstraintViolation):
        c.validate(-1)


# =============================================================================
# 3 — float strict bounds
# =============================================================================


def test_of_float_strict_bounds():
    c = Constraint.of(float_gt=0.0, float_lt=1.0)
    assert c.validate(0.5) == 0.5
    with pytest.raises(ConstraintViolation):
        c.validate(0.0)  # gt → must be strictly above 0
    with pytest.raises(ConstraintViolation):
        c.validate(1.0)


# =============================================================================
# 4 — one_of literal
# =============================================================================


def test_of_one_of_literal():
    c = Constraint.of(one_of=["red", "green", "blue"])
    assert c.validate("red") == "red"
    with pytest.raises(ConstraintViolation):
        c.validate("yellow")


# =============================================================================
# 5 — str pattern + length combined
# =============================================================================


def test_of_str_pattern_length():
    c = Constraint.of(str_pattern=r"^[A-Z]+$", str_length=(2, 5))
    assert c.validate("ABC") == "ABC"
    with pytest.raises(ConstraintViolation):
        c.validate("ab")  # lowercase, also too short
    with pytest.raises(ConstraintViolation):
        c.validate("A")  # too short
    with pytest.raises(ConstraintViolation):
        c.validate("ABCDEF")  # too long


# =============================================================================
# 6 — list_of + list_length + unique + sorted
# =============================================================================


def test_of_list_of_with_modifiers():
    c = Constraint.of(
        list_of=Constraint.of(int_range=(0, 9)),
        list_length=(3, 5),
        unique=True,
        sorted="asc",
    )
    assert c.validate([1, 2, 3]) == [1, 2, 3]
    with pytest.raises(ConstraintViolation):
        c.validate([1, 1, 2])  # not unique
    with pytest.raises(ConstraintViolation):
        c.validate([3, 2, 1])  # not asc
    with pytest.raises(ConstraintViolation):
        c.validate([1, 2])  # too short
    with pytest.raises(ConstraintViolation):
        c.validate([1, 2, 99])  # 99 outside elem range


# =============================================================================
# 7 — dict_of
# =============================================================================


def test_of_dict_of():
    c = Constraint.of(dict_of=(str, Constraint.of(float_range=(0.0, 1.0))))
    out = c.validate({"a": 0.5, "b": 0.9})
    assert out == {"a": 0.5, "b": 0.9}
    with pytest.raises(ConstraintViolation):
        c.validate({"a": 1.5})  # out of range


# =============================================================================
# 8 — instance_of an arbitrary class
# =============================================================================


def test_of_instance_of_arbitrary_class():
    class MCTS:
        def __init__(self, depth):
            self.depth = depth

    c = Constraint.of(instance_of=MCTS, description="an MCTS policy")
    m = MCTS(depth=3)
    assert c.validate(m) is m

    with pytest.raises(ConstraintViolation):
        c.validate("not an MCTS")


# =============================================================================
# 9 — Constraint.of(predicate=fn)
# =============================================================================


def test_of_predicate_passes_and_fails():
    def is_prime(x):
        if x < 2 or any(x % i == 0 for i in range(2, int(x**0.5) + 1)):
            raise ValueError("not prime")

    c = Constraint.of(predicate=is_prime, description="prime")
    assert c.validate(7) == 7
    with pytest.raises(ConstraintViolation):
        c.validate(8)


# =============================================================================
# 10 — plain function predicate pass + fail
# =============================================================================


def test_predicate_callable_pass_and_fail():
    def even(x):
        if x % 2 != 0:
            raise ValueError("not even")

    c = Constraint.of(predicate=even)
    assert c.validate(4) == 4
    with pytest.raises(ConstraintViolation):
        c.validate(5)


# =============================================================================
# 11 — built-in kind registry: semver
# =============================================================================


def test_of_kind_registry():
    c = Constraint.of(kind="semver")
    assert c.validate("1.2.3") == "1.2.3"
    with pytest.raises(ConstraintViolation):
        c.validate("not-semver")


# =============================================================================
# 12 — runtime registry extension
# =============================================================================


def test_runtime_registry_extension():
    def _is_palindrome(s):
        if not isinstance(s, str) or s != s[::-1]:
            raise ValueError("not a palindrome")

    Constraint.registry["palindrome"] = _is_palindrome
    try:
        c = Constraint.from_kind("palindrome")
        assert c.validate("racecar") == "racecar"
        with pytest.raises(ConstraintViolation):
            c.validate("hello")
    finally:
        del Constraint.registry["palindrome"]


# =============================================================================
# 13 — AND composition
# =============================================================================


def test_and_composition():
    def is_even(x):
        if x % 2 != 0:
            raise ValueError("not even")

    c = Constraint.of(int_range=(0, 100)) & Constraint.of(predicate=is_even)
    assert c.validate(42) == 42
    with pytest.raises(ConstraintViolation):
        c.validate(43)  # not even
    with pytest.raises(ConstraintViolation):
        c.validate(200)  # out of range


# =============================================================================
# 14 — OR composition (first success wins)
# =============================================================================


def test_or_composition_first_success():
    c = Constraint.of(int_range=(0, 9)) | Constraint.of(one_of=["x", "y"])
    assert c.validate(5) == 5
    assert c.validate("x") == "x"
    with pytest.raises(ConstraintViolation):
        c.validate("z")


# =============================================================================
# 15 — NOT on a predicate-form constraint
# =============================================================================


def test_not_negation_on_predicate():
    c = ~Constraint.of(int_lt=0)  # NOT (x < 0)  i.e.  x >= 0
    assert c.validate(0) == 0
    assert c.validate(5) == 5
    with pytest.raises(ConstraintViolation):
        c.validate(-1)


# =============================================================================
# 16 — NOT on a structural class raises TypeError at compose time
# =============================================================================


def test_not_on_structural_raises_at_compose():
    class Some(Constraint):
        x: int

    with pytest.raises(TypeError):
        _ = ~Some  # NOT a structural class doesn't make sense


# =============================================================================
# 17 — XOR exactly-one semantics
# =============================================================================


def test_xor_exactly_one():
    c = Constraint.of(int_range=(0, 9)) ^ Constraint.of(str_length=(1, 1))
    assert c.validate(5) == 5
    assert c.validate("a") == "a"
    with pytest.raises(ConstraintViolation):
        c.validate(99)  # neither
    # Both passing — only happens if value is both an int 0..9 AND a single-char str.
    # Python won't have such a value naturally; the structural constraint already
    # rejects mismatched types, so we test "neither" and "exactly-one" cases above.


# =============================================================================
# 18 — describe() renders a struct subclass
# =============================================================================


def test_describe_renders_struct_subclass():
    class Q(Constraint):
        score: int = Field(ge=0, le=100, description="confidence")
        label: Literal["yes", "no"]

        @constraint("score must be even")
        @model_validator(mode="after")
        def _even(self):
            if self.score % 2 != 0:
                raise ValueError("odd")
            return self

    desc = Q.describe()
    assert "score" in desc and "label" in desc
    assert "ge=0" in desc and "le=100" in desc
    assert "confidence" in desc
    assert "score must be even" in desc


# =============================================================================
# 19 — describe() renders factory kwargs
# =============================================================================


def test_describe_renders_factory_kwargs():
    c = Constraint.of(int_range=(0, 9), description="a digit")
    desc = c.describe()
    assert "int" in desc and "0" in desc and "9" in desc
    assert "a digit" in desc


# =============================================================================
# 20 — describe() of a composition
# =============================================================================


def test_describe_renders_composition():
    a = Constraint.of(int_range=(0, 100))
    b = Constraint.of(predicate=lambda x: None, description="some rule")

    composed = a | ~b
    desc = composed.describe()
    assert "OR" in desc
    assert "NOT" in desc


# =============================================================================
# 21 — @constraint decorator description surfaces
# =============================================================================


def test_cross_field_validator_with_decorator():
    class Order(Constraint):
        limit: float
        stop: float

        @constraint("stop must be below limit")
        @model_validator(mode="after")
        def _stop_below_limit(self):
            if self.stop >= self.limit:
                raise ValueError("stop must be < limit")
            return self

    with pytest.raises(ConstraintViolation):
        Order.validate({"limit": 100.0, "stop": 110.0})
    assert "stop must be below limit" in Order.describe()


# =============================================================================
# 22 — @Constraint.exactly_one_of decorator
# =============================================================================


def test_exactly_one_of_decorator():
    @Constraint.exactly_one_of("score", "label", "error")
    class Result(Constraint):
        score: Optional[float] = None
        label: Optional[str] = None
        error: Optional[str] = None

    Result.validate({"score": 0.5})
    Result.validate({"label": "ok"})
    Result.validate({"error": "boom"})

    with pytest.raises(ConstraintViolation):
        Result.validate({"score": 0.5, "label": "x"})  # two set
    with pytest.raises(ConstraintViolation):
        Result.validate({})  # none set


# =============================================================================
# 23 — callable + behaves_like
# =============================================================================


def test_callable_with_behaves_like():
    c = Constraint.of(
        callable=True,
        behaves_like=[((2,), 4), ((3,), 9), ((10,), 100)],
        description="squares its int input",
    )
    assert c.validate(lambda x: x * x)(5) == 25
    with pytest.raises(ConstraintViolation):
        c.validate(lambda x: x + 1)
    with pytest.raises(ConstraintViolation):
        c.validate(42)  # not callable


# =============================================================================
# 24 — approx equality
# =============================================================================


def test_approx_equality():
    c = Constraint.of(approx=(3.14159, 1e-3))
    assert c.validate(3.142) == 3.142
    assert c.validate(3.141) == 3.141
    with pytest.raises(ConstraintViolation):
        c.validate(3.5)


# =============================================================================
# 25 — negative kwargs (not_one_of, not_pattern, not_empty)
# =============================================================================


def test_negative_kwargs():
    c1 = Constraint.of(not_one_of=[1, 2, 3])
    assert c1.validate(5) == 5
    with pytest.raises(ConstraintViolation):
        c1.validate(2)

    c2 = Constraint.of(not_pattern=r"\bsecret\b")
    assert c2.validate("hello world") == "hello world"
    with pytest.raises(ConstraintViolation):
        c2.validate("this is a secret")

    c3 = Constraint.of(not_empty=True)
    assert c3.validate("hi") == "hi"
    assert c3.validate([1]) == [1]
    with pytest.raises(ConstraintViolation):
        c3.validate("")
    with pytest.raises(ConstraintViolation):
        c3.validate([])


# =============================================================================
# 26 — OR aggregates child error reasons
# =============================================================================


def test_constraint_violation_aggregation_or():
    c = Constraint.of(int_range=(0, 9)) | Constraint.of(one_of=["x", "y"])
    try:
        c.validate("z")
    except ConstraintViolation as e:
        msg = str(e)
        assert "OR:" in msg
        assert "any branch" in msg or "branches" in msg


# =============================================================================
# 27 — ordinary pydantic Field passthrough
# =============================================================================


def test_pydantic_field_passthrough():
    class C(Constraint):
        s: str = Field(min_length=3, max_length=10, pattern=r"^[a-z]+$")
        n: int = Field(ge=0, le=10)

    C.validate({"s": "hello", "n": 5})
    with pytest.raises(ConstraintViolation):
        C.validate({"s": "X", "n": 5})  # wrong pattern + too short
    with pytest.raises(ConstraintViolation):
        C.validate({"s": "hello", "n": 11})


# =============================================================================
# 28 — pydantic coercion: "5" → 5
# =============================================================================


def test_validate_returns_coerced_value():
    c = Constraint.of(int_range=(0, 9))
    assert c.validate("5") == 5  # pydantic coerces string→int
    assert isinstance(c.validate("5"), int)


# =============================================================================
# 29 — recursive structure
# =============================================================================


def test_recursive_structure_constraint():
    class Node(Constraint):
        value: int = Field(ge=0)
        children: list["Node"] = []

    Node.model_rebuild()
    tree = Node.validate(
        {"value": 0, "children": [{"value": 1, "children": [{"value": 2}]}]}
    )
    assert tree.value == 0
    assert tree.children[0].children[0].value == 2

    with pytest.raises(ConstraintViolation):
        Node.validate({"value": -1, "children": []})


# =============================================================================
# 30 — validate(value, context=...)
# =============================================================================


def test_validate_with_context():
    class Bounded(Constraint):
        value: int

        @model_validator(mode="after")
        def _check(self, info):
            max_v = info.context.get("max") if info.context else None
            if max_v is not None and self.value > max_v:
                raise ValueError(f"value {self.value} exceeds max {max_v}")
            return self

    Bounded.validate({"value": 5}, context={"max": 10})
    with pytest.raises(ConstraintViolation):
        Bounded.validate({"value": 50}, context={"max": 10})


# =============================================================================
# 31 — JSON Schema export for a struct subclass
# =============================================================================


def test_json_schema_struct_export():
    class Order(Constraint):
        side: Literal["buy", "sell"]
        qty: int = Field(gt=0, le=100)

    schema = Order.json_schema()
    assert schema.get("type") == "object"
    assert "properties" in schema
    assert "side" in schema["properties"]
    assert "qty" in schema["properties"]
    qty_schema = schema["properties"]["qty"]
    # Pydantic emits exclusiveMinimum / maximum
    assert qty_schema.get("exclusiveMinimum") == 0 or qty_schema.get("minimum") == 0
    assert qty_schema.get("maximum") == 100


# =============================================================================
# 32 — JSON Schema export for predicate-form: x-description present
# =============================================================================


def test_json_schema_predicate_x_description():
    def is_prime(x):
        if x < 2 or any(x % i == 0 for i in range(2, int(x**0.5) + 1)):
            raise ValueError("not prime")

    c = Constraint.of(int_range=(0, 9), predicate=is_prime, description="prime digit")
    schema = c.json_schema()
    # Structural bits come through (int with min/max).
    assert schema.get("type") == "integer"
    assert schema.get("minimum") == 0
    assert schema.get("maximum") == 9
    # NL description on x-description.
    assert "x-description" in schema
    assert "prime digit" in schema["x-description"]


# =============================================================================
# 33 — Functional constraint on a structural one via &
# =============================================================================


def test_functional_on_structural_via_and():
    class Trade(Constraint):
        side: Literal["buy", "sell"]
        qty: int = Field(gt=0)

    # Compose: shape + predicate on the instance.
    small_trade = Trade & Constraint.of(
        predicate=lambda t: None if t.qty < 1000 else (_ for _ in ()).throw(
            ValueError("qty >= 1000")
        ),
        description="qty < 1000",
    )

    small_trade.validate({"side": "buy", "qty": 500})
    with pytest.raises(ConstraintViolation):
        small_trade.validate({"side": "buy", "qty": 5000})


# =============================================================================
# 34 — struct-only & struct-only auto-merges
# =============================================================================


def test_struct_only_and_auto_merges():
    class HasScore(Constraint):
        score: float = Field(ge=0.0, le=1.0)

    class HasLabel(Constraint):
        label: Literal["yes", "no"]

    Merged = HasScore & HasLabel

    # Should be a single merged model — not a run-both wrapper.
    assert getattr(Merged, "_constraint_is_composite", False) is False
    assert "score" in Merged.model_fields
    assert "label" in Merged.model_fields

    inst = Merged.validate({"score": 0.8, "label": "yes"})
    assert inst.score == 0.8
    assert inst.label == "yes"


# =============================================================================
# 35 — overlapping same-type field intersects field constraints
# =============================================================================


def test_struct_and_with_overlapping_field_intersects_constraints():
    class HasScoreGe0(Constraint):
        score: int = Field(ge=0)

    class HasScoreLe100(Constraint):
        score: int = Field(le=100)

    Merged = HasScoreGe0 & HasScoreLe100
    Merged.validate({"score": 50})
    with pytest.raises(ConstraintViolation):
        Merged.validate({"score": -1})
    with pytest.raises(ConstraintViolation):
        Merged.validate({"score": 200})


# =============================================================================
# 36 — incompatible-type field: merge fails cleanly, falls back to run-both
# =============================================================================


def test_struct_and_with_incompatible_fields_falls_back_or_errors():
    class HasFieldInt(Constraint):
        x: int

    class HasFieldStr(Constraint):
        x: str

    composed = HasFieldInt & HasFieldStr
    # We allow either: a graceful error at merge time OR fallback to run-both.
    # In our impl: merge returns None on conflict → run-both wrapper. Run-both
    # then fails on validate (no int can be a str). Just assert validate fails:
    with pytest.raises(ConstraintViolation):
        composed.validate({"x": 5})
    with pytest.raises(ConstraintViolation):
        composed.validate({"x": "hello"})


# =============================================================================
# 37 — merged class inherits @model_validators from BOTH parents
# =============================================================================


def test_merge_inherits_validators_from_both_parents():
    class HasScore(Constraint):
        score: int

        @constraint("score must be even")
        @model_validator(mode="after")
        def _score_even(self):
            if self.score % 2 != 0:
                raise ValueError("odd score")
            return self

    class HasLabel(Constraint):
        label: str

        @constraint("label must be lowercase")
        @model_validator(mode="after")
        def _label_lower(self):
            if self.label != self.label.lower():
                raise ValueError("uppercase label")
            return self

    Merged = HasScore & HasLabel
    # Both validators should be registered on the merged class.
    vnames = set(Merged.__pydantic_decorators__.model_validators.keys())
    assert "_score_even" in vnames
    assert "_label_lower" in vnames

    Merged.validate({"score": 4, "label": "ok"})  # both pass
    with pytest.raises(ConstraintViolation):
        Merged.validate({"score": 3, "label": "ok"})  # odd score
    with pytest.raises(ConstraintViolation):
        Merged.validate({"score": 4, "label": "BAD"})  # uppercase label


# =============================================================================
# 38 — merged-class instance is isinstance of both parents
# =============================================================================


def test_merge_instance_is_isinstance_of_both_parents():
    class HasScore(Constraint):
        score: int

    class HasLabel(Constraint):
        label: str

    Merged = HasScore & HasLabel
    inst = Merged.validate({"score": 1, "label": "x"})
    assert isinstance(inst, HasScore)
    assert isinstance(inst, HasLabel)
    assert isinstance(inst, Constraint)


# =============================================================================
# 39 — sorted=False and monotonic=False are explicit opt-outs (skip the check)
# =============================================================================


def test_sorted_false_and_monotonic_false_skip():
    # sorted=False should NOT install a sorted check (no error, no validation).
    c1 = Constraint.of(list_of=int, sorted=False)
    assert c1.validate([3, 1, 2]) == [3, 1, 2]  # would fail if check were on

    # monotonic=False should also skip.
    c2 = Constraint.of(list_of=int, monotonic=False)
    assert c2.validate([3, 1, 2]) == [3, 1, 2]


# =============================================================================
# 40 — AND fails fast on first failing child (later children don't run)
# =============================================================================


def test_predicate_coercer_style_fails_loudly():
    """A coercer-style predicate (returns a transformed value) must fail
    loudly with a clear error instead of silently discarding the return —
    the I5 detect-and-warn behavior."""

    def bad_normalize(s):
        # User mistake: wrote a coercer instead of a predicate.
        return s.lower().strip()

    c = Constraint.of(predicate=bad_normalize)
    with pytest.raises(ConstraintViolation) as excinfo:
        c.validate("  HELLO  ")
    # The error must explain the protocol so the LLM author can fix the mistake.
    msg = str(excinfo.value)
    assert "bad_normalize" in msg
    assert "predicates must return None" in msg or "non-None value" in msg
    assert "coercer" in msg  # points at the right tool (coercer= kwargs)

    # Sanity: legitimate predicate styles still work.
    def ok_returns_none(s):
        if not isinstance(s, str):
            raise ValueError("not a string")

    def ok_returns_true(s):
        if not isinstance(s, str):
            raise ValueError("not a string")
        return True

    def ok_returns_input(s):
        if not isinstance(s, str):
            raise ValueError("not a string")
        return s

    for fn in (ok_returns_none, ok_returns_true, ok_returns_input):
        c = Constraint.of(predicate=fn)
        assert c.validate("hello") == "hello"

    # `return False` is a contract violation too — should fail loudly.
    def returns_false(s):
        return False

    c = Constraint.of(predicate=returns_false)
    with pytest.raises(ConstraintViolation) as excinfo:
        c.validate("x")
    assert "returned False" in str(excinfo.value)


def test_coercer_transforms_value():
    """`coercer=fn` transforms the value before validation."""
    c = Constraint.of(coercer=lambda s: s.lower().strip())
    assert c.validate("  HELLO  ") == "hello"
    assert c.validate("WORLD") == "world"


def test_coercer_then_field_constraint():
    """Coercer runs BEFORE field constraints — `Field(pattern=...)` checks
    the *coerced* value, not the original."""
    c = Constraint.of(
        coercer=lambda s: s.lower().strip(),
        str_pattern=r"^[a-z]+$",
    )
    # "  HELLO  " fails pattern as-is (uppercase + spaces) but passes after
    # the coercer normalizes it.
    assert c.validate("  HELLO  ") == "hello"
    # A value the coercer can't rescue still fails.
    with pytest.raises(ConstraintViolation):
        c.validate("hello123")  # contains digits — pattern rejects


def test_multiple_coercers_chain_in_order():
    """`coercers=[fn1, fn2, fn3]` chains: fn1's output feeds fn2, etc."""
    c = Constraint.of(coercers=[
        str.strip,                      # "  HELLO WORLD  " → "HELLO WORLD"
        str.lower,                      # "HELLO WORLD"     → "hello world"
        lambda s: s.replace(" ", "_"),  # "hello world"     → "hello_world"
    ])
    assert c.validate("  HELLO WORLD  ") == "hello_world"


def test_int_range_plus_strict_bound_rejected_at_compose():
    """B5: mixing int_range with int_gt/int_lt/etc. raises at compose time
    instead of producing an opaque pydantic SchemaError later."""
    with pytest.raises(ValueError, match="int_range.*cannot be combined"):
        Constraint.of(int_range=(0, 100), int_gt=5)
    with pytest.raises(ValueError, match="float_range.*cannot be combined"):
        Constraint.of(float_range=(0.0, 1.0), float_le=0.5)
    # Pure strict bounds (no range) still work.
    Constraint.of(int_gt=0, int_lt=10)


def test_invalid_monotonic_value_rejected():
    """B17: monotonic= must be True/False/'strict'; typos fail at compose."""
    with pytest.raises(ValueError, match="monotonic.*must be"):
        Constraint.of(list_of=int, monotonic="strictt")
    with pytest.raises(ValueError, match="monotonic.*must be"):
        Constraint.of(list_of=int, monotonic="ascending")


def test_describe_skips_falsey_sorted_and_monotonic():
    """B9: sorted=False and monotonic=False are opt-outs and should NOT
    appear in the rendered describe text."""
    c1 = Constraint.of(list_of=int, sorted=False, not_empty=True)
    desc1 = c1.describe()
    assert "sorted" not in desc1.lower()  # must not render the opt-out
    assert "non-empty" in desc1  # other kwargs still render

    c2 = Constraint.of(list_of=int, monotonic=False)
    desc2 = c2.describe()
    assert "monotonic" not in desc2.lower()


def test_coercer_then_predicate_ordering():
    """Coercer must run BEFORE the predicate so the predicate sees the
    coerced value."""
    captured = []

    def check_lowercase(s):
        captured.append(s)
        if s != s.lower():
            raise ValueError("predicate saw uppercase — coercer didn't run first")

    c = Constraint.of(
        coercer=lambda s: s.lower().strip(),
        predicate=check_lowercase,
    )
    assert c.validate("  HELLO  ") == "hello"
    assert captured == ["hello"]  # predicate saw the coerced value


def test_single_coercer_and_coercers_list_compose():
    """coercer= and coercers=[...] in the same call should both apply.
    The single coercer is added before the list (per declaration order in
    _interpret_kwargs)."""
    c = Constraint.of(
        coercer=str.strip,
        coercers=[str.lower, lambda s: s + "!"],
    )
    # Pipeline: strip → lower → add bang.
    assert c.validate("  HELLO  ") == "hello!"


def test_w3_kind_plus_int_range_rejected_at_compose():
    """W3: `kind="semver"` + `int_range=(0,9)` would silently produce a broken
    constraint (str-shaped predicate against int value); now rejected at compose."""
    with pytest.raises(ValueError, match="multiple type-shifting groups"):
        Constraint.of(kind="semver", int_range=(0, 9))
    with pytest.raises(ValueError, match="multiple type-shifting groups"):
        Constraint.of(instance_of=int, str_pattern=r"^x$")
    with pytest.raises(ValueError, match="multiple type-shifting groups"):
        Constraint.of(one_of=["a", "b"], int_range=(0, 9))

    # Same-group combinations still work fine.
    Constraint.of(str_pattern=r"^[a-z]+$", str_length=(1, 10))  # both in 'str' group
    Constraint.of(list_of=int)                                   # 'list' group alone


def test_b22_composite_model_validate_fires_children():
    """B22: calling pydantic's `model_validate` directly on a composite wrapper
    used to bypass the children and return an empty instance. Now it
    delegates to our `validate`, exercising the children correctly."""

    def must_be_5(x):
        if x != 5:
            raise ValueError("not 5")

    c = Constraint.of(int_range=(0, 10)) & Constraint.of(predicate=must_be_5)
    # Both interfaces should behave identically:
    assert c.validate(5) == 5
    assert c.model_validate(5) == 5
    with pytest.raises(ConstraintViolation):
        c.model_validate(7)  # must NOT silently return an empty composite instance


def test_nb1_exactly_one_of_empty_rejected():
    """NB1: `@Constraint.exactly_one_of()` with no field names would create
    an always-failing class. Now raises at decoration time."""
    with pytest.raises(ValueError, match="at least one field name"):
        @Constraint.exactly_one_of()  # type: ignore[call-arg]
        class _Result(Constraint):
            x: int


def test_b33_list_length_plus_int_range_rejected():
    """B33: list_length + int_range targets two conflicting base types
    (int + list-shape). Now rejected at compose time instead of producing
    a confusing pydantic SchemaError about min_length on an int."""
    with pytest.raises(ValueError, match="multiple type-shifting groups"):
        Constraint.of(int_range=(0, 9), list_length=(1, 5))
    with pytest.raises(ValueError, match="multiple type-shifting groups"):
        Constraint.of(list_of=int, dict_of=(str, int))

    # list_length alone (with list_of) still works.
    c = Constraint.of(list_of=int, list_length=(2, 4))
    c.validate([1, 2, 3])


def test_i6_describe_dedups_aliased_descriptions():
    """I6: if a @constraint-tagged validator is reachable via multiple names
    (aliased), the description should still appear exactly once in
    .describe() output."""
    class C(Constraint):
        x: int

        @constraint("x must be even")
        @model_validator(mode="after")
        def _check(self):
            if self.x % 2 != 0:
                raise ValueError("odd")
            return self

    # Alias the validator under a second attribute name on the class. The
    # de-dup logic should still only emit "x must be even" once.
    C._aliased_check = C._check  # type: ignore[attr-defined]
    desc = C.describe()
    assert desc.count("x must be even") == 1


def test_i10_type_compatibility_error_lists_conflicts_clearly():
    """I10: the multi-group error message should make it easy for PLM
    to see WHICH kwargs from WHICH groups conflicted."""
    with pytest.raises(ValueError) as excinfo:
        Constraint.of(kind="semver", int_range=(0, 9), str_pattern=r"^x$")
    msg = str(excinfo.value)
    # Three groups present: kind, int, str — each should appear by name.
    assert "'kind'" in msg
    assert "'int'" in msg
    assert "'str'" in msg
    assert "int_range" in msg or "int_range" in str(msg)
    # The corrective hint about composing with `&` should be present.
    assert "compose" in msg or "&" in msg


def test_nb2_a_and_a_short_circuits():
    """NB2: `A & A` returns `A` directly — no merged class, no run-both
    wrapper. Python would refuse duplicate bases anyway, and the semantics
    are identical."""
    class HasScore(Constraint):
        score: int = Field(ge=0)

    assert (HasScore & HasScore) is HasScore


def test_coercer_can_raise_to_reject():
    """A coercer that raises ValueError causes ConstraintViolation."""
    def parse_int(s):
        return int(s)  # raises ValueError on bad input — pydantic propagates

    c = Constraint.of(coercer=parse_int, int_range=(0, 100))
    assert c.validate("42") == 42
    with pytest.raises(ConstraintViolation):
        c.validate("not-an-int")  # parse_int raises → ConstraintViolation
    with pytest.raises(ConstraintViolation):
        c.validate("200")  # parses to 200 but fails int_range


def test_and_fails_fast():
    side_effect = []

    def first_fails(x):
        side_effect.append("first")
        raise ValueError("first child rejects")

    def second_runs(x):
        side_effect.append("second")

    # Use AND-wrapper path (factory & factory → run-both wrapper):
    c = Constraint.of(predicate=first_fails) & Constraint.of(predicate=second_runs)
    with pytest.raises(ConstraintViolation):
        c.validate(42)
    # second predicate must NOT have run — fail-fast.
    assert "first" in side_effect
    assert "second" not in side_effect
