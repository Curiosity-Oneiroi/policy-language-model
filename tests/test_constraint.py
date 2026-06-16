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
    c = Constraint.of(type=int, int_range=(0, 9))
    assert c.validate(5) == 5
    with pytest.raises(ConstraintViolation):
        c.validate(10)
    with pytest.raises(ConstraintViolation):
        c.validate(-1)


# =============================================================================
# 3 — float strict bounds
# =============================================================================


def test_of_float_strict_bounds():
    c = Constraint.of(type=float, float_gt=0.0, float_lt=1.0)
    assert c.validate(0.5) == 0.5
    with pytest.raises(ConstraintViolation):
        c.validate(0.0)  # gt → must be strictly above 0
    with pytest.raises(ConstraintViolation):
        c.validate(1.0)


# =============================================================================
# 4 — one_of literal
# =============================================================================


def test_of_one_of_literal():
    c = Constraint.of(type=Literal["red", "green", "blue"])
    assert c.validate("red") == "red"
    with pytest.raises(ConstraintViolation):
        c.validate("yellow")


# =============================================================================
# 5 — str pattern + length combined
# =============================================================================


def test_of_str_pattern_length():
    c = Constraint.of(type=str, str_pattern=r"^[A-Z]+$", str_length=(2, 5))
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
        type=list[Constraint.field(type=int, int_range=(0, 9))],
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
    c = Constraint.of(type=dict[str, Constraint.field(type=float, float_range=(0.0, 1.0))])
    out = c.validate({"a": 0.5, "b": 0.9})
    assert out == {"a": 0.5, "b": 0.9}
    with pytest.raises(ConstraintViolation):
        c.validate({"a": 1.5})  # out of range


# =============================================================================
# 8 — is_instance_of an arbitrary class
# =============================================================================


def test_of_is_instance_of_arbitrary_class():
    class MCTS:
        def __init__(self, depth):
            self.depth = depth

    # is_instance_of is a PURE check now; pair it with type=object (no coercion) so the
    # isinstance predicate is the sole gate.
    c = Constraint.of(type=object, is_instance_of=MCTS, description="an MCTS policy")
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

    c = Constraint.of(type=int, predicate=is_prime, description="prime")
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

    c = Constraint.of(type=int, predicate=even)
    assert c.validate(4) == 4
    with pytest.raises(ConstraintViolation):
        c.validate(5)


# =============================================================================
# 11 — built-in kind registry: semver
# =============================================================================


def test_of_kind_registry():
    c = Constraint.of(type=str, kind="semver")
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

    c = Constraint.of(type=int, int_range=(0, 100)) & Constraint.of(type=int, predicate=is_even)
    assert c.validate(42) == 42
    with pytest.raises(ConstraintViolation):
        c.validate(43)  # not even
    with pytest.raises(ConstraintViolation):
        c.validate(200)  # out of range


# =============================================================================
# 14 — OR composition (first success wins)
# =============================================================================


def test_or_composition_first_success():
    c = Constraint.of(type=int, int_range=(0, 9)) | Constraint.of(type=Literal["x", "y"])
    assert c.validate(5) == 5
    assert c.validate("x") == "x"
    with pytest.raises(ConstraintViolation):
        c.validate("z")


# =============================================================================
# 15 — NOT on a predicate-form constraint
# =============================================================================


def test_not_negation_on_predicate():
    c = ~Constraint.of(type=int, int_lt=0)  # NOT (x < 0)  i.e.  x >= 0
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
    c = Constraint.of(type=int, int_range=(0, 9)) ^ Constraint.of(type=str, str_length=(1, 1))
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
    c = Constraint.of(type=int, int_range=(0, 9), description="a digit")
    desc = c.describe()
    assert "int" in desc and "0" in desc and "9" in desc
    assert "a digit" in desc


# =============================================================================
# 20 — describe() of a composition
# =============================================================================


def test_describe_renders_composition():
    a = Constraint.of(type=int, int_range=(0, 100))
    b = Constraint.of(type=object, predicate=lambda x: None, description="some rule")

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
        type=object,
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
    c = Constraint.of(type=float, approx=(3.14159, 1e-3))
    assert c.validate(3.142) == 3.142
    assert c.validate(3.141) == 3.141
    with pytest.raises(ConstraintViolation):
        c.validate(3.5)


# =============================================================================
# 25 — negative kwargs (not_one_of, not_pattern, not_empty)
# =============================================================================


def test_negative_kwargs():
    c1 = Constraint.of(type=int, not_one_of=[1, 2, 3])
    assert c1.validate(5) == 5
    with pytest.raises(ConstraintViolation):
        c1.validate(2)

    c2 = Constraint.of(type=str, not_pattern=r"\bsecret\b")
    assert c2.validate("hello world") == "hello world"
    with pytest.raises(ConstraintViolation):
        c2.validate("this is a secret")

    c3 = Constraint.of(type=object, not_empty=True)
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
    c = Constraint.of(type=int, int_range=(0, 9)) | Constraint.of(type=Literal["x", "y"])
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
    c = Constraint.of(type=int, int_range=(0, 9))
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

    c = Constraint.of(type=int, int_range=(0, 9), predicate=is_prime, description="prime digit")
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
        type=object,
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

    # Two structs sharing a field name with incompatible types cannot be merged
    # (int vs str can't be intersected), and the run-both fallback is BROKEN for
    # two structs — it threads HasFieldInt's instance into HasFieldStr.validate,
    # which rejects EVERY input. So we fail LOUD at compose time (the `&`) with a
    # specific message, rather than emit a composite that silently validates nothing.
    with pytest.raises(TypeError):
        HasFieldInt & HasFieldStr


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
    c1 = Constraint.of(type=list[int], sorted=False)
    assert c1.validate([3, 1, 2]) == [3, 1, 2]  # would fail if check were on

    # monotonic=False should also skip.
    c2 = Constraint.of(type=list[int], monotonic=False)
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

    c = Constraint.of(type=str, predicate=bad_normalize)
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
        c = Constraint.of(type=str, predicate=fn)
        assert c.validate("hello") == "hello"

    # `return False` is a contract violation too — should fail loudly.
    def returns_false(s):
        return False

    c = Constraint.of(type=str, predicate=returns_false)
    with pytest.raises(ConstraintViolation) as excinfo:
        c.validate("x")
    assert "returned False" in str(excinfo.value)


def test_coercer_transforms_value():
    """`coercer=fn` transforms the value before validation."""
    c = Constraint.of(type=str, coercer=lambda s: s.lower().strip())
    assert c.validate("  HELLO  ") == "hello"
    assert c.validate("WORLD") == "world"


def test_coercer_then_field_constraint():
    """Coercer runs BEFORE field constraints — `Field(pattern=...)` checks
    the *coerced* value, not the original."""
    c = Constraint.of(
        type=str,
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
    c = Constraint.of(type=str, coercers=[
        str.strip,                      # "  HELLO WORLD  " → "HELLO WORLD"
        str.lower,                      # "HELLO WORLD"     → "hello world"
        lambda s: s.replace(" ", "_"),  # "hello world"     → "hello_world"
    ])
    assert c.validate("  HELLO WORLD  ") == "hello_world"


def test_int_range_plus_strict_bound_compose_as_predicates():
    """Reframe: int_range and int_gt/int_lt are now both PREDICATES (no pydantic Field), so
    they freely COMPOSE on one type coercer — each is an independent AfterValidator check
    (no more SchemaError, no mutual-exclusion needed)."""
    c = Constraint.of(type=int, int_range=(0, 100), int_gt=5)   # in [0,100] AND > 5  -> (5,100]
    assert c.validate(50) == 50
    with pytest.raises(ConstraintViolation):
        c.validate(5)                                           # fails int_gt
    with pytest.raises(ConstraintViolation):
        c.validate(101)                                         # fails int_range
    f = Constraint.of(type=float, float_range=(0.0, 1.0), float_le=0.5)
    assert f.validate(0.25) == 0.25
    with pytest.raises(ConstraintViolation):
        f.validate(0.75)                                        # fails float_le
    # Pure strict bounds (no range) still work.
    Constraint.of(type=int, int_gt=0, int_lt=10)


def test_invalid_monotonic_value_rejected():
    """B17: monotonic= must be True/False/'strict'; typos fail at compose."""
    with pytest.raises(ValueError, match="monotonic.*must be"):
        Constraint.of(type=list[int], monotonic="strictt")
    with pytest.raises(ValueError, match="monotonic.*must be"):
        Constraint.of(type=list[int], monotonic="ascending")


def test_describe_skips_falsey_sorted_and_monotonic():
    """B9: sorted=False and monotonic=False are opt-outs and should NOT
    appear in the rendered describe text."""
    c1 = Constraint.of(type=list[int], sorted=False, not_empty=True)
    desc1 = c1.describe()
    assert "sorted" not in desc1.lower()  # must not render the opt-out
    assert "non-empty" in desc1  # other kwargs still render

    c2 = Constraint.of(type=list[int], monotonic=False)
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
        type=str,
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
        type=str,
        coercer=str.strip,
        coercers=[str.lower, lambda s: s + "!"],
    )
    # Pipeline: strip → lower → add bang.
    assert c.validate("  HELLO  ") == "hello!"


def test_w3_missing_or_nullable_type_rejected_at_compose():
    """Reframe: the type coercer is mandatory and ALWAYS spelled `type=`. A field with only
    predicates errors; a nullable / union type errors (alternatives go through the `|` algebra,
    not inside one field's type)."""
    import typing
    with pytest.raises(ValueError, match="no type coercer"):
        Constraint.of(kind="semver", int_range=(0, 9))
    with pytest.raises(ValueError, match="no type coercer"):
        Constraint.of(is_instance_of=int, str_pattern=r"^x$")
    with pytest.raises(ValueError, match="nullable"):
        Constraint.of(type=typing.Optional[int])
    with pytest.raises(ValueError, match="Union"):
        Constraint.of(type=int | str)
    with pytest.raises(ValueError, match="nullable"):                # nested nullable too
        Constraint.of(type=list[typing.Optional[int]])

    # One type coercer + same-shape refinement predicates compose fine.
    Constraint.of(type=str, str_pattern=r"^[a-z]+$", str_length=(1, 10))
    Constraint.of(type=list[int])


def test_b22_composite_model_validate_fires_children():
    """B22: calling pydantic's `model_validate` directly on a composite wrapper
    used to bypass the children and return an empty instance. Now it
    delegates to our `validate`, exercising the children correctly."""

    def must_be_5(x):
        if x != 5:
            raise ValueError("not 5")

    c = Constraint.of(type=int, int_range=(0, 10)) & Constraint.of(type=int, predicate=must_be_5)
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


def test_b33_removed_container_sugars_are_unknown_kwargs():
    """Reframe: list_of/dict_of/tuple_of/set_of/frozenset_of/one_of are GONE — express them
    through `type=` (type=list[int], type=Literal[...]). The old spellings are unknown kwargs."""
    for sugar in (dict(list_of=int), dict(dict_of=(str, int)), dict(tuple_of=int),
                  dict(set_of=int), dict(frozenset_of=int), dict(one_of=["a", "b"])):
        with pytest.raises(TypeError, match="unknown kwargs"):
            Constraint.of(**sugar)

    # list_length (a predicate) composes onto a type=list[...] coercer.
    c = Constraint.of(type=list[int], list_length=(2, 4))
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


def test_i10_type_error_messages_are_clear():
    """The mandatory-type and nullable errors name the problem clearly so PLM can fix it
    (point at `type=`, and at the `|` algebra for alternatives)."""
    import typing
    with pytest.raises(ValueError) as e1:
        Constraint.of(int_range=(0, 9))
    assert "no type coercer" in str(e1.value) and "type=" in str(e1.value)
    with pytest.raises(ValueError) as e2:
        Constraint.of(type=typing.Optional[int])
    assert "nullable" in str(e2.value) and "| algebra" in str(e2.value)


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

    c = Constraint.of(type=int, coercer=parse_int, int_range=(0, 100))
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
    c = Constraint.of(type=int, predicate=first_fails) & Constraint.of(type=int, predicate=second_runs)
    with pytest.raises(ConstraintViolation):
        c.validate(42)
    # second predicate must NOT have run — fail-fast.
    assert "first" in side_effect
    assert "second" not in side_effect


def test_all_checks_described_walks_tree():
    """`_all_checks_described` (natural_llm's M12 gate, which replaced `_contains_factory`) detects an
    UNDESCRIBED user check anywhere in the &/|/^ tree OR a struct field; built-ins and described user
    checks pass."""
    from plm.constraint.base import _all_checks_described
    from plm.constraint import constraint

    class S(Constraint):
        x: int
    Builtin = Constraint.of(type=int, int_range=(0, 9))           # built-in refinement -> auto-described
    Undesc = Constraint.of(type=int, predicate=lambda v: v > 0)   # untagged user predicate

    @constraint("positive")
    def pos(v):
        if v <= 0:
            raise ValueError("nope")
    Desc = Constraint.of(type=int, predicate=pos)

    assert _all_checks_described(S) is True
    assert _all_checks_described(Builtin) is True
    assert _all_checks_described(Desc) is True
    assert _all_checks_described(Undesc) is False
    assert _all_checks_described(S & Undesc) is False             # undescribed hidden in a composite
    assert _all_checks_described(S & Desc) is True

    class HasField(Constraint):
        f: Constraint.field(type=int, predicate=lambda v: v > 0)  # undescribed in a struct FIELD
    assert _all_checks_described(HasField) is False               # the old struct-field blind spot, caught


def test_json_schema_factory_preserves_defs_for_nested_model():
    """C22: flattening a factory value schema must carry $defs so nested-model
    $refs still resolve."""
    import json as _json

    class Inner(Constraint):
        x: int
    C = Constraint.of(type=list[Inner])                 # value = List[Inner] -> $ref + $defs
    sch = C.json_schema()
    assert "$defs" in sch and "Inner" in _json.dumps(sch) and "$ref" in _json.dumps(sch)


# =============================================================================
# 37 — predicate exceptions stay within the ConstraintViolation contract
#      (#12 sorted=/monotonic=, #13 kind="path_exists", C24 not_one_of):
#      a predicate hitting a wrong-type/unhashable/non-comparable value must
#      raise ConstraintViolation, NOT a raw TypeError that escapes validate().
# =============================================================================

@pytest.mark.parametrize("ctor,bad_value", [
    (lambda: Constraint.of(type=object, sorted=True),          [1, "a", 2]),     # not mutually comparable
    (lambda: Constraint.of(type=object, sorted=True),          123),             # not iterable
    (lambda: Constraint.of(type=object, monotonic=True),       [1, "a"]),        # not comparable
    (lambda: Constraint.of(type=object, monotonic=True),       123),             # not iterable
    (lambda: Constraint.of(type=str, kind="path_exists"),   123),             # wrong type
])
def test_37_predicate_typeerror_becomes_constraint_violation(ctor, bad_value):
    c = ctor()
    with pytest.raises(ConstraintViolation):       # NOT a raw TypeError
        c.validate(bad_value)


def test_37b_sorted_does_not_consume_generator_then_pass():
    """_check_sorted materializes once: a generator that IS sorted must pass
    (the old code consumed it in sorted() then compared against an empty list)."""
    c = Constraint.of(type=object, sorted=True)
    c.validate(iter([1, 2, 3]))                    # sorted generator → no ConstraintViolation
    with pytest.raises(ConstraintViolation):
        c.validate(iter([3, 1, 2]))                # genuinely unsorted still rejected


# =============================================================================
# 38 — composite validate keeps non-ConstraintViolation child errors inside the
#      contract: a structural child whose @model_validator raises a raw
#      non-ValueError (KeyError/TypeError) must surface as ConstraintViolation,
#      and OR/XOR must still try the OTHER branch instead of letting it escape.
# =============================================================================

def test_38_composite_non_violation_exception_stays_in_contract():
    class Boom(Constraint):
        x: int
        @model_validator(mode="after")
        def _boom(self):
            raise KeyError("structural validator raised a non-ValueError")

    okdict = Constraint.of(type=object, is_instance_of=dict)

    # AND: erroring child -> ConstraintViolation (NOT a raw KeyError).
    with pytest.raises(ConstraintViolation):
        (Boom & okdict).validate({"x": 1})

    # OR: first branch errors (non-CV) but is recoverable -> second branch is
    # still tried and passes (without the fix the KeyError escaped raw).
    assert (Boom | okdict).validate({"x": 1}) == {"x": 1}

    # XOR: erroring branch counts as a failed branch; exactly one passes -> ok.
    assert (Boom ^ okdict).validate({"x": 1}) == {"x": 1}


# =============================================================================
# 39 — not_one_of= with UNHASHABLE members builds (no raw TypeError) and still
#      enforces membership; mirrors one_of= tolerance.
# =============================================================================

def test_39_not_one_of_unhashable_members():
    c = Constraint.of(type=list, not_one_of=[[1], [2]])       # unhashable members -> must BUILD, not raise
    with pytest.raises(ConstraintViolation):
        c.validate([1])                            # forbidden
    c.validate([3])                                # allowed


# =============================================================================
# 40 — NOT composite describe() uses a single-child-appropriate header for a
#      multi-line child, never the misleading "NOT all of:".
# =============================================================================

def test_40_not_describe_single_child_header():
    class A(Constraint):
        a: int
    class B(Constraint):
        b: int
    desc = (~(A | B)).describe()                    # multi-line OR child under a NOT
    assert "NOT the following (must NOT hold):" in desc
    assert "NOT all of:" not in desc


# =============================================================================
# 41 — stacked exactly_one_of with field-name tuples that JOIN to the same
#      string keep BOTH rules: ("a_b","c") and ("a","b_c") both ->
#      "_exactly_one_of_a_b_c"; neither validator may shadow the other.
# =============================================================================

def test_41_exactly_one_of_colliding_names_both_enforced():
    class M(Constraint):
        a_b: Optional[int] = None
        c: Optional[int] = None
        a: Optional[int] = None
        b_c: Optional[int] = None

    M2 = Constraint.exactly_one_of("a", "b_c")(Constraint.exactly_one_of("a_b", "c")(M))
    M2.validate({"a_b": 1, "c": None, "a": 1, "b_c": None})        # both rules satisfied
    with pytest.raises(ConstraintViolation):
        M2.validate({"a_b": 1, "c": 2, "a": 1, "b_c": None})       # rule (a_b,c) violated
    with pytest.raises(ConstraintViolation):
        M2.validate({"a_b": 1, "c": None, "a": 1, "b_c": 2})       # rule (a,b_c) violated -> was DROPPED


# =============================================================================
# 42 — struct-only `&` auto-merge keeps BOTH parents' validators even when they
#      share a method name — model_validators and field_validators.
# =============================================================================

def test_42_struct_and_keeps_both_colliding_model_validators():
    class P(Constraint):
        a: int = 0
        @model_validator(mode="after")
        def _check(self):
            if self.a < 0:
                raise ValueError("P: a must be >= 0")
            return self

    class Q(Constraint):
        b: int = 0
        @model_validator(mode="after")
        def _check(self):                       # SAME method name as P._check
            if self.b < 0:
                raise ValueError("Q: b must be >= 0")
            return self

    M = P & Q
    assert getattr(M, "_constraint_is_composite", False) is False     # a real merged class
    inst = M.validate({"a": 1, "b": 1})
    assert isinstance(inst, P) and isinstance(inst, Q)                 # isinstance BOTH
    with pytest.raises(ConstraintViolation):
        M.validate({"a": -1, "b": 1})           # P's check
    with pytest.raises(ConstraintViolation):
        M.validate({"a": 1, "b": -1})           # Q's check -> was silently dropped pre-fix


def test_42b_struct_and_keeps_both_colliding_field_validators():
    from pydantic import field_validator

    class P(Constraint):
        a: int = 0
        @field_validator("a")
        @classmethod
        def _fv(cls, v):
            if v == 7:
                raise ValueError("P: no 7 in a")
            return v

    class Q(Constraint):
        b: int = 0
        @field_validator("b")
        @classmethod
        def _fv(cls, v):                        # SAME method name as P._fv
            if v == 9:
                raise ValueError("Q: no 9 in b")
            return v

    M = P & Q
    M.validate({"a": 1, "b": 1})
    with pytest.raises(ConstraintViolation):
        M.validate({"a": 7, "b": 1})            # P's field check
    with pytest.raises(ConstraintViolation):
        M.validate({"a": 1, "b": 9})            # Q's field check -> was silently dropped pre-fix


# =============================================================================
# 43 — struct-`&` auto-merge TIGHTENS conflicting field bounds (intersection),
#      not pydantic last-wins which could LOOSEN them. Order-independent.
# =============================================================================

def test_43_struct_and_tightens_numeric_bounds():
    class Lo(Constraint):
        x: int = Field(ge=10)
    class Hi(Constraint):
        x: int = Field(ge=0)
    for M in (Lo & Hi, Hi & Lo):                 # order must not matter
        M.validate({"x": 12})                    # satisfies both
        with pytest.raises(ConstraintViolation):
            M.validate({"x": 5})                 # >=0 but <10 -> A's bound rejects (no loosening)
        with pytest.raises(ConstraintViolation):
            M.validate({"x": -1})


def test_43b_struct_and_tightens_upper_bound():
    class A(Constraint):
        x: int = Field(le=50)
    class B(Constraint):
        x: int = Field(le=100)
    M = A & B
    M.validate({"x": 30})
    with pytest.raises(ConstraintViolation):
        M.validate({"x": 80})                    # <=100 but >50 -> rejected (tightest <=50)


def test_43c_struct_and_tightens_length_and_pattern():
    class A(Constraint):
        s: str = Field(min_length=5, pattern=r"^a")
    class B(Constraint):
        s: str = Field(min_length=2, pattern=r"z$")
    M = A & B
    M.validate({"s": "aaaaz"})                   # len>=5, starts 'a', ends 'z' -> all hold
    with pytest.raises(ConstraintViolation):
        M.validate({"s": "abc"})                 # len 3 < 5
    with pytest.raises(ConstraintViolation):
        M.validate({"s": "zzzzz"})               # doesn't start with 'a' (A's pattern dropped pre-fix)


def test_43d_struct_and_preserves_coercer_on_merged_field():
    """#17: a coercer (BeforeValidator) on a merged shared field must still
    TRANSFORM the value (threaded through), not just be validated and dropped."""
    from typing import Annotated
    from pydantic import BeforeValidator

    class A(Constraint):
        s: Annotated[str, BeforeValidator(lambda x: x.strip().lower())]
    class B(Constraint):
        s: str
    out = (A & B).validate({"s": "  HELLO  "})
    assert out.s == "hello"                      # coercion preserved through the merge


def test_43e_struct_and_arbitrary_typed_field_with_metadata():
    """#1 (round-3 regression): a shared field of an ARBITRARY (non-schema) type
    carrying metadata must still merge — _tighten builds its TypeAdapter with
    arbitrary_types_allowed (retry) instead of raising PydanticSchemaGenerationError
    out of the merge (which escaped _and entirely)."""
    from typing import Annotated
    from pydantic import AfterValidator

    class Arb:
        def __init__(self, v):
            self.v = v

    class A(Constraint):
        x: Annotated[Arb, AfterValidator(lambda v: v)]

    class B(Constraint):
        model_config = {"arbitrary_types_allowed": True}
        x: Arb

    M = A & B                                    # must NOT raise PydanticSchemaGenerationError
    out = M.validate({"x": Arb(7)})
    assert out.x.v == 7


def test_44_describe_one_shot_iterable_kwargs():
    """#7: factory kwargs passed as one-shot iterables (generators) must still
    describe() correctly and not crash — they're materialized before the describe
    snapshot, while the live validation path is unchanged."""
    # one_of from a generator: describe shows the options, not "one of []"
    c1 = Constraint.of(type=Literal["a", "b", "c"])
    d1 = c1.describe()
    assert "a" in d1 and "b" in d1 and "c" in d1 and "one of []" not in d1
    assert c1.validate("a") == "a"                          # live path intact
    with pytest.raises(ConstraintViolation):
        c1.validate("z")

    # coercers from a generator: describe must not crash on len()
    c2 = Constraint.of(type=str, coercers=(f for f in (str.strip, str.lower)))
    assert "2 coercer(s)" in c2.describe()
    assert c2.validate("  AB  ") == "ab"                    # both coercers applied

    # not_one_of from a generator: describe shows the forbidden values
    c3 = Constraint.of(type=int, not_one_of=(x for x in (1, 2)))
    assert "1" in c3.describe() and "2" in c3.describe()
    with pytest.raises(ConstraintViolation):
        c3.validate(1)


def test_45_unique_one_shot_generator_with_unhashable_dup():
    """#8: unique= over a one-shot generator with unhashable elements + a dup must
    still reject — the set fast path no longer half-consumes the generator before
    the unhashable fallback (materialized once, like sorted/monotonic)."""
    c = Constraint.of(type=object, unique=True)
    # generator yielding an unhashable duplicate ([1] twice)
    with pytest.raises(ConstraintViolation):
        c.validate(x for x in ([1], [2], [1]))
    # a clean generator passes
    assert c.validate(x for x in ([1], [2], [3])) is not None or True
    # hashable dup still rejected
    with pytest.raises(ConstraintViolation):
        c.validate(x for x in (1, 2, 1))


def test_46_json_schema_survives_describe_failure():
    """#6 (completeness): a describe() failure must NOT propagate out of
    json_schema() — natural_llm calls it to build response_format OUTSIDE its
    retry loop, so an escape there would abort the call. x-description is omitted."""
    c = Constraint.of(type=int, int_ge=0, description="d")

    def _boom(cls):
        raise RuntimeError("describe boom")

    c.describe = classmethod(_boom)            # force describe() to raise on this class
    with pytest.raises(Exception):
        c.describe()
    sch = c.json_schema()                      # must NOT raise
    assert "x-description" not in sch


def test_47_validate_wraps_raw_non_value_errors():
    """#R4-2: validate() funnels a raw non-ValueError into ConstraintViolation —
    a coercer raising on the wrong type (via _wrap_coercer), and a struct
    @model_validator raising KeyError (via the widened validate()). Closes the
    contract gap the predicate-only net left."""
    c1 = Constraint.of(type=str, coercer=str.lower)
    assert c1.validate("AB") == "ab"               # coercer transforms on success
    with pytest.raises(ConstraintViolation):
        c1.validate(5)                             # str.lower(5) -> TypeError -> CV

    class S(Constraint):
        x: int = 0
        @model_validator(mode="after")
        def _chk(self):
            raise KeyError("boom")                 # raw non-ValueError, pydantic won't wrap
    with pytest.raises(ConstraintViolation):
        S.validate({"x": 1})


def test_48_list_of_composite_element_validates():
    """#R4-3: list_of/dict_of of a COMPOSITE element runs the composite's own
    validate() per element (not pydantic against the composite's empty model)."""
    elem = Constraint.of(type=int, int_range=(0, 9)) | Constraint.of(type=str, str_pattern="^x")
    c = Constraint.of(type=list[elem])
    assert c.validate([5, "xyz"]) == [5, "xyz"]    # 5 -> int branch, 'xyz' -> str branch
    with pytest.raises(ConstraintViolation):
        c.validate([99])                           # out of range AND not ^x
    with pytest.raises(ConstraintViolation):
        c.validate([{}])                           # garbage the empty-model path used to ACCEPT


def test_49_not_one_of_hashable_members_allow_unhashable_value():
    """#R4-4: not_one_of with HASHABLE members must ALLOW an unhashable value
    (it can't equal a scalar) — symmetric with the unhashable-members path."""
    c = Constraint.of(type=object, not_one_of=[1, 2, 3])        # hashable members -> set
    c.validate([1])                                # unhashable non-member -> ALLOWED (was rejected)
    c.validate(9)                                  # hashable non-member -> allowed
    with pytest.raises(ConstraintViolation):
        c.validate(2)                              # genuine member -> rejected


def test_50_composite_json_schema_hoists_nested_defs():
    """#R5-5: a structural OR/XOR composite whose children reference NESTED models
    must hoist each child's `$defs` to the composite ROOT so the root-relative
    `$ref: #/$defs/<Model>` refs resolve. pydantic emits a child's defs at THAT
    child's root; nesting it under anyOf/oneOf/allOf/not strands the defs a level
    down and dangles the ref, so a strict structured-output backend (exactly what
    natural_llm hard-sets as response_format for an accepted structural composite)
    can't follow the schema. Composite analog of the factory $defs-hoist."""
    import json as _json

    class Inner(Constraint):
        x: int

    class HasInner(Constraint):
        inner: Inner                               # nested model -> $ref + $defs

    class Other(Constraint):
        n: int                                     # no nested model -> no $defs

    comp = HasInner | Other                        # OR -> anyOf composite (NOT auto-merged)
    sch = comp.json_schema()
    assert "anyOf" in sch
    assert "$defs" in sch and "Inner" in sch["$defs"]          # defs hoisted to the ROOT
    assert all("$defs" not in child for child in sch["anyOf"])  # not stranded in a child
    assert "#/$defs/Inner" in _json.dumps(sch)                 # the ref now resolves at root


def test_51_struct_and_shared_coercer_applied_once():
    """#R6-7: when A and B inherit the IDENTICAL coercing field from a common base,
    `A & B` must apply that field's coercer EXACTLY ONCE (as A alone does), not
    twice — a non-idempotent transform would otherwise return a wrong value (the
    `_intersect` merge threaded the value through one adapter per side)."""
    from typing import Annotated
    from pydantic import BeforeValidator

    class Base(Constraint):
        n: Annotated[int, BeforeValidator(lambda x: x + 1)]   # non-idempotent coercer

    class A(Base):
        pass

    class B(Base):
        pass

    assert A.validate({"n": 0}).n == 1            # coercer applied once
    assert (A & B).validate({"n": 0}).n == 1      # STILL once (was 2 under double-apply)
    assert (B & A).validate({"n": 0}).n == 1      # order-independent


def test_52_composite_list_element_threads_context():
    """#R6-1: a composite used as a list_of/dict_of ELEMENT must thread the outer
    `validate(context=)` into the per-element composite validate — parity with the
    top-level composite path, so a context-dependent @model_validator in a
    structural child of the composite actually sees the context."""
    class Bounded(Constraint):
        value: int

        @model_validator(mode="after")
        def _check(self, info):
            mx = info.context.get("max") if info.context else None
            if mx is not None and self.value > mx:
                raise ValueError(f"value {self.value} exceeds context max {mx}")
            return self

    class Other(Constraint):
        other: int

    Comp = Bounded | Other
    # top-level parity (already worked): context enforced.
    with pytest.raises(ConstraintViolation):
        Comp.validate({"value": 50}, context={"max": 10})
    # element-level: the context must now reach the per-element composite validate.
    LC = Constraint.of(type=list[Comp])
    with pytest.raises(ConstraintViolation):
        LC.validate([{"value": 50}], context={"max": 10})      # was ACCEPTED (context dropped)
    LC.validate([{"value": 5}], context={"max": 10})           # within the context bound -> ok


def test_53_struct_and_surfaces_intersected_bounds_in_schema():
    """#R6-2: a struct&struct merge's INTERSECTED numeric/length bound must stay
    VISIBLE in json_schema()/describe() (the sub-LLM's two steering channels), not
    be erased by the opaque enforcer — while enforcement remains exact."""
    class Lo(Constraint):
        score: int = Field(ge=10)

    class Hi(Constraint):
        score: int = Field(ge=0)

    M = Lo & Hi
    assert M.json_schema()["properties"]["score"].get("minimum") == 10   # tightest shown
    assert "ge=10" in M.describe()                       # and surfaced in the description
    M.validate({"score": 12})                            # enforcement still exact
    with pytest.raises(ConstraintViolation):
        M.validate({"score": 5})

    # length bound + TWO patterns: minLength shown; the two patterns must NOT make
    # the model build choke (they stay enforced by the AfterValidator only).
    class S1(Constraint):
        s: str = Field(min_length=5, pattern=r"^a")

    class S2(Constraint):
        s: str = Field(min_length=2, pattern=r"z$")

    MS = S1 & S2
    assert MS.json_schema()["properties"]["s"].get("minLength") == 5
    MS.validate({"s": "aaaaz"})
    with pytest.raises(ConstraintViolation):
        MS.validate({"s": "abc"})                        # len 3 < 5 (intersection enforced)

    # GATE: a side with a COERCER keeps the opaque form — the visible marker would
    # check the RAW input pre-coercion and wrongly reject. Enforcement (post-coerce)
    # stays correct and the coercer still runs once.
    from typing import Annotated
    from pydantic import BeforeValidator
    class C1(Constraint):
        n: Annotated[int, BeforeValidator(lambda x: x * 2), Field(ge=10)]

    class C2(Constraint):
        n: int

    MC = C1 & C2
    assert MC.validate({"n": 6}).n == 12                 # 6*2=12 >=10 (coercer applied, not raw-rejected)
    assert "minimum" not in MC.json_schema()["properties"]["n"]   # opaque (gated), no spurious raw bound


def test_54_struct_and_three_way_keeps_bound_visible():
    """#H1: a 3+-way AND-merge of pure-bound constraints must keep the intersected
    bound VISIBLE — the fix's own synthetic enforcer (left in a prior merge's
    metadata) must not trip the transformer-gate and hide the bound."""
    class A(Constraint):
        x: int = Field(ge=0)

    class B(Constraint):
        x: int = Field(ge=10)

    class C(Constraint):
        x: int = Field(ge=5)

    M = A & B & C
    assert M.json_schema()["properties"]["x"].get("minimum") == 10   # tightest, still shown
    assert "ge=10" in M.describe()
    M.validate({"x": 12})
    with pytest.raises(ConstraintViolation):
        M.validate({"x": 9})


def test_55_struct_and_shared_coercer_with_distinct_bounds_applies_once():
    """#H2/#H5: a coercer SHARED via a common base, co-existing with DISTINCT
    per-side bounds (or only one side adding a bound), must apply EXACTLY once —
    the dedup is by func identity, not whole-metadata-list equality."""
    from typing import Annotated
    from pydantic import BeforeValidator

    calls = []

    def add1(v):
        calls.append(v)
        return v + 1

    class Base(Constraint):
        n: Annotated[int, BeforeValidator(add1)]

    class A(Base):
        n: Annotated[int, BeforeValidator(add1)] = Field(ge=0)

    class B(Base):
        n: Annotated[int, BeforeValidator(add1)] = Field(le=100)

    calls.clear()
    out = (A & B).validate({"n": 5})
    assert out.n == 6 and len(calls) == 1                # once (was 7 / twice), bounds still enforced
    with pytest.raises(ConstraintViolation):
        (A & B).validate({"n": 100})                     # add1(100)=101 > le=100 -> intersection rejects
    # partial overlap: B2 adds no bound, just inherits the shared coercer
    class B2(Base):
        pass

    calls.clear()
    assert (A & B2).validate({"n": 5}).n == 6 and len(calls) == 1


def test_56_struct_and_surfaces_interval_and_len_bounds():
    """#H3/#H4: bounds declared via annotated_types Interval/Len (not Field(ge=...))
    must still surface in the merged schema, and a Len-vs-MinLen mix yields the
    COMPLETE intersected bound, not a misleading half."""
    from typing import Annotated
    import annotated_types as at

    class A(Constraint):
        age: Annotated[int, at.Interval(ge=10, le=100)]

    class B(Constraint):
        age: Annotated[int, at.Interval(ge=0, le=50)]

    sch = (A & B).json_schema()["properties"]["age"]
    assert sch.get("minimum") == 10 and sch.get("maximum") == 50   # both surfaced + intersected

    class S1(Constraint):
        s: Annotated[str, at.MinLen(5)]

    class S2(Constraint):
        s: Annotated[str, at.Len(2, 8)]

    ss = (S1 & S2).json_schema()["properties"]["s"]
    assert ss.get("minLength") == 5 and ss.get("maxLength") == 8    # COMPLETE, not a half-bound


def test_57_tighten_dedup_tolerates_raising_eq():
    """#H6: the merge dedup compares metadata by func IDENTITY, never `==`, so a
    field metadata object whose __eq__ raises cannot leak out of `&`."""
    from pydantic import BeforeValidator
    from pydantic.fields import FieldInfo
    from plm.constraint.algebra import _tighten_field_infos

    class Boom:
        def __eq__(self, other):
            raise RuntimeError("cannot compare Boom")
        __hash__ = None

    fn = lambda s: s + "X"                               # noqa: E731
    a = FieldInfo(annotation=str)
    a.metadata = [BeforeValidator(fn), Boom()]
    b = FieldInfo(annotation=str)
    b.metadata = [BeforeValidator(fn)]
    _tighten_field_infos(a, b)                           # must NOT raise RuntimeError from Boom.__eq__


def test_58_composite_json_schema_renames_colliding_defs():
    """#H7: two composite branches defining DIFFERENT models with the SAME class
    name must each resolve to their OWN shape (rename the loser), not first-wins
    into a schema that misdescribes one branch."""
    import pydantic as pd

    def make_L():
        class Item(pd.BaseModel):
            price: float

        class L(Constraint):
            v1: Item
        return L

    def make_R():
        class Item(pd.BaseModel):
            sku: str
            qty: int

        class R(Constraint):
            v2: Item
        return R

    sch = (make_L() | make_R()).json_schema()
    defs = sch["$defs"]
    assert "Item" in defs and any(n != "Item" for n in defs)        # both shapes coexist
    refmap = {}
    for branch in sch["anyOf"]:
        for fld, spec in branch.get("properties", {}).items():
            refmap[fld] = spec["$ref"].split("/")[-1]
    assert sorted(defs[refmap["v1"]].get("properties", {})) == ["price"]          # L's Item
    assert sorted(defs[refmap["v2"]].get("properties", {})) == ["qty", "sku"]     # R's Item


def test_59_validate_funnel_tolerates_raising_str():
    """#H8: validate() must surface a ConstraintViolation even when the underlying
    exception's own __str__ raises (the funnel builds the message defensively)."""
    class ExplodingStr(KeyError):
        def __str__(self):
            raise RuntimeError("str detonated")

    class StructBomb(Constraint):
        x: int

        @model_validator(mode="after")
        def _v(self):
            raise ExplodingStr("boom")

    with pytest.raises(ConstraintViolation):             # NOT a raw RuntimeError
        StructBomb.validate({"x": 1})


def test_60_struct_and_describe_surfaces_noncombinable_constraints():
    """#H3/#H4 follow-up: an auto-merged `A & B` field's describe() must tell the
    model about ALL its constraints upfront — not only the combinable bounds, but
    patterns, multiple_of, and custom validators (recovering @constraint
    descriptions), AND the bounds even when a validator gates the visible markers.
    Enforcement is unchanged."""
    from typing import Annotated
    from pydantic import Field, AfterValidator
    from plm.constraint import constraint

    @constraint("must read the same forwards and backwards")
    def _palindrome(v):
        if v != v[::-1]:
            raise ValueError("not a palindrome")
        return v

    class A(Constraint):
        name: Annotated[str, AfterValidator(_palindrome)] = Field(min_length=3, pattern="^[A-Z]")
        qty: int = Field(ge=0)

    class B(Constraint):
        name: str = Field(max_length=9)
        qty: int = Field(multiple_of=5)

    desc = (A & B).describe()
    # name: bounds (gated -> text) + pattern + the recovered @constraint description
    assert "min_length=3" in desc and "max_length=9" in desc
    assert "/^[A-Z]/" in desc
    assert "must read the same forwards and backwards" in desc
    # qty: visible bound marker + the multiple_of note
    assert "ge=0" in desc and "multiple of 5" in desc

    M = A & B
    M.validate({"name": "ABA", "qty": 10})               # all hold
    for bad in ({"name": "aba", "qty": 10},              # pattern ^[A-Z]
                {"name": "ABC", "qty": 10},              # not a palindrome
                {"name": "ABA", "qty": 7},               # qty not multiple of 5
                {"name": "AA", "qty": 10}):              # min_length 3
        with pytest.raises(ConstraintViolation):
            M.validate(bad)


# ===========================================================================
# Constraint.field(...) — transparent factory: flat field-embed + standalone +
# composable; and the type-completing kwargs across every built-in type.
# ===========================================================================


def test_61_field_embeds_flat_in_struct():
    class Order(Constraint):
        currency: Constraint.field(coercer=str.upper, type=Literal["USD", "EUR"])
        qty:      Constraint.field(type=int, int_range=(1, 1000))
    o = Order.validate({"currency": "usd", "qty": 10})
    assert o.currency == "USD" and type(o.currency) is str        # FLAT + coerced
    assert o.qty == 10 and type(o.qty) is int                     # not nested {value: ..}
    for bad in ({"currency": "yen", "qty": 10},                   # not in one_of
                {"currency": "USD", "qty": 0}):                   # below range
        with pytest.raises(ConstraintViolation):
            Order.validate(bad)


def test_62_field_standalone_parity_with_of():
    f = Constraint.field(type=int, int_range=(0, 9), description="a digit")
    o = Constraint.of(type=int, int_range=(0, 9), description="a digit")
    assert f.validate(5) == 5
    with pytest.raises(ConstraintViolation):
        f.validate(10)
    assert f.describe() == o.describe()                            # delegate to the factory
    assert f.json_schema() == o.json_schema()


def test_63_field_composes_like_a_constraint():
    nf = ~Constraint.field(type=Literal["banned"])
    assert nf.validate("ok") == "ok"
    with pytest.raises(ConstraintViolation):
        nf.validate("banned")

    class S(Constraint):
        n: int
    (S & Constraint.field(type=object, predicate=lambda d: None)).validate({"n": 1})


def test_64_field_describe_in_struct_is_complete():
    class Account(Constraint):
        email: Constraint.field(type=str, kind="email")
        role:  Constraint.field(type=Literal["admin", "user"])
        score: Constraint.field(type=int, int_range=(0, 100), multiple_of=5)
    d = Account.describe()
    assert "email (str)" in d and "kind='email'" in d
    assert "one of ['admin', 'user']" in d
    assert "a multiple of 5" in d
    js = Account.json_schema()["properties"]
    assert js["score"]["multipleOf"] == 5
    assert js["role"]["enum"] == ["admin", "user"]


def test_65_multiple_of_and_finite():
    import math
    for ctor in (Constraint.of, Constraint.field):
        c = ctor(type=int, multiple_of=5)
        assert c.validate(15) == 15
        with pytest.raises(ConstraintViolation):
            c.validate(7)
        fin = ctor(type=float, finite=True)
        assert fin.validate(1.5) == 1.5
        for bad in (math.inf, math.nan, -math.inf):
            with pytest.raises(ConstraintViolation):
                fin.validate(bad)


def test_66_tuple_set_frozenset_bytes():
    for ctor in (Constraint.of, Constraint.field):
        t = ctor(type=tuple[int, str])
        assert t.validate((1, "a")) == (1, "a")
        with pytest.raises(ConstraintViolation):
            t.validate((1, 2))
        assert ctor(type=tuple[int, ...]).validate((1, 2, 3)) == (1, 2, 3)   # homogeneous
        s = ctor(type=set[int])
        assert s.validate({1, 2}) == {1, 2}
        with pytest.raises(ConstraintViolation):
            s.validate({1, "a"})
        assert ctor(type=frozenset[str]).validate(frozenset({"a"})) == frozenset({"a"})
        b = ctor(type=bytes, bytes_length=(2, 4))
        assert b.validate(b"abc") == b"abc"
        with pytest.raises(ConstraintViolation):
            b.validate(b"x")


def test_67_string_and_collection_conveniences():
    for ctor in (Constraint.of, Constraint.field):
        assert ctor(type=str, str_prefix="AB").validate("ABC") == "ABC"
        with pytest.raises(ConstraintViolation):
            ctor(type=str, str_prefix="AB").validate("xyz")
        assert ctor(type=str, str_suffix="Z").validate("abZ") == "abZ"
        assert ctor(type=str, str_contains="@").validate("a@b") == "a@b"
        assert ctor(type=dict, dict_length=(1, 2)).validate({"a": 1}) == {"a": 1}
        with pytest.raises(ConstraintViolation):
            ctor(type=dict, dict_length=(1, 2)).validate({"a": 1, "b": 2, "c": 3})
        assert ctor(type=set, set_length=(1, 2)).validate({1, 2}) == {1, 2}
        assert ctor(type=list, subset_of=[1, 2, 3]).validate([1, 2]) == [1, 2]
        with pytest.raises(ConstraintViolation):
            ctor(type=list, subset_of=[1, 2, 3]).validate([1, 9])
        assert ctor(type=list, superset_of=[1, 2]).validate([1, 2, 3]) == [1, 2, 3]
        assert ctor(type=list, list_contains=7).validate([5, 7]) == [5, 7]
        with pytest.raises(ConstraintViolation):
            ctor(type=list, list_contains=7).validate([5, 9])


def test_68_complex_constraints():
    for ctor in (Constraint.of, Constraint.field):
        a = ctor(type=complex, abs_range=(0, 2))
        assert a.validate(1 + 1j) == 1 + 1j
        with pytest.raises(ConstraintViolation):
            a.validate(3 + 3j)
        r = ctor(type=complex, real_range=(0, 5))
        assert r.validate(3 + 9j) == 3 + 9j
        with pytest.raises(ConstraintViolation):
            r.validate(9 + 1j)
        im = ctor(type=complex, imag_range=(-1, 1))
        assert im.validate(5 + 0.5j) == 5 + 0.5j
        with pytest.raises(ConstraintViolation):
            im.validate(5 + 9j)


def test_69_new_kwargs_registered_and_conflict_checked():
    c = Constraint.of(type=int, int_range=(0, 100), multiple_of=10)  # both predicates on one type
    assert c.validate(50) == 50
    with pytest.raises(ConstraintViolation):
        c.validate(55)
    with pytest.raises(ValueError, match="no type coercer"):       # refinements, no type coercer
        Constraint.of(int_range=(0, 9), str_prefix="A")
    with pytest.raises(TypeError):                                 # removed sugar -> unknown kwarg
        Constraint.of(set_of=int)
    with pytest.raises(TypeError):                                 # unknown kwarg still rejected
        Constraint.of(nonsense=1)


def test_70_multiple_of_zero_rejected_at_build():
    """multiple_of=0 is nonsensical AND unsafe — int 0 panics pydantic-core's Rust
    validator (a raw panic escaping validate(), breaking the airtight invariant) and
    float 0.0 silently accepts everything. Both must be rejected at construction, for
    .of and .field alike."""
    for ctor in (Constraint.of, Constraint.field):
        for zero in (0, 0.0):
            with pytest.raises(ValueError, match="non-zero"):
                ctor(type=int, multiple_of=zero)
    # a valid multiple_of still works
    assert Constraint.of(type=int, multiple_of=3).validate(9) == 9


def test_71_field_usable_as_collection_element():
    """A .field is a strict superset of .of, so it must work as a list_of/set_of/
    dict_of element too (it has no `value` field, so _resolve_elem_type uses its
    stored flat annotation). The element constraint must actually apply."""
    LF = Constraint.of(type=list[Constraint.field(type=int, int_range=(0, 9))])
    assert LF.validate([1, 5, 9]) == [1, 5, 9]
    with pytest.raises(ConstraintViolation):
        LF.validate([1, 99])
    DF = Constraint.of(type=dict[str, Constraint.field(type=int, int_range=(0, 9))])
    assert DF.validate({"a": 5}) == {"a": 5}
    with pytest.raises(ConstraintViolation):
        DF.validate({"a": 50})
    # nested: .field(type=list[.field(...)])
    assert Constraint.field(type=list[Constraint.field(type=int, int_range=(0, 9))]).validate([3]) == [3]


def test_72_struct_and_merges_same_name_field_fields():
    """struct&struct AUTO-MERGE must work when overlapping fields are built with
    Constraint.field(...) — each .field mints a fresh wrapper class, so the merge
    must normalize them to their flat type and INTERSECT (not fall to the broken
    run-both composite, which for two structs rejects every input)."""
    class A(Constraint):
        code: Constraint.field(type=int, int_ge=0)
    class B(Constraint):
        code: Constraint.field(type=int, int_le=100)
    M = A & B
    assert getattr(M, "_constraint_is_composite", False) is False   # merged, not run-both
    assert M.validate({"code": 50}).code == 50
    for bad in ({"code": -1}, {"code": 101}):
        with pytest.raises(ConstraintViolation):
            M.validate(bad)
    # .field merges with a plain typed field of the same name too
    class D(Constraint):
        n: Constraint.field(type=int, int_ge=0)
    class E(Constraint):
        n: int = Field(le=9)
    assert (D & E).validate({"n": 5}).n == 5
    with pytest.raises(ConstraintViolation):
        (D & E).validate({"n": 50})
    # genuinely incompatible same-name field types fail LOUD at compose time
    class F(Constraint):
        code: Constraint.field(type=int, int_ge=0)
    class G(Constraint):
        code: Constraint.field(type=str, str_pattern="^x$")
    with pytest.raises(TypeError):
        F & G


def test_73_subset_superset_tolerate_unhashable_members():
    """subset_of/superset_of must not leak a raw TypeError on UNHASHABLE members
    (mirrors not_one_of) — they compare element-wise (==), not via set()."""
    for ctor in (Constraint.of, Constraint.field):
        c = ctor(type=list, subset_of=[[1], [2], [3]])          # unhashable members: builds cleanly
        assert c.validate([[1], [2]]) == [[1], [2]]
        with pytest.raises(ConstraintViolation):
            c.validate([[1], [9]])
        s = ctor(type=list, superset_of=[[1], [2]])
        assert s.validate([[1], [2], [3]]) == [[1], [2], [3]]
        with pytest.raises(ConstraintViolation):
            s.validate([[1]])
    # hashable members still work
    assert Constraint.of(type=list, subset_of=[1, 2, 3]).validate([1, 2]) == [1, 2]
    with pytest.raises(ConstraintViolation):
        Constraint.of(type=list, superset_of=[1, 2]).validate([1])


# =============================================================================
# predicates=[...] — plural list, mirroring coercers=[...]
# =============================================================================


def test_predicates_list_all_pass():
    """`predicates=[fn1, fn2]` where both pass -> value validates and is returned."""
    def positive(x):
        if x <= 0:
            raise ValueError("not positive")

    def even(x):
        if x % 2 != 0:
            raise ValueError("not even")

    c = Constraint.of(type=int, predicates=[positive, even])
    assert c.validate(4) == 4
    assert c.validate(10) == 10


def test_predicates_list_second_fails():
    """`predicates=[fn1, fn2]` where the 2nd fails -> ConstraintViolation raised."""
    def positive(x):
        if x <= 0:
            raise ValueError("not positive")

    def even(x):
        if x % 2 != 0:
            raise ValueError("not even")

    c = Constraint.of(type=int, predicates=[positive, even])
    with pytest.raises(ConstraintViolation):
        c.validate(3)  # positive passes, even fails


def test_predicate_and_predicates_compose():
    """`predicate=fn` AND `predicates=[...]` supplied together -> all enforced."""
    def positive(x):
        if x <= 0:
            raise ValueError("not positive")

    def even(x):
        if x % 2 != 0:
            raise ValueError("not even")

    def lt_100(x):
        if x >= 100:
            raise ValueError("not < 100")

    c = Constraint.of(type=int, predicate=positive, predicates=[even, lt_100])
    assert c.validate(4) == 4
    with pytest.raises(ConstraintViolation):
        c.validate(-2)  # fails positive (single)
    with pytest.raises(ConstraintViolation):
        c.validate(3)   # fails even (list[0])
    with pytest.raises(ConstraintViolation):
        c.validate(200) # fails lt_100 (list[1])


def test_predicates_list_non_value_error_surfaces_as_violation():
    """A predicate in the list raising a non-ValueError (e.g. TypeError) still
    surfaces as ConstraintViolation (consistent with _wrap_predicate)."""
    def boom(v):
        raise TypeError("wrong shape")

    c = Constraint.of(type=int, predicates=[boom])
    with pytest.raises(ConstraintViolation) as excinfo:
        c.validate(5)
    # _wrap_predicate re-raises non-ValueError as ValueError mentioning the
    # original exception type.
    assert "TypeError" in str(excinfo.value)


def test_predicates_empty_list_is_noop():
    """`predicates=[]` empty list -> no-op, value validates."""
    c = Constraint.of(type=object, predicates=[])
    assert c.validate(5) == 5
    assert c.validate("anything") == "anything"


def test_predicates_non_callable_rejected():
    """A non-callable in `predicates=` raises a clear TypeError at compose
    time, matching the `coercers=` behavior."""
    with pytest.raises(TypeError, match="predicates= contains a non-callable item"):
        Constraint.of(type=object, predicates=[lambda x: None, "not-a-fn"])
    with pytest.raises(TypeError, match="predicates= must be an iterable of callables"):
        Constraint.of(type=object, predicates=123)


def test_predicates_work_in_constraint_field():
    """`predicates=[...]` must work at the API level in BOTH Constraint.of and
    Constraint.field (the kwargs flow through the same _interpret_kwargs)."""
    def positive(x):
        if x <= 0:
            raise ValueError("not positive")

    def even(x):
        if x % 2 != 0:
            raise ValueError("not even")

    F = Constraint.field(type=int, predicates=[positive, even])
    assert F.validate(4) == 4
    with pytest.raises(ConstraintViolation):
        F.validate(3)

    # And as a struct field — embeds FLAT just like coercers=.
    class Container(Constraint):
        n: Constraint.field(type=int, predicates=[positive, even])

    inst = Container.validate({"n": 4})
    assert inst.n == 4
    with pytest.raises(ConstraintViolation):
        Container.validate({"n": 3})


def test_predicates_list_ordering_preserved():
    """Predicates in the list are ANDed in declaration order — verify by
    capturing the order in which they observe a value."""
    seen: list[str] = []

    def p1(v):
        seen.append("p1")

    def p2(v):
        seen.append("p2")

    def p3(v):
        seen.append("p3")

    c = Constraint.of(type=str, predicates=[p1, p2, p3])
    assert c.validate("x") == "x"
    assert seen == ["p1", "p2", "p3"]


def test_predicates_plus_coercers_compose():
    """coercers run BEFORE predicates — a predicate in the list must see the
    coerced value (mirrors test_coercer_then_predicate_ordering, plural side)."""
    def must_be_lowercase(s):
        if s != s.lower():
            raise ValueError("not lowercase")

    def must_be_short(s):
        if len(s) > 10:
            raise ValueError("too long")

    c = Constraint.of(
        type=str,
        coercers=[str.strip, str.lower],
        predicates=[must_be_lowercase, must_be_short],
    )
    assert c.validate("  HELLO  ") == "hello"


def test_74_list_contains_edge_cases():
    """Coverage for list_contains beyond the happy path (finding D): non-list
    input, empty list, None probe, unhashable probe, interaction with list_of,
    and describe()."""
    for ctor in (Constraint.of, Constraint.field):
        with pytest.raises(ConstraintViolation):           # non-list -> base_type=list rejects
            ctor(type=list, list_contains=5).validate(42)
        with pytest.raises(ConstraintViolation):           # empty list can't contain the probe
            ctor(type=list, list_contains="x").validate([])
        assert ctor(type=list, list_contains=None).validate([1, None, 3]) == [1, None, 3]   # None is a valid probe
        with pytest.raises(ConstraintViolation):
            ctor(type=list, list_contains=None).validate([1, 2, 3])
        # unhashable probe -> element equality, no hashing
        assert ctor(type=list, list_contains=[1, 2]).validate([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]
        with pytest.raises(ConstraintViolation):
            ctor(type=list, list_contains=[1, 2]).validate([[5, 6]])
        # with list_of: element constraint AND membership both enforced
        c = ctor(type=list[Constraint.field(type=int, int_range=(0, 9))], list_contains=5)
        assert c.validate([5, 7]) == [5, 7]
        with pytest.raises(ConstraintViolation):
            c.validate([1, 2])                             # no 5
        with pytest.raises(ConstraintViolation):
            c.validate([5, 99])                            # 99 out of element range
    assert "42" in Constraint.field(type=list, list_contains=42).describe()


def test_75_list_contains_composes_with_type_list():
    """Reframe: list_contains is a PREDICATE (no type) — it composes onto a type=list[...] coercer."""
    # list_contains (a predicate) composes with its list type coercer
    c = Constraint.field(type=list[int], list_length=(1, 3), list_contains=2)
    assert c.validate([1, 2]) == [1, 2]
    with pytest.raises(ConstraintViolation):
        c.validate([1, 3])                                 # no 2


def test_76_predicate_constraint_descriptions_surface_in_describe():
    """@constraint('...') tags on factory predicates (single + plural) are surfaced
    in describe() — matching the structural path; untagged predicates fall back to
    the 'N predicate(s) checked' count, and tags flow through a .field struct member."""
    @constraint("must be even")
    def is_even(n):
        if n % 2:
            raise ValueError("not even")

    @constraint("must not equal 42")
    def not_42(n):
        if n == 42:
            raise ValueError("is 42")

    def untagged(n):
        if n < 0:
            raise ValueError("neg")

    # single: tagged surfaces its text; untagged falls back
    assert "must be even" in Constraint.field(type=int, predicate=is_even).describe()
    assert Constraint.field(type=int, predicate=untagged).describe() == "int, 1 predicate(s) checked"

    # plural all tagged: each text, no count line
    d = Constraint.field(type=int, predicates=[is_even, not_42]).describe()
    assert "must be even" in d and "must not equal 42" in d and "predicate(s) checked" not in d

    # plural mixed: tagged surface individually, untagged collapse to a count
    d2 = Constraint.field(type=int, predicates=[is_even, untagged, not_42]).describe()
    assert "must be even" in d2 and "must not equal 42" in d2 and "1 predicate(s) checked" in d2

    # flows through a struct .field member
    class Form(Constraint):
        n: Constraint.field(type=int, int_range=(0, 100), predicates=[is_even])
    assert "must be even" in Form.describe()


def test_77_coercer_constraint_descriptions_render_as_transform():
    """@constraint('...') tags on coercers (single + plural) surface in describe(),
    but rendered as 'coerced (...)' — DISTINCT from a predicate's bare phrase — so
    coercing is distinguishable from testing. Untagged coercers fall back to count."""
    @constraint("strips whitespace")
    def trim(s):
        return s.strip()

    @constraint("uppercases")
    def up(s):
        return s.upper()

    def untagged(s):
        return s.lower()

    assert Constraint.field(type=str, coercer=trim).describe() == "str, coerced (strips whitespace)"
    assert Constraint.field(type=str, coercers=[trim, up]).describe() == "str, coerced (strips whitespace), coerced (uppercases)"
    # mixed: tagged render individually, untagged collapse to a count
    d = Constraint.field(type=str, coercers=[trim, untagged]).describe()
    assert "coerced (strips whitespace)" in d and "1 coercer(s) applied" in d
    # untagged-only stays a count
    assert Constraint.field(type=str, coercer=untagged).describe() == "str, 1 coercer(s) applied"

    # coerce vs test are visibly different in the SAME constraint
    @constraint("must be uppercase")
    def is_upper(s):
        if not s.isupper():
            raise ValueError("not upper")
    full = Constraint.field(type=str, coercer=up, predicate=is_upper).describe()
    assert "coerced (uppercases)" in full and "must be uppercase" in full


def test_78_constraint_tag_surfaces_in_any_decoration_order():
    """@constraint('...') must reach describe() regardless of decoration order —
    including placed OUTSIDE a @field_validator + @classmethod stack, the order that
    used to silently drop the description."""
    from pydantic import field_validator, model_validator

    class FVouter(Constraint):
        n: str
        @constraint("fv-outer")
        @field_validator("n")
        @classmethod
        def _v(cls, v):
            return v

    class FVinner(Constraint):
        n: str
        @field_validator("n")
        @classmethod
        @constraint("fv-inner")
        def _v(cls, v):
            return v

    class MV(Constraint):
        a: int
        @constraint("mv")
        @model_validator(mode="after")
        def _v(self):
            return self

    assert "fv-outer" in FVouter.describe()
    assert "fv-inner" in FVinner.describe()
    assert "mv" in MV.describe()


def test_79_describe_recurses_into_nested_struct_fields():
    """describe() expands nested structural Constraint fields inline (so the sub-LLM
    sees the full contract), unwraps Optional, and terminates on cycles."""
    class Leg(Constraint):
        symbol: Constraint.field(type=str, str_length=(1, 5))
        qty:    Constraint.field(type=int, int_range=(1, 100))

    class Order(Constraint):
        leg:  Leg                                    # nested structural
        opt:  Optional[Leg] = None                   # Optional nested -> still expanded
        note: str                                    # plain field (not expanded)

    d = Order.describe()
    assert "- leg (Leg):" in d
    assert "symbol (str)" in d and "qty (int)" in d  # inner fields surfaced
    assert "- opt (Leg):" in d                       # Optional unwrapped + expanded
    assert "    - symbol" in d                       # inner fields indented deeper

    # a self-referential constraint must TERMINATE (no infinite recursion)
    class Node(Constraint):
        val:  Constraint.field(type=int, int_ge=0)
        next: Optional["Node"] = None
    Node.model_rebuild()
    nd = Node.describe()
    assert "- next (Node)" in nd and "val (int)" in nd


def test_80_generator_subset_superset_survive_describe():
    """G1: a generator passed to subset_of/superset_of must be materialized before
    the describe() snapshot — else describe() re-reads an exhausted generator and
    drops the elements ('a subset of []'). Validation was always correct."""
    c = Constraint.field(type=list, subset_of=(x for x in [1, 2, 3]))
    assert "[1, 2, 3]" in c.describe()
    assert c.validate([1, 2]) == [1, 2]
    s = Constraint.field(type=list, superset_of=(x for x in [9, 8]))
    assert "[9, 8]" in s.describe()


def test_81_range_kwargs_reject_swapped_bounds_at_build():
    """Range guard: a (lo, hi) range kwarg with lo > hi is a typo that would reject
    every value — reject it at BUILD time (like multiple_of=0). Open/equal bounds ok."""
    for t, k, v in [(int, "int_range", (10, 1)), (float, "float_range", (1.0, 0.0)),
                    (str, "str_length", (5, 2)), (list, "list_length", (3, 1)),
                    (bytes, "bytes_length", (4, 2)), (complex, "abs_range", (2, 0)),
                    (complex, "real_range", (5, 0)), (complex, "imag_range", (1, -1))]:
        with pytest.raises(ValueError, match="lo > hi"):
            Constraint.field(**{"type": t, k: v})
    # valid / open / single-value bounds still build + validate
    assert Constraint.field(type=int, int_range=(0, 9)).validate(5) == 5
    assert Constraint.field(type=int, int_range=(0, None)).validate(999) == 999
    assert Constraint.field(type=int, int_range=(5, 5)).validate(5) == 5


def test_crash_restart_recipe_roundtrip():
    """to_recipe/from_recipe survive a constraint built by CALLS (Constraint.field /
    & | ^ ~) across a dill round-trip — the recipe is picklable (the dynamic class is
    NOT) and the rebuilt constraint validates IDENTICALLY. Structural subclasses return
    None (they pickle by value)."""
    import dill
    from plm.constraint import to_recipe, from_recipe

    def rejects(c, v):
        try:
            c.validate(v)
            return False
        except Exception:
            return True

    def roundtrip(c):
        r = to_recipe(c)
        assert r is not None
        r2 = dill.loads(dill.dumps(r))             # the recipe pickles (the dynamic class does NOT)
        return from_recipe(r2)

    A = Constraint.field(type=object, is_instance_of=int)
    B = Constraint.field(type=object, is_instance_of=int, predicate=lambda x: x > 0)
    cases = [
        (Constraint.field(type=object, is_instance_of=int, predicate=lambda x: x > 1000), 2000, 5),         # field + predicate
        (A & B, 5, -1),                                                                      # AND
        (A | B, 5, "str"),                                                                   # OR
        (~B, -1, 5),                                                                         # NOT
        (A & (B | Constraint.field(type=object, is_instance_of=int, predicate=lambda x: x == 0)), 5, -1),    # nested
    ]
    for c, good, bad in cases:
        c2 = roundtrip(c)
        c2.validate(good)                          # rebuilt accepts the good value
        assert rejects(c2, bad)                    # rebuilt rejects the bad value
        assert rejects(c, bad) == rejects(c2, bad)  # ... identically to the original

    # structural subclass -> reconstructed from its fields/defaults/bounds/validators
    # (pydantic classes can't cross-process dill-pickle, so the snapshot recipes them too).
    from pydantic import Field, model_validator

    class S(Constraint):
        name: str = "anon"
        n: int = Field(ge=0)

        @model_validator(mode="after")
        def _chk(self):
            if self.name == "BAD":
                raise ValueError("bad")
            return self

    rs = to_recipe(S)
    assert rs is not None and rs[0] == "structural"
    S2 = from_recipe(dill.loads(dill.dumps(rs)))    # recipe pickles; class rebuilt
    assert S2.validate({"n": 3}).name == "anon"     # default preserved
    assert rejects(S2, {"n": -1})                   # Field bound (ge=0) preserved
    assert rejects(S2, {"name": "BAD", "n": 0})     # validator preserved


def test_crash_restart_recipe_nested_constraints():
    """H1: a Constraint NESTED inside another (struct field typed as a Constraint, a
    `Constraint.field(...)`-typed field, a Constraint inside a factory kwarg, or one wrapped
    in Optional/List) is itself an unpicklable class — the recipe must recurse and store it
    as a NESTED recipe, not keep it live. Before the fix to_recipe kept the inner class:
    a by-reference struct (`addr: Address`) made dill `loads` fail (sinking the whole
    snapshot), and a `.field()`-typed annotation simply failed to pickle (silent drop)."""
    import dill
    from typing import Optional
    from plm.constraint import to_recipe, from_recipe

    def rejects(c, v):
        try:
            c.validate(v)
            return False
        except Exception:
            return True

    def roundtrip(c):
        r = to_recipe(c)
        assert r is not None
        dill.dumps(r)                                  # the recipe MUST pickle (the old bug: it could not)
        return from_recipe(dill.loads(dill.dumps(r)))

    # (a) struct field typed as another STRUCTURAL Constraint (was: by-ref -> sinks snapshot)
    class Address(Constraint):
        city: str

    class Person(Constraint):
        name: str
        addr: Address

    P2 = roundtrip(Person)
    assert P2.validate({"name": "x", "addr": {"city": "NYC"}}).addr.city == "NYC"
    assert rejects(P2, {"name": "x", "addr": {"city": 123}})   # nested rule still enforced

    # (b) `Constraint.field(...)`-typed field (the README's canonical Pattern-2)
    class Order(Constraint):
        currency: Constraint.field(type=Literal["USD", "EUR"])
        qty: Constraint.field(type=int, int_range=(1, 100))

    O2 = roundtrip(Order)
    g = O2.validate({"currency": "USD", "qty": 10})
    assert g.currency == "USD" and g.qty == 10
    assert rejects(O2, {"currency": "GBP", "qty": 10})         # one_of preserved
    assert rejects(O2, {"currency": "USD", "qty": 999})        # int_range preserved

    # (c) a Constraint nested inside a factory kwarg (type=list[Constraint.field(...)])
    N2 = roundtrip(Constraint.field(type=list[Constraint.field(type=int, int_range=(1, 5))]))
    assert N2.validate([1, 2, 3]) == [1, 2, 3]
    assert rejects(N2, [1, 99])

    # (d) a nested Constraint wrapped in a generic (Optional[Address])
    class Maybe(Constraint):
        who: Optional[Address] = None

    M2 = roundtrip(Maybe)
    assert M2.validate({}).who is None
    assert M2.validate({"who": {"city": "LA"}}).who.city == "LA"


def test_crash_restart_recipe_same_name_validators_survive():
    """H2: two validators that share a func __name__ must BOTH survive a recipe round-trip.
    Before the fix from_recipe keyed the rebuilt namespace by func.__name__, so a colliding
    name dropped one rule silently. Two reachable triggers: stacked @exactly_one_of (both
    inner funcs named `_check`) and an auto-merged `A & B` (both named `_ck`)."""
    import dill
    from typing import Optional
    from pydantic import model_validator
    from plm.constraint import to_recipe, from_recipe

    def rejects(c, v):
        try:
            c.validate(v)
            return False
        except Exception:
            return True

    def roundtrip(c):
        return from_recipe(dill.loads(dill.dumps(to_recipe(c))))

    # (1) stacked exactly_one_of — both inner validators are named `_check`
    @Constraint.exactly_one_of("a", "b")
    @Constraint.exactly_one_of("c", "d")
    class X(Constraint):
        a: Optional[int] = None
        b: Optional[int] = None
        c: Optional[int] = None
        d: Optional[int] = None

    X2 = roundtrip(X)
    assert not rejects(X2, {"a": 1, "c": 1})         # one of (a,b) + one of (c,d) -> ok
    assert rejects(X2, {"a": 1, "c": 1, "d": 1})     # (c,d) rule must STILL fire (was dropped)
    assert rejects(X2, {"a": 1, "b": 1, "c": 1})     # (a,b) rule must STILL fire

    # (2) auto-merged A & B — both model_validators are named `_ck`
    class A(Constraint):
        lo: int

        @model_validator(mode="after")
        def _ck(self):
            if self.lo < 0:
                raise ValueError("lo<0")
            return self

    class B(Constraint):
        hi: int

        @model_validator(mode="after")
        def _ck(self):
            if self.hi < 0:
                raise ValueError("hi<0")
            return self

    M2 = roundtrip(A & B)
    assert not rejects(M2, {"lo": 1, "hi": 1})
    assert rejects(M2, {"lo": -1, "hi": 1})          # A's rule survives
    assert rejects(M2, {"lo": 1, "hi": -1})          # B's rule survives (was dropped by the collision)


def test_crash_restart_recipe_field_fidelity():
    """H3 (universal): a structural recipe must reconstruct fields FAITHFULLY, not from a
    shallow read. The old code carried only annotation + bounds + the bare `fi.default`, so
    it silently lost `default_factory` (field -> REQUIRED), every other FieldInfo attr
    (alias/description/json_schema_extra/...), AND `model_config` (extra/frozen/...). The fix
    captures `fi._attributes_set` + metadata + `model_config`."""
    import dill
    from typing import Optional
    from pydantic import Field, ConfigDict
    from plm.constraint import to_recipe, from_recipe

    def rejects(c, v):
        try:
            c.validate(v)
            return False
        except Exception:
            return True

    class Rich(Constraint):
        model_config = ConfigDict(extra="forbid")
        req: int                                       # required, no default
        bare: int = 7                                  # bare default
        opt: Optional[int] = None
        tags: list = Field(default_factory=list)       # the H3 headline
        cur: int = Field(default=1, alias="currency", description="the currency")
        score: int = Field(ge=0, le=100, default=50)
        notes: str = Field(default="x", json_schema_extra={"hint": "yo"})

    R = from_recipe(dill.loads(dill.dumps(to_recipe(Rich))))
    base = {"req": 1, "currency": 2}                    # 'currency' is cur's alias
    g = R.validate(base)
    assert g.tags == []                                # default_factory -> [] (NOT required)
    assert g.bare == 7 and g.opt is None and g.score == 50
    assert g.cur == 2                                  # alias honored
    assert R.model_fields["cur"].description == "the currency"
    assert R.model_fields["notes"].json_schema_extra == {"hint": "yo"}
    assert rejects(R, {"currency": 2})                 # 'req' still required
    assert rejects(R, {**base, "score": 999})          # ge/le bound preserved
    assert rejects(R, {**base, "zzz": 9})              # model_config extra=forbid preserved


def test_crash_restart_recipe_generator_kwargs_survive():
    """H4: a one-shot iterable passed to Constraint.field (one_of/coercers/...) is consumed
    by the time field() snapshots its recipe kwargs. The facade must record INNER's
    already-MATERIALIZED kwargs, not its own raw (now-exhausted) generator — else
    from_recipe rebuilds with an empty sequence (crash on one_of, silent no-op on
    coercers/predicates)."""
    import dill
    from plm.constraint import to_recipe, from_recipe

    def rejects(c, v):
        try:
            c.validate(v)
            return False
        except Exception:
            return True

    def roundtrip(c):
        return from_recipe(dill.loads(dill.dumps(to_recipe(c))))

    # the recipe must carry the type= membership and survive a roundtrip
    c = Constraint.field(type=Literal["USD", "EUR"])
    assert to_recipe(c)[1] == {"type": Literal["USD", "EUR"]}     # recorded at snapshot time
    c2 = roundtrip(c)
    assert not rejects(c2, "USD") and rejects(c2, "GBP")

    # generator coercers: normalize-then-check must survive (a silent no-op if dropped)
    c3 = roundtrip(Constraint.field(coercers=(f for f in [str.strip, str.upper]), type=Literal["USD", "EUR"]))
    assert not rejects(c3, "  usd ")                          # coerced to "USD" -> accepted
    assert rejects(c3, "gbp")


def test_crash_restart_recipe_namespace_fidelity():
    """NC-6 + structural-recipe audit: the recipe must reconstruct the FULL user namespace.
    Before: validators were stored as BOUND classmethods (dragging the whole old class into
    the pickle AND binding the rebuilt validator to the DEAD class), and ClassVars + methods
    were dropped — so a `cls.MAX`-reading validator only 'worked' by reading the OLD class via
    the bound-method bug. Now validators are UNWRAPPED to plain functions and ClassVars/methods
    survive, so the validator binds to the NEW class."""
    import dill
    from typing import ClassVar
    from pydantic import field_validator
    from plm.constraint import to_recipe, from_recipe

    def rejects(c, v):
        try:
            c.validate(v)
            return False
        except Exception:
            return True

    class Acc(Constraint):
        MAX: ClassVar[int] = 100
        amount: int

        @field_validator("amount")
        @classmethod
        def _under_max(cls, v):
            if v > cls.MAX:                          # reads the ClassVar -> must bind to the NEW class
                raise ValueError("over max")
            return v

        def doubled(self):                           # a plain method -> must survive
            return self.amount * 2

    blob = dill.dumps(to_recipe(Acc))
    assert len(blob) < 2500, len(blob)               # no old-class drag (the bound classmethod was ~5 KB)
    A2 = from_recipe(dill.loads(blob))
    assert A2 is not Acc
    assert getattr(A2, "MAX", None) == 100           # ClassVar preserved
    assert A2.validate({"amount": 5}).doubled() == 10  # method preserved
    assert rejects(A2, {"amount": 200})              # validator runs against MAX
    A2.MAX = 10                                       # the validator binds to the NEW class...
    assert rejects(A2, {"amount": 50})               # 50 > new MAX(10) -> reject (was masked by old-class bind)
    assert not rejects(A2, {"amount": 5})


def test_nc34_facade_factory_kwargs_deepcopied():
    """NC3-4: a field facade's `_constraint_factory_kwargs` must DEEP-copy the inner's, not alias
    its mutable list values — mutating the facade's copy must not leak into the constraint."""
    from plm.constraint import Constraint
    c = Constraint.field(type=str, not_one_of=["USD", "EUR"])
    c._constraint_factory_kwargs["not_one_of"].append("GBP")
    assert "GBP" not in c.describe()


def test_f6_field_kwargs_copy_tolerates_non_deepcopyable_value():
    """F6: NC3-4's deepcopy must not turn an exotic, non-deepcopyable kwarg value into a
    construction-time crash. `_safe_copy_kwargs` falls back to the reference per-value — a crash is
    strictly worse than the rare residual aliasing for such a value."""
    import threading
    from plm.constraint import Constraint
    Constraint.field(type=object, not_one_of=[threading.Lock()])  # must NOT raise (old shallow dict didn't; bare deepcopy did)
    assert Constraint.field(type=Literal["USD", "EUR"]).validate("USD") == "USD"   # normal case still fine


def test_nc31_recursive_constraint_recipe_no_stack_overflow():
    """NC3-1: `to_recipe` on a self-recursive structural Constraint (a field annotation pointing
    back at the class) must NOT infinite-recurse. The cycle guard breaks it: the cyclic field keeps
    its live annotation (so the recipe is later dropped cleanly by the dumps-probe), but the call
    itself returns instead of blowing the stack with RecursionError."""
    from typing import Optional
    from plm.constraint import Constraint
    from plm.constraint.snapshot import to_recipe

    class Node(Constraint):
        val: int
        nxt: "Optional[Node]" = None
    Node.model_rebuild()
    r = to_recipe(Node)                              # returns without RecursionError
    assert r is None or r[0] == "structural"


def test_f7_recipe_cycle_guard_isolated_per_context():
    """F7: the recipe-recursion cycle guard is a per-context ContextVar, not a shared module set —
    so concurrent `parallel()` branches (each runs via `copy_context().run(...)` in a thread pool)
    recipe-ing the SAME class don't collide (a shared set would let one branch see another's
    mid-build id as a spurious cycle and drop a valid recipe). Mirrors parallel()'s exact model."""
    from contextvars import copy_context
    from concurrent.futures import ThreadPoolExecutor
    from plm.constraint import Constraint
    from plm.constraint.snapshot import to_recipe

    class Money(Constraint):
        amount: int

    def _run():
        return to_recipe(Money)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = [f.result() for f in [pool.submit(copy_context().run, _run) for _ in range(64)]]
    assert all(r and r[0] == "structural" for r in results)      # 0 spurious-None


def test_nc32_super_in_constraint_rejected_at_definition():
    """NC3-2 (resolved via the constraint contract): a Constraint is a FLAT structural validator
    (fields + validators + predicates), NOT an inheritance hierarchy. A method using bare super()
    is REJECTED at DEFINITION with a clear error — rather than failing mysteriously on a recipe
    round-trip / silently dropping on crash-restart. The library's own base/field/composite classes
    use no bare super(), so this only ever fires on misuse; normal constraints are unaffected."""
    from plm.constraint import Constraint
    from plm.constraint.snapshot import to_recipe, from_recipe

    with pytest.raises(TypeError, match="super"):
        class S(Constraint):
            x: int
            def who(self):
                return super().__repr_name__()      # zero-arg super -> rejected at the class statement

    with pytest.raises(TypeError, match="super"):    # F5: EXPLICIT super(C, self) is rejected too
        class E(Constraint):
            x: int
            def who(self):
                return super(Constraint, self).model_dump()

    # F5: a method whose NESTED helper does NOT call super is allowed (no false positive from the
    # old `__class__`-freevar check, which the bytecode-scan discriminator avoids).
    class Helped(Constraint):
        x: int
        def label(self):
            def _fmt(v):
                return f"x={v}"
            return _fmt(self.x)
    assert Helped(x=3).label() == "x=3"

    # F35: a method that reads an ATTRIBUTE/field named `super` (LOAD_ATTR, not the super BUILTIN)
    # must be ALLOWED — the bytecode scan only flags `super` loaded as a global/name/free var. A
    # `'super' in co_names` membership test wrongly rejected this (co_names also holds attr names).
    from pydantic import field_validator
    class HasSuperField(Constraint):
        node: object
        @field_validator("node")
        @classmethod
        def _v(cls, n):
            _ = getattr(n, "super", None)            # also exercise: a value with a `super` member
            return n
    assert HasSuperField(node=object()).node is not None

    class Money(Constraint):                         # a normal structural constraint is unaffected
        amount: int
    assert from_recipe(to_recipe(Money))(amount=5).amount == 5      # ...and still round-trips


def test_all_constraint_shapes_rehydrate_via_recipe():
    """EXHAUSTIVE: every constraint shape, all four ops (& | ^ ~) and their combinations, and
    every structural feature must round-trip through `to_recipe -> from_recipe` and BEHAVE
    identically. Proves this round's recipe rework missed no shape. (Cross-process survival —
    incl. lambda predicates + validators dill-pickling across the kernel socket — is covered by
    the kernel crash-restart integration tests.)"""
    from typing import ClassVar, Optional
    from pydantic import field_validator, BaseModel
    from plm.constraint.snapshot import to_recipe, from_recipe

    def rt(c):
        r = to_recipe(c)
        assert r is not None, "expected a recipe (not the dill fallback)"
        return from_recipe(r)

    def outcome(c, v):
        try:
            out = c.validate(v)
            return ("ok", out.model_dump() if isinstance(out, BaseModel) else out)
        except Exception as e:
            return ("err", type(e).__name__)

    of = Constraint.of
    cases = [
        ("int_range", of(type=int, int_range=(0, 9)), [5, -1, 10]),
        ("float_gt/lt", of(type=float, float_gt=0.0, float_lt=1.0), [0.5, 0.0, 1.0]),
        ("one_of", of(type=Literal["a", "b"]), ["a", "c"]),
        ("not_one_of", of(type=str, not_one_of=["x"]), ["y", "x"]),
        ("str_pattern+len", of(type=str, str_pattern=r"^[A-Z]+$", str_length=(2, 5)), ["ABC", "ab", "ABCDEF"]),
        ("list unique+sorted+nested", of(type=list[Constraint.field(type=int, int_range=(0, 9))], list_length=(1, 3),
                                         unique=True, sorted=True), [[1, 2, 3], [1, 1], [3, 2, 1], [1, 2, 99]]),
        ("dict_of nested", of(type=dict[str, Constraint.field(type=float, float_le=1.0)]), [{"a": 0.5}, {"a": 1.5}]),
        ("set_of", of(type=set[int]), [{1, 2}, {1, 2.5}]),
        ("tuple_of", of(type=tuple[int, ...]), [(1, 2), (1, "a")]),
        ("coercer", of(coercer=str.upper, type=Literal["AB"]), ["ab", "xy"]),
        ("coercers chain", of(coercers=[str.strip, str.upper], type=Literal["AB"]), [" ab ", "x"]),
        ("multiple_of", of(type=int, multiple_of=3), [9, 10]),
        ("str_prefix/suffix", of(type=str, str_prefix="a", str_suffix="z"), ["abz", "bz", "ab"]),
        ("is_instance_of", of(type=object, is_instance_of=int), [5, "5", 5.0, True]),   # strict survives rehydrate
    ]
    A, B, C = of(type=int, int_ge=0), of(type=int, int_le=10), of(type=int, not_one_of=[5])
    cases += [
        ("A & B", A & B, [3, -1, 11]), ("A | B", A | B, [3, -1, 11, 100]),
        ("A ^ B", A ^ B, [3, -1, 100]), ("~A", ~A, [-1, 3]),
        ("A & B & C", A & B & C, [3, 5, -1]), ("(A|B)&C", (A | B) & C, [3, 5, -1]),
        ("~(A&B)", ~(A & B), [3, -1]), ("A & ~C", A & ~C, [5, 3, -1]),
        ("(A^B)|C", (A ^ B) | C, [3, 100, 5]),
    ]

    class Basic(Constraint):
        name: str
        score: int = Field(ge=0, le=100)

    class WithFV(Constraint):
        x: int
        @field_validator("x")
        @classmethod
        def _pos(cls, v):
            if v <= 0:
                raise ValueError("pos")
            return v

    class WithMV(Constraint):
        a: int
        b: int
        @model_validator(mode="after")
        def _s(self):
            if self.a + self.b > 10:
                raise ValueError("sum")
            return self

    class WithCV(Constraint):
        MAX: ClassVar[int] = 7
        v: int
        @field_validator("v")
        @classmethod
        def _le(cls, val):
            if val > cls.MAX:
                raise ValueError("max")
            return val

    class WithOpt(Constraint):
        x: Optional[int] = None

    class Inner(Constraint):
        k: int = Field(ge=0)

    class Outer(Constraint):
        inner: Inner

    @Constraint.exactly_one_of("a", "b")
    class XOne(Constraint):
        a: Optional[int] = None
        b: Optional[int] = None

    class Lo(Constraint):
        x: int = Field(ge=10)

    class Hi(Constraint):
        x: int = Field(ge=0)

    cases += [
        ("struct Basic+Field", Basic, [{"name": "x", "score": 50}, {"name": "x", "score": 200}]),
        ("struct field_validator", WithFV, [{"x": 5}, {"x": -1}]),
        ("struct model_validator", WithMV, [{"a": 2, "b": 3}, {"a": 9, "b": 9}]),
        ("struct ClassVar+method", WithCV, [{"v": 5}, {"v": 8}]),
        ("struct Optional", WithOpt, [{"x": 1}, {}, {"x": None}]),
        ("struct nested constraint", Outer, [{"inner": {"k": 1}}, {"inner": {"k": -1}}]),
        ("struct exactly_one_of", XOne, [{"a": 1}, {"a": 1, "b": 2}, {}]),
        ("struct composite Lo&Hi", Lo & Hi, [{"x": 12}, {"x": 5}, {"x": -1}]),
    ]

    for label, c, probes in cases:
        assert (c.describe() or "").strip(), label   # every shape also DESCRIBES without crashing
        rebuilt = rt(c)
        for v in probes:
            assert outcome(c, v) == outcome(rebuilt, v), (label, v)
    assert len(cases) == 31                          # 14 scalar + 9 composite + 8 structural


def test_describe_robust_against_pathological_member():
    """NC-8: describe() must not crash on a class member whose descriptor `__get__` raises a
    non-AttributeError — the `dir(cls)` walk's `getattr`, and `get_description`'s `.wrapped` /
    `.__func__` descend, are both guarded. The bad member is skipped; the rest renders. And
    `get_description` is robust for EVERY caller (returns None, never raises)."""
    from typing import ClassVar
    from plm.constraint.decorator import get_description

    class Raising:
        def __get__(self, obj, owner):
            raise RuntimeError("boom-getter")

    class Bad(Constraint):
        x: int = 0
        sneaky: ClassVar = Raising()                 # descriptor that raises on class access

    d = Bad.describe()                               # must NOT raise
    assert isinstance(d, str) and "x" in d           # renders; the raising member is skipped
    assert get_description(Raising()) is None         # robust for any caller, never propagates


def test_is_instance_of_is_strict_no_coercion():
    """Reframe: `is_instance_of=T` is a PURE isinstance predicate (sets NO type, NO schema). Paired
    with `type=object` (which does NO coercion) it is a STRICT type check: for a builtin it rejects
    other types INCLUDING a bool for int (a bool is not a real int); for an arbitrary class it is a
    plain isinstance check; and there is NO promotion / NO build-from-dict / NO subclass flatten,
    because object passes the value through untouched and the predicate is the sole gate."""
    def of(**kw):                                     # every is_instance_of field gets type=object
        return Constraint.of(type=object, **kw)

    ci = of(is_instance_of=int)
    assert ci.validate(5) == 5
    for bad in ("5", 5.0, True):                      # string / float / bool all rejected (no coercion)
        with pytest.raises(ConstraintViolation):
            ci.validate(bad)

    cs = of(is_instance_of=str)
    assert cs.validate("x") == "x"
    with pytest.raises(ConstraintViolation):
        cs.validate(5)

    cb = of(is_instance_of=bool)
    assert cb.validate(True) is True
    with pytest.raises(ConstraintViolation):
        cb.validate(1)                               # an int is not a bool

    class W:                                          # arbitrary class -> isinstance
        pass
    w = W()
    cw = of(is_instance_of=W)
    assert cw.validate(w) is w
    with pytest.raises(ConstraintViolation):
        cw.validate("x")

    # No coercion: promotion / build-from-dict / subclass-flatten are all closed.
    with pytest.raises(ConstraintViolation):
        of(is_instance_of=float).validate(5)          # int is NOT promoted to float
    class P(Constraint):
        name: str
    with pytest.raises(ConstraintViolation):
        of(is_instance_of=P).validate({"name": "x"})  # a dict is NOT built into the model
    class MyInt(int):
        pass
    mi = MyInt(7)
    assert of(is_instance_of=int).validate(mi) is mi  # a subclass instance passes through UNCHANGED

    with pytest.raises(Exception):                    # old name removed
        of(instance_of=int)
    assert "strictly an instance of int" in of(is_instance_of=int).describe()


def test_is_instance_of_target_kinds_and_rejections():
    """L3: is_instance_of accepts a plain class, a STRUCT Constraint, or a TUPLE of those (Python's
    native type-union). It REJECTS a composite (`& | ^ ~` — that algebra is VALIDATION, not type
    membership; for 'A or B' use a tuple) and a `.field`/`.of` factory (a value is never an instance
    of the dynamic wrapper). describe() renders cleanly — no mangled wrapper name."""
    of = lambda **kw: Constraint.of(type=object, **kw)

    class Addr(Constraint):
        city: str
    class Person(Constraint):
        name: str

    # struct target: accepts a REAL instance, rejects a dict (use type= to build from a dict)
    addr = Addr.validate({"city": "NYC"})
    assert of(is_instance_of=Addr).validate(addr) is addr
    with pytest.raises(ConstraintViolation):
        of(is_instance_of=Addr).validate({"city": "NYC"})

    # tuple target = the native "instance of A OR B"
    either = of(is_instance_of=(Addr, Person))
    assert either.validate(addr) is addr
    assert either.validate(Person.validate({"name": "x"})) is not None
    with pytest.raises(ConstraintViolation):
        either.validate(123)
    assert "(Addr / Person)" in either.describe()                       # clean tuple render

    # composite target -> rejected at build, routed to the tuple / algebra-as-type
    with pytest.raises(TypeError, match="composite"):
        Constraint.field(type=object, is_instance_of=(Addr | Person))
    # .field / .of factory target -> rejected at build (not a Python instance type)
    with pytest.raises(TypeError, match="factory"):
        Constraint.field(type=object, is_instance_of=Constraint.field(type=int, int_range=(0, 9)))


def test_is_instance_of_json_schema_is_airtight():
    """Regression (was the H4 crash): `is_instance_of` is a pure predicate now (no
    is_instance_schema), so json_schema() must NEVER raise for ANY class — it degrades to
    an x-description, contributing no `type`. Covers builtins AND an arbitrary class."""
    class Weird:
        pass
    for target in (int, str, Weird):
        c = Constraint.of(type=object, is_instance_of=target)
        sch = c.json_schema()                          # must NOT raise (the old crash)
        assert isinstance(sch, dict)
        assert "type" not in sch                       # is_instance_of contributes NO type
        assert "x-description" in sch                  # the check surfaces here instead
        assert "instance of" in sch["x-description"]


def test_h5_composite_struct_field_with_nested_defs_resolves():
    """Regression (H5): a composite whose structural children carry NESTED model refs (so each
    child schema emits `$ref: #/$defs/<Model>` + a `$defs`) used as a DIRECT struct field used to
    crash `Parent.model_json_schema()` with `KeyError: '#/$defs/<Model>'` — pydantic owns the
    parent document's `$defs` and won't hoist the composite hook's injected one, so the ref dangled.
    The composite's json-schema hook now INLINES its `$defs`, making the field schema self-contained."""
    class City(Constraint):
        name: str
    class Leg(Constraint):
        origin: City                  # nested ref -> Leg schema uses $ref + $defs[City]
    class Hub(Constraint):
        center: City

    class Route(Constraint):
        stop: Leg | Hub

    sch = Route.model_json_schema()    # must NOT raise (was KeyError: '#/$defs/City')
    assert isinstance(sch, dict)
    # every $ref in the field schema resolves at the document root (or is inlined away)
    import json as _json
    txt = _json.dumps(sch["properties"]["stop"])
    top_defs = set(sch.get("$defs", {}))
    for ref in [r.split("/")[-1] for r in __import__("re").findall(r'"#/\$defs/([^"]+)"', txt)]:
        assert ref in top_defs, f"dangling $ref to {ref}"
    # validation still correct: either branch accepted, garbage rejected
    assert Route.validate({"stop": {"origin": {"name": "NYC"}}}).stop.origin.name == "NYC"
    assert Route.validate({"stop": {"center": {"name": "LA"}}}).stop.center.name == "LA"
    with pytest.raises(ConstraintViolation):
        Route.validate({"stop": {"nope": 1}})


def test_d1_composite_as_direct_struct_field():
    """D1: a composite (`A | B`, `&`, `^`, `~`) used as a DIRECT struct-field annotation must
    validate the field value through the composite's own `.validate()` — NOT against an empty-fields
    model (which rejected EVERY scalar and accepted garbage), and must `describe()` the OR/AND tree
    rather than leak the mangled internal class name. Fixed via the composite's
    `__get_pydantic_core_schema__` hook + a composite branch in `_struct_body`."""
    A = Constraint.field(type=int, int_ge=0)
    B = Constraint.field(type=str, str_pattern="^x")

    class S(Constraint):
        v: A | B

    # correct OR semantics on the field value
    assert S.validate({"v": 5}) is not None              # int >= 0 -> A
    assert S.validate({"v": "xyz"}) is not None           # matches ^x -> B
    for bad in ({"v": -1}, {"v": "nope"}):
        with pytest.raises(Exception):
            S.validate(bad)

    # describe renders the tree, not `_OrConstraint__...`
    d = S.describe()
    assert "OR" in d and "_OrConstraint" not in d and "_FieldConstraint" not in d, d
    # json-schema generation over a composite field must not crash
    S.model_json_schema()

    # structural children as a direct field work too (accept either shape, reject garbage)
    class P(Constraint):
        x: int

    class Q(Constraint):
        y: str

    class T(Constraint):
        obj: P | Q

    assert T.validate({"obj": {"x": 1}}) is not None
    assert T.validate({"obj": {"y": "hi"}}) is not None
    with pytest.raises(Exception):
        T.validate({"obj": {"z": 9}})
    assert "_OrConstraint" not in T.describe()

    # top-level composite use is unaffected
    assert (A | B).validate(5) == 5


def test_algebra_rejects_non_constraint_operands():
    """Constraints have NO nullable/Optional: a field always satisfies its constraint. So a bare
    non-Constraint operand to the algebra (`A | None`, `A & 5`, `A ^ "x"`) — a model's likely way of
    fumbling 'optional' — must FAIL LOUD at composition, not silently build a dead 'branch' that
    quietly rejects everything (a D1-class trap). Legitimate Constraint-only algebra is unaffected."""
    A = Constraint.field(type=int, int_ge=0)
    B = Constraint.field(type=str, str_pattern="^x")
    for bad in (lambda: A | None, lambda: None | A, lambda: A & 5, lambda: A ^ "x", lambda: A & None):
        with pytest.raises(TypeError, match="combines Constraints"):
            bad()
    # L2: the bare ABSTRACT base Constraint is also rejected — `A & Constraint` would reject
    # everything and `A | Constraint` is a dead branch; both are silent traps.
    for bad in (lambda: A | Constraint, lambda: Constraint & A, lambda: A ^ Constraint):
        with pytest.raises(TypeError, match="bare base"):
            bad()
    # constraint-only algebra still composes
    assert (A | B).validate(5) == 5
    assert (A | B).validate("xyz") == "xyz"
    assert (A & Constraint.field(type=int, int_le=10)).validate(5) == 5
    assert (A ^ B).validate(5) == 5
    assert (~A).validate("not-an-int") == "not-an-int"
