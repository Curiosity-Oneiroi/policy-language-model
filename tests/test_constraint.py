"""Test suite for plm.constraint — covers structural + factory authoring, JSON Schema export,
context-passing, the @constraint decorator, and the kind registry."""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import model_validator

from plm.constraint import (
    Constraint,
    ConstraintViolation,
    REGISTRY,
    constraint,
)


# =============================================================================
# 1 — structural subclass validates a struct
# =============================================================================


def test_subclass_validates_struct():
    class MyAnswer(Constraint):
        score: Constraint.field(type=int, int_range=(0, 100))
        label: Constraint.field(type=str)

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


def test_registry_is_read_only_custom_checks_use_predicate():
    # The kind= table is a FIXED set of built-ins and is READ-ONLY (a MappingProxyType): runtime
    # writes (item-set, deletion) are refused, so the shared table is never mutated at runtime —
    # no mutable global to race on in a parallel() branch, and no runtime-kind crash-restart hole.
    # A built-in kind works via kind=/from_kind; a CUSTOM check is authored inline with predicate=fn
    # (same effect, nothing global to manage, and it survives crash-restart).
    with pytest.raises(TypeError):
        REGISTRY["_custom"] = lambda v: None
    with pytest.raises(TypeError):
        del REGISTRY["email"]
    with pytest.raises(TypeError):
        Constraint.registry["_custom"] = lambda v: None
    assert "_custom" not in REGISTRY and "email" in REGISTRY

    # a built-in kind still works:
    c = Constraint.from_kind("email")
    assert c.validate("a@b.co") == "a@b.co"
    with pytest.raises(ConstraintViolation):
        c.validate("nope")

    # the way to do a CUSTOM check is an inline predicate (crash-restart-safe):
    def _is_palindrome(s):
        if not isinstance(s, str) or s != s[::-1]:
            raise ValueError("not a palindrome")
    p = Constraint.field(type=str, predicate=_is_palindrome, description="a palindrome")
    assert p.validate("racecar") == "racecar"
    with pytest.raises(ConstraintViolation):
        p.validate("hello")


# =============================================================================
# 18 — describe() renders a struct subclass
# =============================================================================


def test_describe_renders_struct_subclass():
    class Q(Constraint):
        score: Constraint.field(type=int, int_range=(0, 100), description="confidence")
        label: Constraint.field(type=Literal["yes", "no"])

        @constraint("score must be even")
        @model_validator(mode="after")
        def _even(self):
            if self.score % 2 != 0:
                raise ValueError("odd")
            return self

    desc = Q.describe()
    assert "score" in desc and "label" in desc
    assert "0" in desc and "100" in desc
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
# 21 — @constraint decorator description surfaces
# =============================================================================


def test_cross_field_validator_with_decorator():
    class Order(Constraint):
        limit: Constraint.field(type=float)
        stop: Constraint.field(type=float)

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
        value:    Constraint.field(type=int, int_ge=0)
        children: Constraint.field(type=list["Node"])

    Node.model_rebuild()
    tree = Node.validate(
        {"value": 0, "children": [{"value": 1, "children": [{"value": 2, "children": []}]}]}
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
        value: Constraint.field(type=int)

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
        side: Constraint.field(type=Literal["buy", "sell"])
        qty:  Constraint.field(type=int, int_gt=0, int_le=100)

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
    predicates errors; a nullable / union type errors (alternatives go through a @model_validator,
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
        x: Constraint.field(type=int)

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
    (point at `type=`)."""
    import typing
    with pytest.raises(ValueError) as e1:
        Constraint.of(int_range=(0, 9))
    assert "no type coercer" in str(e1.value) and "type=" in str(e1.value)
    with pytest.raises(ValueError) as e2:
        Constraint.of(type=typing.Optional[int])
    assert "nullable" in str(e2.value)


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


def test_all_checks_described_walks_tree():
    """`_all_checks_described` (llm's M12 gate, which replaced `_contains_factory`) detects an
    UNDESCRIBED user check in a factory's predicate/coercer OR a struct field; built-ins and described
    user checks pass."""
    from plm.constraint.base import _all_checks_described
    from plm.constraint import constraint

    class S(Constraint):
        x: Constraint.field(type=int)
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

    class HasField(Constraint):
        f: Constraint.field(type=int, predicate=lambda v: v > 0)  # undescribed in a struct FIELD
    assert _all_checks_described(HasField) is False               # the old struct-field blind spot, caught


def test_json_schema_factory_preserves_defs_for_nested_model():
    """C22: flattening a factory value schema must carry $defs so nested-model
    $refs still resolve."""
    import json as _json

    class Inner(Constraint):
        x: Constraint.field(type=int)
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
# 39 — not_one_of= with UNHASHABLE members builds (no raw TypeError) and still
#      enforces membership; mirrors one_of= tolerance.
# =============================================================================

def test_39_not_one_of_unhashable_members():
    c = Constraint.of(type=list, not_one_of=[[1], [2]])       # unhashable members -> must BUILD, not raise
    with pytest.raises(ConstraintViolation):
        c.validate([1])                            # forbidden
    c.validate([3])                                # allowed


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
    json_schema() — llm calls it to build response_format OUTSIDE its
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
        x: Constraint.field(type=int)
        @model_validator(mode="after")
        def _chk(self):
            raise KeyError("boom")                 # raw non-ValueError, pydantic won't wrap
    with pytest.raises(ConstraintViolation):
        S.validate({"x": 1})


def test_49_not_one_of_hashable_members_allow_unhashable_value():
    """#R4-4: not_one_of with HASHABLE members must ALLOW an unhashable value
    (it can't equal a scalar) — symmetric with the unhashable-members path."""
    c = Constraint.of(type=object, not_one_of=[1, 2, 3])        # hashable members -> set
    c.validate([1])                                # unhashable non-member -> ALLOWED (was rejected)
    c.validate(9)                                  # hashable non-member -> allowed
    with pytest.raises(ConstraintViolation):
        c.validate(2)                              # genuine member -> rejected


def test_59_validate_funnel_tolerates_raising_str():
    """#H8: validate() must surface a ConstraintViolation even when the underlying
    exception's own __str__ raises (the funnel builds the message defensively)."""
    class ExplodingStr(KeyError):
        def __str__(self):
            raise RuntimeError("str detonated")

    class StructBomb(Constraint):
        x: Constraint.field(type=int)

        @model_validator(mode="after")
        def _v(self):
            raise ExplodingStr("boom")

    with pytest.raises(ConstraintViolation):             # NOT a raw RuntimeError
        StructBomb.validate({"x": 1})


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
        n: Constraint.field(type=str)
        @constraint("fv-outer")
        @field_validator("n")
        @classmethod
        def _v(cls, v):
            return v

    class FVinner(Constraint):
        n: Constraint.field(type=str)
        @field_validator("n")
        @classmethod
        @constraint("fv-inner")
        def _v(cls, v):
            return v

    class MV(Constraint):
        a: Constraint.field(type=int)
        @constraint("mv")
        @model_validator(mode="after")
        def _v(self):
            return self

    assert "fv-outer" in FVouter.describe()
    assert "fv-inner" in FVinner.describe()
    assert "mv" in MV.describe()


def test_79_describe_recurses_into_nested_struct_fields():
    """describe() expands nested structural Constraint fields inline (so the sub-LLM
    sees the full contract) and terminates on cycles (self-reference)."""
    class Leg(Constraint):
        symbol: Constraint.field(type=str, str_length=(1, 5))
        qty:    Constraint.field(type=int, int_range=(1, 100))

    class Order(Constraint):
        leg:  Constraint.field(type=Leg)             # nested structural -> expanded inline
        note: Constraint.field(type=str)             # plain field (not expanded)

    d = Order.describe()
    assert "leg (Leg)" in d
    assert "symbol (str)" in d and "qty (int)" in d  # nested inner fields surfaced inline

    # a self-referential (tree) constraint must TERMINATE describe (no infinite recursion)
    class Node(Constraint):
        val:      Constraint.field(type=int, int_ge=0)
        children: Constraint.field(type=list["Node"])
    Node.model_rebuild()
    nd = Node.describe()
    assert "val (int)" in nd and "children" in nd


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
    """to_recipe/from_recipe survive a constraint built by a CALL (Constraint.field) across a dill
    round-trip — the recipe is picklable (the dynamic class is NOT) and the rebuilt constraint
    validates IDENTICALLY. A structural subclass is recipe'd from its fields/validators too."""
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

    cases = [
        (Constraint.field(type=object, is_instance_of=int, predicate=lambda x: x > 1000), 2000, 5),         # field + predicate
    ]
    for c, good, bad in cases:
        c2 = roundtrip(c)
        c2.validate(good)                          # rebuilt accepts the good value
        assert rejects(c2, bad)                    # rebuilt rejects the bad value
        assert rejects(c, bad) == rejects(c2, bad)  # ... identically to the original

    # structural subclass -> reconstructed from its fields/bounds/validators
    # (pydantic classes can't cross-process dill-pickle, so the snapshot recipes them too).
    from pydantic import model_validator

    class S(Constraint):
        name: Constraint.field(type=str)
        n:    Constraint.field(type=int, int_ge=0)

        @model_validator(mode="after")
        def _chk(self):
            if self.name == "BAD":
                raise ValueError("bad")
            return self

    rs = to_recipe(S)
    assert rs is not None and rs[0] == "structural"
    S2 = from_recipe(dill.loads(dill.dumps(rs)))    # recipe pickles; class rebuilt
    assert S2.validate({"name": "ok", "n": 3}).name == "ok"   # fields rebuilt
    assert rejects(S2, {"name": "ok", "n": -1})     # int_ge bound preserved
    assert rejects(S2, {"name": "BAD", "n": 0})     # validator preserved


def test_crash_restart_recipe_nested_constraints():
    """H1: a Constraint NESTED inside another (struct field typed as a Constraint, a
    `Constraint.field(...)`-typed field, a Constraint inside a factory kwarg, or one wrapped
    in Optional/List) is itself an unpicklable class — the recipe must recurse and store it
    as a NESTED recipe, not keep it live. Before the fix to_recipe kept the inner class:
    a by-reference struct (`addr: Address`) made dill `loads` fail (sinking the whole
    snapshot), and a `.field()`-typed annotation simply failed to pickle (silent drop)."""
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
        dill.dumps(r)                                  # the recipe MUST pickle (the old bug: it could not)
        return from_recipe(dill.loads(dill.dumps(r)))

    # (a) struct field typed as another STRUCTURAL Constraint (was: by-ref -> sinks snapshot)
    class Address(Constraint):
        city: Constraint.field(type=str)

    class Person(Constraint):
        name: Constraint.field(type=str)
        addr: Constraint.field(type=Address)

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


def test_crash_restart_recipe_field_fidelity():
    """A structural recipe reconstructs fields FAITHFULLY: every field is a required
    Constraint.field (no defaults/Optional/alias under the .field-only model), the bounds
    are preserved, and a class-level `model_config` (e.g. extra='forbid') survives the recipe."""
    import dill
    from pydantic import ConfigDict
    from plm.constraint import to_recipe, from_recipe

    def rejects(c, v):
        try:
            c.validate(v)
            return False
        except Exception:
            return True

    class Rich(Constraint):
        model_config = ConfigDict(extra="forbid")
        req:   Constraint.field(type=int)
        score: Constraint.field(type=int, int_range=(0, 100))

    R = from_recipe(dill.loads(dill.dumps(to_recipe(Rich))))
    g = R.validate({"req": 1, "score": 50})
    assert g.req == 1 and g.score == 50                # fields rebuilt, both required
    assert rejects(R, {"score": 50})                   # 'req' still required
    assert rejects(R, {"req": 1, "score": 999})        # int_range bound preserved
    assert rejects(R, {"req": 1, "score": 50, "zzz": 9})  # model_config extra=forbid preserved


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
        amount: Constraint.field(type=int)

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
    from plm.constraint import Constraint
    from plm.constraint.snapshot import to_recipe

    class Node(Constraint):
        val:      Constraint.field(type=int)
        children: Constraint.field(type=list["Node"])
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
        amount: Constraint.field(type=int)

    def _run():
        return to_recipe(Money)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = [f.result() for f in [pool.submit(copy_context().run, _run) for _ in range(64)]]
    assert all(r and r[0] == "structural" for r in results)      # 0 spurious-None


def test_nc32_super_in_constraint_rejected_at_definition():
    """NC3-2 (resolved via the constraint contract): a Constraint is a FLAT structural validator
    (fields + validators + predicates), NOT an inheritance hierarchy. A method using bare super()
    is REJECTED at DEFINITION with a clear error — rather than failing mysteriously on a recipe
    round-trip / silently dropping on crash-restart. The library's own base/field classes
    use no bare super(), so this only ever fires on misuse; normal constraints are unaffected."""
    from plm.constraint import Constraint
    from plm.constraint.snapshot import to_recipe, from_recipe

    with pytest.raises(TypeError, match="super"):
        class S(Constraint):
            x: Constraint.field(type=int)
            def who(self):
                return super().__repr_name__()      # zero-arg super -> rejected at the class statement

    with pytest.raises(TypeError, match="super"):    # F5: EXPLICIT super(C, self) is rejected too
        class E(Constraint):
            x: Constraint.field(type=int)
            def who(self):
                return super(Constraint, self).model_dump()

    # F5: a method whose NESTED helper does NOT call super is allowed (no false positive from the
    # old `__class__`-freevar check, which the bytecode-scan discriminator avoids).
    class Helped(Constraint):
        x: Constraint.field(type=int)
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
        node: Constraint.field(type=object)
        @field_validator("node")
        @classmethod
        def _v(cls, n):
            _ = getattr(n, "super", None)            # also exercise: a value with a `super` member
            return n
    assert HasSuperField(node=object()).node is not None

    class Money(Constraint):                         # a normal structural constraint is unaffected
        amount: Constraint.field(type=int)
    assert from_recipe(to_recipe(Money))(amount=5).amount == 5      # ...and still round-trips


def test_all_constraint_shapes_rehydrate_via_recipe():
    """EXHAUSTIVE: every constraint shape and every structural feature must round-trip through
    `to_recipe -> from_recipe` and BEHAVE identically. Proves this round's recipe rework missed no
    shape. (Cross-process survival —
    incl. lambda predicates + validators dill-pickling across the kernel socket — is covered by
    the kernel crash-restart integration tests.)"""
    from typing import ClassVar
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
        ("kind=email", of(type=str, kind="email"), ["a@b.co", "nope"]),                 # built-in kind re-resolves post-rehydrate
    ]
    class Basic(Constraint):
        name:  Constraint.field(type=str)
        score: Constraint.field(type=int, int_range=(0, 100))

    class WithFV(Constraint):
        x: Constraint.field(type=int)
        @field_validator("x")
        @classmethod
        def _pos(cls, v):
            if v <= 0:
                raise ValueError("pos")
            return v

    class WithMV(Constraint):
        a: Constraint.field(type=int)
        b: Constraint.field(type=int)
        @model_validator(mode="after")
        def _s(self):
            if self.a + self.b > 10:
                raise ValueError("sum")
            return self

    class WithCV(Constraint):
        MAX: ClassVar[int] = 7
        v: Constraint.field(type=int)
        @field_validator("v")
        @classmethod
        def _le(cls, val):
            if val > cls.MAX:
                raise ValueError("max")
            return val

    class Inner(Constraint):
        k: Constraint.field(type=int, int_ge=0)

    class Outer(Constraint):
        inner: Constraint.field(type=Inner)

    cases += [
        ("struct Basic+Field", Basic, [{"name": "x", "score": 50}, {"name": "x", "score": 200}]),
        ("struct field_validator", WithFV, [{"x": 5}, {"x": -1}]),
        ("struct model_validator", WithMV, [{"a": 2, "b": 3}, {"a": 9, "b": 9}]),
        ("struct ClassVar+method", WithCV, [{"v": 5}, {"v": 8}]),
        ("struct nested constraint", Outer, [{"inner": {"k": 1}}, {"inner": {"k": -1}}]),
    ]

    for label, c, probes in cases:
        assert (c.describe() or "").strip(), label   # every shape also DESCRIBES without crashing
        rebuilt = rt(c)
        for v in probes:
            assert outcome(c, v) == outcome(rebuilt, v), (label, v)
    assert len(cases) == 20                          # 15 scalar + 5 structural


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
        x: Constraint.field(type=int)
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
        name: Constraint.field(type=str)
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
    native type-union). It REJECTS a `.field`/`.of` factory (a value is never an instance of the
    dynamic wrapper). describe() renders cleanly — no mangled wrapper name."""
    of = lambda **kw: Constraint.of(type=object, **kw)

    class Addr(Constraint):
        city: Constraint.field(type=str)
    class Person(Constraint):
        name: Constraint.field(type=str)

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


def test_structural_json_schema_airtight_for_arbitrary_field():
    """(A) Companion to the factory airtight test: a STRUCTURAL class with a field typed as a
    non-renderable arbitrary class must NOT crash json_schema() — the structural branch degrades
    to {} + x-description like the factory branch (llm builds response_format from it
    OUTSIDE its retry loop, so an escape there would abort the call)."""
    class Foo:                                         # arbitrary, non-JSON-renderable
        pass
    class Holder(Constraint):
        f: Constraint.field(type=Foo)
    sch = Holder.json_schema()                         # was PydanticInvalidForJsonSchema; must NOT raise
    assert isinstance(sch, dict) and "x-description" in sch
    assert Holder.describe()                           # describe stays airtight on this shape too


def test_structural_fields_must_be_dot_field_not_bare():
    """(B) The .field-only authoring rule is ENFORCED at class definition: a bare primitive, a bare
    arbitrary class, a bare nested Constraint, AND a .field carrying a default (or default_factory)
    are each rejected with a clear TypeError naming the field — every field is a REQUIRED
    Constraint.field. An all-.field struct (incl. a nested struct via type=) builds + validates."""
    from pydantic import Field as _PydField
    class Leg(Constraint):
        qty: Constraint.field(type=int)
    class Foo:
        pass

    with pytest.raises(TypeError, match="Constraint.field"):
        class _Bare(Constraint):
            v: int                                     # bare primitive
    with pytest.raises(TypeError, match="Constraint.field"):
        class _Arb(Constraint):
            f: Foo                                     # bare arbitrary class
    with pytest.raises(TypeError, match="Constraint.field"):
        class _Nest(Constraint):
            leg: Leg                                   # bare nested Constraint -> must be type=Leg
    with pytest.raises(TypeError, match="default"):
        class _Def(Constraint):
            v: Constraint.field(type=int) = 3          # .field WITH a default -> non-required -> rejected
    with pytest.raises(TypeError, match="default"):
        class _Fac(Constraint):
            tags: Constraint.field(type=list[int]) = _PydField(default_factory=list)   # default_factory too

    # the one elegant way: every field a REQUIRED .field (nested via type=), and it validates
    class Order(Constraint):
        side: Constraint.field(type=str)
        leg:  Constraint.field(type=Leg)
    assert Order.validate({"side": "buy", "leg": {"qty": 5}}).leg.qty == 5


def test_kind_constraint_concurrent_reads_of_readonly_registry():
    """(Gap 2a) A parallel()-style fan-out (copy_context + a thread pool, mirroring how parallel()
    runs each branch) validating a kind= constraint concurrently READS the now-read-only REGISTRY
    mappingproxy. Reads need no lock and never race — every branch returns the correct verdict."""
    from contextvars import copy_context
    from concurrent.futures import ThreadPoolExecutor

    c = Constraint.field(type=str, kind="email")

    def _ok(v):
        try:
            c.validate(v)
            return True
        except ConstraintViolation:
            return False

    valid = ["a@b.co"] * 32
    invalid = ["nope"] * 32
    with ThreadPoolExecutor(max_workers=8) as pool:
        rv = [f.result() for f in [pool.submit(copy_context().run, _ok, v) for v in valid]]
        ri = [f.result() for f in [pool.submit(copy_context().run, _ok, v) for v in invalid]]
    assert all(rv) and not any(ri)      # all valid pass, all invalid reject — no concurrent-read race
