"""Constraint — generic constraint class with a full composition algebra.

PLM authors constraints two ways:

  Pattern 1 — Structural (subclass):
      class HypothesisReduction(Constraint):
          facts: list[str]
          ambiguities: list[str]
          hypotheses: dict[str, list[str]]

  Pattern 2 — One-shot factory:
      c = Constraint.of(int_range=(0, 9))
      c = Constraint.of(predicate=is_prime, description="must be prime")

Two kinds of user-supplied functions, with deliberately different contracts:

  Predicate (`predicate=fn`): *checks*. Runs AFTER coercion + type validation.
    - signature: `(value) -> None | raises ValueError`
    - return None / True / the input on success; raise ValueError to reject
    - predicates do NOT transform. If a predicate returns a non-canonical
      value, the wrapper raises a clear error pointing at `coercer=` /
      `coercers=` as the right tool.

  Coercer (`coercer=fn` or `coercers=[fn1, fn2, ...]`): *transforms*. Runs
    BEFORE type validation and field constraints.
    - signature: `(value) -> transformed_value | raises ValueError`
    - return value becomes the new value passed downstream
    - raise ValueError to reject (value can't be coerced)
    - multiple coercers chain in declaration order: fn1's output feeds fn2

Execution order in a single Constraint.of call:
  coercer(s)  →  type & Field(...) constraints  →  predicate(s)

Both patterns expose the same API on the class:

  .validate(value, *, context=None) -> value
  .describe() -> str
  .json_schema() -> dict
  &, |, ~, ^                       (class-level via metaclass)

Built on pydantic v2.
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    get_args,
    get_origin,
)

import pydantic as pd
from pydantic import AfterValidator, BeforeValidator, Field, model_validator
from typing_extensions import Annotated

from .decorator import _DESCRIPTION_ATTR, get_description
from .registry import REGISTRY


# ============================================================================
# Exceptions
# ============================================================================


class ConstraintViolation(ValueError):
    """Raised when a value fails any layer of a Constraint check."""


# ============================================================================
# Metaclass — class-level operator composition
# ============================================================================


_PydanticMeta: type = type(pd.BaseModel)


class _ConstraintMeta(_PydanticMeta):  # type: ignore[misc, valid-type]
    """Metaclass enabling `Cls1 & Cls2`, `Cls1 | Cls2`, `~Cls`, `Cls1 ^ Cls2`
    at the class level."""

    def __and__(cls, other: type) -> type:
        from .algebra import _and
        return _and(cls, other)

    def __rand__(cls, other: type) -> type:
        from .algebra import _and
        return _and(other, cls)

    def __or__(cls, other: type) -> type:
        from .algebra import _or
        return _or(cls, other)

    def __ror__(cls, other: type) -> type:
        from .algebra import _or
        return _or(other, cls)

    def __invert__(cls) -> type:
        from .algebra import _not
        return _not(cls)

    def __xor__(cls, other: type) -> type:
        from .algebra import _xor
        return _xor(cls, other)

    def __rxor__(cls, other: type) -> type:
        from .algebra import _xor
        return _xor(other, cls)


# ============================================================================
# Constraint base
# ============================================================================


class Constraint(pd.BaseModel, metaclass=_ConstraintMeta):
    """Base class for all constraints.

    Subclass for structural constraints (fields + pydantic validators).
    Or call `Constraint.of(...)` for a one-shot factory-built constraint."""

    model_config = pd.ConfigDict(arbitrary_types_allowed=True)

    # Class-level marker — set by the factory builder; structural subclasses
    # have it False. Distinguishes "wraps a single scalar value" from
    # "structural object with fields."
    _constraint_is_factory: ClassVar[bool] = False

    # Original kwargs passed to Constraint.of (for .describe()).
    _constraint_factory_kwargs: ClassVar[Optional[Dict[str, Any]]] = None

    # NL description carried verbatim through .describe().
    _constraint_description: ClassVar[str] = ""

    # Registry of common-kind predicates.
    registry: ClassVar[Dict[str, Callable[[Any], None]]] = REGISTRY

    # ---- Authoring entry points ------------------------------------------------

    @classmethod
    def of(cls, **kwargs: Any) -> "type[Constraint]":
        """Build a one-shot Constraint subclass from kwargs.

        See module docstring for the full kwarg surface."""
        return _build_factory(**kwargs)

    @classmethod
    def from_kind(cls, name: str) -> "type[Constraint]":
        """Build a Constraint that runs the named predicate from the registry."""
        if name not in cls.registry:
            raise KeyError(
                f"unknown kind {name!r}; "
                f"available: {sorted(cls.registry)}"
            )
        pred = cls.registry[name]
        return cls.of(predicate=pred, description=f"kind={name!r}")

    @classmethod
    def exactly_one_of(cls, *fields: str) -> Callable[[type], type]:
        """Class decorator: enforce that *exactly one* of `fields` is non-None.

        Builds a new class (a thin subclass of the decorated one) whose body
        includes the `@model_validator` — this is the only reliable way to
        attach a pydantic validator at class-decoration time, since pydantic
        scans the class namespace during metaclass construction.

        Usage:
            @Constraint.exactly_one_of("score", "label", "error")
            class Result(Constraint):
                score: float | None = None
                label: str | None = None
                error: str | None = None
        """
        field_names = tuple(fields)
        if not field_names:
            raise ValueError(
                "Constraint.exactly_one_of() requires at least one field name"
            )

        def deco(klass: type) -> type:
            from .decorator import _DESCRIPTION_ATTR

            def _check(self):  # noqa: ANN001
                count = sum(
                    1 for f in field_names if getattr(self, f, None) is not None
                )
                if count != 1:
                    raise ValueError(
                        f"exactly one of {field_names} must be non-None; got {count}"
                    )
                return self

            setattr(
                _check,
                _DESCRIPTION_ATTR,
                f"exactly one of {list(field_names)} must be non-None",
            )
            method_name = f"_exactly_one_of_{'_'.join(field_names)}"
            wrapped = model_validator(mode="after")(_check)

            # Build a NEW class that subclasses the decorated one and includes
            # the validator in its namespace. Pydantic's metaclass sees it
            # during construction and registers it as a model_validator.
            ns: Dict[str, Any] = {
                method_name: wrapped,
                "__annotations__": {},  # no new fields
                "__module__": getattr(klass, "__module__", __name__),
                "__qualname__": klass.__qualname__,
            }
            new_klass = type(klass)(klass.__name__, (klass,), ns)
            return new_klass

        return deco

    # ---- Validation -----------------------------------------------------------

    @classmethod
    def validate(  # type: ignore[override]
        cls, value: Any, *, context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Validate a value against this constraint.

        Returns the (possibly coerced) value on success; raises
        ConstraintViolation on failure.
        """
        try:
            if cls._constraint_is_factory:
                # Factory: stored as a single `value` field; wrap + unwrap.
                instance = cls.model_validate(
                    {"value": value}, context=context
                )
                return instance.value  # type: ignore[attr-defined]
            else:
                return cls.model_validate(value, context=context)
        except pd.ValidationError as e:
            raise ConstraintViolation(str(e)) from e

    # ---- Description ----------------------------------------------------------

    @classmethod
    def describe(cls) -> str:
        """Render this constraint as a natural-language string."""
        if cls._constraint_is_factory:
            return _describe_factory(cls)
        return _describe_structural(cls)

    # ---- JSON Schema ----------------------------------------------------------

    @classmethod
    def json_schema(cls) -> Dict[str, Any]:
        """Export as JSON Schema.

        Structural Constraints export fully via pydantic. Factory-built
        Constraints export the structural bits pydantic can express plus an
        `x-description` field carrying the NL description (since
        free-form predicates have no JSON Schema equivalent).
        """
        schema = cls.model_json_schema()
        if cls._constraint_is_factory:
            # Pydantic wraps the value in a "value" field; flatten for export.
            value_schema = schema.get("properties", {}).get("value", schema)
            # Carry x-description from the factory kwargs + user description.
            x_desc = cls.describe()
            if x_desc:
                value_schema = dict(value_schema)
                value_schema["x-description"] = x_desc
            return value_schema
        return schema


# ============================================================================
# Factory — Constraint.of(**kwargs)
# ============================================================================


_KNOWN_KWARGS = frozenset(
    {
        "int_range", "int_gt", "int_ge", "int_lt", "int_le",
        "float_range", "float_gt", "float_ge", "float_lt", "float_le",
        "instance_of", "one_of", "not_one_of",
        "str_pattern", "str_length", "not_pattern", "not_empty",
        "list_of", "dict_of", "list_length", "unique", "sorted", "monotonic",
        "approx", "callable", "behaves_like",
        "kind", "predicate", "coercer", "coercers", "description",
    }
)


def _resolve_elem_type(arg: Any) -> Any:
    """If arg is a factory Constraint, unwrap to its underlying *annotated*
    type — preserving Field metadata + AfterValidator predicates so element
    constraints actually apply inside list_of / dict_of. Otherwise return arg
    as-is."""
    if isinstance(arg, type) and issubclass(arg, Constraint):
        if getattr(arg, "_constraint_is_factory", False):
            info = arg.model_fields["value"]
            base = info.annotation
            metadata = list(info.metadata or ())
            if metadata:
                return Annotated[(base, *metadata)]  # type: ignore[valid-type]
            return base
        return arg
    return arg


def _wrap_predicate(pred: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Wrap a user predicate as a pydantic AfterValidator callable.

    User predicate protocol:
      - signature: `(value) -> None | raises ValueError`
      - return `None` (or `True`, or the original value) on success
      - raise `ValueError` to reject

    The wrapper enforces this contract. If the predicate returns anything
    other than `None`, `True`, or the input value, we raise a clear
    `ValueError` explaining the protocol — predicates must not be coercers.
    For value coercion, use `Annotated[type, AfterValidator(fn)]` directly
    inside a structural Constraint subclass.

    This catches the common LLM mistake of writing a coercer-style predicate
    (e.g. `def normalize(s): return s.lower().strip()`) where the returned
    value would otherwise be silently discarded.
    """
    pred_name = getattr(pred, "__name__", "<predicate>")

    def _wrapped(v: Any) -> Any:
        result = pred(v)
        if result is None or result is True or result is v:
            return v
        if result is False:
            raise ValueError(
                f"predicate {pred_name!r} returned False — predicates must "
                f"RAISE ValueError to reject, not return False."
            )
        raise ValueError(
            f"predicate {pred_name!r} returned a non-None value "
            f"({type(result).__name__}: {result!r}). Predicates must return "
            f"None (or True, or the input value) and raise ValueError to "
            f"reject. For value transformation, use `coercer=fn` / "
            f"`coercers=[fn1, fn2, ...]` on Constraint.of instead."
        )
    _wrapped.__plm_underlying__ = pred  # type: ignore[attr-defined]
    return _wrapped


_TYPE_KWARG_GROUPS: Dict[str, frozenset] = {
    "int":      frozenset({"int_range", "int_gt", "int_ge", "int_lt", "int_le"}),
    "float":    frozenset({"float_range", "float_gt", "float_ge", "float_lt", "float_le"}),
    "str":      frozenset({"str_pattern", "str_length"}),
    # `list_length` applies `min_length`/`max_length` Field constraints which
    # work on both lists and strings — but combining with int/float/dict groups
    # is meaningless (Field would reject), so we route it to the list group.
    # `unique`/`sorted`/`monotonic` install predicates that iterate; they work
    # on any iterable (including strings) and stay un-grouped.
    "list":     frozenset({"list_of", "list_length"}),
    "dict":     frozenset({"dict_of"}),
    "instance": frozenset({"instance_of"}),
    "literal":  frozenset({"one_of"}),
    "kind":     frozenset({"kind"}),
}


def _check_type_compatibility(kwargs: Dict[str, Any]) -> None:
    """Reject kwargs that target conflicting base types — catches W3-style
    mistakes like `kind="semver"` + `int_range=(0,9)` at compose time
    instead of letting the resulting broken constraint fail far from cause."""
    keys = set(kwargs)
    present = {name: sorted(group & keys) for name, group in _TYPE_KWARG_GROUPS.items() if group & keys}
    if len(present) > 1:
        lines = "\n".join(f"  - {group!r}: {kws}" for group, kws in present.items())
        raise ValueError(
            "Constraint.of: kwargs from multiple type-shifting groups:\n"
            f"{lines}\n"
            "Each constraint targets a single base type — pick one group "
            "(e.g. don't combine `kind` with `int_range`). If you need both "
            "shapes, compose two factories with `&`."
        )


def _build_factory(**kwargs: Any) -> Type[Constraint]:
    """Build a dynamic factory Constraint subclass from kwargs."""
    unknown = set(kwargs) - _KNOWN_KWARGS
    if unknown:
        raise TypeError(
            f"Constraint.of: unknown kwargs {sorted(unknown)}; "
            f"known: {sorted(_KNOWN_KWARGS)}"
        )

    _check_type_compatibility(kwargs)

    # Save originals for .describe()
    original_kwargs = dict(kwargs)

    description = kwargs.pop("description", "") or ""

    # Determine base type + field constraints + predicates + coercers.
    base_type, field_kwargs, predicates, coercers = _interpret_kwargs(kwargs)

    # Build the annotated type. Execution order pydantic gives us:
    #   BeforeValidators (LIFO from metadata) → coercers run first
    #   type coerce + Field(...) constraints  → structural check
    #   AfterValidators  (FIFO from metadata) → predicates run last
    #
    # Pydantic v2 quirk: BeforeValidators execute in REVERSE metadata order
    # (last-added runs first), while AfterValidators execute in metadata
    # order. To preserve user-facing left→right declaration semantics for
    # both, we iterate coercers in REVERSE when wrapping and predicates in
    # FORWARD order.
    type_ann: Any = base_type
    for coercer in reversed(coercers):
        type_ann = Annotated[type_ann, BeforeValidator(coercer)]
    if field_kwargs:
        type_ann = Annotated[type_ann, Field(**field_kwargs)]
    for pred in predicates:
        type_ann = Annotated[type_ann, AfterValidator(_wrap_predicate(pred))]

    # Build a dynamic Constraint subclass with a single `value` field.
    cls_dict: Dict[str, Any] = {
        "__annotations__": {"value": type_ann},
        "_constraint_is_factory": True,
        "_constraint_factory_kwargs": original_kwargs,
        "_constraint_description": description,
        "model_config": pd.ConfigDict(arbitrary_types_allowed=True),
    }

    cls = _ConstraintMeta(  # type: ignore[call-arg]
        "_FactoryConstraint",
        (Constraint,),
        cls_dict,
    )
    return cls  # type: ignore[return-value]


def _interpret_kwargs(
    kwargs: Dict[str, Any],
) -> Tuple[Any, Dict[str, Any], List[Callable[[Any], None]], List[Callable[[Any], Any]]]:
    """Translate Constraint.of kwargs into a 4-tuple.

    Returns:
      base_type:    the underlying Python type the value must conform to.
      field_kwargs: pydantic `Field()` kwargs (ge/le/pattern/min_length/...)
                    applied during type validation.
      predicates:   `(value) -> None | raises ValueError` callables — run AFTER
                    type validation as AfterValidators.
      coercers:     `(value) -> transformed_value | raises ValueError` callables
                    — run BEFORE type validation as BeforeValidators, in
                    declaration order (fn1's output feeds fn2).
    """
    base_type: Any = Any
    field_kwargs: Dict[str, Any] = {}
    predicates: List[Callable[[Any], None]] = []
    coercers: List[Callable[[Any], Any]] = []

    # --- kind (registry-backed predicate) ---
    if "kind" in kwargs:
        name = kwargs.pop("kind")
        if name not in REGISTRY:
            raise KeyError(
                f"unknown kind {name!r}; available: {sorted(REGISTRY)}"
            )
        predicates.append(REGISTRY[name])
        # Default base_type to str for the string-shaped kinds; others stay Any.
        if name in {"semver", "url", "json", "identifier", "regex",
                    "email", "iso_date", "uuid", "path_exists"}:
            base_type = str

    # --- numeric ranges (closed) ---
    # Reject mixing range= with strict-bound kwargs on the same axis; pydantic
    # would raise an opaque SchemaError far from the user's call site otherwise.
    _int_strict = {"int_gt", "int_ge", "int_lt", "int_le"} & set(kwargs)
    _float_strict = {"float_gt", "float_ge", "float_lt", "float_le"} & set(kwargs)
    if "int_range" in kwargs and _int_strict:
        raise ValueError(
            f"Constraint.of: int_range= cannot be combined with strict bounds "
            f"{sorted(_int_strict)}. Pick one form."
        )
    if "float_range" in kwargs and _float_strict:
        raise ValueError(
            f"Constraint.of: float_range= cannot be combined with strict bounds "
            f"{sorted(_float_strict)}. Pick one form."
        )

    if "int_range" in kwargs:
        lo, hi = kwargs.pop("int_range")
        base_type = int
        if lo is not None:
            field_kwargs["ge"] = lo
        if hi is not None:
            field_kwargs["le"] = hi
    if "float_range" in kwargs:
        lo, hi = kwargs.pop("float_range")
        base_type = float
        if lo is not None:
            field_kwargs["ge"] = lo
        if hi is not None:
            field_kwargs["le"] = hi

    # --- numeric strict bounds ---
    for k_int, key in (
        ("int_gt", "gt"), ("int_ge", "ge"), ("int_lt", "lt"), ("int_le", "le"),
    ):
        if k_int in kwargs:
            base_type = int
            field_kwargs[key] = kwargs.pop(k_int)
    for k_float, key in (
        ("float_gt", "gt"), ("float_ge", "ge"),
        ("float_lt", "lt"), ("float_le", "le"),
    ):
        if k_float in kwargs:
            base_type = float
            field_kwargs[key] = kwargs.pop(k_float)

    # --- instance_of ---
    if "instance_of" in kwargs:
        target_cls = kwargs.pop("instance_of")
        base_type = target_cls

    # --- one_of (Literal[...]) ---
    if "one_of" in kwargs:
        options = list(kwargs.pop("one_of"))
        if not options:
            raise ValueError("one_of= requires a non-empty sequence")
        base_type = Literal[tuple(options)]  # type: ignore[valid-type]

    # --- not_one_of ---
    if "not_one_of" in kwargs:
        bad = set(kwargs.pop("not_one_of"))

        def _check_not_one_of(v, _bad=bad):
            if v in _bad:
                raise ValueError(f"{v!r} is in the forbidden set {sorted(_bad)}")

        predicates.append(_check_not_one_of)

    # --- str ---
    if "str_pattern" in kwargs:
        base_type = str
        field_kwargs["pattern"] = kwargs.pop("str_pattern")
    if "str_length" in kwargs:
        lo, hi = kwargs.pop("str_length")
        base_type = str
        if lo is not None:
            field_kwargs["min_length"] = lo
        if hi is not None:
            field_kwargs["max_length"] = hi
    if "not_pattern" in kwargs:
        import re as _re
        pat = _re.compile(kwargs.pop("not_pattern"))
        base_type = str if base_type is Any else base_type

        def _check_not_pattern(v, _pat=pat):
            if isinstance(v, str) and _pat.search(v):
                raise ValueError(f"{v!r} matches forbidden pattern /{_pat.pattern}/")

        predicates.append(_check_not_pattern)
    if "not_empty" in kwargs:
        if kwargs.pop("not_empty"):
            def _check_not_empty(v):
                try:
                    if len(v) == 0:
                        raise ValueError(f"value must not be empty")
                except TypeError:
                    raise ValueError(f"value of type {type(v).__name__} has no length")
            predicates.append(_check_not_empty)

    # --- list_of / dict_of ---
    if "list_of" in kwargs:
        elem = _resolve_elem_type(kwargs.pop("list_of"))
        base_type = List[elem]  # type: ignore[valid-type]
    if "dict_of" in kwargs:
        k_t, v_t = kwargs.pop("dict_of")
        k_t = _resolve_elem_type(k_t)
        v_t = _resolve_elem_type(v_t)
        base_type = Dict[k_t, v_t]  # type: ignore[valid-type]

    # --- list_length / unique / sorted / monotonic ---
    if "list_length" in kwargs:
        lo, hi = kwargs.pop("list_length")
        if lo is not None:
            field_kwargs["min_length"] = lo
        if hi is not None:
            field_kwargs["max_length"] = hi
    if "unique" in kwargs and kwargs.pop("unique"):
        def _check_unique(v):
            try:
                seen = set()
                for x in v:
                    if x in seen:
                        raise ValueError(f"list contains duplicate: {x!r}")
                    seen.add(x)
            except TypeError:
                # Unhashable elements — fall back to O(n^2) check.
                seen_list = []
                for x in v:
                    if x in seen_list:
                        raise ValueError(f"list contains duplicate: {x!r}")
                    seen_list.append(x)
        predicates.append(_check_unique)
    if "sorted" in kwargs:
        order = kwargs.pop("sorted")
        if order is False:
            pass  # explicit opt-out — no sorted check
        else:
            if order is True or order == "asc":
                reverse = False
            elif order == "desc":
                reverse = True
            else:
                raise ValueError(
                    f"sorted= must be True/False, 'asc', or 'desc'; got {order!r}"
                )

            def _check_sorted(v, _reverse=reverse):
                sorted_v = sorted(v, reverse=_reverse)
                if list(v) != sorted_v:
                    raise ValueError(
                        f"list not sorted {'desc' if _reverse else 'asc'}"
                    )
            predicates.append(_check_sorted)
    if "monotonic" in kwargs:
        mode = kwargs.pop("monotonic")
        if mode is False:
            pass  # explicit opt-out — no monotonic check
        else:
            if mode is True:
                strict = False
            elif mode == "strict":
                strict = True
            else:
                raise ValueError(
                    f"monotonic= must be True/False or 'strict'; got {mode!r}"
                )

            def _check_monotonic(v, _strict=strict):
                for i in range(1, len(v)):
                    a, b = v[i - 1], v[i]
                    if _strict and not (a < b):
                        raise ValueError(f"not strictly monotonic at index {i}")
                    if (not _strict) and not (a <= b):
                        raise ValueError(f"not monotonic at index {i}")
            predicates.append(_check_monotonic)

    # --- approx ---
    if "approx" in kwargs:
        target, eps = kwargs.pop("approx")

        def _check_approx(v, _target=target, _eps=eps):
            try:
                if abs(v - _target) > _eps:
                    raise ValueError(
                        f"{v!r} not within {_eps} of {_target!r}"
                    )
            except TypeError as e:
                raise ValueError(f"approx check failed: {e}") from None
        predicates.append(_check_approx)

    # --- callable / behaves_like ---
    if "callable" in kwargs:
        if kwargs.pop("callable"):
            def _check_callable(v):
                if not callable(v):
                    raise ValueError(f"value of type {type(v).__name__} is not callable")
            predicates.append(_check_callable)
            base_type = Any  # functions aren't easily typed
    if "behaves_like" in kwargs:
        cases = list(kwargs.pop("behaves_like"))

        def _check_behaves_like(v, _cases=cases):
            if not callable(v):
                raise ValueError("behaves_like: value is not callable")
            for args, expected in _cases:
                try:
                    got = v(*args)
                except Exception as e:
                    raise ValueError(
                        f"behaves_like: f{args!r} raised "
                        f"{type(e).__name__}: {e}"
                    ) from None
                if got != expected:
                    raise ValueError(
                        f"behaves_like: f{args!r} = {got!r}, expected {expected!r}"
                    )
        predicates.append(_check_behaves_like)

    # --- user predicate ---
    if "predicate" in kwargs:
        pred = kwargs.pop("predicate")
        if not callable(pred):
            raise TypeError("predicate= must be a callable")
        predicates.append(pred)

    # --- user coercers (single + plural) ---
    if "coercer" in kwargs:
        fn = kwargs.pop("coercer")
        if not callable(fn):
            raise TypeError("coercer= must be a callable")
        coercers.append(fn)
    if "coercers" in kwargs:
        fns = kwargs.pop("coercers")
        try:
            fns_list = list(fns)
        except TypeError:
            raise TypeError("coercers= must be an iterable of callables")
        for fn in fns_list:
            if not callable(fn):
                raise TypeError("coercers= contains a non-callable item")
            coercers.append(fn)

    # Any leftover kwargs are silently passed through as field_kwargs?
    # No — _build_factory already rejected unknowns. Here we should be empty.
    # (description was popped before this fn was called.)

    return base_type, field_kwargs, predicates, coercers


# ============================================================================
# Description rendering
# ============================================================================


def _render_elem(elem: Any) -> str:
    """Render a list_of/dict_of element type — recursively call .describe()
    for Constraint elements, otherwise use the short type name."""
    if isinstance(elem, type) and issubclass(elem, Constraint):
        return elem.describe()
    return _short(elem)


# Descriptor table for _describe_factory. Each rule is:
#   (kwarg_name, only_if_truthy, render_function)
# `only_if_truthy=True` means we skip the render when the value is False/0/None
# (used for opt-out flags like sorted=False, monotonic=False, unique=False).
_DESCRIBE_RULES: Tuple[Tuple[str, bool, Callable[[Any], str]], ...] = (
    ("int_range",   False, lambda v: _range_phrase("int", v[0], v[1], ge=True, le=True)),
    ("float_range", False, lambda v: _range_phrase("float", v[0], v[1], ge=True, le=True)),
    ("int_gt",      False, lambda v: f"int gt {v}"),
    ("int_ge",      False, lambda v: f"int ge {v}"),
    ("int_lt",      False, lambda v: f"int lt {v}"),
    ("int_le",      False, lambda v: f"int le {v}"),
    ("float_gt",    False, lambda v: f"float gt {v}"),
    ("float_ge",    False, lambda v: f"float ge {v}"),
    ("float_lt",    False, lambda v: f"float lt {v}"),
    ("float_le",    False, lambda v: f"float le {v}"),
    ("instance_of", False, lambda v: f"an instance of {v.__name__}"),
    ("one_of",      False, lambda v: f"one of {list(v)}"),
    ("not_one_of",  False, lambda v: f"not one of {list(v)}"),
    ("str_pattern", False, lambda v: f"matches pattern /{v}/"),
    ("str_length",  False, lambda v: _range_phrase("length", v[0], v[1], ge=True, le=True)),
    ("not_pattern", False, lambda v: f"does NOT match /{v}/"),
    ("not_empty",   True,  lambda v: "non-empty"),
    ("list_of",     False, lambda v: f"list of [{_render_elem(v)}]"),
    ("dict_of",     False, lambda v: f"dict[{_render_elem(v[0])}, {_render_elem(v[1])}]"),
    ("list_length", False, lambda v: _range_phrase("length", v[0], v[1], ge=True, le=True)),
    ("unique",      True,  lambda v: "unique elements"),
    ("sorted",      True,  lambda v: f"sorted {v if v is not True else 'asc'}"),
    ("monotonic",   True,  lambda v: f"monotonic ({v})"),
    ("approx",      False, lambda v: f"~ {v[0]} (eps={v[1]})"),
    ("callable",    True,  lambda v: "a callable"),
    ("behaves_like",False, lambda v: f"callable behaving as {v}"),
    ("kind",        False, lambda v: f"kind={v!r}"),
    ("predicate",   False, lambda v: "predicate-checked"),
)


def _describe_factory(cls: Type[Constraint]) -> str:
    """Render an NL description for a factory-built Constraint.

    Driven by `_DESCRIBE_RULES` — adding a new kwarg means adding one row.
    """
    kwargs = cls._constraint_factory_kwargs or {}
    parts: List[str] = []

    for key, only_if_truthy, render in _DESCRIBE_RULES:
        if key not in kwargs:
            continue
        v = kwargs[key]
        if only_if_truthy and not v:
            continue
        parts.append(render(v))

    # Coercers — combine single + plural into one count line.
    n_coercers = (1 if "coercer" in kwargs else 0) + len(kwargs.get("coercers") or [])
    if n_coercers:
        parts.append(f"{n_coercers} coercer(s) applied")

    desc = cls._constraint_description
    if desc:
        parts.append(desc)
    return ", ".join(parts) if parts else "no constraint"


def _describe_structural(cls: Type[Constraint]) -> str:
    """Render an NL description for a structural Constraint subclass."""
    lines = [f"Return an object of type {cls.__name__} with fields:"]
    for name, info in cls.model_fields.items():
        ann = info.annotation
        constraints = _field_constraint_summary(info)
        field_desc = info.description or ""
        line = f"  - {name} ({_short(ann)})"
        if constraints:
            line += f" — {constraints}"
        if field_desc:
            line += f": {field_desc}"
        lines.append(line)

    # Collect @constraint-tagged descriptions from validators on the class.
    # De-duplicated via `seen` so the same description never appears twice,
    # regardless of whether it was found via dir() or via __pydantic_decorators__.
    extras: List[str] = []
    seen: set[str] = set()

    def _add(d: Optional[str]) -> None:
        if d and d not in seen:
            seen.add(d)
            extras.append(d)

    for name in dir(cls):
        _add(get_description(getattr(cls, name, None)))

    decs = getattr(cls, "__pydantic_decorators__", None)
    if decs is not None:
        for v in list(getattr(decs, "model_validators", {}).values()):
            _add(get_description(v.func))
        for v in list(getattr(decs, "field_validators", {}).values()):
            _add(get_description(v.func))

    if extras:
        lines.append("")
        lines.append("Additionally:")
        for e in extras:
            lines.append(f"  - {e}")
    return "\n".join(lines)


def _field_constraint_summary(info: pd.fields.FieldInfo) -> str:
    """Render pydantic Field constraints (ge, le, pattern, ...) as a string."""
    bits: List[str] = []
    for meta in info.metadata or ():
        for attr in ("ge", "le", "gt", "lt", "min_length", "max_length", "pattern"):
            val = getattr(meta, attr, None)
            if val is not None:
                bits.append(f"{attr}={val!r}")
    return ", ".join(bits)


def _range_phrase(label: str, lo: Any, hi: Any, *, ge: bool, le: bool) -> str:
    lo_part = f"{lo}" if lo is not None else "-∞"
    hi_part = f"{hi}" if hi is not None else "∞"
    lo_op = "≤" if ge else "<"
    hi_op = "≤" if le else "<"
    return f"{label} in [{lo_part} {lo_op} x {hi_op} {hi_part}]"


def _short(t: Any) -> str:
    """Short type name for description rendering."""
    if t is type(None):
        return "None"
    if hasattr(t, "__name__"):
        return t.__name__
    if get_origin(t) is not None:
        origin = get_origin(t)
        args = get_args(t)
        origin_name = getattr(origin, "__name__", repr(origin))
        if args:
            return f"{origin_name}[{', '.join(_short(a) for a in args)}]"
        return origin_name
    return repr(t)
