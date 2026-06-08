"""Constraint — generic constraint class with a full composition algebra.

PLM authors constraints two ways:

  Pattern 1 — Structural (subclass): an object SHAPE with named fields.
      class HypothesisReduction(Constraint):
          facts: list[str]
          ambiguities: list[str]
          hypotheses: dict[str, list[str]]
      # per-field value rules → Constraint.field(...) (below); cross-field
      # invariants → @model_validator(mode="after").

  Pattern 2 — Value / field rule:  Constraint.field(**kw)
      The single public authoring entry for a value rule. It works BOTH standalone
      (validate a bare value, compose with & | ^ ~, pass to a sub-LLM) AND as a
      struct FIELD, where it embeds FLAT (the field IS the bare value, not a
      {value:..} wrapper) while remaining a real Constraint (validate / describe /
      json_schema / & | ^ ~ all work):
          class Order(Constraint):
              currency: Constraint.field(coercer=str.upper, one_of=["USD","EUR"])
              qty:      Constraint.field(int_range=(1, 1000), multiple_of=5)

The Constraint.field(**kw) kwarg surface, by built-in type:
  numeric : int_range / float_range, int_ge/gt/le/lt (+ float_*), multiple_of,
            approx=(target,eps), finite=True  (reject nan/inf)
  member  : one_of=[...], not_one_of=[...]
  str     : str_pattern, str_length=(lo,hi), not_pattern, not_empty,
            str_prefix, str_suffix, str_contains
  list    : list_of=T, list_length=(lo,hi), unique, sorted, monotonic,
            list_contains=X  (the list must contain X as an element, via `in`)
  tuple   : tuple_of=(T1,T2,..)  (fixed)  |  tuple_of=T  (homogeneous)
  set     : set_of=T, frozenset_of=T, set_length, subset_of=[..], superset_of=[..]
  dict    : dict_of=(K,V), dict_length=(lo,hi)
  bytes   : bytes_length=(lo,hi)   (a bytearray is accepted but normalized to bytes)
  complex : abs_range=(lo,hi), real_range=(lo,hi), imag_range=(lo,hi)
  bool/None: no dedicated kwargs by design — use one_of=[True]/[False] / one_of=[None]
             (or is_instance_of=bool); Optional[...] handles nullable struct fields
  type    : is_instance_of=T (STRICT isinstance, no coercion), callable=True, behaves_like=[((args),out),..]
  named   : kind="email|url|uuid|semver|iso_date|identifier|json|regex|path_exists"
  custom  : predicate=fn / predicates=[fn,..], coercer=fn / coercers=[fn,..], description="..."
  (element types T in list_of/set_of/tuple_of/dict_of may themselves be Constraints.)

No finite kwarg set is complete: for relations / aggregates / domain logic
("sums to 100", "is prime", "compiles") drop to predicate=fn WITH a description=
(the description is what still steers the model). Compose anything with & | ^ ~;
~ negates value-tests only (a structural NOT is a @model_validator).

Two kinds of user-supplied functions, with deliberately different contracts:

  Predicate (`predicate=fn`): *checks*. Runs AFTER coercion + type validation.
    - signature: `(value) -> None | raises ValueError`
    - return None / True / the input on success; raise ValueError to reject
    - predicates do NOT transform. If a predicate returns a non-canonical
      value, the wrapper raises a clear error pointing at `coercer=` /
      `coercers=` as the right tool.
    - the wrapper also catches ANY exception raised by the predicate (not
      just ValueError — e.g. TypeError from a non-comparable / non-iterable
      value) and re-raises it as a ValueError reporting the original
      exception type, so it funnels through pydantic into a
      ConstraintViolation. Predicates therefore need not be defensive
      about value shape.
    - tag a predicate `def` with `@constraint("...")` to give it a description
      that describe() surfaces — for the single `predicate=` AND for each item
      of `predicates=[...]`. Untagged predicates collapse to a generic
      "N predicate(s) checked" count. (Lambdas can't be tagged; use a `def`,
      or set the whole-constraint `description=` for one blanket phrase.)

  Coercer (`coercer=fn` or `coercers=[fn1, fn2, ...]`): *transforms*. Runs
    BEFORE type validation and field constraints.
    - signature: `(value) -> transformed_value | raises ValueError`
    - return value becomes the new value passed downstream
    - raise ValueError to reject (value can't be coerced)
    - multiple coercers chain in declaration order: fn1's output feeds fn2
    - tag a coercer `def` with `@constraint("...")` to describe the transform;
      describe() renders it as "coerced (...)" — deliberately DISTINCT from a
      predicate's bare phrase — so a reader (and the sub-LLM) can tell input
      preprocessing apart from a check the value must satisfy. Untagged coercers
      collapse to a "N coercer(s) applied" count.

Execution order in a single Constraint.field call:
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

    Subclass for structural constraints (fields + pydantic validators), or call
    `Constraint.field(...)` for a value/field rule — the single public authoring
    entry for non-structural constraints."""

    model_config = pd.ConfigDict(arbitrary_types_allowed=True)

    # Class-level marker — set by the factory builder; structural subclasses
    # have it False. Distinguishes "wraps a single scalar value" from
    # "structural object with fields."
    _constraint_is_factory: ClassVar[bool] = False

    # `Constraint.field(...)` marker — a TRANSPARENT factory that embeds FLAT as a
    # struct field. `_constraint_field_inner_type` is the bare value type (str/int/…)
    # so `describe()` can show the real field type instead of `_FieldConstraint`.
    _constraint_is_field: ClassVar[bool] = False
    _constraint_field_inner_type: ClassVar[Any] = None
    _constraint_field_annotation: ClassVar[Any] = None   # flat Annotated[type, *meta] for element use

    # Composition markers — set by the `&`/`|`/`^`/`~` wrapper (algebra.py).
    # Declared ClassVar here (NOT bare `_`-attrs) so they are real, introspectable
    # class attributes; a bare `_constraint_*` on a pydantic model becomes a
    # ModelPrivateAttr (instance-only) and reads back as a descriptor on the class.
    _constraint_is_composite: ClassVar[bool] = False
    _constraint_children: ClassVar[tuple] = ()

    # Composite operator label ("AND"/"OR"/"XOR"/"NOT") — set by the algebra wrappers.
    # Declared here (not just set in the wrapper's class dict) so it reads back as the
    # plain string, not a pydantic ModelPrivateAttr — needed by the crash-restart recipe.
    _constraint_op_label: ClassVar[Optional[str]] = None

    # Original kwargs passed to the factory builder (for .describe() + crash-restart recipe).
    _constraint_factory_kwargs: ClassVar[Optional[Dict[str, Any]]] = None

    # NL description carried verbatim through .describe().
    _constraint_description: ClassVar[str] = ""

    # Registry of common-kind predicates.
    registry: ClassVar[Dict[str, Callable[[Any], None]]] = REGISTRY

    # ---- Authoring entry points ------------------------------------------------

    @classmethod
    def of(cls, **kwargs: Any) -> "type[Constraint]":
        """INTERNAL engine — private to the constraint module. External code uses
        `Constraint.field(...)` (the public, transparent superset), never `.of`
        directly. Kept as the shared builder that `.field` / `from_kind` ride on.
        See the module docstring for the kwarg surface."""
        return _build_factory(**kwargs)

    @classmethod
    def field(cls, **kwargs: Any) -> Any:
        """The public authoring entry for a value/field rule. Builds a constraint
        from the kwarg surface (see the module docstring); the result works BOTH
        standalone (`.validate` a bare value, compose with `& | ^ ~`, pass to a
        sub-LLM) AND as a struct FIELD, where it embeds FLAT — the field IS the
        bare value, carrying the constraints — while remaining a real `Constraint`
        (`.validate` / `.describe` / `.json_schema` all work):

            class Order(Constraint):
                currency: Constraint.field(coercer=str.upper, one_of=["USD","EUR"])
                qty:      Constraint.field(int_range=(1, 1000), multiple_of=5)
            Order.validate({"currency": "usd", "qty": 10})   # currency -> "USD" (flat)
        """
        inner = _build_factory(**kwargs)
        _vfi = inner.model_fields["value"]
        # INVARIANT: _flat must carry EVERYTHING enforcement-relevant from inner's
        # value field, because two code paths must agree — _validate delegates to
        # inner.validate (the {value:..} model), while _get_core embeds _flat as a
        # struct member / for model_validate. Today that is annotation + metadata
        # (Field bounds + Before/After validators); FieldInfo default/alias are
        # schema-only and not enforcement, so they are intentionally not carried.
        _flat = (Annotated[tuple([_vfi.annotation, *_vfi.metadata])]
                 if _vfi.metadata else _vfi.annotation)

        def _get_core(c, source, handler, _ann=_flat):
            return handler(_ann)                       # embed FLAT (the inner value schema)

        def _validate(c, value, *, context=None, _i=inner):
            return _i.validate(value, context=context)  # delegate to the proven factory

        def _describe(c, _i=inner):
            return _i.describe()

        def _json_schema(c, _i=inner):
            return _i.json_schema()

        ns: Dict[str, Any] = {
            # pydantic's namespace inspection reads `__module__` when a class-attr
            # value is itself a type (here `_constraint_field_inner_type`); a
            # metaclass-constructed class doesn't get it auto-set, so provide it.
            "__module__": Constraint.__module__,
            "__get_pydantic_core_schema__": classmethod(_get_core),
            "validate": classmethod(_validate),
            "describe": classmethod(_describe),
            "json_schema": classmethod(_json_schema),
            "_constraint_is_factory": True,            # so natural_llm/_contains_factory reject it
            "_constraint_is_field": True,              # so _describe_structural renders it fully
            "_constraint_field_inner_type": _vfi.annotation,
            "_constraint_field_annotation": _flat,     # so .field works as a list_of/set_of/.. element
            # describe()/json_schema() delegate to `inner` (the facade shadows the BASE
            # ones), so these are NOT used for rendering. `_constraint_factory_kwargs` IS
            # carried, but for ONE purpose only: crash-restart RECIPE replay — the snapshot
            # can't dill-pickle this dynamic class, so it stores these kwargs and rehydrate
            # rebuilds via `Constraint.field(**kwargs)` (see constraint/snapshot.py). Derive
            # it from INNER — like every other facade attr (`_constraint_field_inner_type`,
            # `_constraint_field_annotation`) reads from inner — NOT from the facade's own raw
            # `kwargs`: inner captured them AFTER materializing one-shot iterables, whereas the
            # raw `kwargs` here hold ALREADY-EXHAUSTED generators (one_of/coercers/...), so a
            # raw copy would recipe an empty/garbage sequence.
            "_constraint_factory_kwargs": dict(inner._constraint_factory_kwargs),
            "model_config": pd.ConfigDict(arbitrary_types_allowed=True),
        }
        return _ConstraintMeta("_FieldConstraint", (Constraint,), ns)

    @classmethod
    def from_kind(cls, name: str) -> "type[Constraint]":
        """Build a Constraint that runs the named predicate from the registry."""
        if name not in cls.registry:
            raise KeyError(
                f"unknown kind {name!r}; "
                f"available: {sorted(cls.registry)}"
            )
        pred = cls.registry[name]
        return cls.field(predicate=pred, description=f"kind={name!r}")

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
            # The method name is derived from the field names joined by '_',
            # which is AMBIGUOUS: exactly_one_of("a_b","c") and
            # exactly_one_of("a","b_c") both join to "_exactly_one_of_a_b_c".
            # Each decoration builds a subclass, so a prior decoration's
            # validator is an attribute of `klass`; reusing the name would
            # SHADOW it (one rule silently dropped). Bump a suffix until the
            # name is free so BOTH validators survive.
            base_method = f"_exactly_one_of_{'_'.join(field_names)}"
            method_name = base_method
            _suffix = 1
            while hasattr(klass, method_name):
                method_name = f"{base_method}__{_suffix}"
                _suffix += 1
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
        except ConstraintViolation:
            raise                                          # already in-contract; don't double-wrap
        except Exception as e:
            # A raw non-ValueError can still escape model_validate even with the
            # predicate net: a Field constraint on an `Any` base (e.g. `list_length=`
            # alone) makes pydantic raise TypeError on a non-sized value, and a
            # coercer can raise anything. Funnel any such failure into the
            # documented ConstraintViolation contract instead of leaking a raw
            # exception to a direct `.validate()` caller. Build the message
            # DEFENSIVELY: a pathological exception's own `__str__` can itself raise
            # (e.g. a custom validator error whose `__str__` blows up), and an
            # unguarded f-string interpolation of it would leak THAT raw exception
            # out of validate(), breaking the contract for the wrapper too.
            try:
                _detail = str(e)
            except Exception:
                _detail = "<error message unavailable>"
            raise ConstraintViolation(f"{type(e).__name__}: {_detail}") from e

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
            value_schema = dict(schema.get("properties", {}).get("value", schema))
            # Preserve the document's `$defs`: when the value annotation references
            # another model (e.g. is_instance_of=Model, list_of=Model), the flattened
            # value schema carries `$ref: #/$defs/Model` but the definitions live
            # at the top level — drop them and the refs dangle. Carry them along so
            # the refs still resolve in the exported (now top-level) schema.
            if "$defs" in schema and "$defs" not in value_schema:
                value_schema["$defs"] = schema["$defs"]
            # Carry x-description from the factory kwargs + user description.
            # describe() can raise on a pathological constraint; never let that
            # abort json_schema() — it's called e.g. by natural_llm to build
            # response_format, OUTSIDE its retry loop. Omit x-desc on failure.
            try:
                x_desc = cls.describe()
            except Exception:
                x_desc = ""
            if x_desc:
                value_schema["x-description"] = x_desc
            return value_schema
        return schema


def _contains_factory(constraint: type) -> bool:
    """True iff `constraint` carries a factory-built node anywhere in its tree —
    i.e. a Python predicate/coercer that a JSON schema cannot express.

    Walks `&`/`|`/`^`/`~` composites recursively (a composite exposes
    `_constraint_is_composite` + `_constraint_children`). A struct-only
    constraint, or an auto-merged struct-only `&` (a real pydantic model, not a
    composite wrapper), contains no factory node and returns False.

    Used by `natural_llm` to reject constraints it cannot honor: only a complete
    JSON schema can steer a single-shot model, so a factory hidden inside a
    composite (e.g. `StructA & Constraint.field(predicate=...)`) must be routed to
    `react_llm` (which runs the predicate against the model's code) — not silently
    accepted and retried to exhaustion against a schema that omits the predicate.
    """
    if getattr(constraint, "_constraint_is_factory", False):
        return True
    if getattr(constraint, "_constraint_is_composite", False):
        return any(_contains_factory(c)
                   for c in getattr(constraint, "_constraint_children", ()) or ())
    return False


# ============================================================================
# Factory engine (internal) — _build_factory(**kwargs); public entry is Constraint.field
# ============================================================================


_KNOWN_KWARGS = frozenset(
    {
        "int_range", "int_gt", "int_ge", "int_lt", "int_le",
        "float_range", "float_gt", "float_ge", "float_lt", "float_le",
        "is_instance_of", "one_of", "not_one_of",
        "str_pattern", "str_length", "not_pattern", "not_empty",
        "list_of", "dict_of", "list_length", "unique", "sorted", "monotonic",
        "approx", "callable", "behaves_like",
        "kind", "predicate", "predicates", "coercer", "coercers", "description",
        # type-completing additions:
        "multiple_of", "finite",
        "tuple_of", "set_of", "frozenset_of", "bytes_length",
        "str_prefix", "str_suffix", "str_contains",
        "dict_length", "set_length", "subset_of", "superset_of", "list_contains",
        "abs_range", "real_range", "imag_range",
    }
)


def _resolve_elem_type(arg: Any) -> Any:
    """If arg is a factory Constraint, unwrap to its underlying *annotated*
    type — preserving Field metadata + AfterValidator predicates so element
    constraints actually apply inside list_of / dict_of. Otherwise return arg
    as-is."""
    if isinstance(arg, type) and issubclass(arg, Constraint):
        if getattr(arg, "_constraint_is_field", False):
            # A Constraint.field(...) is a TRANSPARENT factory — it has no `value`
            # field (its schema lives in the core-schema hook), so use the flat
            # annotation stored at build time. This makes `.field` usable as a
            # list_of/set_of/dict_of/tuple_of element, matching `.of`.
            return arg._constraint_field_annotation   # field() always sets this (the flat annotation)
        if getattr(arg, "_constraint_is_factory", False):
            info = arg.model_fields["value"]
            base = info.annotation
            metadata = list(info.metadata or ())
            if metadata:
                return Annotated[(base, *metadata)]  # type: ignore[valid-type]
            return base
        if getattr(arg, "_constraint_is_composite", False):
            # A composite (A|B, A&B, ~A, A^B) has NO pydantic fields and a custom
            # `.validate()`. Returning it bare would make pydantic validate
            # List[composite]/Dict[..,composite] against the composite's EMPTY
            # model — rejecting valid scalar elements and accepting garbage. Route
            # each element through an AfterValidator that runs the composite's own
            # validate(), so OR/AND/XOR/NOT actually apply per element.
            # Use a 2-arg validator so pydantic injects ValidationInfo, and thread
            # the outer `context=` into the child validate() — matching the
            # top-level composite path and every other `context=context` hand-off.
            # A 1-arg lambda silently dropped the context, so a context-dependent
            # @model_validator in a structural child of a COMPOSITE element never
            # saw it (validate(value, *, context=) is a documented public API).
            return Annotated[Any, AfterValidator(
                lambda v, _info, _c=arg: _c.validate(v, context=_info.context))]
        return arg          # plain structural subclass: pydantic validates it as a model
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
        try:
            result = pred(v)
        except ValueError:
            raise                                       # documented reject path: pydantic turns
                                                        # this into a ConstraintViolation.
        except Exception as e:
            # A predicate that raises anything OTHER than ValueError (e.g. a
            # TypeError because the value is unhashable, not comparable, not
            # iterable, or just the wrong shape for the check) would otherwise
            # escape pydantic RAW and bypass the ConstraintViolation contract
            # (validate() only converts pydantic ValidationError). Treat
            # "couldn't evaluate this value" as a rejection: re-raise as
            # ValueError — chained, so the original type/message stays visible —
            # so it funnels through pydantic into a ConstraintViolation like any
            # other failure. This is the contract guarantee for the WHOLE class
            # (sorted=/monotonic=/not_one_of=/kind= registry preds + any user
            # predicate), not a per-predicate patch.
            raise ValueError(
                f"predicate {pred_name!r} could not evaluate the value "
                f"({type(e).__name__}: {e})"
            ) from e
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
            f"`coercers=[fn1, fn2, ...]` on Constraint.field instead."
        )
    _wrapped.__plm_underlying__ = pred  # type: ignore[attr-defined]
    return _wrapped


def _bounded_predicate(project: Callable[[Any], Any], label: str, lo: Any, hi: Any) -> Callable[[Any], None]:
    """Build a predicate enforcing `lo <= project(value) <= hi` (a None bound is
    open). Shared by the len / abs / real / imag bound checks so they don't repeat
    the None-guarded comparison idiom — the interpret-side analogue of how
    describe() shares `_range_phrase`. `project` raising (e.g. `.real` on a list)
    funnels through `_wrap_predicate` into a ConstraintViolation like any other."""
    def _check(v: Any) -> None:
        m = project(v)
        if lo is not None and m < lo:
            raise ValueError(f"{label} = {m} < {lo}")
        if hi is not None and m > hi:
            raise ValueError(f"{label} = {m} > {hi}")
    return _check


def _wrap_coercer(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Wrap a user coercer (a TRANSFORMING `(value) -> new_value`) so a
    non-ValueError it raises becomes a ValueError — the coercer counterpart to
    `_wrap_predicate`. A coercer applied to the wrong-typed value (e.g.
    `coercer=str.lower` on an int) would otherwise escape pydantic RAW and bypass
    the ConstraintViolation contract. Unlike a predicate, a coercer RETURNS the
    transformed value, so we return `fn(v)` unchanged on success."""
    fn_name = getattr(fn, "__name__", "<coercer>")

    def _wrapped(v: Any) -> Any:
        try:
            return fn(v)
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(
                f"coercer {fn_name!r} could not transform the value "
                f"({type(e).__name__}: {e})"
            ) from e
    return _wrapped


_TYPE_KWARG_GROUPS: Dict[str, frozenset] = {
    "int":      frozenset({"int_range", "int_gt", "int_ge", "int_lt", "int_le"}),
    "float":    frozenset({"float_range", "float_gt", "float_ge", "float_lt", "float_le"}),
    "str":      frozenset({"str_pattern", "str_length", "str_prefix", "str_suffix", "str_contains"}),
    # `list_length` applies `min_length`/`max_length` Field constraints which
    # work on both lists and strings — but combining with int/float/dict groups
    # is meaningless (Field would reject), so we route it to the list group.
    # `unique`/`sorted`/`monotonic` install predicates that iterate; they work
    # on any iterable (including strings) and stay un-grouped. `multiple_of`/
    # `finite` also stay un-grouped (they refine an existing int/float group).
    "list":     frozenset({"list_of", "list_length", "list_contains"}),
    "dict":     frozenset({"dict_of", "dict_length"}),
    "tuple":    frozenset({"tuple_of"}),
    "set":      frozenset({"set_of", "frozenset_of", "set_length"}),
    "bytes":    frozenset({"bytes_length"}),
    "complex":  frozenset({"abs_range", "real_range", "imag_range"}),
    "instance": frozenset({"is_instance_of"}),
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
            "Constraint.field: kwargs from multiple type-shifting groups:\n"
            f"{lines}\n"
            "Each constraint targets a single base type — pick one group "
            "(e.g. don't combine `kind` with `int_range`). If you need both "
            "shapes, compose two factories with `&`."
        )


_RANGE_KWARGS: Tuple[str, ...] = (
    "int_range", "float_range", "str_length", "list_length", "set_length",
    "dict_length", "bytes_length", "abs_range", "real_range", "imag_range",
)


def _check_range_bounds(kwargs: Dict[str, Any]) -> None:
    """Reject a `(lo, hi)` range kwarg whose `lo > hi` at BUILD time — otherwise the
    constraint builds fine and silently rejects EVERYTHING (a classic author typo,
    e.g. `int_range=(10, 1)`). Mirrors the `multiple_of=0` build guard. A `None`
    bound (open range) is skipped; `lo == hi` (a single value) is allowed."""
    for _k in _RANGE_KWARGS:
        v = kwargs.get(_k)
        if not isinstance(v, (tuple, list)) or len(v) != 2:
            continue
        lo, hi = v
        if lo is None or hi is None:
            continue
        try:
            swapped = lo > hi
        except TypeError:
            swapped = False
        if swapped:
            raise ValueError(
                f"Constraint.field: {_k}={tuple(v)!r} has lo > hi — it would reject "
                f"every value. Did you swap the bounds?"
            )


def _build_factory(**kwargs: Any) -> Type[Constraint]:
    """Build a dynamic factory Constraint subclass from kwargs."""
    unknown = set(kwargs) - _KNOWN_KWARGS
    if unknown:
        raise TypeError(
            f"Constraint.field: unknown kwargs {sorted(unknown)}; "
            f"known: {sorted(_KNOWN_KWARGS)}"
        )

    _check_type_compatibility(kwargs)
    _check_range_bounds(kwargs)

    # #7: materialize one-shot iterables BEFORE snapshotting `original_kwargs`.
    # `_interpret_kwargs` below CONSUMES one_of/not_one_of/coercers/behaves_like
    # (it list()s / iterates them), so a shallow dict() snapshot would otherwise
    # capture an EXHAUSTED generator that `describe()` later re-iterates to
    # garbage (`one of []`) or crashes on (`len()` over coercers). Lists are
    # re-iterable, so both the live consumption and `describe()` see the full
    # sequence — and the live path already list()s these, so nothing changes
    # there. A non-iterable value is left as-is for `_interpret_kwargs` to reject
    # with its specific message.
    for _key in ("one_of", "not_one_of", "coercers", "predicates", "behaves_like",
                 "subset_of", "superset_of"):
        if kwargs.get(_key) is not None:
            try:
                kwargs[_key] = list(kwargs[_key])
            except TypeError:
                pass

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
        type_ann = Annotated[type_ann, BeforeValidator(_wrap_coercer(coercer))]
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


def _strict_base(target_cls: Any) -> Any:
    """Annotation for `is_instance_of=target_cls` that does NOT coerce.

    For a coercible type (int/float/str/bytes/bool, containers, std-lib types) pydantic's
    `Strict()` turns off coercion — so `is_instance_of=int` rejects "5", 5.0 AND True (a bool is
    not a real int). For an ARBITRARY class pydantic uses an is-instance schema that `Strict()`
    can't annotate (and which never coerces anyway), so we fall back to the plain type — already
    an isinstance-only check. Probing a tiny TypeAdapter is the robust way to tell the two apart
    without enumerating every coercible std-lib type."""
    ann = Annotated[target_cls, pd.Strict()]
    try:
        pd.TypeAdapter(ann)              # builds the schema eagerly; raises if Strict can't apply
        return ann
    except Exception:
        return target_cls                # is-instance schema (arbitrary class): plain == isinstance


def _interpret_kwargs(
    kwargs: Dict[str, Any],
) -> Tuple[Any, Dict[str, Any], List[Callable[[Any], None]], List[Callable[[Any], Any]]]:
    """Translate the factory kwarg surface into a 4-tuple.

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
            f"Constraint.field: int_range= cannot be combined with strict bounds "
            f"{sorted(_int_strict)}. Pick one form."
        )
    if "float_range" in kwargs and _float_strict:
        raise ValueError(
            f"Constraint.field: float_range= cannot be combined with strict bounds "
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

    # --- is_instance_of (STRICT: pydantic Strict() -> a real isinstance check, NO coercion) ---
    if "is_instance_of" in kwargs:
        target_cls = kwargs.pop("is_instance_of")
        base_type = _strict_base(target_cls)             # reject "5"/5.0/True for int; arbitrary class
                                                         # stays isinstance-only (Strict can't annotate it)

    # --- one_of (Literal[...]) ---
    if "one_of" in kwargs:
        options = list(kwargs.pop("one_of"))
        if not options:
            raise ValueError("one_of= requires a non-empty sequence")
        base_type = Literal[tuple(options)]  # type: ignore[valid-type]

    # --- not_one_of ---
    if "not_one_of" in kwargs:
        seq = kwargs.pop("not_one_of")
        try:
            bad = set(seq)                            # hashable members: fast set membership
        except TypeError:                             # unhashable members (e.g. lists): fall
            bad = list(seq)                           # back to list membership, mirroring one_of=

        def _check_not_one_of(v, _bad=bad):
            try:
                forbidden = v in _bad
            except TypeError:
                # An UNHASHABLE value (list/dict) vs a hashable-member SET: it can
                # never equal a hashable scalar, so it is genuinely NOT forbidden —
                # but `in set` raises. Fall back to element-wise == (-> False), so
                # such a value is ALLOWED, matching the unhashable-MEMBERS path
                # (acceptance must not depend on member hashability).
                forbidden = any(v == b for b in _bad)
            if forbidden:
                raise ValueError(f"{v!r} is in the forbidden set {list(_bad)!r}")

        predicates.append(_check_not_one_of)

    # --- str ---
    if "str_pattern" in kwargs:
        base_type = str if base_type is Any else base_type
        field_kwargs["pattern"] = kwargs.pop("str_pattern")
    if "str_length" in kwargs:
        lo, hi = kwargs.pop("str_length")
        base_type = str if base_type is Any else base_type
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
            # Materialize ONCE (mirrors _check_sorted/_check_monotonic): the set
            # fast path would otherwise partially consume a one-shot generator
            # before an unhashable element raises TypeError, and the fallback
            # would then re-iterate an EXHAUSTED stream and miss the duplicate.
            try:
                items = list(v)
            except TypeError as e:
                raise ValueError(f"unique=: value is not iterable ({e})") from None
            try:
                seen = set()
                for x in items:
                    if x in seen:
                        raise ValueError(f"list contains duplicate: {x!r}")
                    seen.add(x)
            except TypeError:
                # Unhashable elements — fall back to an O(n^2) check over the SAME
                # materialized `items` (not a re-iterated, possibly-spent `v`).
                seen_list = []
                for x in items:
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
                try:
                    items = list(v)                     # materialize ONCE (also avoids
                                                        # double-consuming a generator)
                except TypeError as e:
                    raise ValueError(f"sorted=: value is not iterable ({e})") from None
                try:
                    ordered = sorted(items, reverse=_reverse)
                except TypeError as e:
                    raise ValueError(
                        f"sorted=: items are not mutually comparable ({e})"
                    ) from None
                if items != ordered:
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
                try:
                    items = list(v)                     # accept any iterable (set has len
                                                        # but no indexing; generators etc.)
                except TypeError as e:
                    raise ValueError(f"monotonic=: value is not iterable ({e})") from None
                for i in range(1, len(items)):
                    a, b = items[i - 1], items[i]
                    try:
                        ok = (a < b) if _strict else (a <= b)
                    except TypeError as e:
                        raise ValueError(
                            f"monotonic=: items at indices {i - 1}/{i} are not "
                            f"comparable ({e})"
                        ) from None
                    if not ok:
                        raise ValueError(
                            f"not {'strictly ' if _strict else ''}monotonic at index {i}"
                        )
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

    # --- numeric: multiple_of (divisibility) + finite (reject nan/inf) ---
    if "multiple_of" in kwargs:
        _mo = kwargs.pop("multiple_of")
        # multiple_of=0 is nonsensical AND unsafe: int 0 panics pydantic-core's Rust
        # validator (remainder-by-zero) — a raw panic that escapes validate() — and
        # float 0.0 silently accepts everything. Reject it at build time.
        if _mo == 0:
            raise ValueError("multiple_of= must be a non-zero number")
        field_kwargs["multiple_of"] = _mo
        if base_type is Any:
            base_type = int if isinstance(_mo, int) else float
    if "finite" in kwargs:
        if kwargs.pop("finite"):
            if base_type is Any:
                base_type = float

            def _check_finite(v):
                import math as _math
                try:
                    ok = _math.isfinite(v)
                except TypeError:
                    raise ValueError(f"value of type {type(v).__name__} is not a finite-checkable number")
                if not ok:
                    raise ValueError(f"{v!r} is not finite (nan/inf not allowed)")
            predicates.append(_check_finite)

    # --- tuple / set / frozenset (typed; element type via _resolve_elem_type) ---
    if "tuple_of" in kwargs:
        _t = kwargs.pop("tuple_of")
        if isinstance(_t, tuple):
            base_type = tuple[tuple(_resolve_elem_type(x) for x in _t)]   # fixed-arity Tuple[T1,T2,..]
        else:
            base_type = tuple[_resolve_elem_type(_t), ...]                # homogeneous Tuple[T, ...]
    if "set_of" in kwargs:
        base_type = set[_resolve_elem_type(kwargs.pop("set_of"))]         # type: ignore[misc]
    if "frozenset_of" in kwargs:
        base_type = frozenset[_resolve_elem_type(kwargs.pop("frozenset_of"))]  # type: ignore[misc]

    # --- bytes length ---
    if "bytes_length" in kwargs:
        _lo, _hi = kwargs.pop("bytes_length")
        base_type = bytes
        if _lo is not None:
            field_kwargs["min_length"] = _lo
        if _hi is not None:
            field_kwargs["max_length"] = _hi

    # --- string prefix / suffix / contains (predicate forms; regex covers more) ---
    if "str_prefix" in kwargs:
        _p = kwargs.pop("str_prefix")
        base_type = str if base_type is Any else base_type

        def _check_prefix(v, _p=_p):
            if not (isinstance(v, str) and v.startswith(_p)):
                raise ValueError(f"{v!r} must start with {_p!r}")
        predicates.append(_check_prefix)
    if "str_suffix" in kwargs:
        _s = kwargs.pop("str_suffix")
        base_type = str if base_type is Any else base_type

        def _check_suffix(v, _s=_s):
            if not (isinstance(v, str) and v.endswith(_s)):
                raise ValueError(f"{v!r} must end with {_s!r}")
        predicates.append(_check_suffix)
    if "str_contains" in kwargs:
        _c = kwargs.pop("str_contains")
        base_type = str if base_type is Any else base_type

        def _check_contains(v, _c=_c):
            if not (isinstance(v, str) and _c in v):
                raise ValueError(f"{v!r} must contain {_c!r}")
        predicates.append(_check_contains)

    # --- mapping / set size (shared bound-check helper) ---
    if "dict_length" in kwargs:
        _lo, _hi = kwargs.pop("dict_length")
        if base_type is Any:
            base_type = dict
        predicates.append(_bounded_predicate(len, "dict size", _lo, _hi))
    if "set_length" in kwargs:
        _lo, _hi = kwargs.pop("set_length")
        if base_type is Any:
            base_type = set
        predicates.append(_bounded_predicate(len, "set size", _lo, _hi))

    # --- set relations / membership ---
    # Element-wise (==) rather than set ops, so UNHASHABLE members AND unhashable
    # values are tolerated (mirrors not_one_of's spirit) — no eager set()
    # leaks a raw TypeError out of the factory.
    if "subset_of" in kwargs:
        _sub = list(kwargs.pop("subset_of"))

        def _check_subset(v, _sub=_sub):
            if not all(any(x == m for m in _sub) for x in v):
                raise ValueError(f"{v!r} is not a subset of {_sub!r}")
        predicates.append(_check_subset)
    if "superset_of" in kwargs:
        _sup = list(kwargs.pop("superset_of"))

        def _check_superset(v, _sup=_sup):
            if not all(any(m == x for x in v) for m in _sup):
                raise ValueError(f"{v!r} is not a superset of {_sup!r}")
        predicates.append(_check_superset)
    if "list_contains" in kwargs:
        _x = kwargs.pop("list_contains")
        if base_type is Any:
            base_type = list

        def _check_list_contains(v, _x=_x):
            if _x not in v:
                raise ValueError(f"value must contain {_x!r}")
        predicates.append(_check_list_contains)

    # --- complex: magnitude / real / imaginary bounds (shared helper) ---
    if "abs_range" in kwargs:
        _lo, _hi = kwargs.pop("abs_range")
        if base_type is Any:
            base_type = complex
        predicates.append(_bounded_predicate(abs, "|z|", _lo, _hi))
    if "real_range" in kwargs:
        _lo, _hi = kwargs.pop("real_range")
        if base_type is Any:
            base_type = complex
        predicates.append(_bounded_predicate(lambda v: v.real, "Re", _lo, _hi))
    if "imag_range" in kwargs:
        _lo, _hi = kwargs.pop("imag_range")
        if base_type is Any:
            base_type = complex
        predicates.append(_bounded_predicate(lambda v: v.imag, "Im", _lo, _hi))

    # --- user predicates (single + plural) ---
    if "predicate" in kwargs:
        pred = kwargs.pop("predicate")
        if not callable(pred):
            raise TypeError("predicate= must be a callable")
        predicates.append(pred)
    if "predicates" in kwargs:
        preds = kwargs.pop("predicates")
        try:
            preds_list = list(preds)
        except TypeError:
            raise TypeError("predicates= must be an iterable of callables")
        for pred in preds_list:
            if not callable(pred):
                raise TypeError("predicates= contains a non-callable item")
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
    ("is_instance_of", False, lambda v: f"strictly an instance of {v.__name__} (no coercion)"),
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
    # type-completing additions:
    ("multiple_of", False, lambda v: f"a multiple of {v}"),
    ("finite",      True,  lambda v: "finite (not nan/inf)"),
    ("tuple_of",    False, lambda v: (f"tuple of {[_render_elem(x) for x in v]}"
                                      if isinstance(v, tuple) else f"tuple of [{_render_elem(v)}, ...]")),
    ("set_of",      False, lambda v: f"set of [{_render_elem(v)}]"),
    ("frozenset_of",False, lambda v: f"frozenset of [{_render_elem(v)}]"),
    ("bytes_length",False, lambda v: _range_phrase("byte length", v[0], v[1], ge=True, le=True)),
    ("str_prefix",  False, lambda v: f"starts with {v!r}"),
    ("str_suffix",  False, lambda v: f"ends with {v!r}"),
    ("str_contains",False, lambda v: f"contains {v!r}"),
    ("dict_length", False, lambda v: _range_phrase("dict size", v[0], v[1], ge=True, le=True)),
    ("set_length",  False, lambda v: _range_phrase("set size", v[0], v[1], ge=True, le=True)),
    ("subset_of",   False, lambda v: f"a subset of {list(v)}"),
    ("superset_of", False, lambda v: f"a superset of {list(v)}"),
    ("list_contains",False,lambda v: f"contains {v!r}"),
    ("abs_range",   False, lambda v: _range_phrase("|z|", v[0], v[1], ge=True, le=True)),
    ("real_range",  False, lambda v: _range_phrase("Re", v[0], v[1], ge=True, le=True)),
    ("imag_range",  False, lambda v: _range_phrase("Im", v[0], v[1], ge=True, le=True)),
    ("predicate",   False, lambda v: get_description(v) or "predicate-checked"),
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

    # Coercers — surface each @constraint("...") description, but rendered as a
    # TRANSFORM ("coerced (...)") so it reads as input preprocessing, NOT a check
    # the value must satisfy (predicates render their description BARE). This is the
    # signal that lets PLM tell coercing apart from testing. Untagged coercers
    # collapse to a "N coercer(s) applied" count.
    _coercers = ([kwargs["coercer"]] if "coercer" in kwargs else []) + list(kwargs.get("coercers") or [])
    if _coercers:
        _ctagged = [d for c in _coercers if (d := get_description(c)) is not None]
        parts.extend(f"coerced ({d})" for d in _ctagged)
        _c_untagged = len(_coercers) - len(_ctagged)
        if _c_untagged:
            parts.append(f"{_c_untagged} coercer(s) applied")

    # Predicates plural — surface each predicate's @constraint("...") description
    # individually (same as the single `predicate=` rule above and the structural
    # path), and count the UNTAGGED remainder as one "N predicate(s) checked" line.
    # (Coercers stay a pure count — they transform, they don't assert a property.)
    _preds_extra = list(kwargs.get("predicates") or [])
    if _preds_extra:
        _tagged = [d for p in _preds_extra if (d := get_description(p)) is not None]
        parts.extend(_tagged)
        _n_untagged = len(_preds_extra) - len(_tagged)
        if _n_untagged:
            parts.append(f"{_n_untagged} predicate(s) checked")

    desc = cls._constraint_description
    if desc:
        parts.append(desc)
    return ", ".join(parts) if parts else "no constraint"


def _nested_struct(ann: Any) -> Optional[Type["Constraint"]]:
    """If `ann` (or the inner type of `Optional[...]` / `X | None`) is a STRUCTURAL
    Constraint subclass — NOT a `.field`/factory or a composite, which describe
    themselves — return it, else None. Lets describe() recurse into nested object
    fields. Containers (`List[...]`/`Dict[...]`) are left to their flat rendering."""
    args = get_args(ann)
    if args and type(None) in args:                      # Optional[X] / X | None
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            ann = non_none[0]
    if (isinstance(ann, type) and issubclass(ann, Constraint)
            and not getattr(ann, "_constraint_is_factory", False)
            and not getattr(ann, "_constraint_is_composite", False)):
        return ann
    return None


def _collect_validator_descriptions(cls: Type[Constraint]) -> List[str]:
    """The `@constraint('...')` descriptions on `cls`'s validators (model + field),
    de-duplicated — for describe()'s 'Additionally:' block. Found via both `dir(cls)`
    and `__pydantic_decorators__` (covers either decoration order)."""
    extras: List[str] = []
    seen: set = set()

    def _add(d: Optional[str]) -> None:
        if d and d not in seen:
            seen.add(d)
            extras.append(d)

    for name in dir(cls):
        try:
            _member = getattr(cls, name, None)        # a descriptor __get__ can raise (non-AttributeError)
        except Exception:
            continue                                  # skip a member that can't even be read off the class
        _add(get_description(_member))
    decs = getattr(cls, "__pydantic_decorators__", None)
    if decs is not None:
        for v in list(getattr(decs, "model_validators", {}).values()):
            _add(get_description(v.func))
        for v in list(getattr(decs, "field_validators", {}).values()):
            _add(get_description(v.func))
    return extras


def _struct_body(cls: Type[Constraint], base: int, seen: set) -> List[str]:
    """Field lines (+ this class's 'Additionally:' block) for `cls`: field lines at
    indent `base+2`, the 'Additionally:' header at `base`. A field whose type is a
    nested structural Constraint (not already on the path — `seen` breaks cycles) is
    expanded inline two spaces deeper."""
    fpad = " " * (base + 2)
    pad = " " * base
    lines: List[str] = []
    for name, info in cls.model_fields.items():
        ann = info.annotation
        field_desc = info.description or ""

        nested = _nested_struct(ann)
        if nested is not None:
            head = f"{fpad}- {name} ({nested.__name__})"
            if nested in seen:                            # cycle — show the type, don't expand
                if field_desc:
                    head += f": {field_desc}"
                lines.append(head)
                continue
            if field_desc:
                head += f" — {field_desc}"
            lines.append(head + ":")
            lines.extend(_struct_body(nested, base + 2, seen | {nested}))
            continue

        if getattr(ann, "_constraint_is_field", False):
            # A `Constraint.field(...)`: its constraints live inside its core-schema
            # hook (not on this FieldInfo's metadata), so render the bare value TYPE
            # plus the field's COMPLETE factory describe (predicate/kind/one_of/...).
            t = _short(getattr(ann, "_constraint_field_inner_type", None) or ann)
            summary = ann.describe()
            line = f"{fpad}- {name} ({t})"
            if summary:
                line += f" — {summary}"
            if field_desc:
                line += f": {field_desc}"
            lines.append(line)
            continue

        constraints = _field_constraint_summary(info)
        line = f"{fpad}- {name} ({_short(ann)})"
        if constraints:
            line += f" — {constraints}"
        if field_desc:
            line += f": {field_desc}"
        lines.append(line)

    extras = _collect_validator_descriptions(cls)
    if extras:
        # Top-level "Additionally:" sits at the header column (base 0); a NESTED
        # class's sits at its own field column (fpad) so it clearly belongs to that
        # nested block rather than the enclosing object.
        a_pad = pad if base == 0 else fpad
        lines.append("")
        lines.append(f"{a_pad}Additionally:")
        for e in extras:
            lines.append(f"{a_pad}  - {e}")
    return lines


def _describe_structural(cls: Type[Constraint]) -> str:
    """Render an NL description for a structural Constraint subclass: a header, one
    line per field (nested structural fields expanded inline, cycle-guarded), and an
    'Additionally:' block for cross-field `@constraint`-tagged validators."""
    lines = [f"Return an object of type {cls.__name__} with fields:"]
    lines.extend(_struct_body(cls, base=0, seen={cls}))
    return "\n".join(lines)


def _field_constraint_summary(info: pd.fields.FieldInfo) -> str:
    """Render pydantic Field constraints (ge, le, pattern, ...) as a string."""
    bits: List[str] = []
    for meta in info.metadata or ():
        for attr in ("ge", "le", "gt", "lt", "min_length", "max_length", "pattern"):
            val = getattr(meta, attr, None)
            if val is not None:
                bits.append(f"{attr}={val!r}")
        # A struct-&-struct AUTO-MERGE keeps each side's NON-combinable constraints
        # (patterns, multiple_of, custom validators + their @constraint descriptions)
        # only inside its merge enforcer, since they can't reduce to a single visible
        # marker. The enforcer stashes them as NL phrases on its func, so surface them
        # here — otherwise the model wouldn't be TOLD about them upfront and would
        # only meet them on a violation. (#H3/#H4 follow-up)
        extra = getattr(getattr(meta, "func", None), "_plm_extra_desc", None)
        if extra:
            bits.extend(extra)
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
