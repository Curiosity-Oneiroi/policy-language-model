"""Constraint — generic constraint class.

PLM authors constraints two ways:

  Pattern 1 — Structural (subclass): an object SHAPE — EVERY field a Constraint.field(...).
      class HypothesisReduction(Constraint):
          facts:       Constraint.field(type=list[str])
          ambiguities: Constraint.field(type=list[str])
          hypotheses:  Constraint.field(type=dict[str, list[str]])
      # every field is a REQUIRED Constraint.field(...); cross-field invariants →
      # @model_validator(mode="after"). Bare annotations / Optional / defaults are
      # rejected at class definition.

  Pattern 2 — Value / field rule:  Constraint.field(**kw)
      The single public authoring entry for a value rule. It works BOTH standalone
      (validate a bare value, pass to a sub-LLM) AND as a struct FIELD, where it embeds
      FLAT (the field IS the bare value, not a {value:..} wrapper) while remaining a real
      Constraint (validate / describe / json_schema all work):
          class Order(Constraint):
              currency: Constraint.field(coercer=str.upper, type=Literal["USD","EUR"])
              qty:      Constraint.field(type=int, int_range=(1, 1000), multiple_of=5)

Every field compiles to a THREE-stage pipeline: (1) user COERCERS transform the value
first, (2) the MANDATORY TYPE COERCER establishes the value's type (and is the SOLE
source of the JSON-schema "type"), (3) PREDICATES are pure pass/fail checks that run last
and NEVER set a type (though some contribute non-"type" schema keywords like minimum/
pattern/minLength to enrich it). You MUST give exactly one type coercer; predicate kwargs
alone (e.g. `int_range=` with no `type=`) are an error.

The Constraint.field(**kw) kwarg surface:
  TYPE COERCER (mandatory, exactly one — ALWAYS spelled `type=`):
    type    : type=int|float|str|bool|bytes|complex
              | type=list[T] | dict[K,V] | tuple[T1,..] | tuple[T,...] | set[T] | frozenset[T]
              | type=Literal["a","b",..]   (an enum of values — the old `one_of`)
              | type=YourStruct            (a Constraint/model class — coerces a dict into an instance)
              The element T in a generic may ITSELF be a Constraint (e.g.
              type=list[Constraint.field(type=int, int_range=(0,9))]) — pydantic embeds it natively.
              NOT nullable, NOT a Union/`X|Y` (top-level OR nested) — model alternatives with a
              @model_validator (or type=object + a predicate), not inside one field's type.
  PREDICATES (pure checks; never set a type; some enrich the schema):
    numeric : int_range / float_range, int_ge/gt/le/lt (+ float_*), multiple_of,
              approx=(target,eps), finite=True  (reject nan/inf)
    member  : not_one_of=[...]
    str     : str_pattern, str_length=(lo,hi), not_pattern, not_empty,
              str_prefix, str_suffix, str_contains
    list    : list_length=(lo,hi), unique, sorted, monotonic,
              list_contains=X  (the list must contain X as an element, via `in`)
    set     : set_length, subset_of=[..], superset_of=[..]
    dict    : dict_length=(lo,hi)
    bytes   : bytes_length=(lo,hi)   (a bytearray is accepted but normalized to bytes)
    complex : abs_range=(lo,hi), real_range=(lo,hi), imag_range=(lo,hi)
    type    : is_instance_of=T (a PURE isinstance check — sets no type; pair with type=object
              for a STRICT no-coercion gate), callable=True, behaves_like=[((args),out),..]
    named   : kind="email|url|uuid|semver|iso_date|identifier|json|regex|path_exists" (needs type=str)
    custom  : predicate=fn / predicates=[fn,..], coercer=fn / coercers=[fn,..], description="..."
  bool/None : no dedicated kwargs by design — use type=Literal[True]/[False] / type=Literal[None].

No finite kwarg set is complete: for relations / aggregates / domain logic
("sums to 100", "is prime", "compiles") drop to predicate=fn WITH a description=
(the description is what still steers the model). To require several rules/shapes at once,
author ONE constraint with all of them (+ a @model_validator for cross-field rules).

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
  coercer(s)  →  type coercer  →  predicate(s)

Both patterns expose the same API on the class:

  .validate(value, *, context=None) -> value
  .describe() -> str
  .json_schema() -> dict

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
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
)

import copy
import pydantic as pd
from pydantic import AfterValidator, BeforeValidator, model_validator
from typing_extensions import Annotated

from .decorator import _DESCRIPTION_ATTR, get_description
from .registry import REGISTRY


# ============================================================================
# Exceptions
# ============================================================================


class ConstraintViolation(ValueError):
    """Raised when a value fails any layer of a Constraint check."""


# ============================================================================
# Metaclass — field-authoring guard (every field a required Constraint.field) + super() rejection
# ============================================================================


_PydanticMeta: type = type(pd.BaseModel)


def _safe_copy_kwargs(d: Dict[str, Any]) -> Dict[str, Any]:
    """Copy a factory-kwargs dict, deep-copying each value to avoid the facade aliasing inner's
    mutable values — but PER VALUE, falling back to the original reference for any value
    that is NOT deepcopyable (a lock / file handle / live resource). A construction-time crash is
    strictly worse than the rare residual aliasing for such an exotic value, and one non-copyable
    value must not cost the others their deep copy."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        try:
            out[k] = copy.deepcopy(v)
        except Exception:
            out[k] = v
    return out


def _calls_super_builtin(code: Any) -> bool:
    """True iff `code` loads the `super` BUILTIN as a global/name/free var — i.e. the method itself
    calls `super()` (zero-arg) or `super(C, self)` (explicit). Precise, with NO gotchas:
      * an ATTRIBUTE/method named `super` (`n.super`, `n.super()`) is a LOAD_ATTR/LOAD_METHOD -> NOT
        flagged (allowed);
      * a LOCAL named `super` (`super = ...; super()`) is a LOAD_FAST -> NOT flagged (it shadows the
        builtin, isn't inheritance);
      * a different name (`fsuper`, `super_helper`) -> exact `argval == "super"` -> NOT flagged;
      * a NESTED helper's `super()` lives in its OWN code object — a `def`/`lambda`/generator-expr
        in co_consts — not in `code`'s instruction stream -> NOT flagged. (PEP-709 caveat, Py3.12+:
        a list/set/dict COMPREHENSION is INLINED into the enclosing code object, so a `super()`
        *inside a comprehension* IS in `code` and WOULD be flagged. `super()` inside a comprehension
        in a flat constraint method is pathological, so this is noted, not worked around.)
    Replaces a `'super' in co_names` membership test, which wrongly flagged `n.super` (co_names also
    holds LOAD_ATTR names)."""
    import dis as _dis
    for _ins in _dis.get_instructions(code):
        if _ins.opname in ("LOAD_GLOBAL", "LOAD_NAME", "LOAD_DEREF") and _ins.argval == "super":
            return True
    return False


class _ConstraintMeta(_PydanticMeta):  # type: ignore[misc, valid-type]
    """Constraint metaclass: at class-definition time, rejects any constraint method that calls
    `super()`. A Constraint is a FLAT structural validator (fields + validators + predicates), not
    an inheritance hierarchy, and a `super()` method's hidden `__class__` cell can't dill-pickle for
    crash-restart — so catch the misuse loudly at definition rather than failing later."""

    def __init__(cls, *args, **kwargs):
        # Done in __init__, NOT __new__: pydantic resolves forward references by walking the call
        # frames in its metaclass `__new__`, so an extra `__new__` here shifts that frame and breaks
        # nested/forward-ref constraints. `__init__` runs AFTER the class is fully built, so the scan
        # can't disturb it.
        super().__init__(*args, **kwargs)
        # A Constraint is a FLAT structural validator — fields + @field/@model_validator +
        # predicates — NOT an OOP inheritance hierarchy. A method that CALLS `super()` is delegating
        # to a parent constraint, which is unsupported: it doesn't fit the model AND its hidden
        # `__class__` cell can't dill-pickle. Reject it at DEFINITION with a clear message instead of
        # failing mysteriously later. None of the library's own classes (base / field)
        # call super(), so this only fires on a misuse. (Constraint contract; supersedes)
        #
        # Detection: `_calls_super_builtin` — does THIS method LOAD the `super` builtin (global/name/
        # free var)? Catches BOTH zero-arg `super()` and explicit `super(C, self)`. It does NOT
        # false-positive on an ATTRIBUTE/field named `super` (`n.super` — LOAD_ATTR), a LOCAL named
        # `super`, a different name (`fsuper`/`super_x`), or a NESTED helper's super (its own code
        # object). An earlier `'super' in co_names` test wrongly flagged `n.super` (co_names also
        # holds attribute names). The rarer residual `__class__`-cell cases the check intentionally
        # allows (a nested-helper super, a bare `__class__` reference) are handled defensively by
        # `from_recipe`'s cell-rebind. (F35)
        import types as _types
        for _k, _v in vars(cls).items():
            _fn = _v.__func__ if isinstance(_v, (classmethod, staticmethod)) else _v
            # Gate on FunctionType (a pure type check) BEFORE touching `.__code__`: a pathological
            # class member with a raising __getattr__ must NOT break constraint definition (NC-8).
            if isinstance(_fn, _types.FunctionType) and _calls_super_builtin(_fn.__code__):
                raise TypeError(
                    f"Constraint {cls.__name__!r}: method {_k!r} calls super(), which constraints do "
                    f"NOT support — a Constraint is a flat structural validator (fields + validators "
                    f"+ predicates), not an inheritance hierarchy. Remove the super()/inheritance."
                )

        # FIELD-AUTHORING GUARD: every field of a STRUCTURAL constraint must be authored through
        # `Constraint.field(...)` AND be REQUIRED — the single, elegant authoring path, enforced by TWO
        # checks: (1) SHAPE — a bare annotation (`x: int`, `x: Foo`, a bare nested `leg: Leg`) or a raw
        # pydantic `Field(...)` is rejected; only a `.field`-built annotation (carrying
        # `_constraint_is_factory`) is accepted. (2) REQUIRED — a default / default_factory makes a
        # `.field` omittable (`= 3` keeps the factory annotation but yields a non-required field), which
        # the strict model forbids, so reject any non-required field too. Factory classes are exempt
        # (their own `value` field is the engine's internal annotation). `model_fields[name].annotation`
        # is the resolved object (works under `from __future__ import annotations`), and a forward-ref
        # recursion field `Constraint.field(type=list["X"])` still presents as a required factory field.
        if not getattr(cls, "_constraint_is_factory", False):
            for _fname, _finfo in getattr(cls, "model_fields", {}).items():
                if not getattr(_finfo.annotation, "_constraint_is_factory", False):
                    raise TypeError(
                        f"Constraint {cls.__name__!r}: field {_fname!r} must be authored via "
                        f"Constraint.field(...), not a bare annotation — write "
                        f"`{_fname}: Constraint.field(type=...)`. Bare annotations, a bare nested "
                        f"Constraint, Optional, defaults, and raw pydantic Field(...) are not supported "
                        f"(every field is a required Constraint.field; nest via type=YourStruct, "
                        f"recurse via type=list[\"YourClass\"])."
                    )
                if not _finfo.is_required():
                    raise TypeError(
                        f"Constraint {cls.__name__!r}: field {_fname!r} has a default — every field is a "
                        f"REQUIRED Constraint.field(...). A default value or default_factory makes a field "
                        f"omittable, which the strict .field-only model does not support; drop the default."
                    )


# ============================================================================
# Constraint base
# ============================================================================


class Constraint(pd.BaseModel, metaclass=_ConstraintMeta):
    """Base class for all constraints.

    Subclass for structural constraints (fields + pydantic validators), or call
    `Constraint.field(...)` for a value/field rule — the single public authoring
    entry for non-structural constraints.

    A Constraint is a FLAT structural validator — fields + `@field_validator` /
    `@model_validator` + predicates. It is NOT an OOP inheritance hierarchy: a
    method using bare `super()` (delegating to a parent constraint) is REJECTED at
    definition. Constraints stay simple and self-contained; nest one as another's
    field type instead of subclassing+super()."""

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

    # Original kwargs passed to the factory builder (for .describe() + crash-restart recipe).
    _constraint_factory_kwargs: ClassVar[Optional[Dict[str, Any]]] = None

    # Non-"type" JSON-Schema fragments each predicate contributes (range/pattern/length/...),
    # in declaration order — merged by json_schema() to enrich the field schema. Declared
    # ClassVar (not a bare `_`-attr) so pydantic treats it as a real introspectable class var,
    # not a ModelPrivateAttr. Empty for structural / predicate-less constraints.
    _constraint_schema_fragments: ClassVar[List[Dict[str, Any]]] = []

    # NL description carried verbatim through .describe().
    _constraint_description: ClassVar[str] = ""

    # Read-only table of built-in `kind=` predicates (a MappingProxyType — never mutated at
    # runtime). For a custom check use `Constraint.field(..., predicate=fn)`, not a runtime write.
    registry: ClassVar[Mapping[str, Callable[[Any], None]]] = REGISTRY

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
        standalone (`.validate` a bare value, pass to a sub-LLM) AND as a struct
        FIELD, where it embeds FLAT — the field IS the
        bare value, carrying the constraints — while remaining a real `Constraint`
        (`.validate` / `.describe` / `.json_schema` all work):

            class Order(Constraint):
                currency: Constraint.field(coercer=str.upper, type=Literal["USD","EUR"])
                qty:      Constraint.field(type=int, int_range=(1, 1000), multiple_of=5)
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

        def _get_json(c, _cs, handler, _i=inner):
            # When a `.field` is embedded as a STRUCT FIELD, pydantic builds the parent schema
            # from the FLAT core schema (the bare type) — which no longer carries the bounds,
            # since refinements are now predicate FRAGMENTS, not pydantic Field() kwargs. Merge
            # those fragments here so the struct-field schema keeps surfacing minimum/maximum/
            # pattern/multipleOf/... for structured-output steering (symmetric with the
            # standalone `json_schema()` path). Predicates still never inject a "type".
            schema = dict(handler(_cs))
            for frag in getattr(_i, "_constraint_schema_fragments", ()) or ():
                _merge_schema_fragment(schema, frag)
            return schema

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
            "__get_pydantic_json_schema__": classmethod(_get_json),
            "validate": classmethod(_validate),
            "describe": classmethod(_describe),
            "json_schema": classmethod(_json_schema),
            "_constraint_is_factory": True,            # so _describe_factory / natural_llm's gate see it
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
            # per-value deepcopy (not a shallow dict()): a shallow copy aliases the inner's mutable
            # VALUES (e.g. the `one_of` list), so mutating the facade's copy would silently change
            # inner's. `_safe_copy_kwargs` falls back to the reference for a non-deepcopyable
            # value so an exotic kwarg can't crash construction .
            "_constraint_factory_kwargs": _safe_copy_kwargs(inner._constraint_factory_kwargs),
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
        # All registry kinds are string-shaped validators (semver/url/email/uuid/iso_date/
        # json/regex/identifier/path_exists). The reframe needs an explicit type coercer, and
        # `kind=` is itself a predicate now (with a `format` schema hint where expressible), so
        # build a string field running that kind's check.
        return cls.field(type=str, kind=name)

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
        """Export as JSON Schema — AIRTIGHT (never raises), compositional.

        Structural Constraints export via pydantic; a field typed as a non-renderable
        arbitrary class degrades to `{}` + x-description rather than raising. For a factory-built
        Constraint the schema is built in three steps from the reframe pipeline:
          1. the "type" SKELETON comes ONLY from the type coercer (the value
             annotation) — the sole source of `"type"`. A non-renderable type
             (an arbitrary-class `type=`/`is_instance_of`) degrades to `{}`
             rather than crashing (this is what closed the is_instance_of crash).
          2. each predicate's NON-"type" `fragment` (minimum/maximum/pattern/
             minLength/uniqueItems/format/...) is MERGED in to enrich the schema;
             predicates never set a type.
          3. `x-description` carries the NL description for everything the schema
             can't express (is_instance_of, unique, sorted, user predicates, ...).
        """
        if not cls._constraint_is_factory:
            # AIRTIGHT (mirrors the factory branch below): a structural field typed as an
            # arbitrary class with no JSON-Schema form makes model_json_schema() raise
            # PydanticInvalidForJsonSchema. json_schema() must NEVER raise (natural_llm builds
            # response_format from it OUTSIDE its retry loop), so degrade to a type-less skeleton
            # + x-description from the (airtight) describe(), exactly as the factory branch does.
            try:
                return cls.model_json_schema()
            except Exception:
                try:
                    x_desc = cls.describe()
                except Exception:
                    x_desc = ""
                return {"x-description": x_desc} if x_desc else {}

        # 1. Type skeleton from the type coercer. model_json_schema() can raise for an
        #    arbitrary-class type coercer (no JSON-Schema form) — json_schema() must NEVER
        #    raise (natural_llm builds response_format from it OUTSIDE its retry loop), so
        #    fall back to an empty (type-less) skeleton and let x-description carry the intent.
        try:
            schema = cls.model_json_schema()
            value_schema = dict(schema.get("properties", {}).get("value", schema))
            # Preserve the document's `$defs`: when the value annotation references another
            # model (e.g. type=Model, type=list[Model]), the flattened value schema carries
            # `$ref: #/$defs/Model` but the definitions live at the top level — carry them
            # along so the refs still resolve in the exported (now top-level) schema.
            if "$defs" in schema and "$defs" not in value_schema:
                value_schema["$defs"] = schema["$defs"]
        except Exception:
            value_schema = {}

        # 2. Merge each predicate's non-"type" fragment (tightest bound wins; "type" dropped).
        for frag in getattr(cls, "_constraint_schema_fragments", ()) or ():
            _merge_schema_fragment(value_schema, frag)

        # 3. x-description. describe() can raise on a pathological constraint; never let that
        #    abort json_schema(). Omit x-desc on failure.
        try:
            x_desc = cls.describe()
        except Exception:
            x_desc = ""
        if x_desc:
            value_schema["x-description"] = x_desc
        return value_schema


def _iter_constraints_in(ann: Any):
    """Yield every Constraint subclass reachable in a type annotation — the annotation itself
    if it's a Constraint (a `.field` facade / struct), or any nested inside a typing
    generic (`list[X]`, `dict[K,V]`, `tuple[..]`, ...). Used to walk a factory's `type=` spec and
    a struct's field annotations for the natural_llm description gate."""
    if isinstance(ann, type) and issubclass(ann, Constraint) and ann is not Constraint:
        yield ann
        return
    for a in get_args(ann):
        yield from _iter_constraints_in(a)


def _all_checks_described(constraint: type, _seen: Optional[set] = None) -> bool:
    """natural_llm GATE (not a general-constraint rule): True iff EVERY user-supplied
    predicate / coercer / `@model_validator` / `@field_validator` anywhere in `constraint`'s
    tree carries a description (built-in refinements like int_range/str_pattern auto-describe,
    so they always pass).

    natural_llm steers a single-shot model with the schema (shape) + `describe()` (the rules);
    a check with NO description is invisible to `describe()` — the model can never be told about
    it, so single-shot generation can only fail it. Such a constraint is rejected up front and
    routed to `react_llm`. (Walks struct fields and `type=` generic elements.)"""
    if _seen is None:
        _seen = set()
    if id(constraint) in _seen:
        return True
    _seen.add(id(constraint))

    # factory (.of / .field) -> user predicate/coercer fns must be described; recurse type= elements
    if getattr(constraint, "_constraint_is_factory", False):
        kw = getattr(constraint, "_constraint_factory_kwargs", None) or {}
        fns: List[Any] = []
        if "predicate" in kw:
            fns.append(kw["predicate"])
        fns += list(kw.get("predicates") or [])
        if "coercer" in kw:
            fns.append(kw["coercer"])
        fns += list(kw.get("coercers") or [])
        if any(get_description(fn) is None for fn in fns):
            return False
        for nested in _iter_constraints_in(kw.get("type")):
            if not _all_checks_described(nested, _seen):
                return False
        return True

    # structural Constraint -> every model/field validator must be described; recurse field types
    if isinstance(constraint, type) and issubclass(constraint, Constraint):
        decs = getattr(constraint, "__pydantic_decorators__", None)
        if decs is not None:
            for v in list(getattr(decs, "model_validators", {}).values()):
                if get_description(v.func) is None:
                    return False
            for v in list(getattr(decs, "field_validators", {}).values()):
                if get_description(v.func) is None:
                    return False
        for finfo in getattr(constraint, "model_fields", {}).values():
            for nested in _iter_constraints_in(finfo.annotation):
                if not _all_checks_described(nested, _seen):
                    return False
        return True

    return True   # a plain (non-Constraint) annotation carries no checks


def _has_generatable_structure(schema: Any) -> bool:
    """True iff a JSON schema pins a concrete generation target the model can produce against —
    a `type` / `enum` / `const` / `properties` / `items` / `$ref`, or a combinator (anyOf/allOf/
    oneOf) whose members ALL qualify. A schema that is only `{title, x-description}` (no type —
    e.g. a bare predicate / is_instance_of on `type=object`) pins NO shape. natural_llm uses this
    to decide whether to hard-set `response_format` (structured steering) or fall back to
    describe-only generation — NOT to accept/reject (that's `_all_checks_described`)."""
    if not isinstance(schema, dict):
        return False
    if any(k in schema for k in ("type", "enum", "const", "properties", "items", "$ref")):
        return True
    for comb in ("anyOf", "allOf", "oneOf"):
        members = schema.get(comb)
        if isinstance(members, list) and members and all(_has_generatable_structure(m) for m in members):
            return True
    return False


# ============================================================================
# Factory engine (internal) — _build_factory(**kwargs); public entry is Constraint.field
# ============================================================================


_KNOWN_KWARGS = frozenset(
    {
        # the mandatory type coercer (sole source of the value's type + schema "type")
        "type",
        "int_range", "int_gt", "int_ge", "int_lt", "int_le",
        "float_range", "float_gt", "float_ge", "float_lt", "float_le",
        "is_instance_of", "not_one_of",
        "str_pattern", "str_length", "not_pattern", "not_empty",
        "list_length", "unique", "sorted", "monotonic",
        "approx", "callable", "behaves_like",
        "kind", "predicate", "predicates", "coercer", "coercers", "description",
        # type-completing additions:
        "multiple_of", "finite",
        "bytes_length",
        "str_prefix", "str_suffix", "str_contains",
        "dict_length", "set_length", "subset_of", "superset_of", "list_contains",
        "abs_range", "real_range", "imag_range",
    }
)


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


# ---- The reframe: type coercer (sole type source) vs predicates (pure checks) ----
#
# A field compiles to THREE stages: (1) user coercers (BeforeValidators, run first),
# (2) the MANDATORY type coercer (the value annotation — the sole source of the schema's
# "type"; spelled ONLY via `type=`: a primitive, a typing generic like list[int]/dict[str,int],
# a Literal[...], or a Constraint struct), and (3) predicates (AfterValidators — pure pass/fail
# checks that NEVER set a type but MAY contribute non-"type" JSON-Schema keywords to enrich it).


class _Predicate:
    """A stage-3 check: a runtime pass/fail `check` plus an OPTIONAL non-"type" JSON-Schema
    `fragment` (e.g. {"minimum": 1}) that enriches the field schema. A predicate NEVER sets a
    type — `fragment` carries only refinement keywords (and any "type" key is dropped on merge)."""

    __slots__ = ("check", "fragment")

    def __init__(self, check: Callable[[Any], Any], fragment: Optional[Dict[str, Any]] = None):
        self.check = check
        self.fragment = fragment or None


# JSON-Schema keywords where merging two values keeps the TIGHTER bound (the exported schema
# must not advertise a looser constraint than the predicates actually enforce).
_TIGHTEN_MAX: frozenset = frozenset(
    {"minimum", "exclusiveMinimum", "minLength", "minItems", "minProperties"})
_TIGHTEN_MIN: frozenset = frozenset(
    {"maximum", "exclusiveMaximum", "maxLength", "maxItems", "maxProperties"})


def _merge_schema_fragment(schema: Dict[str, Any], frag: Dict[str, Any]) -> None:
    """Merge a predicate's non-"type" fragment into the field schema in place. `"type"` is
    dropped (predicates never type). Same-key numeric/length bounds keep the tighter value;
    a second `pattern` is kept out (both patterns still run as AfterValidators — the schema
    just can't express two)."""
    for k, val in frag.items():
        if k == "type":
            continue                                  # structural guarantee: predicates never type
        if k not in schema:
            schema[k] = val
        elif k in _TIGHTEN_MAX:
            try:
                schema[k] = max(schema[k], val)
            except TypeError:
                schema[k] = val
        elif k in _TIGHTEN_MIN:
            try:
                schema[k] = min(schema[k], val)
            except TypeError:
                schema[k] = val
        elif k == "pattern":
            pass                                      # keep the first; AfterValidators enforce both
        else:
            schema[k] = val


def _reject_nullable_union(spec: Any, where: str = "type=") -> None:
    """A field is NEVER nullable and NEVER a type-union: reject `None`/`NoneType`, `Optional[...]`,
    and any union (`Union[...]` / `X | Y`) — at the top level AND nested inside a generic
    (`list[Optional[int]]`). A field has exactly ONE type; model alternatives with a
    `@model_validator` (or `type=object` + a predicate), never inside one field's type.
    (`Literal[...]` VALUES are not types, so its args are not recursed — `Literal[None]` means
    'the constant None', which is allowed.)"""
    if spec is None or spec is type(None):
        raise ValueError(
            f"Constraint.field: {where} cannot be None/NoneType — a field is never nullable. "
            "Model alternatives with a @model_validator (or type=object + a predicate)."
        )
    origin = get_origin(spec)
    import types as _t
    if origin is Union or (hasattr(_t, "UnionType") and isinstance(spec, _t.UnionType)):
        if type(None) in get_args(spec):                  # Optional[X] / X | None
            raise ValueError(
                f"Constraint.field: {where} cannot be Optional / nullable (a Union with None) — "
                "a field is never nullable. Model alternatives with a @model_validator "
                "(or type=object + a predicate)."
            )
        raise ValueError(
            f"Constraint.field: {where} cannot be a Union / `X | Y` — a field has exactly ONE type. "
            "Model alternatives with a @model_validator (or type=object + a predicate)."
        )
    if origin is not None and origin is not Literal:
        for arg in get_args(spec):
            _reject_nullable_union(arg, where=where)


def _resolve_type_coercer(kwargs: Dict[str, Any]) -> Any:
    """Resolve and POP the one mandatory type coercer from `type=`. It is the SOLE spelling of the
    type — a builtin (int/float/str/bool/bytes/complex), a typing generic (`list[int]`,
    `dict[str,int]`, `tuple[..]`, `set[int]`, `frozenset[int]`), a `Literal[...]` (an enum of
    values), or a Constraint struct / model class. Generic element types may themselves be
    Constraints (e.g. `list[Constraint.field(...)]`) — pydantic embeds their flat schema natively.
    Raises loudly on a missing type or a nullable/union spec. Everything else in `kwargs` is a
    stage-3 predicate / stage-1 coercer."""
    if "type" not in kwargs:
        raise ValueError(
            "Constraint.field: no type coercer — every field must declare its type via `type=`: "
            "a builtin (int/float/str/bool/bytes/complex), a typing generic (list[int], "
            "dict[str,int], tuple[..], set[int], frozenset[int]), a Literal[...] (an enum of "
            "values), or a Constraint struct. A predicate (int_range, str_pattern, is_instance_of, "
            "kind=, ...) does NOT establish a type."
        )
    spec = kwargs.pop("type")
    _reject_nullable_union(spec)
    return spec


def _cmp_check(bound: Any, kind: str) -> Callable[[Any], None]:
    """Build a single-sided comparison predicate (`gt`/`ge`/`lt`/`le`). A non-comparable value
    funnels through `_wrap_predicate` into a ConstraintViolation like any other reject."""
    sym = {"gt": ">", "ge": ">=", "lt": "<", "le": "<="}[kind]

    def _check(v: Any, _b=bound, _k=kind, _sym=sym) -> None:
        try:
            ok = (v > _b) if _k == "gt" else (v >= _b) if _k == "ge" else (v < _b) if _k == "lt" else (v <= _b)
        except TypeError as e:
            raise ValueError(f"cannot compare {v!r} to {_b!r} ({e})") from None
        if not ok:
            raise ValueError(f"{v!r} is not {_sym} {_b}")
    return _check


_CMP_FRAG_KEY: Dict[str, str] = {
    "gt": "exclusiveMinimum", "ge": "minimum", "lt": "exclusiveMaximum", "le": "maximum",
}


# ---- is_instance_of: a STRICT Python isinstance check (type membership) ----
# It accepts a plain class, a STRUCT Constraint, or a TUPLE of those (Python's native "instance of
# any of these"). For "an instance of A or B" use a tuple `is_instance_of=(A, B)`. It does NOT
# accept a `.field`/factory — a value is never an instance of the dynamic wrapper class.
def _isinstance_target_reason(t: Any) -> Optional[str]:
    """Return WHY `t` is not a valid is_instance_of target, or None if it's fine (a plain class, a
    struct Constraint, or a tuple of those)."""
    if isinstance(t, tuple):
        if not t:
            return "an empty tuple — give at least one class"
        for x in t:
            r = _isinstance_target_reason(x)
            if r:
                return f"tuple element {x!r}: {r}"
        return None
    if isinstance(t, type) and issubclass(t, Constraint):
        if getattr(t, "_constraint_is_factory", False) or getattr(t, "_constraint_is_field", False):
            return ("a Constraint.field(...)/.of(...) factory — a value is never a Python instance of "
                    "the dynamic wrapper class. Use a struct Constraint or a plain class (or use it as "
                    "the type=)")
        return None                                   # a STRUCT Constraint is isinstance-able
    if isinstance(t, type):
        return None                                   # any plain class
    return f"not a class (got {type(t).__name__})"


def _isinstance_describe(target: Any) -> str:
    """Render an accepted is_instance_of target for describe(): a class -> its short name; a tuple ->
    `(A / B)`."""
    if isinstance(target, tuple):
        return "(" + " / ".join(_short(t) for t in target) + ")"
    return _short(target)


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
    for _key in ("not_one_of", "coercers", "predicates", "behaves_like",
                 "subset_of", "superset_of"):
        if kwargs.get(_key) is not None:
            try:
                kwargs[_key] = list(kwargs[_key])
            except TypeError:
                pass

    # Save originals for .describe()
    original_kwargs = dict(kwargs)

    description = kwargs.pop("description", "") or ""

    # The reframe pipeline: stage-2 type coercer (the value annotation, sole "type" source),
    # stage-1 user coercers, stage-3 predicate records (check + optional non-"type" schema fragment).
    type_spec, coercers, predicate_records = _interpret_kwargs(kwargs)

    # Build the annotated type. Execution order pydantic gives us:
    #   BeforeValidators (LIFO from metadata) → user coercers run FIRST
    #   type coerce (the type_spec annotation)→ establishes the value's type
    #   AfterValidators  (FIFO from metadata) → predicate checks run LAST
    #
    # Pydantic v2 quirk: BeforeValidators execute in REVERSE metadata order
    # (last-added runs first), while AfterValidators execute in metadata
    # order. To preserve user-facing left→right declaration semantics for
    # both, we iterate coercers in REVERSE when wrapping and predicates in
    # FORWARD order. NOTE: refinement bounds are NO LONGER pydantic Field()
    # kwargs — they are predicate checks carrying a json-schema fragment
    # (merged in json_schema()); only the type coercer feeds the schema "type".
    type_ann: Any = type_spec
    for coercer in reversed(coercers):
        type_ann = Annotated[type_ann, BeforeValidator(_wrap_coercer(coercer))]
    for prec in predicate_records:
        type_ann = Annotated[type_ann, AfterValidator(_wrap_predicate(prec.check))]

    # The non-"type" schema keywords each predicate contributes (range/pattern/length/...),
    # in declaration order — the channel json_schema() merges to enrich the field schema
    # (replacing the bounds pydantic's Field used to emit).
    schema_fragments = [prec.fragment for prec in predicate_records if prec.fragment]

    # Build a dynamic Constraint subclass with a single `value` field.
    cls_dict: Dict[str, Any] = {
        "__annotations__": {"value": type_ann},
        "_constraint_is_factory": True,
        "_constraint_factory_kwargs": original_kwargs,
        "_constraint_schema_fragments": schema_fragments,
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
) -> Tuple[Any, List[Callable[[Any], Any]], List["_Predicate"]]:
    """Translate the factory kwarg surface into the 3-stage pipeline parts.

    Returns:
      type_spec:  the MANDATORY type coercer — the value annotation pydantic validates
                  against and the SOLE source of the schema's "type" (resolved + popped
                  from `type=` xor one container/enum sugar by `_resolve_type_coercer`).
      coercers:   stage-1 user `(value) -> new_value` transforms (BeforeValidators), in
                  declaration order (fn1's output feeds fn2).
      predicates: stage-3 `_Predicate` records — a pure `(value) -> None | raises ValueError`
                  check plus an OPTIONAL non-"type" JSON-Schema `fragment`. Predicates NEVER
                  set a type; they only check (and may enrich the schema).
    """
    # Stage 2 — resolve + pop the one mandatory type coercer FIRST (raises on zero / conflict /
    # nullable / union). Everything left in `kwargs` is a stage-3 predicate or stage-1 coercer.
    type_spec: Any = _resolve_type_coercer(kwargs)

    predicates: List[_Predicate] = []
    coercers: List[Callable[[Any], Any]] = []

    # --- kind (registry-backed predicate; `format` hint where JSON-expressible) ---
    if "kind" in kwargs:
        name = kwargs.pop("kind")
        if name not in REGISTRY:
            raise KeyError(f"unknown kind {name!r}; available: {sorted(REGISTRY)}")
        _KIND_FORMAT = {"email": "email", "url": "uri", "uuid": "uuid",
                        "iso_date": "date", "regex": "regex"}
        predicates.append(_Predicate(
            REGISTRY[name],
            {"format": _KIND_FORMAT[name]} if name in _KIND_FORMAT else None,
        ))

    # --- numeric ranges (closed) — check + minimum/maximum fragment ---
    if "int_range" in kwargs:
        lo, hi = kwargs.pop("int_range")
        frag: Dict[str, Any] = {}
        if lo is not None:
            frag["minimum"] = lo
        if hi is not None:
            frag["maximum"] = hi
        predicates.append(_Predicate(_bounded_predicate(lambda v: v, "value", lo, hi), frag or None))
    if "float_range" in kwargs:
        lo, hi = kwargs.pop("float_range")
        frag = {}
        if lo is not None:
            frag["minimum"] = lo
        if hi is not None:
            frag["maximum"] = hi
        predicates.append(_Predicate(_bounded_predicate(lambda v: v, "value", lo, hi), frag or None))

    # --- numeric single bounds (strict + inclusive) ---
    for _kk in ("int_gt", "int_ge", "int_lt", "int_le",
                "float_gt", "float_ge", "float_lt", "float_le"):
        if _kk in kwargs:
            _bound = kwargs.pop(_kk)
            _kind = _kk.split("_", 1)[1]               # gt / ge / lt / le
            predicates.append(_Predicate(_cmp_check(_bound, _kind), {_CMP_FRAG_KEY[_kind]: _bound}))

    # --- is_instance_of (PURE isinstance check — sets NO type, contributes NO schema) ---
    if "is_instance_of" in kwargs:
        target = kwargs.pop("is_instance_of")
        # Accept a plain class / struct Constraint / tuple of those; reject a factory (a value is
        # never a Python instance of the dynamic wrapper).
        _reason = _isinstance_target_reason(target)
        if _reason is not None:
            raise TypeError(f"is_instance_of= must be a class, a struct Constraint, or a tuple of "
                            f"those; got {_reason}.")

        def _check_isinstance(v, _t=target):
            # bool IS an int subclass; `is_instance_of=int` must reject it (use is_instance_of=bool).
            if _t is int and isinstance(v, bool):
                raise ValueError(
                    "strictly an instance of int (a bool is not accepted; use is_instance_of=bool)")
            if not isinstance(v, _t):
                raise ValueError(f"value of type {type(v).__name__} is not strictly an instance of "
                                 f"{_isinstance_describe(_t)}")
        predicates.append(_Predicate(_check_isinstance, None))

    # --- not_one_of ---
    if "not_one_of" in kwargs:
        seq = kwargs.pop("not_one_of")
        try:
            bad = set(seq)                            # hashable members: fast set membership
        except TypeError:                             # unhashable members (e.g. lists): fall
            bad = list(seq)                           # back to list membership (hashable-member fast path + element-wise fallback)

        def _check_not_one_of(v, _bad=bad):
            try:
                forbidden = v in _bad
            except TypeError:
                # unhashable value vs hashable-member set: it can never equal a hashable scalar,
                # so it is genuinely NOT forbidden — element-wise == keeps it allowed.
                forbidden = any(v == b for b in _bad)
            if forbidden:
                raise ValueError(f"{v!r} is in the forbidden set {list(_bad)!r}")
        predicates.append(_Predicate(_check_not_one_of, None))

    # --- str pattern / length ---
    if "str_pattern" in kwargs:
        import re as _re
        _pat = _re.compile(kwargs.pop("str_pattern"))

        def _check_pattern(v, _p=_pat):
            if not isinstance(v, str) or _p.search(v) is None:
                raise ValueError(f"{v!r} does not match pattern /{_p.pattern}/")
        predicates.append(_Predicate(_check_pattern, {"pattern": _pat.pattern}))
    if "str_length" in kwargs:
        lo, hi = kwargs.pop("str_length")
        frag = {}
        if lo is not None:
            frag["minLength"] = lo
        if hi is not None:
            frag["maxLength"] = hi
        predicates.append(_Predicate(_bounded_predicate(len, "length", lo, hi), frag or None))
    if "not_pattern" in kwargs:
        import re as _re
        _np = _re.compile(kwargs.pop("not_pattern"))

        def _check_not_pattern(v, _p=_np):
            if isinstance(v, str) and _p.search(v):
                raise ValueError(f"{v!r} matches forbidden pattern /{_p.pattern}/")
        predicates.append(_Predicate(_check_not_pattern, None))
    if "not_empty" in kwargs:
        if kwargs.pop("not_empty"):
            def _check_not_empty(v):
                try:
                    if len(v) == 0:
                        raise ValueError("value must not be empty")
                except TypeError:
                    raise ValueError(f"value of type {type(v).__name__} has no length")
            predicates.append(_Predicate(_check_not_empty, None))

    # --- list_length / unique / sorted / monotonic ---
    if "list_length" in kwargs:
        lo, hi = kwargs.pop("list_length")
        frag = {}
        if lo is not None:
            frag["minItems"] = lo
        if hi is not None:
            frag["maxItems"] = hi
        predicates.append(_Predicate(_bounded_predicate(len, "length", lo, hi), frag or None))
    if "unique" in kwargs and kwargs.pop("unique"):
        def _check_unique(v):
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
                seen_list = []
                for x in items:
                    if x in seen_list:
                        raise ValueError(f"list contains duplicate: {x!r}")
                    seen_list.append(x)
        predicates.append(_Predicate(_check_unique, {"uniqueItems": True}))
    if "sorted" in kwargs:
        order = kwargs.pop("sorted")
        if order is not False:                        # False = explicit opt-out
            if order is True or order == "asc":
                reverse = False
            elif order == "desc":
                reverse = True
            else:
                raise ValueError(f"sorted= must be True/False, 'asc', or 'desc'; got {order!r}")

            def _check_sorted(v, _reverse=reverse):
                try:
                    items = list(v)
                except TypeError as e:
                    raise ValueError(f"sorted=: value is not iterable ({e})") from None
                try:
                    ordered = sorted(items, reverse=_reverse)
                except TypeError as e:
                    raise ValueError(f"sorted=: items are not mutually comparable ({e})") from None
                if items != ordered:
                    raise ValueError(f"list not sorted {'desc' if _reverse else 'asc'}")
            predicates.append(_Predicate(_check_sorted, None))
    if "monotonic" in kwargs:
        mode = kwargs.pop("monotonic")
        if mode is not False:                         # False = explicit opt-out
            if mode is True:
                strict = False
            elif mode == "strict":
                strict = True
            else:
                raise ValueError(f"monotonic= must be True/False or 'strict'; got {mode!r}")

            def _check_monotonic(v, _strict=strict):
                try:
                    items = list(v)
                except TypeError as e:
                    raise ValueError(f"monotonic=: value is not iterable ({e})") from None
                for i in range(1, len(items)):
                    a, b = items[i - 1], items[i]
                    try:
                        ok = (a < b) if _strict else (a <= b)
                    except TypeError as e:
                        raise ValueError(
                            f"monotonic=: items at indices {i - 1}/{i} are not comparable ({e})"
                        ) from None
                    if not ok:
                        raise ValueError(
                            f"not {'strictly ' if _strict else ''}monotonic at index {i}")
            predicates.append(_Predicate(_check_monotonic, None))

    # --- approx ---
    if "approx" in kwargs:
        target, eps = kwargs.pop("approx")

        def _check_approx(v, _target=target, _eps=eps):
            try:
                if abs(v - _target) > _eps:
                    raise ValueError(f"{v!r} not within {_eps} of {_target!r}")
            except TypeError as e:
                raise ValueError(f"approx check failed: {e}") from None
        predicates.append(_Predicate(_check_approx, None))

    # --- callable / behaves_like ---
    if "callable" in kwargs:
        if kwargs.pop("callable"):
            def _check_callable(v):
                if not callable(v):
                    raise ValueError(f"value of type {type(v).__name__} is not callable")
            predicates.append(_Predicate(_check_callable, None))
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
                        f"behaves_like: f{args!r} raised {type(e).__name__}: {e}") from None
                if got != expected:
                    raise ValueError(f"behaves_like: f{args!r} = {got!r}, expected {expected!r}")
        predicates.append(_Predicate(_check_behaves_like, None))

    # --- numeric: multiple_of (divisibility) + finite (reject nan/inf) ---
    if "multiple_of" in kwargs:
        _mo = kwargs.pop("multiple_of")
        if _mo == 0:                                  # 0 is nonsensical (div-by-zero / accept-all)
            raise ValueError("multiple_of= must be a non-zero number")

        def _check_multiple_of(v, _m=_mo):
            try:
                r = v % _m
            except TypeError as e:
                raise ValueError(f"multiple_of: cannot check {v!r} ({e})") from None
            if r != 0:
                raise ValueError(f"{v!r} is not a multiple of {_m}")
        predicates.append(_Predicate(_check_multiple_of, {"multipleOf": _mo}))
    if "finite" in kwargs:
        if kwargs.pop("finite"):
            def _check_finite(v):
                import math as _math
                try:
                    ok = _math.isfinite(v)
                except TypeError:
                    raise ValueError(
                        f"value of type {type(v).__name__} is not a finite-checkable number")
                if not ok:
                    raise ValueError(f"{v!r} is not finite (nan/inf not allowed)")
            predicates.append(_Predicate(_check_finite, None))

    # --- bytes length ---
    if "bytes_length" in kwargs:
        _lo, _hi = kwargs.pop("bytes_length")
        frag = {}
        if _lo is not None:
            frag["minLength"] = _lo
        if _hi is not None:
            frag["maxLength"] = _hi
        predicates.append(_Predicate(_bounded_predicate(len, "byte length", _lo, _hi), frag or None))

    # --- string prefix / suffix / contains ---
    if "str_prefix" in kwargs:
        _pp = kwargs.pop("str_prefix")

        def _check_prefix(v, _p=_pp):
            if not (isinstance(v, str) and v.startswith(_p)):
                raise ValueError(f"{v!r} must start with {_p!r}")
        predicates.append(_Predicate(_check_prefix, None))
    if "str_suffix" in kwargs:
        _ss = kwargs.pop("str_suffix")

        def _check_suffix(v, _s=_ss):
            if not (isinstance(v, str) and v.endswith(_s)):
                raise ValueError(f"{v!r} must end with {_s!r}")
        predicates.append(_Predicate(_check_suffix, None))
    if "str_contains" in kwargs:
        _cc = kwargs.pop("str_contains")

        def _check_contains(v, _c=_cc):
            if not (isinstance(v, str) and _c in v):
                raise ValueError(f"{v!r} must contain {_c!r}")
        predicates.append(_Predicate(_check_contains, None))

    # --- mapping / set size ---
    if "dict_length" in kwargs:
        _lo, _hi = kwargs.pop("dict_length")
        frag = {}
        if _lo is not None:
            frag["minProperties"] = _lo
        if _hi is not None:
            frag["maxProperties"] = _hi
        predicates.append(_Predicate(_bounded_predicate(len, "dict size", _lo, _hi), frag or None))
    if "set_length" in kwargs:
        _lo, _hi = kwargs.pop("set_length")
        frag = {}
        if _lo is not None:
            frag["minItems"] = _lo
        if _hi is not None:
            frag["maxItems"] = _hi
        predicates.append(_Predicate(_bounded_predicate(len, "set size", _lo, _hi), frag or None))

    # --- set relations / membership (element-wise, tolerates unhashable members/values) ---
    if "subset_of" in kwargs:
        _sub = list(kwargs.pop("subset_of"))

        def _check_subset(v, _sub=_sub):
            if not all(any(x == m for m in _sub) for x in v):
                raise ValueError(f"{v!r} is not a subset of {_sub!r}")
        predicates.append(_Predicate(_check_subset, None))
    if "superset_of" in kwargs:
        _sup = list(kwargs.pop("superset_of"))

        def _check_superset(v, _sup=_sup):
            if not all(any(m == x for x in v) for m in _sup):
                raise ValueError(f"{v!r} is not a superset of {_sup!r}")
        predicates.append(_Predicate(_check_superset, None))
    if "list_contains" in kwargs:
        _x = kwargs.pop("list_contains")

        def _check_list_contains(v, _x=_x):
            if _x not in v:
                raise ValueError(f"value must contain {_x!r}")
        predicates.append(_Predicate(_check_list_contains, None))

    # --- complex: magnitude / real / imaginary bounds ---
    if "abs_range" in kwargs:
        _lo, _hi = kwargs.pop("abs_range")
        predicates.append(_Predicate(_bounded_predicate(abs, "|z|", _lo, _hi), None))
    if "real_range" in kwargs:
        _lo, _hi = kwargs.pop("real_range")
        predicates.append(_Predicate(_bounded_predicate(lambda v: v.real, "Re", _lo, _hi), None))
    if "imag_range" in kwargs:
        _lo, _hi = kwargs.pop("imag_range")
        predicates.append(_Predicate(_bounded_predicate(lambda v: v.imag, "Im", _lo, _hi), None))

    # --- user predicates (single + plural) ---
    if "predicate" in kwargs:
        pred = kwargs.pop("predicate")
        if not callable(pred):
            raise TypeError("predicate= must be a callable")
        predicates.append(_Predicate(pred, None))
    if "predicates" in kwargs:
        preds = kwargs.pop("predicates")
        try:
            preds_list = list(preds)
        except TypeError:
            raise TypeError("predicates= must be an iterable of callables")
        for pred in preds_list:
            if not callable(pred):
                raise TypeError("predicates= contains a non-callable item")
            predicates.append(_Predicate(pred, None))

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

    return type_spec, coercers, predicates


# ============================================================================
# Description rendering
# ============================================================================


def _render_elem(elem: Any) -> str:
    """Render a type spec for describe(): a Constraint element → its `.describe()`; a typing
    generic → `origin[arg, ...]` with each arg rendered recursively (so `type=list[.field]`
    expands the element); a `Literal[...]` → `one of [...]`; otherwise the short type name."""
    if isinstance(elem, type) and issubclass(elem, Constraint):
        return elem.describe()
    origin = get_origin(elem)
    if origin is not None:
        args = get_args(elem)
        if origin is Literal:
            return f"one of {list(args)}"
        name = getattr(origin, "__name__", None) or _short(origin)
        return f"{name}[{', '.join(_render_elem(a) for a in args)}]" if args else name
    return _short(elem)


# Descriptor table for _describe_factory. Each rule is:
#   (kwarg_name, only_if_truthy, render_function)
# `only_if_truthy=True` means we skip the render when the value is False/0/None
# (used for opt-out flags like sorted=False, monotonic=False, unique=False).
_DESCRIBE_RULES: Tuple[Tuple[str, bool, Callable[[Any], str]], ...] = (
    # The type coercer renders FIRST (it establishes the value's type); a struct recurses.
    ("type",        False, lambda v: _render_elem(v)),
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
    ("is_instance_of", False, lambda v: f"strictly an instance of {_isinstance_describe(v)} (no coercion)"),
    ("not_one_of",  False, lambda v: f"not one of {list(v)}"),
    ("str_pattern", False, lambda v: f"matches pattern /{v}/"),
    ("str_length",  False, lambda v: _range_phrase("length", v[0], v[1], ge=True, le=True)),
    ("not_pattern", False, lambda v: f"does NOT match /{v}/"),
    ("not_empty",   True,  lambda v: "non-empty"),
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
    ("predicate",   False, lambda v: get_description(v) or "1 predicate(s) checked"),
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


def _struct_body(cls: Type[Constraint]) -> List[str]:
    """Field lines (+ this class's 'Additionally:' block) for `cls`: each field at indent 2,
    the 'Additionally:' header at column 0. Every field is a Constraint.field(...) (or a
    Constraint.of(...) factory) whose COMPLETE contract is rendered inline from the facade's own
    describe(); a nested struct is a `.field(type=Struct)` facade rendered the same way (its
    describe() recurses)."""
    fpad = "  "
    lines: List[str] = []
    for name, info in cls.model_fields.items():
        ann = info.annotation
        field_desc = info.description or ""

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

        # A non-.field factory field (a Constraint.of(...) annotation — rare; .field is the
        # authoring path): render its bare type. There is no pydantic-Field metadata to summarize
        # under the .field-only model (raw Field is rejected), so the line is just `name (type)`.
        line = f"{fpad}- {name} ({_short(ann)})"
        if field_desc:
            line += f": {field_desc}"
        lines.append(line)

    extras = _collect_validator_descriptions(cls)
    if extras:
        lines.append("")
        lines.append("Additionally:")
        for e in extras:
            lines.append(f"  - {e}")
    return lines


def _describe_structural(cls: Type[Constraint]) -> str:
    """Render an NL description for a structural Constraint subclass: a header, one line per
    field (each a Constraint.field(...) rendered with its full contract inline), and an
    'Additionally:' block for cross-field `@constraint`-tagged validators."""
    lines = [f"Return an object of type {cls.__name__} with fields:"]
    lines.extend(_struct_body(cls))
    return "\n".join(lines)


def _range_phrase(label: str, lo: Any, hi: Any, *, ge: bool, le: bool) -> str:
    lo_part = f"{lo}" if lo is not None else "-∞"
    hi_part = f"{hi}" if hi is not None else "∞"
    lo_op = "≤" if ge else "<"
    hi_op = "≤" if le else "<"
    return f"{label} in [{lo_part} {lo_op} x {hi_op} {hi_part}]"


def _short(t: Any) -> str:
    """Short type name for description rendering.

    Generic / subscripted types (e.g. `List[int]`, `Optional[int]`, `Tuple[int, ...]`)
    keep their parametrization — checking get_origin/get_args BEFORE the bare
    `.__name__` fallback (typing aliases like `List[int]` still expose `__name__`
    as just "List", which would drop the `[int]` bit).
    """
    if t is type(None):
        return "None"
    origin = get_origin(t)
    args = get_args(t)
    if origin is not None:
        # `_name` covers typing aliases (List/Tuple/Dict/Optional/Union -> "List"/...);
        # `__name__` covers PEP 585 builtin generics (list[int] -> "list"/...);
        # repr is the last-resort fallback for exotic forms.
        origin_name = (getattr(t, "_name", None)
                       or getattr(origin, "__name__", None)
                       or repr(origin))
        if args:
            return f"{origin_name}[{', '.join(_short(a) for a in args)}]"
        return origin_name
    if hasattr(t, "__name__"):
        return t.__name__
    return repr(t)
