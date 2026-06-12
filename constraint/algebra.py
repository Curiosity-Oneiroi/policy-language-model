"""Operator composition for Constraint — `&`, `|`, `~`, `^`.

`&` for struct-only pair auto-merges into a single pydantic model with the
union of fields; for anything else, falls back to a run-both wrapper.
`|`, `~`, `^` always produce thin wrappers (no auto-merge — semantics
don't reduce to a single pydantic model).
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional, Type

import pydantic as pd
from pydantic_core import core_schema

from .base import (
    Constraint,
    ConstraintViolation,
    _ConstraintMeta,
)


# ============================================================================
# AND  (with auto-merge for struct-only pair)
# ============================================================================


def _is_struct_only(cls: type) -> bool:
    """A struct-only Constraint is a direct subclass that:
    - is a Constraint subclass,
    - is not a factory-built (single `value` field) class,
    - is not itself a composition wrapper.
    """
    if not (isinstance(cls, type) and issubclass(cls, Constraint)):
        return False
    if getattr(cls, "_constraint_is_factory", False):
        return False
    if getattr(cls, "_constraint_is_composite", False):
        return False
    return True


def _require_constraints(op_symbol: str, *ops: Any) -> None:
    """Every operand of the constraint algebra MUST be a Constraint. Reject anything else LOUDLY at
    composition time (e.g. `A | None`, `A | 5`) — otherwise a bare non-constraint gets silently
    wrapped as a dead "branch" that quietly rejects everything, with no signal (a D1-class trap).
    Constraints have NO nullable/Optional: a field always satisfies its constraint, so `| None` is
    not valid — build constraints ONLY from `Constraint.field(...)` and the algebra `& | ^ ~`."""
    for o in ops:
        if not (isinstance(o, type) and issubclass(o, Constraint)):
            raise TypeError(
                f"`{op_symbol}` combines Constraints; got {o!r} ({type(o).__name__}). A field must "
                f"satisfy a constraint — compose `Constraint.field(...)` constraints with `& | ^ ~`. "
                f"(Constraints have no nullable/Optional; `| None` is not valid.)")


def _and(a: type, b: type) -> type:
    """`a & b` — if both are struct-only, try to merge; else build a run-both
    wrapper. Result is always a Constraint subclass.

    NB2: short-circuit when both children are the same class — `A & A == A`
    semantically, and Python disallows duplicate bases anyway."""
    _require_constraints("&", a, b)
    if a is b:
        return a
    if _is_struct_only(a) and _is_struct_only(b):
        merged = _try_merge_pydantic(a, b)
        if merged is not None:
            return merged
        # Two structs that cannot merge share a field with genuinely incompatible
        # types. Do NOT fall through to the run-both wrapper: for two structs that
        # path threads a's validated instance into b.validate, which pydantic
        # rejects for EVERY input — a composite that silently validates nothing.
        # Fail loud, at compose time, with the specific conflict.
        raise TypeError(
            f"Cannot merge structural constraints {a.__name__} & {b.__name__}: "
            f"they share a field name whose types are incompatible (cannot be "
            f"intersected). Reconcile the field types, or express the combination "
            f"with a @model_validator instead of &."
        )
    return _make_and_wrapper(a, b)


def _preserve_collided_validators(a: type, b: type, namespace: Dict[str, Any]) -> None:
    """Re-bind `b`'s validators that collide (same method NAME, different func)
    with `a`'s into `namespace` under unique names, so the MI-merged class keeps
    BOTH instead of MRO silently dropping `b`'s.

    Handles the realistic cases — `@model_validator` (cross-field `_check`-style
    checks, `exactly_one_of`) and `@field_validator` — reconstructing each from
    its pydantic `Decorator.info` (mode / fields). A shared validator (same func
    inherited from a common base) is NOT a collision and is left to MI.
    """
    from pydantic import field_validator, model_validator

    a_decs = getattr(a, "__pydantic_decorators__", None)
    b_decs = getattr(b, "__pydantic_decorators__", None)
    if a_decs is None or b_decs is None:
        return

    taken = set(namespace) | set(dir(a)) | set(dir(b))

    def _uniq(base: str) -> str:
        name, i = f"{base}__b", 1
        while name in taken:
            name, i = f"{base}__b{i}", i + 1
        taken.add(name)
        return name

    a_mv = getattr(a_decs, "model_validators", {}) or {}
    for name, dec in (getattr(b_decs, "model_validators", {}) or {}).items():
        if name in a_mv and getattr(a_mv[name], "func", None) is not dec.func:
            namespace[_uniq(name)] = model_validator(mode=dec.info.mode)(dec.func)

    a_fv = getattr(a_decs, "field_validators", {}) or {}
    for name, dec in (getattr(b_decs, "field_validators", {}) or {}).items():
        if name in a_fv and getattr(a_fv[name], "func", None) is not dec.func:
            kw: Dict[str, Any] = {"mode": dec.info.mode}
            if getattr(dec.info, "check_fields", None) is not None:
                kw["check_fields"] = dec.info.check_fields
            namespace[_uniq(name)] = field_validator(*dec.info.fields, **kw)(dec.func)


def _intersected_bound_markers(a_meta, b_meta):
    """The single TIGHTEST `annotated_types` marker per combinable bound type
    across both sides — e.g. `Ge(10)` from `Ge(10)` & `Ge(0)`, `Le(50)` from
    `Le(50)` & `Le(100)`, `MinLen(5)` from `MinLen(5)` & `MinLen(2)`.

    `_tighten_field_infos` uses this to keep a merged struct-&-struct field's
    numeric/length bounds VISIBLE in `json_schema()`/`describe()` (the sub-LLM's
    two steering channels) while the appended `AfterValidator` stays the
    authoritative enforcer. Exactly ONE marker per type is returned, so
    the visible metadata never carries a duplicate constraint that pydantic would
    choke on at model-build time.

    GroupedMetadata (`Interval`, `Len` — what you get from
    `Annotated[int, Interval(...)]` / `Len(...)` written directly, vs the
    `Field(ge=...)`-decomposed `Ge`/`Le`) is DECOMPOSED into its atomic markers
    first, so those bounds surface too and a Len-vs-MinLen mix yields the COMPLETE
    intersected bound rather than a misleading half. Non-combinable
    metadata (patterns, multiple_of, validators/coercers, custom markers) is
    intentionally OMITTED — it can't reduce to one marker safely, so it stays
    enforced by the AfterValidator alone.
    """
    import annotated_types as _at
    _reducers = {
        _at.Ge: ("ge", max), _at.Gt: ("gt", max),
        _at.Le: ("le", min), _at.Lt: ("lt", min),
        _at.MinLen: ("min_length", max), _at.MaxLen: ("max_length", min),
    }
    buckets: Dict[type, list] = {}
    for _m in (*a_meta, *b_meta):
        # Interval/Len are GroupedMetadata: iterating yields their atomic
        # Ge/Gt/Le/Lt/MinLen/MaxLen so they bucket like a decomposed Field would.
        for _atom in (list(_m) if isinstance(_m, _at.GroupedMetadata) else (_m,)):
            _spec = _reducers.get(type(_atom))
            if _spec is not None:
                _v = getattr(_atom, _spec[0], None)
                if _v is not None:
                    buckets.setdefault(type(_atom), []).append(_v)
    out = []
    for _mt, (_attr, _reduce) in _reducers.items():
        _vals = buckets.get(_mt)
        if _vals:
            out.append(_mt(_reduce(_vals)))
    return out


def _summarize_extra_constraints(a_meta, b_meta, include_bounds=False):
    """Render a merged struct-&-struct field's NON-combinable constraints — patterns,
    `multiple_of`, and custom validators (recovering any `@constraint` description) —
    as a list of short natural-language phrases.

    The combinable Ge/Gt/Le/Lt/MinLen/MaxLen bounds are normally surfaced as VISIBLE
    markers (rendered by `_field_constraint_summary`); these OTHERS are enforced by
    the merge's `AfterValidator` but otherwise invisible to `describe()`, so without
    this an auto-merged `A & B` field would not TELL the model about them UPFRONT — it
    would only discover them on a violation. We stash the result on the enforcer
    function (`_plm_extra_desc`); the describe path reads it back, giving parity with
    how a COMPOSITE renders each child's describe verbatim. (#H3/#H4 follow-up)

    `include_bounds=True` (passed when the visibility gate suppressed the combinable
    markers because a coercer is present) ALSO lists the intersected Ge/Le/.. bounds
    here as text, so the describe stays COMPLETE even for a field that has both a
    coercer and a bound (enforcement is unchanged — these are text only).
    """
    import annotated_types as _at
    from pydantic import BeforeValidator, AfterValidator, PlainValidator, WrapValidator
    from .decorator import get_description

    _combinable = (_at.Ge, _at.Gt, _at.Le, _at.Lt, _at.MinLen, _at.MaxLen)
    out: list = []
    n_generic = 0

    def _add(s):
        if s and s not in out:
            out.append(s)

    if include_bounds:
        _bound_attr = {
            _at.Ge: "ge", _at.Gt: "gt", _at.Le: "le", _at.Lt: "lt",
            _at.MinLen: "min_length", _at.MaxLen: "max_length",
        }
        for _mk in _intersected_bound_markers(a_meta, b_meta):
            _attr = _bound_attr.get(type(_mk))
            if _attr is not None:
                _add(f"{_attr}={getattr(_mk, _attr)!r}")

    for _m in (*a_meta, *b_meta):
        if isinstance(_m, _combinable) or isinstance(_m, _at.GroupedMetadata):
            continue                                   # already surfaced as a visible marker
        _p = getattr(_m, "pattern", None)
        if _p is not None:
            _add(f"must match /{_p}/")
            continue
        _mo = getattr(_m, "multiple_of", None)
        if _mo is not None:
            _add(f"must be a multiple of {_mo}")
            continue
        if isinstance(_m, (BeforeValidator, AfterValidator, PlainValidator, WrapValidator)):
            _d = get_description(getattr(_m, "func", None))
            if _d:
                _add(_d)                               # recover a @constraint-tagged description
            else:
                n_generic += 1
            continue
    if n_generic:
        _add(f"plus {n_generic} custom validator{'s' if n_generic > 1 else ''}")
    return out


def _rewrite_schema_refs(node, rename):
    """Recursively rewrite every JSON-Schema `$ref: #/$defs/<old>` to
    `#/$defs/<new>` for each old->new in `rename`. Used by the composite
    `_json_schema` $defs-hoist to keep a branch's refs pointing at its OWN def
    after a same-name/different-shape collision forced a rename."""
    if isinstance(node, dict):
        _ref = node.get("$ref")
        if isinstance(_ref, str) and _ref.startswith("#/$defs/"):
            _name = _ref[len("#/$defs/"):]
            if _name in rename:
                node["$ref"] = "#/$defs/" + rename[_name]
        for _v in node.values():
            _rewrite_schema_refs(_v, rename)
    elif isinstance(node, list):
        for _it in node:
            _rewrite_schema_refs(_it, rename)


def _tighten_field_infos(a_info, b_info):
    """Merge two same-annotation `FieldInfo`s so the result enforces the
    CONJUNCTION (intersection) of BOTH sides' constraints — `A & B` means
    intersection, so a shared field must satisfy A's AND B's bounds.

    `FieldInfo.merge_field_infos` is **last-wins on the whole metadata list**,
    which can LOOSEN a bound (`Field(ge=10) & Field(ge=0)` -> `ge=0`, accepting
    values A rejects). A naive per-type union of `.metadata` is ALSO unreliable:
    pydantic folds duplicate metadata inconsistently (duplicate `Ge` conjuncts,
    but duplicate `MinLen`/`pattern` are last-wins). So we keep
    `merge_field_infos` for the non-constraint bits (default / alias /
    description), make a single `AfterValidator` the AUTHORITATIVE enforcer (it
    re-validates against EACH side's ORIGINAL constraints via a `TypeAdapter` —
    sound, order-independent, agnostic to constraint type), AND additionally
    surface the intersected numeric/length bounds as visible markers so the
    merged field's `json_schema()`/`describe()` still show them for steering
    (#R6-2; gated off when a side coerces — see below).
    """
    from pydantic import (
        AfterValidator, BeforeValidator, ConfigDict,
        PlainValidator, TypeAdapter, WrapValidator,
    )
    from pydantic.fields import FieldInfo

    merged = FieldInfo.merge_field_infos(a_info, b_info)   # default/alias/desc (last-wins is fine)
    a_meta, b_meta = list(a_info.metadata), list(b_info.metadata)
    if not a_meta and not b_meta:
        return merged                                      # neither side constrains the field
    ann = a_info.annotation

    def _adapter(meta):
        # Build a TypeAdapter for this side's (annotation + constraints). Try
        # PLAIN first: that works for primitives AND pydantic models/dataclasses
        # (which carry their own config — passing `config` to those is a hard
        # PydanticUserError). Only on a schema-generation failure (an ARBITRARY,
        # non-schema field type — the base Constraint default is
        # arbitrary_types_allowed) do we retry WITH that config. Without this
        # retry, an arbitrary-typed shared field raised PydanticSchemaGenerationError
        # at construction and ESCAPED the merge instead of tightening. Metadata
        # here came from a valid struct (which built under arbitrary_types_allowed),
        # so the retry is guaranteed to build.
        annotated = Annotated[tuple([ann, *meta])]
        try:
            return TypeAdapter(annotated)
        except Exception:
            return TypeAdapter(annotated, config=ConfigDict(arbitrary_types_allowed=True))

    # Identify a validator/coercer marker's wrapped callable (else None). Used to
    # dedup a coercer SHARED across both sides by FUNC IDENTITY.
    def _validator_func(_m):
        return getattr(_m, "func", None) if isinstance(
            _m, (BeforeValidator, AfterValidator, PlainValidator, WrapValidator)) else None

    # Build the b-side adapter from b's metadata MINUS any validator whose wrapped
    # func is ALREADY present in a_meta (identity match). This applies a coercer
    # shared via a common base EXACTLY once even when each side adds its own
    # DISTINCT bound — the old whole-list `b_meta != a_meta` skip missed that and
    # double-applied the coercer (#R6-7 / /). Compared by func IDENTITY
    # (`id`), NEVER `==`, so a metadata object whose `__eq__` raises cannot leak out
    # of the `&` operator. b's declarative bound markers are kept — double-
    # checking a bound is harmless/idempotent, and a's adapter applies the shared
    # coercer's transform that b's distinct bound is then checked against.
    _a_validator_funcs = {id(_f) for _m in a_meta if (_f := _validator_func(_m)) is not None}
    _b_distinct = [
        _m for _m in b_meta
        if not ((_f := _validator_func(_m)) is not None and id(_f) in _a_validator_funcs)
    ]
    adapters = []
    if a_meta:
        adapters.append(_adapter(a_meta))
    if _b_distinct:
        adapters.append(_adapter(_b_distinct))

    def _intersect(v, _ads=tuple(adapters)):
        # THREAD the value through each side: validate_python both checks the
        # bounds (raises -> field rejected: conjunction) AND applies that side's
        # coercers/BeforeValidators, so a transforming validator on a merged
        # field isn't silently dropped (A then B; a no-coercer side is a no-op).
        for _ta in _ads:
            v = _ta.validate_python(v)
        return v

    # Mark our synthetic enforcer so a NESTED merge ((A&B)&C) does NOT mistake a
    # prior merge's AfterValidator for a user coercer and trip the visibility gate
    # below — which would hide the (still-correctly-enforced) bound from a 3+-way
    # merge of pure bounds. The mark rides on the function, read via .func.
    _intersect._plm_merge_enforcer = True

    def _is_merge_enforcer(_m):
        return isinstance(_m, AfterValidator) and getattr(
            getattr(_m, "func", None), "_plm_merge_enforcer", False)

    # Keep the INTERSECTED declarative bounds VISIBLE in the merged field's
    # json_schema()/describe() so the sub-LLM is steered with the real bound (e.g.
    # `minimum: 10` for `Field(ge=10) & Field(ge=0)`); without this the metadata is
    # just the opaque AfterValidator and the bound vanishes from BOTH steering
    # channels. Only single, non-duplicate, combinable annotated_types
    # markers are surfaced (Ge/Gt/Le/Lt/MinLen/MaxLen, incl. decomposed Interval/Len)
    # — patterns/multiple_of/custom stay enforced by _intersect alone (duplicating
    # them risks a pydantic build error). GATE on "no GENUINE transformer either
    # side" (our own marked merge-enforcer excluded): a coercer means the bound must
    # be checked POST-transform (which _intersect does), whereas a visible marker
    # checks the RAW input and could wrongly reject — so when a side carries a real
    # Before/After/Plain/Wrap validator we keep the opaque form for that field.
    # AfterValidator(_intersect) stays the AUTHORITATIVE enforcer either way.
    _has_transformer = any(
        isinstance(_m, (BeforeValidator, AfterValidator, PlainValidator, WrapValidator))
        and not _is_merge_enforcer(_m)
        for _m in (*a_meta, *b_meta)
    )
    _visible = [] if _has_transformer else _intersected_bound_markers(a_meta, b_meta)

    # Stash the NON-combinable constraints (patterns/multiple_of/custom validators)
    # as NL phrases on the enforcer func, so describe() surfaces them upfront —
    # they're enforced here but invisible to the field-metadata summary (#H3/#H4
    # follow-up). FieldInfo has __slots__ (no private-attr slot), but the
    # AfterValidator carrying this func IS in the field's metadata, so the describe
    # path reads it back via `meta.func._plm_extra_desc`. When the gate suppressed
    # the visible combinable markers (a coercer is present), ALSO list the
    # intersected bounds here (text only — no enforcing marker) so describe stays
    # COMPLETE; enforcement remains in the AfterValidator either way.
    _intersect._plm_extra_desc = _summarize_extra_constraints(
        a_meta, b_meta, include_bounds=_has_transformer)

    merged.metadata = [*_visible, AfterValidator(_intersect)]
    return merged


def _try_merge_pydantic(a: Type[Constraint], b: Type[Constraint]) -> Optional[type]:
    """Merge two struct-only Constraint classes into one via multi-inheritance.

    Strategy: build `merged = _ConstraintMeta(name, (a, b), namespace)`. This
    means:
      - **Validators (`@model_validator`, `@field_validator`) inherit
        naturally** — pydantic's MI handling picks them up from both parents.
      - **`isinstance(merged_instance, a) and isinstance(merged_instance, b)`
        is True** — the merged class is genuinely a subclass of both.
      - **Field types are union'd** — disjoint fields land cleanly.
      - **Same-name same-type fields:** we explicitly intersect their
        `FieldInfo`s into the namespace (so bounds tighten — pydantic's MI
        alone would pick whichever parent comes first in MRO).
      - **Same-name *different-type* fields:** return None (the type check below,
        or a caught pydantic construction error) — the caller then raises a loud
        TypeError, because the run-both wrapper is broken for two structs (it
        threads a's instance into b.validate, rejecting every input).
      - **`.field`-typed fields:** normalized to their flat (base+metadata) form
        first (see `_unwrap`), so two freshly-minted `.field` wrappers intersect
        like plain typed fields instead of comparing unequal by class identity.
    """
    from pydantic.fields import FieldInfo

    a_fields = dict(a.model_fields)
    b_fields = dict(b.model_fields)

    def _unwrap(info: FieldInfo) -> FieldInfo:
        """Normalize a field typed with a `Constraint.field(...)` (a TRANSPARENT
        wrapper, freshly minted per `.field` call) down to its flat (base type +
        metadata) FieldInfo. Without this, two same-name `.field` fields compare
        as distinct wrapper classes, abort the merge, and fall to the run-both
        path (broken for two structs). Unwrapped, they intersect like plain typed
        fields via `_tighten_field_infos`. Non-`.field` annotations pass through
        (a nested `.of` field keeps its {value:..} shape)."""
        ann = info.annotation
        if (isinstance(ann, type) and issubclass(ann, Constraint)
                and getattr(ann, "_constraint_is_field", False)):
            flat = getattr(ann, "_constraint_field_annotation", None)
            if flat is not None:
                new = FieldInfo.from_annotation(flat)
                if info.metadata:
                    new.metadata = [*new.metadata, *info.metadata]
                return new
        return info

    namespace: Dict[str, Any] = {
        "model_config": pd.ConfigDict(arbitrary_types_allowed=True),
        "__annotations__": {},
    }

    # For overlapping fields: explicitly intersect FieldInfo + check type compat.
    # For non-overlapping fields: rely on MI to inherit them.
    for name in set(a_fields) & set(b_fields):
        a_info = _unwrap(a_fields[name])
        b_info = _unwrap(b_fields[name])
        if a_info.annotation != b_info.annotation:
            # Type conflict — abort merge; the caller raises (run-both is broken
            # for two structs: it threads a's instance into b.validate).
            return None
        # Same annotation: TIGHTEN — the merged field must enforce the
        # CONJUNCTION (intersection) of BOTH sides' constraints (& == intersect).
        namespace["__annotations__"][name] = a_info.annotation
        namespace[name] = _tighten_field_infos(a_info, b_info)

    # Preserve validators MI would otherwise DROP: when `a` and `b` each define a
    # validator with the SAME method name but a DIFFERENT function, MRO keeps only
    # `a`'s (a precedes b in the bases), silently discarding `b`'s check. Re-bind
    # `b`'s colliding validators under unique names in the merge namespace so
    # pydantic registers BOTH. (Run-both is not an option for structs: `_and_validate`
    # threads `a.validate`'s instance into `b.validate`, which pydantic rejects.)
    _preserve_collided_validators(a, b, namespace)

    merged_name = f"{a.__name__}And{b.__name__}"
    try:
        merged = _ConstraintMeta(  # type: ignore[call-arg]
            merged_name,
            (a, b),  # MI: inherits validators + isinstance relationships.
            namespace,
        )
    except Exception:
        return None

    return merged


# ----- run-both wrapper (when merge isn't applicable) -----


def _make_and_wrapper(a: type, b: type) -> type:
    """Build an AndConstraint wrapper that runs both validators."""
    return _make_composite_wrapper("And", "AND", [a, b], _and_validate)


def _and_validate(children: List[type], value: Any, context: Optional[Dict] = None) -> Any:
    """Validate against all children; both must pass. Returns the value
    coerced by the last child.

    Fails fast on the first failing child — subsequent children (which may
    have side effects: network, LLM, etc.) do not run after a failure."""
    cur = value
    for c in children:
        try:
            cur = c.validate(cur, context=context)
        except Exception as e:                          # CV OR any other validator error: a
            # structural child's @model_validator raising KeyError/TypeError/etc.
            # escapes validate() RAW (pydantic only wraps ValueError). Treat any
            # child error as a failure, chained so the original stays visible —
            # keeps the whole composite within the ConstraintViolation contract
            # (same philosophy as the _wrap_predicate net).
            raise ConstraintViolation(
                f"AND: child {getattr(c, '__name__', c)!r} failed: {e}"
            ) from e
    return cur


# ============================================================================
# OR
# ============================================================================


def _or(a: type, b: type) -> type:
    _require_constraints("|", a, b)
    return _make_composite_wrapper("Or", "OR", [a, b], _or_validate)


def _or_validate(children: List[type], value: Any, context: Optional[Dict] = None) -> Any:
    errors: List[str] = []
    for c in children:
        try:
            return c.validate(value, context=context)
        except Exception as e:                          # CV or any other validator error:
            errors.append(str(e))                       # record + try the NEXT branch (a branch
                                                        # that errors must not skip the others)
    raise ConstraintViolation(
        "OR: value did not satisfy any branch: " + " || ".join(errors)
    )


# ============================================================================
# NOT
# ============================================================================


def _not(c: type) -> type:
    """`~c` — meaningful only on predicate-form (factory) constraints. Raises
    TypeError on structural subclasses."""
    if _is_struct_only(c):
        raise TypeError(
            "~ (NOT) is only meaningful for predicate-form constraints; "
            f"got structural {c.__name__}. Use a custom @model_validator instead."
        )
    return _make_composite_wrapper("Not", "NOT", [c], _not_validate)


def _not_validate(children: List[type], value: Any, context: Optional[Dict] = None) -> Any:
    (inner,) = children
    try:
        inner.validate(value, context=context)
    except Exception:                                   # inner failed (CV or any other error)
        return value  # -> value is NOT in inner; negation succeeded
    try:                                                # describe() must not turn the rejection
        _neg = inner.describe()                         # into a raw describe-error escape
    except Exception:
        _neg = "<unavailable>"
    raise ConstraintViolation(
        f"NOT: value satisfied negated constraint ({_neg})"
    )


# ============================================================================
# XOR
# ============================================================================


def _xor(a: type, b: type) -> type:
    _require_constraints("^", a, b)
    return _make_composite_wrapper("Xor", "XOR", [a, b], _xor_validate)


def _xor_validate(children: List[type], value: Any, context: Optional[Dict] = None) -> Any:
    results = []
    for c in children:
        try:
            v = c.validate(value, context=context)
            results.append((True, v))
        except Exception:                               # CV or any other error -> failed branch
            results.append((False, None))
    passed = sum(1 for ok, _ in results if ok)
    if passed == 1:
        return next(v for ok, v in results if ok)
    if passed == 0:
        raise ConstraintViolation("XOR: neither branch satisfied")
    raise ConstraintViolation(f"XOR: {passed} branches satisfied; expected exactly 1")


# ============================================================================
# Composite-wrapper factory — shared shell for AND-runboth, OR, NOT, XOR
# ============================================================================


def _make_composite_wrapper(
    suffix: str,
    label: str,
    children: List[type],
    validate_fn,
) -> type:
    """Build a Constraint subclass that runs `validate_fn(children, value)` as
    its `.validate()` implementation."""
    child_names = "_".join(getattr(c, "__name__", "x") for c in children)
    cls_name = f"_{suffix}Constraint__{child_names}"

    def _validate(cls, value, *, context=None):  # noqa: ANN001
        return validate_fn(children, value, context)

    def _model_validate(cls, obj, *, strict=None, from_attributes=None, context=None, by_alias=None, by_name=None):  # noqa: ANN001
        """B22 fix: pydantic's default `model_validate` would create an
        empty instance (no fields) and bypass the children entirely. Override
        so calling `Cls.model_validate(value)` directly on a composite goes
        through our `validate(value, context=...)` like every other path.

        The extra pydantic kwargs (strict/from_attributes/by_alias/by_name)
        are accepted-and-ignored so callers using the full pydantic signature
        don't crash."""
        return cls.validate(obj, context=context)

    def _describe(cls):  # noqa: ANN001
        inner = [c.describe() if hasattr(c, "describe") else repr(c) for c in children]
        if label == "NOT":
            body = inner[0]
            if "\n" in body:                                # multi-line (structural) child
                indented = "\n".join("  " + ln for ln in body.splitlines())
                # A NOT has exactly ONE child — never a conjunction — so avoid
                # "all of" (which would sit, wrongly, above e.g. an "AT LEAST ONE
                # of..." block). Single-child-appropriate header.
                return "NOT the following (must NOT hold):\n" + indented
            return f"NOT ({body})"
        if not any("\n" in i for i in inner):
            # Compact inline form — the common case (single-line predicate children).
            return f" {label} ".join(f"({i})" for i in inner)
        # At least one child is a multi-line block (e.g. a structural constraint's
        # header + per-field bullets + "Additionally:"). Inlining it inside
        # `( ... ) AND ( ... )` garbles the layout — and this string feeds
        # natural_llm's retry/last-round prompts. Render a numbered vertical list
        # and indent each child's continuation lines under its number so the
        # block stays readable.
        heading = {
            "AND": "ALL of the following must hold",
            "OR":  "AT LEAST ONE of the following must hold",
            "XOR": "EXACTLY ONE of the following must hold",
        }.get(label, label)
        items = []
        for idx, desc in enumerate(inner, 1):
            marker = f"  {idx}. "
            items.append(marker + desc.replace("\n", "\n" + " " * len(marker)))
        return heading + ":\n" + "\n".join(items)

    def _get_core(cls, source, handler):  # noqa: ANN001
        # As a DIRECT field annotation inside another Constraint (`v: A | B`), validate the field
        # value through THIS composite's own validate() — threading the outer `context=` — instead
        # of letting pydantic build a schema for an empty-fields model (a composite has NO pydantic
        # fields), which rejects EVERY scalar and accepts garbage (D1). Mirrors the `.field()`
        # factory's core-schema hook (base.py `_get_core`). Top-level use goes through `.validate()`
        # directly and never hits this; list_of/dict_of elements still route via `_resolve_elem_type`.
        return core_schema.with_info_plain_validator_function(
            lambda v, info, _c=cls: _c.validate(
                v, context=(info.context if info is not None else None)))

    def _get_json_schema(cls, schema, handler):  # noqa: ANN001
        # The plain-validator core schema above carries no auto JSON schema, so a struct WITH a
        # composite field would otherwise raise PydanticInvalidForJsonSchema. Supply the composite's
        # OWN json_schema() (allOf/anyOf/oneOf/not + hoisted child $defs) — the same shape the
        # top-level composite emits — so `Struct.model_json_schema()` / structured-output works (D1).
        return cls.json_schema()

    def _json_schema(cls):  # noqa: ANN001
        sub = [c.json_schema() if hasattr(c, "json_schema") else {} for c in children]
        # Hoist every child's `$defs` to the composite's OWN root. Pydantic emits a
        # structural child's nested-model definitions at THAT child's root with
        # root-relative `$ref: #/$defs/<Model>` refs; nesting the child one level
        # down under allOf/anyOf/oneOf/not (or a flattened factory value-schema,
        # which carries $defs via base.py:303) leaves those refs dangling against
        # the composite document root, so a strict structured-output backend
        # (OpenAI / vLLM guided-json — exactly what natural_llm hard-sets as
        # response_format for an accepted structural composite) cannot resolve them.
        # Collect the defs up and strip them from the children so the refs resolve.
        # This is the composite analog of the factory $defs-hoist at base.py:303.
        defs: dict = {}
        cleaned = []
        for _s in sub:
            if isinstance(_s, dict) and "$defs" in _s:
                _s = dict(_s)
                _rename: dict = {}
                for _dk, _dv in _s.pop("$defs").items():
                    if _dk not in defs:
                        defs[_dk] = _dv                     # first occurrence
                    elif defs[_dk] == _dv:
                        pass                                # same model reused -> share the def
                    else:
                        # GENUINE collision: same name, DIFFERENT shape across two
                        # branches (e.g. each branch defines its own `Item`).
                        # first-wins would silently point THIS branch's $ref at the
                        # OTHER branch's def, so the schema would misdescribe this
                        # branch (schema disagrees with validate();). Rename this
                        # branch's def to a fresh key and rewrite this branch's refs
                        # to it, so both shapes coexist and each $ref resolves to the
                        # right one.
                        _n = 2
                        _new = f"{_dk}__{_n}"
                        while _new in defs and defs[_new] != _dv:
                            _n += 1
                            _new = f"{_dk}__{_n}"
                        defs[_new] = _dv
                        _rename[_dk] = _new
                if _rename:
                    _rewrite_schema_refs(_s, _rename)        # this branch's body refs
                    for _moved in _rename.values():
                        _rewrite_schema_refs(defs[_moved], _rename)   # + the moved def's own refs
            cleaned.append(_s)
        if label == "NOT":
            result = {"not": cleaned[0]}
        elif label == "AND":
            result = {"allOf": cleaned}
        elif label == "OR":
            result = {"anyOf": cleaned}
        elif label == "XOR":
            result = {"oneOf": cleaned}
        else:
            return {}
        if defs:
            result["$defs"] = defs
        return result

    cls_dict = {
        "_constraint_is_composite": True,
        "_constraint_is_factory": False,
        "_constraint_children": tuple(children),   # honest re: ClassVar[tuple]; immutable
        "_constraint_op_label": label,
        "validate": classmethod(_validate),
        "model_validate": classmethod(_model_validate),
        "describe": classmethod(_describe),
        "json_schema": classmethod(_json_schema),
        "__get_pydantic_core_schema__": classmethod(_get_core),   # usable as a DIRECT field (D1)
        "__get_pydantic_json_schema__": classmethod(_get_json_schema),   # + its JSON schema (D1)
        "model_config": pd.ConfigDict(arbitrary_types_allowed=True),
    }
    cls = _ConstraintMeta(cls_name, (Constraint,), cls_dict)  # type: ignore[call-arg]
    return cls
