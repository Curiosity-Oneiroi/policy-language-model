"""Operator composition for Constraint — `&`, `|`, `~`, `^`.

`&` for struct-only pair auto-merges into a single pydantic model with the
union of fields; for anything else, falls back to a run-both wrapper.
`|`, `~`, `^` always produce thin wrappers (no auto-merge — semantics
don't reduce to a single pydantic model).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

import pydantic as pd

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


def _and(a: type, b: type) -> type:
    """`a & b` — if both are struct-only, try to merge; else build a run-both
    wrapper. Result is always a Constraint subclass.

    NB2: short-circuit when both children are the same class — `A & A == A`
    semantically, and Python disallows duplicate bases anyway."""
    if a is b:
        return a
    if _is_struct_only(a) and _is_struct_only(b):
        merged = _try_merge_pydantic(a, b)
        if merged is not None:
            return merged
        # fall through to run-both
    return _make_and_wrapper(a, b)


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
      - **Same-name *different-type* fields:** pydantic raises during class
        construction; we catch and return None so the caller falls back to
        the run-both wrapper.
    """
    from pydantic.fields import FieldInfo

    a_fields = dict(a.model_fields)
    b_fields = dict(b.model_fields)

    namespace: Dict[str, Any] = {
        "model_config": pd.ConfigDict(arbitrary_types_allowed=True),
        "__annotations__": {},
    }

    # For overlapping fields: explicitly intersect FieldInfo + check type compat.
    # For non-overlapping fields: rely on MI to inherit them.
    for name in set(a_fields) & set(b_fields):
        a_info = a_fields[name]
        b_info = b_fields[name]
        if a_info.annotation != b_info.annotation:
            # Type conflict — abort merge; caller will use run-both fallback.
            return None
        # Same annotation: intersect Field metadata (tightest-bounds wins).
        namespace["__annotations__"][name] = a_info.annotation
        namespace[name] = FieldInfo.merge_field_infos(a_info, b_info)

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
        except ConstraintViolation as e:
            raise ConstraintViolation(
                f"AND: child {getattr(c, '__name__', c)!r} failed: {e}"
            ) from e
    return cur


# ============================================================================
# OR
# ============================================================================


def _or(a: type, b: type) -> type:
    return _make_composite_wrapper("Or", "OR", [a, b], _or_validate)


def _or_validate(children: List[type], value: Any, context: Optional[Dict] = None) -> Any:
    errors: List[str] = []
    for c in children:
        try:
            return c.validate(value, context=context)
        except ConstraintViolation as e:
            errors.append(str(e))
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
    except ConstraintViolation:
        return value  # negation succeeded
    raise ConstraintViolation(
        f"NOT: value satisfied negated constraint ({inner.describe()})"
    )


# ============================================================================
# XOR
# ============================================================================


def _xor(a: type, b: type) -> type:
    return _make_composite_wrapper("Xor", "XOR", [a, b], _xor_validate)


def _xor_validate(children: List[type], value: Any, context: Optional[Dict] = None) -> Any:
    results = []
    for c in children:
        try:
            v = c.validate(value, context=context)
            results.append((True, v))
        except ConstraintViolation:
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
            return f"NOT ({inner[0]})"
        return f" {label} ".join(f"({i})" for i in inner)

    def _json_schema(cls):  # noqa: ANN001
        sub = [c.json_schema() if hasattr(c, "json_schema") else {} for c in children]
        if label == "NOT":
            return {"not": sub[0]}
        if label == "AND":
            return {"allOf": sub}
        if label == "OR":
            return {"anyOf": sub}
        if label == "XOR":
            return {"oneOf": sub}
        return {}

    cls_dict = {
        "_constraint_is_composite": True,
        "_constraint_is_factory": False,
        "_constraint_children": children,
        "_constraint_op_label": label,
        "validate": classmethod(_validate),
        "model_validate": classmethod(_model_validate),
        "describe": classmethod(_describe),
        "json_schema": classmethod(_json_schema),
        "model_config": pd.ConfigDict(arbitrary_types_allowed=True),
    }
    cls = _ConstraintMeta(cls_name, (Constraint,), cls_dict)  # type: ignore[call-arg]
    return cls
