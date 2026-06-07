"""Crash-restart RECIPE for constraints — `Constraint.field(...)`, `& | ^ ~`, AND
structural `class X(Constraint)`.

The kernel snapshots user variables with dill, but **no** Constraint CLASS cross-process
dill-pickles:
  * a factory (`Constraint.field(...)`) or composite builds a dynamic class with recursive
    self-references → dill `dumps` itself FAILS;
  * a structural `class X(Constraint)` defined in a cell is pydantic-forced to pickle
    BY-REFERENCE (`__main__.X`) → `dumps` "succeeds" but `loads` fails on the fresh kernel
    (and would otherwise sink the WHOLE snapshot).

So — mirroring how a CLASS POLICY survives by its `_p_source` — every Constraint class is
snapshotted as a small picklable RECIPE and REBUILT on rehydrate, instead of by value:

  * field      -> ("field", kwargs)                       -> Constraint.field(**kwargs)
  * composite  -> ("composite", OP, [child, ...])         -> recompose with & | ^ ~
  * structural -> ("structural", name, fields, defaults,  -> type(name, (Constraint,), ns)
                   model_validators, field_validators)        rebuilt from model_fields +
                                                               the pydantic validator decorators
                                                               (fields, defaults, Field bounds,
                                                               and validators are preserved;
                                                               arbitrary METHODS are not — rare)

A recipe is picklable iff its inputs are (predicate/validator functions are, via dill; one
that closes over something unpicklable is not — the snapshot probes + drops it, as today).
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

_CHILD_KINDS = ("field", "composite", "structural")


def is_constraint_class(c: Any) -> bool:
    """True iff `c` is a Constraint subCLASS (field / composite / structural) other than
    the base itself. Such classes cannot cross-process dill-pickle, so the snapshot must
    recipe them rather than keep them by value."""
    try:
        from plm.constraint import Constraint
        return isinstance(c, type) and issubclass(c, Constraint) and c is not Constraint
    except Exception:
        return False


def to_recipe(c: Any) -> Optional[Tuple]:
    """A picklable recipe that `from_recipe` rebuilds into the same constraint, or None for
    a non-constraint (left alone) / a constraint we can't recipe (the caller then drops it
    rather than keep an un-loadable value)."""
    if not isinstance(c, type):
        return None
    if getattr(c, "_constraint_is_composite", False):
        op = getattr(c, "_constraint_op_label", None)
        children: List[Any] = []
        for ch in getattr(c, "_constraint_children", ()):
            r = to_recipe(ch)
            children.append(r if r is not None else ch)   # a by-value-picklable child kept as-is
        return ("composite", op, children)
    if getattr(c, "_constraint_is_factory", False):
        kw = getattr(c, "_constraint_factory_kwargs", None)
        if kw is None:
            return None
        return ("field", dict(kw))
    if is_constraint_class(c):
        # structural subclass -> reconstruct from its fields + validators (NOT dill / source).
        from typing import Annotated
        fields: dict = {}
        defaults: dict = {}
        for fname, fi in c.model_fields.items():
            ann = fi.annotation
            if fi.metadata:                                # Field bounds (Gt/Le/MinLen/...) -> Annotated
                ann = Annotated[tuple([ann, *fi.metadata])]
            fields[fname] = ann
            if not fi.is_required():
                defaults[fname] = fi.default
        decs = getattr(c, "__pydantic_decorators__", None)
        mvs = [(v.info.mode, v.func) for v in decs.model_validators.values()] if decs else []
        fvs = [(getattr(v.info, "mode", "after"), tuple(v.info.fields), v.func)
               for v in decs.field_validators.values()] if decs else []
        return ("structural", c.__name__, fields, defaults, mvs, fvs)
    return None


def from_recipe(recipe: Tuple) -> Any:
    """Rebuild a constraint from a recipe produced by `to_recipe`."""
    from plm.constraint import Constraint                  # in-call: constraint surface is optional
    kind = recipe[0]
    if kind == "field":
        return Constraint.field(**recipe[1])
    if kind == "composite":
        op, child_specs = recipe[1], recipe[2]
        children = [from_recipe(s) if (isinstance(s, tuple) and s and s[0] in _CHILD_KINDS) else s
                    for s in child_specs]
        if op == "NOT":
            return ~children[0]
        result = children[0]
        for ch in children[1:]:
            if op == "AND":
                result = result & ch
            elif op == "OR":
                result = result | ch
            elif op == "XOR":
                result = result ^ ch
            else:
                raise ValueError(f"unknown composite op {op!r}")
        return result
    if kind == "structural":
        from pydantic import model_validator, field_validator
        _, name, fields, defaults, mvs, fvs = recipe
        ns: dict = {"__annotations__": dict(fields), "__module__": "__main__"}
        ns.update(defaults)
        for mode, func in mvs:
            ns[func.__name__] = model_validator(mode=mode)(func)
        for mode, vfields, func in fvs:
            ns[func.__name__] = field_validator(*vfields, mode=mode)(func)
        return type(name, (Constraint,), ns)
    raise ValueError(f"unknown constraint recipe kind {recipe[0]!r}")
