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
  * structural -> ("structural", name, fields, extra_anns, -> type(name, (Constraint,), ns)
                   extras, model_config, model_validators,     rebuilt from the class's FULL
                   field_validators)                           user namespace

The recipe is RECURSIVE: a Constraint that appears NESTED inside another (a struct field
typed as a Constraint — `addr: Address` / `currency: Constraint.field(...)` — or a
Constraint passed as a factory kwarg — `type=list[Constraint.field(...)]`) is itself a class
that can't pickle, so it is stored as a NESTED recipe rather than kept live. Without this
the inner class re-introduces exactly the unpicklable / by-reference failure the recipe
exists to avoid (and the by-reference case would sink the whole snapshot). `_NESTED` marks
a nested constraint; `_GENERIC` marks a generic (`Optional[X]`/`List[X]`/`Dict[K,V]`)
wrapping one. Each structural field is stored as `(annotation-spec, metadata, field-kwargs)`
so the Field-bound metadata + the full `_attributes_set` (always picklable) travel separately
from the recursed annotation.

A structural recipe reconstructs the class's FULL user namespace — fields (with defaults,
Field bounds, aliases, json_schema_extra, ...), non-field annotations (ClassVars), extra
class members (methods / classmethods / staticmethods / ClassVar values), model_config, and
the model/field validators (stored UNWRAPPED, so a `@classmethod` validator neither drags the
old class nor binds the rebuilt validator to it). A recipe is picklable iff its inputs are
(predicate/validator/method functions are, via dill; one that closes over something
unpicklable is not — the snapshot probes + drops it, as today).

Non-goal (out of scope, NOT a gap): a FOREIGN (non-Constraint) pydantic `BaseModel` used as
a constraint field type — or kept as a plain top-level variable — pickles BY-REFERENCE and
is not recipe'd, so it can't be rebuilt on a fresh kernel (a top-level one even fails the
main blob load). The Constraint authoring surface never produces this: you compose
Constraints (which DO recurse here), not raw pydantic models. Recipe-channel isolation does
not help — such a model is a live variable that sinks via the ordinary var-snapshot path
regardless — so it is left documented rather than worked around.
"""

from __future__ import annotations

import typing
from contextvars import ContextVar
from typing import Annotated, Any, List, Optional, Tuple, get_args, get_origin

_CHILD_KINDS = ("field", "composite", "structural")
# Markers used inside an annotation-spec / factory-kwarg-spec to flag a value that is itself
# a Constraint class (recipe'd recursively), or a generic wrapping one. Distinctive strings
# so they can never collide with a real annotation or kwarg value.
_NESTED = "__plm_nested_constraint_recipe__"
_GENERIC = "__plm_generic_annotation__"

# Class members the structural recipe SKIPS — pydantic internals + abc artifacts, never
# user-authored (users can't use the reserved `model_`/`_pydantic` prefixes).
_STRUCT_SKIP = frozenset({"model_config", "model_fields", "model_computed_fields", "_abc_impl"})


def _unbind(fn: Any) -> Any:
    """A pydantic validator's `.func` may be a BOUND classmethod (the `@field_validator
    @classmethod` form) whose `__self__` is the ORIGINAL class — storing it drags that whole
    (unpicklable) class into the recipe AND binds the rebuilt validator to the dead class.
    Return the underlying PLAIN function; pydantic re-binds it to the new class on rebuild."""
    return getattr(fn, "__func__", fn)


def is_constraint_class(c: Any) -> bool:
    """True iff `c` is a Constraint subCLASS (field / composite / structural) other than
    the base itself. Such classes cannot cross-process dill-pickle, so the snapshot must
    recipe them rather than keep them by value."""
    try:
        from plm.constraint import Constraint
        return isinstance(c, type) and issubclass(c, Constraint) and c is not Constraint
    except Exception:
        return False


# --- recursion helpers: replace any NESTED Constraint class with a picklable recipe -------

def _ann_contains_constraint(ann: Any) -> bool:
    """True iff `ann` is, or (via a generic's args) wraps, a Constraint class."""
    if is_constraint_class(ann):
        return True
    try:
        return any(_ann_contains_constraint(a) for a in get_args(ann))
    except Exception:
        return False


def _ann_to_spec(ann: Any) -> Any:
    """A field ANNOTATION as a picklable spec: a nested Constraint class -> ("__nested__",
    recipe); a generic wrapping one -> ("__generic__", origin, [arg-specs]); everything else
    (plain types, generics of plain types) passes through unchanged."""
    if is_constraint_class(ann):
        r = to_recipe(ann)
        return (_NESTED, r) if r is not None else ann   # un-recipe-able -> leave; dumps-probe drops
    origin = get_origin(ann)
    if origin is not None:
        args = get_args(ann)
        if args and any(_ann_contains_constraint(a) for a in args):
            return (_GENERIC, origin, tuple(_ann_to_spec(a) for a in args))
    return ann


def _spec_to_ann(spec: Any) -> Any:
    """Inverse of `_ann_to_spec` — rebuild the live annotation, re-creating nested
    Constraints via `from_recipe`."""
    if isinstance(spec, tuple) and len(spec) == 2 and spec[0] == _NESTED:
        return from_recipe(spec[1])
    if isinstance(spec, tuple) and len(spec) == 3 and spec[0] == _GENERIC:
        _, origin, arg_specs = spec
        args = tuple(_spec_to_ann(a) for a in arg_specs)
        if origin is typing.Union:
            return typing.Union[args]
        return origin[args[0]] if len(args) == 1 else origin[args]
    return spec


def _value_to_spec(v: Any) -> Any:
    """A factory-kwarg VALUE as a picklable spec: a Constraint class (e.g. `type=SomeStruct`, `is_instance_of=`) -> nested recipe; a typing generic wrapping a Constraint
    (e.g. `type=list[SomeStruct]`) -> ("__generic__", origin, [arg-specs]); list/tuple/dict
    containers are walked; plain values pass through."""
    if is_constraint_class(v):
        r = to_recipe(v)
        return (_NESTED, r) if r is not None else v
    origin = get_origin(v)
    if origin is not None:
        args = get_args(v)
        if args and any(_ann_contains_constraint(a) for a in args):
            return (_GENERIC, origin, tuple(_value_to_spec(a) for a in args))
    if isinstance(v, list):
        return [_value_to_spec(x) for x in v]
    if isinstance(v, tuple):
        return tuple(_value_to_spec(x) for x in v)
    if isinstance(v, dict):
        return {k: _value_to_spec(x) for k, x in v.items()}
    return v


def _spec_to_value(s: Any) -> Any:
    """Inverse of `_value_to_spec`."""
    if isinstance(s, tuple) and len(s) == 2 and s[0] == _NESTED:
        return from_recipe(s[1])
    if isinstance(s, tuple) and len(s) == 3 and s[0] == _GENERIC:
        _, origin, arg_specs = s
        args = tuple(_spec_to_value(a) for a in arg_specs)
        return origin[args[0]] if len(args) == 1 else origin[args]
    if isinstance(s, list):
        return [_spec_to_value(x) for x in s]
    if isinstance(s, tuple):
        return tuple(_spec_to_value(x) for x in s)
    if isinstance(s, dict):
        return {k: _spec_to_value(x) for k, x in s.items()}
    return s


# The recipe-recursion stack (ids of constraint classes mid-build), used ONLY to break a cycle on a
# self-recursive constraint (a field typed as the constraint itself) so `to_recipe` doesn't recurse
# forever. `to_recipe` runs ONLY in the SYNCHRONOUS crash-restart snapshot loop (repl/prefix.py,
# between cells) — never inside a `parallel()` branch or a thread — so it only needs to break
# recursion WITHIN one single-threaded call tree; the per-id add/discard balance keeps the stack
# empty between top-level calls. Kept as a ContextVar (not a module global) merely so it can't be a
# process-global shared across unrelated top-level snapshots — it is NOT a parallel-branch isolation
# guarantee (copy_context() would share the set by reference, but that case can't arise: to_recipe
# is never run concurrently).
_RECIPE_IN_PROGRESS: "ContextVar[Optional[set]]" = ContextVar("_recipe_in_progress", default=None)


def to_recipe(c: Any) -> Optional[Tuple]:
    """A picklable recipe that `from_recipe` rebuilds into the same constraint, or None for
    a non-constraint (left alone) / a constraint we can't recipe (the caller then drops it
    rather than keep an un-loadable value).

    Cycle-guarded: a self-/mutually-recursive structural Constraint (a field annotation pointing
    back at the class) would otherwise recurse `_ann_to_spec -> to_recipe -> ...` forever and blow
    the stack. When a class is reached again while its OWN recipe is still building, we return None
    (a cyclic constraint can't flatten to a finite recipe) — the field then keeps its live
    annotation, so the dumps-probe drops the whole (unpicklable) recipe cleanly. The
    in-progress stack lives in a per-context ContextVar so concurrent `parallel()` branches don't
    share it; add/discard per id balance out, so the stack is empty between top-level calls."""
    if not isinstance(c, type):
        return None
    _seen = _RECIPE_IN_PROGRESS.get()
    if _seen is None:                                # first call in THIS context -> fresh stack
        _seen = set()
        _RECIPE_IN_PROGRESS.set(_seen)
    if id(c) in _seen:
        return None
    _seen.add(id(c))
    try:
        return _to_recipe_build(c)
    finally:
        _seen.discard(id(c))


def _to_recipe_build(c: Any) -> Optional[Tuple]:
    """Recipe body — always reached through the cycle-guarded public `to_recipe`."""
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
        # Recurse: a kwarg value may itself be a Constraint (type=Constraint struct,
        # is_instance_of=SomeConstraint, ...) which can't pickle -> store it as a nested recipe.
        return ("field", {k: _value_to_spec(v) for k, v in dict(kw).items()})
    if is_constraint_class(c):
        # structural subclass -> reconstruct its FULL user namespace (NOT dill / source):
        # fields + non-field annotations (ClassVars) + extra class members (methods/ClassVar
        # values) + model_config + validators. Each field is (annotation-spec, metadata,
        # field-kwargs):
        #   * annotation recursed via _ann_to_spec (a Constraint-typed field -> nested recipe);
        #   * Field-bound metadata (Gt/Le/MinLen/...) re-wrapped as Annotated on rebuild;
        #   * `_attributes_set` is pydantic's OWN record of EVERY kwarg the author set on the
        #     field (default, default_factory, alias(es), description, json_schema_extra,
        #     frozen, exclude, ...) — faithful + future-proof, unlike an attribute allow-list
        #     that grows edge cases as pydantic evolves. `annotation` is carried separately so
        #     it is dropped here (Field() would reject it as an extra kwarg).
        decs = getattr(c, "__pydantic_decorators__", None)
        _vk = (set(decs.model_validators) | set(decs.field_validators)) if decs else set()
        fields: dict = {}
        for fname, fi in c.model_fields.items():
            _fkw = {_ak: _av for _ak, _av in fi._attributes_set.items() if _ak != "annotation"}
            fields[fname] = (_ann_to_spec(fi.annotation), tuple(fi.metadata), _fkw)
        # Non-field annotations (ClassVars etc.) — kept so a validator/method reading a ClassVar
        # still finds it on the rebuilt class (else `cls.MAX` -> AttributeError after respawn).
        extra_anns = {_k: _ann_to_spec(_v) for _k, _v in getattr(c, "__annotations__", {}).items()
                      if _k not in c.model_fields}
        # Extra user-defined class members — ClassVar VALUES, methods, classmethods,
        # staticmethods — that are NOT fields, validators, dunders, or pydantic internals. A
        # classmethod/staticmethod is stored UNWRAPPED (its __func__) and re-wrapped on rebuild,
        # so it carries no binding to the (unpicklable) old class.
        extras: dict = {}
        for _k, _v in c.__dict__.items():
            if (_k.startswith("__") or _k in c.model_fields or _k in _vk
                    or _k in _STRUCT_SKIP or _k.startswith(("model_", "_pydantic", "_abc"))):
                continue
            if isinstance(_v, classmethod):
                extras[_k] = ("cm", _v.__func__)
            elif isinstance(_v, staticmethod):
                extras[_k] = ("sm", _v.__func__)
            else:
                extras[_k] = ("plain", _v)
        cfg = dict(getattr(c, "model_config", None) or {})    # extra/frozen/populate_by_name/...
        # Validators: carry the pydantic decorator KEY (the namespace attr name pydantic
        # identifies the validator by), NOT func.__name__ — two validators can share a __name__
        # (stacked `exactly_one_of` -> both `_check`; auto-merged A & B -> both `_ck`) while the
        # source class kept them under DISTINCT keys; re-deriving from __name__ collides them
        # and silently drops one rule. And store the UNWRAPPED func (`_unbind`): a bound
        # classmethod (`@field_validator @classmethod`) would otherwise drag the whole old class
        # into the pickle AND bind the rebuilt validator to that dead class.
        mvs = [(k, v.info.mode, _unbind(v.func)) for k, v in decs.model_validators.items()] if decs else []
        fvs = [(k, getattr(v.info, "mode", "after"), tuple(v.info.fields), _unbind(v.func))
               for k, v in decs.field_validators.items()] if decs else []
        return ("structural", c.__name__, fields, extra_anns, extras, cfg, mvs, fvs)
    return None


def from_recipe(recipe: Tuple) -> Any:
    """Rebuild a constraint from a recipe produced by `to_recipe`."""
    from plm.constraint import Constraint                  # in-call: constraint surface is optional
    kind = recipe[0]
    if kind == "field":
        return Constraint.field(**{k: _spec_to_value(v) for k, v in recipe[1].items()})
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
        from pydantic import model_validator, field_validator, Field
        _, name, fields, extra_anns, extras, cfg, mvs, fvs = recipe
        annotations: dict = {}
        ns: dict = {"__module__": "__main__"}
        for fname, (spec, meta, fkw) in fields.items():
            ann = _spec_to_ann(spec)
            if meta:
                ann = Annotated[tuple([ann, *meta])]
            annotations[fname] = ann
            if fkw:
                ns[fname] = Field(**fkw)            # default/default_factory/alias/description/...
        for _k, _spec in extra_anns.items():        # ClassVar (and other non-field) annotations
            annotations[_k] = _spec_to_ann(_spec)
        for _k, (_kind, _val) in extras.items():    # ClassVar values, methods, class/staticmethods
            ns[_k] = classmethod(_val) if _kind == "cm" else staticmethod(_val) if _kind == "sm" else _val
        ns["__annotations__"] = annotations
        if cfg:
            ns["model_config"] = cfg               # extra/frozen/populate_by_name/...
        for key, mode, func in mvs:
            ns[key] = model_validator(mode=mode)(func)
        for key, mode, vfields, func in fvs:
            ns[key] = field_validator(*vfields, mode=mode)(func)
        new_cls = type(name, (Constraint,), ns)
        # Re-bind any captured method's `__class__` free-var cell to the REBUILT class. A method that
        # carries such a cell (at capture time it pointed at the ORIGINAL, now-dead class) otherwise
        # raises "obj is not an instance or subtype of type" on the rebuilt instances. NOTE: a method
        # that itself CALLS super() can no longer reach here — the metaclass rejects it at definition
        # (constraint/base.py `_calls_super_builtin`, F35). What DOES still reach here are the
        # residual `__class__`-cell cases that check intentionally allows: a method whose NESTED
        # helper uses super(), or one that references bare `__class__`. The cell is identified by the
        # `__class__` free-var and (Py3.7+) is writable. Covers methods/classmethods/staticmethods +
        # validators. (NC3-2; the defensive complement to the F35 reject — see base.py.)
        for _fn in (*(_v for (_kind, _v) in extras.values()), *(f for *_h, f in mvs),
                    *(f for *_h, f in fvs)):
            _code = getattr(_fn, "__code__", None)
            _closure = getattr(_fn, "__closure__", None)
            if _code is not None and _closure and "__class__" in _code.co_freevars:
                try:
                    _closure[_code.co_freevars.index("__class__")].cell_contents = new_cls
                except Exception:
                    pass
        return new_cls
    raise ValueError(f"unknown constraint recipe kind {recipe[0]!r}")
