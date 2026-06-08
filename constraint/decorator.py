"""@constraint decorator — attach a natural-language description to a
validator function so `.describe()` can include it in the rendered output.

Usage:

    class Order(Constraint):
        limit: float
        stop: float

        @constraint("stop must be below limit")
        @model_validator(mode="after")
        def _stop_below_limit(self):
            if self.stop >= self.limit:
                raise ValueError("stop must be < limit")
            return self

Either decoration order works (`@constraint` inside or outside the pydantic
validator decorator). The decorator handles `PydanticDescriptorProxy` by
reaching into `.wrapped` so the description ends up on the underlying
function that pydantic keeps in `cls.__pydantic_decorators__`.
"""

from __future__ import annotations

from typing import Callable, TypeVar

_DESCRIPTION_ATTR = "_plm_constraint_description"

_F = TypeVar("_F")


def constraint(description: str) -> Callable[[_F], _F]:
    """Decorator factory. Attaches `description` to the decorated callable
    so the Constraint `.describe()` walker can collect it.

    Works in either decoration order with pydantic's `@model_validator` and
    `@field_validator`:

        @constraint("...")
        @model_validator(mode="after")
        def fn(self): ...

        # OR

        @model_validator(mode="after")
        @constraint("...")
        def fn(self): ...

    For pydantic's `PydanticDescriptorProxy` (created by validator
    decorators), the description is attached to the proxy's `.wrapped`
    underlying function — so pydantic's `__pydantic_decorators__.func`
    points to the function carrying the attribute either way.
    """

    def deco(fn_or_obj: _F) -> _F:
        # Attach the description to the DEEPEST plain function — the object pydantic
        # keeps as `Decorator.func` (the classmethod's `__func__` for a validator).
        # Walking both `.wrapped` (pydantic descriptor proxy) and `.__func__`
        # (classmethod/staticmethod/bound method) makes it land in the right place
        # regardless of decoration order — including `@constraint` placed OUTSIDE a
        # `@field_validator` + `@classmethod` stack, where the older single-level
        # unwrap missed it.
        target = fn_or_obj
        for layer in _descend(fn_or_obj):
            target = layer
        try:
            setattr(target, _DESCRIPTION_ATTR, description)
        except (AttributeError, TypeError):
            pass
        return fn_or_obj

    return deco


def _descend(obj):
    """Yield `obj` and each layer beneath it — a pydantic descriptor proxy's
    `.wrapped`, then a classmethod/staticmethod/bound-method's `.__func__` — down
    to the underlying plain function. Cycle-guarded by object id."""
    seen: set = set()
    while obj is not None and id(obj) not in seen:
        seen.add(id(obj))
        yield obj
        obj = getattr(obj, "wrapped", None) or getattr(obj, "__func__", None)


def get_description(fn_or_obj) -> str | None:
    """Read back the `@constraint(...)` description, if any (else None).

    Descends through `.wrapped` (pydantic descriptor proxy) and `.__func__`
    (classmethod/staticmethod/bound method), checking each layer — so it finds the
    description no matter which layer it was attached to or whether pydantic has
    unwrapped the function yet. Order-independent, matching `@constraint`."""
    try:
        for layer in _descend(fn_or_obj):
            d = getattr(layer, _DESCRIPTION_ATTR, None)
            if d is not None:
                return d
    except Exception:
        return None          # a malformed object (a .wrapped/.__func__ getter raising) -> no description,
                             # never propagate: a description is optional metadata and this is read on
                             # arbitrary authored members. Protects EVERY caller, not just describe()'s walk.
    return None
