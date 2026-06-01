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
        # If we're wrapping a pydantic descriptor proxy, attach to its
        # underlying function so the attribute survives pydantic's
        # decorator unwrapping during class construction.
        target = fn_or_obj
        wrapped = getattr(fn_or_obj, "wrapped", None)
        if wrapped is not None and callable(wrapped):
            target = wrapped  # type: ignore[assignment]
        try:
            setattr(target, _DESCRIPTION_ATTR, description)
        except (AttributeError, TypeError):
            pass
        return fn_or_obj

    return deco


def get_description(fn_or_obj) -> str | None:
    """Read back the `@constraint(...)` description, if any. Returns None
    when the object has no attached description.

    Checks the object itself first, then `obj.wrapped` (pydantic descriptor
    proxy case) — so it works regardless of decoration order or whether
    pydantic has unwrapped the function yet."""
    direct = getattr(fn_or_obj, _DESCRIPTION_ATTR, None)
    if direct is not None:
        return direct
    wrapped = getattr(fn_or_obj, "wrapped", None)
    if wrapped is not None:
        return getattr(wrapped, _DESCRIPTION_ATTR, None)
    return None
