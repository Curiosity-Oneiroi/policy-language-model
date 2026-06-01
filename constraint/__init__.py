"""plm.constraint — generic Constraint class with full algebra.

Public API
==========

    from plm.constraint import Constraint, ConstraintViolation, Field, constraint

    # Pattern 1 — structural subclass (PLM writes ordinary pydantic):
    class MyAnswer(Constraint):
        score: int = Field(ge=0, le=100)
        label: str

    # Pattern 2 — one-shot factory (no class needed):
    c = Constraint.of(int_range=(0, 9))
    c = Constraint.of(predicate=is_prime, description="must be prime")
    c = Constraint.of(coercer=str.lower, str_pattern=r"^[a-z]+$")
    c = Constraint.of(coercers=[strip_bom, normalize_spaces, str.lower])

    # Composition (closed under &, |, ~, ^):
    c1 & c2     # both must pass; struct-only pair auto-merges into one model
    c1 | c2     # at least one passes
    ~c          # predicate-form only — must NOT satisfy (structural raises)
    c1 ^ c2     # exactly one passes

    # Shared API on every Constraint subclass:
    c.validate(value, context=None)   # raises ConstraintViolation on failure
    c.describe()                       # NL string for sub-LLM prompts
    c.json_schema()                    # JSON Schema for non-Python sub-LLMs


User-supplied functions: predicate vs coercer
=============================================

Two kwargs accept user-supplied functions, with deliberately different
contracts so each has one clear meaning:

  predicate=fn               — checks. Runs AFTER coercion + type validation.
                               Signature: (value) -> None | raises ValueError
                               Return None / True / the input on success.
                               Raise ValueError to reject.
                               Returning any other value is a contract
                               violation and raises a clear error pointing at
                               `coercer=` as the right tool.

  coercer=fn                 — transforms. Runs BEFORE type validation.
  coercers=[fn1, fn2, ...]     Signature: (value) -> new_value | raises ValueError
                               The return value becomes the new value passed
                               downstream. Raise ValueError to reject.
                               Multiple coercers chain left → right: fn1's
                               output feeds fn2.

Execution order in one `Constraint.of(...)` call:

    coercer(s)  →  type & Field(...) constraints  →  predicate(s)

PLM authors typically use one of:
  - `predicate=` alone for pure checks.
  - `coercer=` for normalize-then-check (output coerced, no rejection).
  - both, when LLM output needs normalization before a check fires.


Built-in `kind=` registry
=========================

`Constraint.of(kind="semver")`, `kind="url"`, `kind="json"`, `kind="email"`,
`kind="iso_date"`, `kind="uuid"`, `kind="identifier"`, `kind="regex"`,
`kind="path_exists"`. PLM can extend at runtime:

    Constraint.registry["my_kind"] = lambda v: ... # raises ValueError on bad


Decorators
==========

  @constraint("NL text")     attaches a natural-language description to a
                             validator method (works with `@model_validator`
                             or `@field_validator`, either decoration order).
                             The description surfaces in `.describe()` so
                             cross-field rules show up in sub-LLM prompts.

  @Constraint.exactly_one_of("a", "b", "c")
                             class decorator enforcing that exactly one of
                             the named fields is non-None. Raises at
                             decoration time if no field names given.
"""

from pydantic import Field

from .base import Constraint, ConstraintViolation
from .decorator import constraint, get_description
from .registry import REGISTRY

__all__ = [
    "Constraint",
    "ConstraintViolation",
    "Field",
    "constraint",
    "get_description",
    "REGISTRY",
]
