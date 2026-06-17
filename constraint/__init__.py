"""plm.constraint — generic Constraint class.

Public API
==========

    from plm.constraint import Constraint, ConstraintViolation, constraint

    # Pattern 1 — structural subclass; EVERY field is a Constraint.field(...):
    class MyAnswer(Constraint):
        score: Constraint.field(type=int, int_range=(0, 100))
        label: Constraint.field(type=str)

    # Pattern 2 — value/field rule via Constraint.field (the public authoring
    # entry; works standalone AND embeds FLAT as a struct field):
    c = Constraint.field(type=int, int_range=(0, 9))
    c = Constraint.field(type=int, predicate=is_prime, description="must be prime")
    c = Constraint.field(type=str, coercer=str.lower, str_pattern=r"^[a-z]+$")
    c = Constraint.field(type=str, coercers=[strip_bom, normalize_spaces, str.lower])

    # Every field has a MANDATORY type coercer, ALWAYS spelled `type=`: a builtin, a typing
    # generic (list[int]/dict[str,int]/tuple[..]/set[int]/frozenset[int]), a Literal[...] (an
    # enum of values), or a Constraint struct. It is the sole source of the schema "type".
    # Everything else (int_range, str_pattern, is_instance_of, kind=, multiple_of, ...) is
    # a PREDICATE — a pure check that never sets a type. The full kwarg surface is
    # catalogued in the `plm.constraint.base` module docstring.

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

Execution order in one `Constraint.field(...)` call (the 3-stage pipeline):

    coercer(s)  →  the type coercer (establishes the type)  →  predicate(s)

PLM authors typically use one of:
  - `predicate=` alone for pure checks.
  - `coercer=` for normalize-then-check (output coerced, no rejection).
  - both, when LLM output needs normalization before a check fires.


Built-in `kind=` registry
=========================

`Constraint.field(type=str, kind="semver")`, `kind="url"`, `kind="json"`, `kind="email"`,
`kind="iso_date"`, `kind="uuid"`, `kind="identifier"`, `kind="regex"`,
`kind="path_exists"`. The table is read-only — a fixed set of built-ins. For a check no
built-in kind expresses, author it inline:

    Constraint.field(type=str, predicate=fn, description="...")  # fn raises ValueError on bad


Decorators
==========

  @constraint("NL text")     attaches a natural-language description to a
                             validator method (works with `@model_validator`
                             or `@field_validator`, either decoration order).
                             The description surfaces in `.describe()` so
                             cross-field rules show up in sub-LLM prompts.
"""

from .base import Constraint, ConstraintViolation
from .decorator import constraint, get_description
from .registry import REGISTRY
from .snapshot import to_recipe, from_recipe, is_constraint_class

__all__ = [
    "Constraint",
    "ConstraintViolation",
    "constraint",
    "get_description",
    "REGISTRY",
    "to_recipe",
    "from_recipe",
    "is_constraint_class",
]
