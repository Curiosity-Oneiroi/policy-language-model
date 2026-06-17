# `plm.constraint`

A constraint is a **machine-checkable contract** PLM hands a sub-LLM: a value-rule or
an object-shape that the sub-LLM's output must satisfy. It exposes three things to the
model — `validate()` (the gate), `describe()` (the natural-language contract), and
`json_schema()` (the formal contract).

> The full kwarg catalog lives in the `base.py` module docstring. This file documents
> the **authoring surface**, the **`describe()` rendering** (and its edges), and the
> **conventions** PLM should follow.

## Authoring — two patterns, one vocabulary

**1. A shape** — subclass `Constraint`; each field is a `Constraint.field(...)`:
```python
class Order(Constraint):
    currency: Constraint.field(coercer=str.upper, type=Literal["USD", "EUR"])  # enum via type=
    qty:      Constraint.field(type=int, int_range=(1, 1000), multiple_of=5)
```

**2. A value rule** — `Constraint.field(...)` standalone (no class):
```python
Prob = Constraint.field(type=float, float_range=(0.0, 1.0))
```

Every field has a **mandatory type coercer, always spelled `type=`**: a builtin, a typing
generic (`list[int]`, `dict[str,int]`, `tuple[..]`, `set[int]`, `frozenset[int]`), a
`Literal[...]` (an enum of values), or a `Constraint` struct. It is the sole source of the
schema's `"type"`. Everything else (`int_range`, `str_pattern`, `is_instance_of`, …) is a
**predicate** — a pure check that never sets a type (some enrich the schema with non-`"type"`
keywords like `minimum`/`pattern`). A field with no `type=` is an error; the type is never
nullable and never a `Union`/`X | Y` (model alternatives with a `@model_validator` instead).

`Constraint.field(...)` is the **single public authoring entry**. It embeds *flat* as a
struct field (the field is the bare value, not a `{value: ..}` wrapper) and works
standalone too. `Constraint.of(...)` is the internal engine `.field` rides on — **not**
used outside this module. EVERY structural field is authored through `.field` — a bare
annotation (`x: int`, `x: Foo`, a bare nested `leg: Leg`), a raw pydantic `Field(...)`, an
`Optional[...]`, or a default value is **rejected at class definition**. The constraint
mechanism covers those needs (`type=Literal[...]`, `multiple_of`, nested via `type=Leg`, …),
only the constraint mechanism is rendered by `describe()`, and every field is required.

## The `.field(**kw)` surface (full catalog in base.py docstring)

**Type coercer** (mandatory, always `type=`):
`type=<int|float|str|bool|bytes|complex | list[T]|dict[K,V]|tuple[..]|set[T]|frozenset[T] |
Literal[...] | YourStruct>`. The element `T` in a generic may itself be a Constraint
(`type=list[Constraint.field(...)]`).
**Predicates** (pure checks; never set a type; some enrich the schema): `int_range/ge/gt/le/lt ·
multiple_of · float_range/approx/finite · not_one_of · str_pattern/length/prefix/suffix/contains ·
list_length/unique/sorted/monotonic/contains · set_length/subset_of/superset_of · dict_length ·
bytes_length · abs_range/real_range/imag_range · is_instance_of (pure isinstance; pair with
type=object)/callable/behaves_like ·
kind="email|url|uuid|semver|iso_date|identifier|json|regex|path_exists" (needs type=str) ·
coercer(s)/predicate(s)/description`.

Custom checks that no kwarg expresses → `predicate=fn` (with `description=`). Transforms →
`coercer=fn`. Execution order: **coercer(s) → type coercer → predicate(s)**.

## Several rules at once — author one constraint

To require several shapes/rules together, write ONE `class X(Constraint)` with all the fields
plus a `@model_validator(mode="after")` for any cross-field or either-of rule. The single
constraint carries the whole contract.

## `describe()` — the natural-language contract

For a structural constraint, `describe()` renders three parts:

1. **Header** — `Return an object of type X with fields:`
2. **One line per field** — `name (type) — <constraints>`:
   - a `.field` member shows its **complete** rule: declarative phrases + tagged
     predicates + tagged coercers + `description=` (see *tagging* below).
   - a **nested structural** field (`leg: Constraint.field(type=Leg)`) renders its type and
     the nested contract inline, with a cycle guard (a self-referential type renders as its
     name, not infinitely).
3. **`Additionally:`** — cross-field rules: every validator **tagged** with
   `@constraint("...")` (works in any decoration order).

Cross-field invariants → `@model_validator(mode="after")` tagged with `@constraint("...")`.

### Tagging predicates / coercers / validators

`@constraint("...")` attaches a description to any callable so `describe()` surfaces it:

```python
@constraint("must be a prime")        # PREDICATE → rendered as a bare clause: "must be a prime"
def is_prime(n): ...
@constraint("trimmed + lowercased")   # COERCER  → rendered as "coerced (trimmed + lowercased)"
def slugify(s): return s.strip().lower()

Constraint.field(type=str, predicate=is_prime, coercer=slugify)
```

- **predicates** → tagged render their text bare (a *check*); untagged collapse to
  `N predicate(s) checked`.
- **coercers** → tagged render as `coerced (...)` (a *transform*, deliberately distinct
  from a check); untagged collapse to `N coercer(s) applied`.
- **validators** → tagged show in `Additionally:`. **Convention: always tag a
  `@model_validator` with `@constraint("...")`** — an *untagged* validator runs but is
  **invisible** to `describe()` (the sub-LLM is never told the cross-field rule exists).

> You can't `@constraint` a lambda — use a `def` (or set the whole-constraint
> `description=` for one blanket phrase).

### Known limitation — collections of nested Constraints are not deep-recursed

`describe()` expands a **nested structural** field (`Constraint.field(type=Leg)`) inline,
but does **not** expand nested Constraints inside a *container*:

```python
class B(Constraint):
    items: Constraint.field(type=list[Item])   # describe() shows: items (List) — Item's NOT expanded
```

The element's shape *is* present in `json_schema()` (via `$defs`), just not in the NL
`describe()`. If a collection element's contract must appear in the NL contract, describe
the element type separately, or rely on `json_schema()` for that field.

## Operations

```python
c.validate(value, context={...})   # → validated value, or raises ConstraintViolation
c.describe()                       # NL contract (the channel above)
c.json_schema()                    # formal contract — nests via $defs, carries enums + predicate bounds (minimum/pattern/…)
```

`describe()` is the *unlimited* NL channel (predicates, coercers, cross-field rules);
`json_schema()` is the *formal, fixed-vocabulary* channel (every field is required + non-null, so
it carries no defaults/optionality — it does carry enum values, predicate-derived keywords
(minimum/pattern/…), and the deep-nested/collection shapes that the NL describe summarizes).
The two are complementary — PLM typically hands a sub-LLM both.
