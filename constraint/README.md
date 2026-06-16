# `plm.constraint`

A constraint is a **machine-checkable contract** PLM hands a sub-LLM: a value-rule or
an object-shape that the sub-LLM's output must satisfy. It exposes three things to the
model — `validate()` (the gate), `describe()` (the natural-language contract), and
`json_schema()` (the formal contract) — and a small algebra (`& | ^ ~`) to compose them.

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
nullable and never a `Union`/`X | Y` (use the `|` algebra across whole constraints instead).

`Constraint.field(...)` is the **single public authoring entry**. It embeds *flat* as a
struct field (the field is the bare value, not a `{value: ..}` wrapper) and works
standalone too. `Constraint.of(...)` is the internal engine `.field` rides on — **not**
used outside this module. Author everything through `.field` + structural subclassing +
the algebra; do **not** reach for plain pydantic `Field(...)`, bare `x: Annotation`
defaults, or raw `Enum` types — the constraint mechanism covers those (`type=Literal[...]`,
`multiple_of`, etc.) and only the constraint mechanism is rendered by `describe()`.

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
`coercer=fn`. Execution order: **coercer(s) → type/Field → predicate(s)**.

## The algebra `& | ^ ~`

```
A & B   two shapes → MERGE into one tighter shape; shape & value-rule → run-both composite
A | B   at least one holds        A ^ B   exactly one holds
~A      negate a value-test  (a shape has no complement → use a @model_validator)
```

## `describe()` — the natural-language contract

For a structural constraint, `describe()` renders three parts:

1. **Header** — `Return an object of type X with fields:`
2. **One line per field** — `name (type) — <constraints>`:
   - a `.field` member shows its **complete** rule: declarative phrases + tagged
     predicates + tagged coercers + `description=` (see *tagging* below).
   - a **nested structural** field (`leg: Leg`, or `Optional[Leg]`) is **expanded
     inline**, indented, with a cycle guard (a self-referential type renders as its
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

`describe()` recurses into **direct** and **`Optional[...]`** nested structural fields,
but **not** into nested Constraints inside a *container*:

```python
class B(Constraint):
    items: list[Item]        # describe() shows:  - items (List)   ← Item's fields NOT expanded
```

The element's shape *is* present in `json_schema()` (via `$defs`), just not in the NL
`describe()`. If a collection element's contract must appear in the NL contract, describe
the element type separately, or rely on `json_schema()` for that field. (`list_of=Item`
on a `.field` shows `list of [Item]` — the element type name, also not deep-expanded.)

## Operations

```python
c.validate(value, context={...})   # → validated value, or raises ConstraintViolation
c.describe()                       # NL contract (the channel above)
c.json_schema()                    # formal contract — nests via $defs, carries optionality/defaults/enums
```

`describe()` is the *unlimited* NL channel (predicates, coercers, cross-field rules);
`json_schema()` is the *formal, fixed-vocabulary* channel (it does carry optionality,
defaults, enum values, and deep-nested/collection shapes that the NL describe summarizes).
The two are complementary — PLM typically hands a sub-LLM both.
