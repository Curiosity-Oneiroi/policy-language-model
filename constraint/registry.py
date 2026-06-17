"""Built-in `kind=` constraint registry — common string predicates PLM would
otherwise re-invent every call.

Each entry is a plain predicate callable: takes a value, returns nothing
on success, raises ValueError with a reason on failure.

The table is a FIXED set of built-ins and is READ-ONLY (a `MappingProxyType`):
it is never mutated at runtime, so there is no shared mutable global to guard in a
parallel() branch and nothing to lose across a crash-restart. For a check no
built-in kind expresses, author it inline —
`Constraint.field(type=..., predicate=fn, description="...")` — the same act with
no global name to manage, and it survives crash-restart (the function travels with
the constraint).
"""

from __future__ import annotations

import json as _json
import keyword as _keyword
import os
import re as _re
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping
from urllib.parse import urlparse
from uuid import UUID


def _semver(s: Any) -> None:
    if not isinstance(s, str):
        raise ValueError(f"semver: expected str, got {type(s).__name__}")
    if not _re.fullmatch(r"\d+\.\d+\.\d+(-[0-9A-Za-z.\-]+)?", s):
        raise ValueError(f"semver: {s!r} does not match MAJOR.MINOR.PATCH[-pre]")


def _url(s: Any) -> None:
    if not isinstance(s, str):
        raise ValueError(f"url: expected str, got {type(s).__name__}")
    parsed = urlparse(s)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"url: {s!r} missing scheme or netloc")


def _json_str(s: Any) -> None:
    if not isinstance(s, str):
        raise ValueError(f"json: expected str, got {type(s).__name__}")
    try:
        _json.loads(s)
    except _json.JSONDecodeError as e:
        raise ValueError(f"json: not parseable ({e})") from None


def _identifier(s: Any) -> None:
    if not isinstance(s, str):
        raise ValueError(f"identifier: expected str, got {type(s).__name__}")
    if not s.isidentifier() or _keyword.iskeyword(s):
        raise ValueError(f"identifier: {s!r} is not a valid Python identifier")


def _regex_compiles(s: Any) -> None:
    if not isinstance(s, str):
        raise ValueError(f"regex: expected str, got {type(s).__name__}")
    try:
        _re.compile(s)
    except _re.error as e:
        raise ValueError(f"regex: {s!r} does not compile ({e})") from None


def _email(s: Any) -> None:
    if not isinstance(s, str):
        raise ValueError(f"email: expected str, got {type(s).__name__}")
    # Basic shape — not a full RFC 5322 validator
    if not _re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", s):
        raise ValueError(f"email: {s!r} does not look like an email")


def _iso_date(s: Any) -> None:
    if not isinstance(s, str):
        raise ValueError(f"iso_date: expected str, got {type(s).__name__}")
    try:
        date.fromisoformat(s)
    except ValueError as e:
        raise ValueError(f"iso_date: {s!r} not ISO-8601 ({e})") from None


def _uuid(s: Any) -> None:
    if not isinstance(s, str):
        raise ValueError(f"uuid: expected str, got {type(s).__name__}")
    try:
        UUID(s)
    except ValueError as e:
        raise ValueError(f"uuid: {s!r} not a valid UUID ({e})") from None


def _path_exists(s: Any) -> None:
    if not isinstance(s, (str, os.PathLike)):
        raise ValueError(
            f"path_exists: expected str/os.PathLike, got {type(s).__name__}"
        )
    if not Path(s).exists():
        raise ValueError(f"path_exists: {s!r} does not exist")


# READ-ONLY: a MappingProxyType over the fixed built-ins. Reads use the dict fast path; there is no
# write surface, so the shared table is never mutated at runtime — no parallel()-branch race to gate
# and no runtime-registered kind to lose across a crash-restart. Custom checks go through predicate=fn.
REGISTRY: Mapping[str, Callable[[Any], None]] = MappingProxyType({
    "semver": _semver,
    "url": _url,
    "json": _json_str,
    "identifier": _identifier,
    "regex": _regex_compiles,
    "email": _email,
    "iso_date": _iso_date,
    "uuid": _uuid,
    "path_exists": _path_exists,
})
