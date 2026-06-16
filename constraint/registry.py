"""Built-in `kind=` constraint registry — common predicates PLM would
otherwise re-invent every call.

Each entry is a plain predicate callable: takes a value, returns nothing
on success, raises ValueError with a reason on failure.

PLM (or downstream code) can extend the registry at runtime:

    from plm.constraint import Constraint
    def _smiles_check(s):
        if rdkit.Chem.MolFromSmiles(s) is None:
            raise ValueError("not a valid SMILES string")
    Constraint.registry["smiles"] = _smiles_check
    Constraint.field(type=str, kind="smiles").validate("CCO")
"""

from __future__ import annotations

import json as _json
import keyword as _keyword
import os
import re as _re
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict
from urllib.parse import urlparse
from uuid import UUID

from plm._branch_state import _no_branch_mutation   # stdlib-only leaf; parallel-branch mutation gate


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


class _GatedRegistry(dict):
    """The `kind=` predicate registry, aliased as `Constraint.registry`. Reads use the C dict fast
    path (unchanged); every WRITE is refused inside a parallel() branch — registering a custom kind
    mutates a shared, process-global table that no single branch owns, so it is parent-only. The
    initial built-ins below load via `dict(...)` (C-level, bypasses these), not `__setitem__`."""

    def __setitem__(self, key, value):
        _no_branch_mutation("registering a Constraint kind (Constraint.registry write)")
        super().__setitem__(key, value)

    def __delitem__(self, key):
        _no_branch_mutation("removing a Constraint kind (Constraint.registry write)")
        super().__delitem__(key)

    def update(self, *a, **k):
        _no_branch_mutation("updating Constraint.registry")
        super().update(*a, **k)

    def setdefault(self, *a, **k):
        _no_branch_mutation("Constraint.registry.setdefault")
        return super().setdefault(*a, **k)

    def pop(self, *a, **k):
        _no_branch_mutation("Constraint.registry.pop")
        return super().pop(*a, **k)

    def popitem(self):
        _no_branch_mutation("Constraint.registry.popitem")
        return super().popitem()

    def clear(self):
        _no_branch_mutation("clearing Constraint.registry")
        super().clear()


REGISTRY: Dict[str, Callable[[Any], None]] = _GatedRegistry({
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
