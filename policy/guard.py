"""The policy guard — "edit only through the edit interface."

Three layered defenses (B, re-decoration=rewrite, lives in the decorator):

  A — AST pre-audit (`_audit_cell`): reject a cell BEFORE exec if a binding
      statement at enclosing-block scope targets a registered policy name. Skips
      function/class bodies (a `for predict in ...` inside a function binds a
      local, not the policy). Best-effort, clear, early.

      Phase 2 extension: also rejects `del <immutable>` (a separate `ast.Delete`
      branch — mutable `del` stays allowed, since it's the intended removal
      path for non-LLM-default policies).

  C — post-cell guard (`_post_cell_guard`): the AUTHORITATIVE catch-all. After
      every cell, restore any policy whose __main__ binding drifted from the
      canonical, and clean up `del`'d names. Catches what the static audit can't
      model (globals()[...]=x, setattr, `global predict; predict=x`, walrus,
      match/case captures).

      Phase 2 extension: for immutable policy names, RESTORE the __main__
      binding on `del`/pop (instead of clearing the registry entry). The
      registry side is already protected by `_sync`'s immutable-skip; this
      restores the global binding so subsequent cells see the policy again.

These run in plm.policy (a real module), so bare `type`/`set`/etc. resolve to
builtins and are immune to a __main__ policy shadowing them — only the kernel's
own __main__ code routes builtins through `_builtins`.
"""

from __future__ import annotations

import ast

from .registry import _SEALED_POLICIES, _MISSING, _PLM_POLICIES, _store_writable


def _names_in_target(target):
    """Yield bound Name ids in an assignment target, recursing tuples/lists and
    unwrapping Starred. Attribute (`a.b = x`) / Subscript (`a[b] = x`) bind no
    Name, so they yield nothing."""
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, ast.Starred):
        yield from _names_in_target(target.value)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            yield from _names_in_target(elt)


def _binding_targets(s):
    """Yield the names a single statement binds at its enclosing scope."""
    if isinstance(s, ast.Assign):
        for t in s.targets:
            yield from _names_in_target(t)
    elif isinstance(s, ast.AugAssign):
        yield from _names_in_target(s.target)
    elif isinstance(s, ast.AnnAssign):
        if s.value is not None:                 # `predict: T` (no value) binds nothing
            yield from _names_in_target(s.target)
    elif isinstance(s, (ast.For, ast.AsyncFor)):
        yield from _names_in_target(s.target)
    elif isinstance(s, (ast.With, ast.AsyncWith)):
        for item in s.items:
            if item.optional_vars is not None:
                yield from _names_in_target(item.optional_vars)
    elif isinstance(s, (ast.Import, ast.ImportFrom)):
        for alias in s.names:
            if alias.asname:
                yield alias.asname
            else:                               # `import a.b` binds the top name `a`
                yield alias.name.split(".", 1)[0]
    # ast.Delete is NOT a binding -> not flagged by _binding_targets (it's the
    # intended removal path for mutables). `_audit_stmts` handles Delete in a
    # separate branch, gated on `immutable_names`.


def _audit_stmts(stmts, names, immutable_names=()):
    for s in stmts:
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue                            # own scope — skip (re-decoration allowed)
        # `del <immutable>` at enclosing-block scope -> reject. `del <mutable>`
        # stays allowed (it's the intended removal path).
        if isinstance(s, ast.Delete):
            for t in s.targets:
                for b in _names_in_target(t):
                    if b in immutable_names:
                        return (f"Cell would `del` immutable policy {b!r}. "
                                f"It is part of the LLM-default base. "
                                f"Use `duplicate_policy({b!r}, '<new_name>')` "
                                f"to fork it.")
            continue                            # nothing else to check on Delete
        for b in _binding_targets(s):
            if b in names:
                return _rebind_msg(b, type(s).__name__, immutable_names)
        for attr in ("body", "orelse", "finalbody"):
            sub = getattr(s, attr, None)
            if sub and (e := _audit_stmts(sub, names, immutable_names)):
                return e
        for h in getattr(s, "handlers", []):    # try/except handler bodies
            if e := _audit_stmts(h.body, names, immutable_names):
                return e
        for c in getattr(s, "cases", []):       # match/case bodies (ast.Match.cases[*].body)
            if e := _audit_stmts(c.body, names, immutable_names):
                return e
    return None


def _rebind_msg(name, form, immutable_names):
    """Friendly Guard-A message for a cell that would rebind a registered policy name."""
    tail = (
        f"It is immutable; use `duplicate_policy({name!r}, '<new_name>')` to fork."
        if name in immutable_names else
        f"Use {name}._rewrite(...) to edit, `del {name}` to remove, or "
        f"`@policy def {name}(...)` to redefine."
    )
    return f"Cell would rebind policy {name!r} via {form}. {tail}"


def _names_in_pattern(p):
    """Yield names a match-case PATTERN binds: MatchAs (`case ... as x` / bare capture
    `case x`), MatchStar (`*x`), MatchMapping (`**rest`); recurses nested patterns."""
    if isinstance(p, ast.MatchAs):
        if p.name:
            yield p.name
        if p.pattern is not None:
            yield from _names_in_pattern(p.pattern)
    elif isinstance(p, ast.MatchStar):
        if p.name:
            yield p.name
    elif isinstance(p, ast.MatchMapping):
        if p.rest:
            yield p.rest
        for sub in p.patterns:
            yield from _names_in_pattern(sub)
    elif isinstance(p, (ast.MatchSequence, ast.MatchOr)):
        for sub in p.patterns:
            yield from _names_in_pattern(sub)
    elif isinstance(p, ast.MatchClass):
        for sub in [*p.patterns, *p.kwd_patterns]:
            yield from _names_in_pattern(sub)


def _enclosing_scope_nodes(child):
    """The subtrees of a nested def/lambda/class that evaluate in the ENCLOSING scope (NOT the
    body): default values, decorators, base classes, class keywords, and every annotation. A
    binding form here (e.g. `def f(x=(predict := 5)): ...`) rebinds an OUTER name, so it must
    still be audited even though the body is a separate scope."""
    out = []
    args = getattr(child, "args", None)
    if args is not None:
        out += [d for d in getattr(args, "defaults", []) if d is not None]
        out += [d for d in getattr(args, "kw_defaults", []) if d is not None]
        for a in (list(getattr(args, "posonlyargs", [])) + list(getattr(args, "args", []))
                  + list(getattr(args, "kwonlyargs", []))
                  + [getattr(args, "vararg", None), getattr(args, "kwarg", None)]):
            if a is not None and getattr(a, "annotation", None) is not None:
                out.append(a.annotation)
    out += list(getattr(child, "decorator_list", []))
    out += list(getattr(child, "bases", []))
    out += list(getattr(child, "keywords", []))
    if getattr(child, "returns", None) is not None:
        out.append(child.returns)
    return out


def _audit_extras(node, names, immutable_names):
    """Reject the enclosing-scope binding forms `_binding_targets` misses — walrus
    (`x := ...`), `except ... as x`, match captures, and 3.12 `type x = ...` — that
    would rebind a registered policy name. Guard C reverts them post-cell regardless;
    this is the friendly FAIL-LOUD before the cell runs. A nested function / class / lambda's
    BODY is a separate scope (skipped), but its SIGNATURE positions evaluate here, so they are
    substituted in and audited too."""
    _TypeAlias = getattr(ast, "TypeAlias", ())
    _SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
    if isinstance(node, _SCOPES):
        # THIS node is itself a nested scope — e.g. a lambda handed in as a default value via
        # `_enclosing_scope_nodes` (the signature of an OUTER def). Audit ONLY its signature: its BODY
        # is a separate scope, so a binding there (a walrus / `except as` / match capture) is LOCAL and
        # is NOT a policy rebind. Without this, recursing into such a lambda walked its body and
        # false-positived on a lambda-local walrus (e.g. `def f(cb=lambda: (predict := 5)): ...`).
        children = list(_enclosing_scope_nodes(node))
    else:
        children = []
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _SCOPES):
                children.extend(_enclosing_scope_nodes(child))   # body skipped; signature audited
            else:
                children.append(child)
    for child in children:
        if isinstance(child, ast.NamedExpr) and isinstance(child.target, ast.Name):
            if child.target.id in names:
                return _rebind_msg(child.target.id, "walrus `:=`", immutable_names)
        elif isinstance(child, ast.ExceptHandler) and child.name in names:
            return _rebind_msg(child.name, "`except ... as`", immutable_names)
        elif isinstance(child, _TypeAlias) and isinstance(getattr(child, "name", None), ast.Name):
            if child.name.id in names:
                return _rebind_msg(child.name.id, "`type` alias", immutable_names)
        elif isinstance(child, ast.match_case):
            for nm in _names_in_pattern(child.pattern):
                if nm in names:
                    return _rebind_msg(nm, "match capture", immutable_names)
        if e := _audit_extras(child, names, immutable_names):
            return e
    return None


def _audit_cell(code, policy_names, immutable_names=()):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None                             # let compile() surface it
    return (_audit_stmts(tree.body, policy_names, immutable_names)
            or _audit_extras(tree, policy_names, immutable_names))


def _post_cell_guard(cell_globals, stderr_buf):
    for name, canonical in list(_PLM_POLICIES.items()):
        binding = cell_globals.get(name, _MISSING)
        if binding is canonical:
            continue                            # OK
        if binding is _MISSING:
            if name in _SEALED_POLICIES:
                # Immutable policy was dynamically del'd / popped from __main__.
                # Restore the binding from the registry (which `_sync` never
                # evicted for immutable names).
                cell_globals[name] = canonical
                stderr_buf.write(
                    f"\n[policy guard] {name!r} is immutable; deletion reverted "
                    f"(restored from the registry). Use "
                    f"`duplicate_policy({name!r}, '<new_name>')` to fork it.\n"
                )
                continue
            with _store_writable():             # Guard C cleanup -> authorize the registry write
                _PLM_POLICIES.pop(name, None)   # mutable del -> cleanup
            continue
        cell_globals[name] = canonical          # rebound -> restore + note
        tail = (
            f"It is immutable; use `duplicate_policy({name!r}, '<new_name>')` to fork."
            if name in _SEALED_POLICIES else
            f"Use {name}._rewrite(...) to edit or `del {name}`/{name}._remove() to remove."
        )
        stderr_buf.write(
            f"\n[policy guard] {name!r} was reassigned to "
            f"{type(binding).__name__}; reverted. {tail}\n"
        )
