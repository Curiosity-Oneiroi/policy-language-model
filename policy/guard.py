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

from .registry import _IMMUTABLE_POLICIES, _MISSING, _PLM_POLICIES


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
                tail = (
                    f"It is immutable; use `duplicate_policy({b!r}, '<new_name>')` to fork."
                    if b in immutable_names else
                    f"Use {b}._rewrite(...) to edit, `del {b}` to remove, or "
                    f"`@policy def {b}(...)` to redefine."
                )
                return f"Cell would rebind policy {b!r} via {type(s).__name__}. {tail}"
        for attr in ("body", "orelse", "finalbody"):
            sub = getattr(s, attr, None)
            if sub and (e := _audit_stmts(sub, names, immutable_names)):
                return e
        for h in getattr(s, "handlers", []):    # try/except handler bodies
            if e := _audit_stmts(h.body, names, immutable_names):
                return e
    return None


def _audit_cell(code, policy_names, immutable_names=()):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None                             # let compile() surface it
    return _audit_stmts(tree.body, policy_names, immutable_names)


def _post_cell_guard(cell_globals, stderr_buf):
    for name, canonical in list(_PLM_POLICIES.items()):
        binding = cell_globals.get(name, _MISSING)
        if binding is canonical:
            continue                            # OK
        if binding is _MISSING:
            if name in _IMMUTABLE_POLICIES:
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
            _PLM_POLICIES.pop(name, None)       # mutable del -> cleanup
            continue
        cell_globals[name] = canonical          # rebound -> restore + note
        tail = (
            f"It is immutable; use `duplicate_policy({name!r}, '<new_name>')` to fork."
            if name in _IMMUTABLE_POLICIES else
            f"Use {name}._rewrite(...) to edit or `del {name}`/{name}._remove() to remove."
        )
        stderr_buf.write(
            f"\n[policy guard] {name!r} was reassigned to "
            f"{type(binding).__name__}; reverted. {tail}\n"
        )
