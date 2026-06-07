"""Kernel bootstrap + main loop (text constants exec'd in the child).

Assembled by `session._assemble_kernel_script` as:

    <KERNEL_BOOTSTRAP>
    try:
        <prefix + KERNEL_LOOP>
    except BaseException:
        _repl_write_frame({"type": "boot_error", "traceback": ...})
"""

from __future__ import annotations


KERNEL_BOOTSTRAP = r'''
import builtins as _builtins
import os as _repl_os, struct as _repl_struct, pickle as _repl_pickle, socket as _repl_socket

# Parent-death detector (Linux): if the parent dies abruptly, deliver SIGTERM so this
# kernel doesn't linger ORPHANED mid-cell — start_new_session put us in our own session,
# so we wouldn't otherwise be reaped with the parent. Best-effort. (S-F16)
try:
    import ctypes as _repl_ctypes, signal as _repl_signal
    _repl_ctypes.CDLL(None).prctl(1, _repl_signal.SIGTERM)   # PR_SET_PDEATHSIG=1
except Exception:
    pass

_repl_sock = _repl_socket.socket(_repl_socket.AF_UNIX, _repl_socket.SOCK_STREAM)
_repl_sock.connect(_repl_os.environ["_REPL_SOCK_PATH"])
_repl_sock_io = _repl_sock.makefile("rwb", buffering=0)
_repl_req_in  = _repl_sock_io
_repl_resp_out = _repl_sock_io


def _repl_read_frame():
    hdr = b""
    while _builtins.len(hdr) < 4:                       # unbuffered SocketIO.read() may return
        _chunk = _repl_req_in.read(4 - _builtins.len(hdr))   # 1-3 bytes for the 4-byte header
        if not _chunk:                                  # under fragmentation; loop until full
            raise EOFError("parent closed request socket")   # (only b'' is a genuine close)
        hdr += _chunk
    (_n,) = _repl_struct.unpack(">I", hdr)
    buf = b""
    while _builtins.len(buf) < _n:
        chunk = _repl_req_in.read(_n - _builtins.len(buf))
        if not chunk:
            raise EOFError("short frame body")
        buf += chunk
    return _repl_pickle.loads(buf)


def _repl_write_frame(obj):
    body = _repl_pickle.dumps(obj)
    _repl_resp_out.write(_repl_struct.pack(">I", _builtins.len(body)))
    _repl_resp_out.write(body)


def _repl_reset_buffers():
    global _repl_stdout_buf, _repl_stderr_buf
    _repl_stdout_buf = _io.StringIO()
    _repl_stderr_buf = _io.StringIO()
    _sys.stdout = _repl_stdout_buf
    _sys.stderr = _repl_stderr_buf
'''


KERNEL_LOOP = r'''
# `boot_stderr`: soft EXTRA-policy install tracebacks captured during PREFIX (a
# failed DEFAULT is a hard boot_error and never reaches here). The parent
# surfaces it once on the next cell so it isn't lost to the buffer reset (#26).
_repl_write_frame({"type": "ready", "boot_stderr": _repl_stderr_buf.getvalue()})

# Kernel-side cell counter → unique <cell-N> linecache filenames, with no
# dependence on the parent and no "<cell>" fallback that could collapse cells
# onto one slot. Reset to 0 on respawn (safe: the dead kernel's <cell-*> are
# gone; policies live under <policy-*>, rebuilt from _p_source).
_repl_cell_n = 0

# EVERY loop variable is _repl_-prefixed (the loop IS __main__ top-level code,
# and cells exec in __main__) so it can't collide with a user/policy var or be
# captured into the snapshot. ALL builtins go through _builtins so a cell binding
# set/list/compile/type/len/globals in __main__ can't break the loop/audit.
while True:
    try:
        _repl_req = _repl_read_frame()
    except EOFError:
        break
    except KeyboardInterrupt:
        # A late SIGINT can land HERE, on the idle top-level read: the parent
        # sends SIGINT on a clean-boundary cell timeout, but if the cell's exec
        # had ALREADY finished and written its result frame (which the parent's
        # grace-read then consumed), the signal arrives while we're blocked
        # waiting for the NEXT frame. Swallow it and re-enter the read — letting
        # it propagate would hit the outer boot_error wrapper, poison the stream
        # with a spurious boot_error, kill the kernel, and silently drop the next
        # cell. The cell that the SIGINT was meant to interrupt already completed
        # and reported, so there is nothing to abort. (#R6-6)
        continue

    _repl_etype = _repl_req.get("type")

    if _repl_etype == "shutdown":
        break

    if _repl_etype == "rehydrate":
        _repl_rehydrate_error = None
        try:
            _repl_blob = _repl_req.get("vars_blob") or b""
            # Single-blob: dill memoization preserves identity across all refs.
            # blob may be b"" (a collect that bailed) -> nothing to restore, and
            # _dill.loads(b"") would raise.
            _repl_restored = _dill.loads(_repl_blob) if _repl_blob else {}

            # Pop the source-fallback registry (policies that couldn't snapshot by
            # value — e.g. class policies) BEFORE globals().update so it never leaks
            # into __main__; it's re-installed from source after the reconcile (#H11).
            _repl_pol_sources = _repl_restored.pop("_PLM_POLICY_SOURCES", None) or {}
            # CALL-built constraints (Constraint.field / & | ^ ~) that couldn't dill-pickle
            # were snapshotted as RECIPES — popped here so they don't reach globals().update;
            # replayed back into __main__ after the reconcile (mirrors the policy-source path).
            _repl_constraint_recipes = _repl_restored.pop("_CONSTRAINT_RECIPES", None) or {}

            # === Phase 2: subscript-poisoning closure ===
            # Discard the snapshot's copies of immutable LLM defaults and keep
            # PREFIX's freshly-installed v0. Guard A skips `ast.Subscript` targets,
            # so a previous-session cell could have written
            # `_PLM_POLICIES["natural_llm"] = evil_proxy` (the dict is in
            # __main__ via `from plm.policy import (...)`); Guard C treats the
            # registry as canonical, so the poisoned proxy round-trips through
            # the snapshot. We keep the existing "snapshot is canonical for
            # mutables" semantic (PLM-deleted extras stay deleted) BUT force the
            # LLM defaults to come from on-disk source.
            from plm.policy.defaults import UNDUPLICABLE_DEFAULTS as _repl_undup_defs
            from plm.policy.defaults import _bless_llm_callers as _repl_bless_after
            from plm.policy.registry import _unsealed as _repl_unsealed

            # Snapshot the FRESH-PREFIX boot policies (name -> proxy) BEFORE the
            # registry reset below, so orphaned __main__ bindings can be reaped
            # after reconciliation (see the drop loop further down, #R5-2).
            _repl_preboot = {
                _repl_pn: _PLM_POLICIES[_repl_pn]
                for _repl_pn in _builtins.list(_PLM_POLICIES)
            }

            if "_PLM_POLICIES" in _repl_restored:
                _repl_saved_defaults = {
                    n: _PLM_POLICIES[n] for n in _repl_undup_defs if n in _PLM_POLICIES
                }
                # The registry protects default entries during normal cell exec;
                # this is a harness-owned full reset, so drop the seal for it. The
                # protected store would also reject the snapshot's poisoned default
                # copies, but `_unsealed` keeps this path explicit + force-restores
                # the freshly-installed v0 defaults regardless of what the snapshot
                # carried.
                with _repl_unsealed():
                    _PLM_POLICIES.clear()
                    _PLM_POLICIES.update(_repl_restored.pop("_PLM_POLICIES"))
                    _PLM_POLICIES.update(_repl_saved_defaults)  # force-overwrite back to v0

            _builtins.globals().update(_repl_restored)

            # Force-rebind LLM-default names in __main__ to the registry's v0
            # proxies. REQUIRED because the snapshot contains
            # `natural_llm`/`react_llm` as __main__ globals (they're NOT in
            # _REPL_INJECTED), so globals().update(...) above would otherwise
            # overwrite PREFIX's v0 binding with the snapshot's OLD/poisoned
            # proxy — leaving _PLM_POLICIES (v0) inconsistent with __main__
            # (OLD). All `natural_llm(...)` calls from cells look up
            # __main__["natural_llm"], so this rebind is what guarantees PLM
            # hits v0. (Known limitation, documented in the plan: a cell-level
            # alias like `f = natural_llm` rehydrates pointing at the OLD
            # proxy via dill identity-preservation; PLM should look up via
            # the global at call time, not cache the reference.)
            _repl_g = _builtins.globals()
            for _repl_dn in _repl_undup_defs:
                if _repl_dn in _PLM_POLICIES:
                    _repl_g[_repl_dn] = _PLM_POLICIES[_repl_dn]

            # Reap orphaned __main__ bindings: a boot policy the PRIOR session
            # DELETED is absent from the snapshot's registry, but THIS kernel's
            # fresh PREFIX re-installed it into __main__ before this rehydrate. The
            # registry reconcile above dropped it from _PLM_POLICIES, yet
            # globals().update() can't remove a name the snapshot simply lacks — so
            # without this it lingers in __main__ as a live orphan proxy (callable,
            # but invisible to list/get/read/edit_policy and Guard C), contradicting
            # the "PLM-deleted policies stay deleted" contract. Drop any preboot name
            # now gone from the registry — but ONLY if __main__ still holds the exact
            # fresh-PREFIX proxy object: if the snapshot re-bound that name to a user
            # value (e.g. a cell did `base_verifier = 42` after deleting the policy),
            # globals().update() restored the user's value and the identity check
            # leaves it untouched (no clobbering a legitimate cell variable). (#R5-2)
            for _repl_pn, _repl_proxy in _repl_preboot.items():
                if _repl_pn not in _PLM_POLICIES and _repl_g.get(_repl_pn) is _repl_proxy:
                    _repl_g.pop(_repl_pn, None)

            # Re-install policies that could NOT be snapshotted by value (a class
            # policy: its depth-cap __call__ wrap closes over the _POLICY_CALL_DEPTH
            # ContextVar, so dill pickles the __main__ class by-reference —
            # unresolvable on the fresh respawn). The snapshot carried their
            # `_p_source` instead; re-exec `@policy` on it via the normal install
            # path so an AUTHORED class policy SURVIVES a hard respawn rather than
            # being silently lost (#H11/#H12). Skip a name a by-value entry already
            # restored; best-effort per source so one broken body can't abort the
            # whole rehydrate.
            if _repl_pol_sources:
                from plm.policy.registry import _install_policy_source as _repl_install_src
                for _repl_pn, _repl_src in _repl_pol_sources.items():
                    if _repl_pn in _PLM_POLICIES:
                        continue
                    try:
                        _repl_install_src(
                            "@policy\n" + _repl_src, "<policy-rehydrate-" + _repl_pn + ">")
                    except Exception as _repl_reinstall_exc:
                        # Surface, don't swallow (K-F5): an authored class policy whose
                        # body fails to re-exec on respawn must NOT vanish silently.
                        _repl_rehydrate_error = (_repl_rehydrate_error or "") + (
                            "[rehydrate] policy " + _repl_pn + " not replayed: "
                            + _builtins.repr(_repl_reinstall_exc) + "; ")

            # Replay constraint RECIPES (CALL-built constraints that couldn't dill-pickle)
            # back into __main__, rebuilding via from_recipe. Best-effort + surfaced like
            # the policy-source replay above; guarded on the constraint surface.
            if _repl_constraint_recipes and _repl_constraint_from_recipe is not None:
                _repl_cg = _builtins.globals()
                for _repl_cn, _repl_crec in _repl_constraint_recipes.items():
                    try:
                        _repl_cg[_repl_cn] = _repl_constraint_from_recipe(_repl_crec)
                    except Exception as _repl_crec_exc:
                        _repl_rehydrate_error = (_repl_rehydrate_error or "") + (
                            "[rehydrate] constraint " + _repl_cn + " not replayed: "
                            + _builtins.repr(_repl_crec_exc) + "; ")

            _rebuild_linecache_from_policies()

            # Refresh `_BLESSED_CALLERS` against the current registry's code
            # objects. Defensively uniform: for the LLM defaults we just
            # kept-from-PREFIX it's effectively a no-op (same code objects
            # PREFIX already blessed), but any future immutable default that
            # legitimately rehydrates would get correctly blessed here.
            _repl_bless_after()
        except Exception as _repl_rehydrate_exc:
            _repl_rehydrate_error = (_repl_rehydrate_error or "") + _builtins.repr(_repl_rehydrate_exc)
        _repl_write_frame({"type": "ready", "rehydrate_error": _repl_rehydrate_error})
        continue

    if _repl_etype != "code":
        _repl_write_frame({
            "type": "result", "vars_blob": b"", "stdout": "", "stderr": ""
        })
        continue

    _repl_code = _repl_req["code"]
    _repl_cell_file = f"<cell-{_repl_cell_n}>"
    _repl_cell_n += 1
    _repl_reset_buffers()
    _linecache.cache[_repl_cell_file] = (
        _builtins.len(_repl_code), None, _repl_code.splitlines(True), _repl_cell_file)

    _repl_g = _builtins.globals()

    # General per-round injection channel: bind each {name: object} in `seed`
    # into __main__ before exec. Frames without a seed carry None -> nothing
    # bound (existing names untouched).
    for _repl_sname, _repl_sobj in (_repl_req.get("seed") or {}).items():
        if _repl_sname == "plm_messages":
            _repl_plm_messages[:] = _repl_sobj          # seeded trajectory REPLACES the accumulator (its source of truth)
        else:
            _repl_g[_repl_sname] = _repl_sobj

    # `plm_messages` (the root's own trajectory) arrives ADDITIVELY: the parent
    # appends only the messages new since the last cell. We extend a hidden
    # accumulator and rebind the PUBLIC `plm_messages` to a fresh shallow copy of
    # it, so a cell mutating `plm_messages` (append/clear/reassign) can't corrupt
    # the accumulation. After a respawn the parent resends the full trajectory
    # (the accumulator is empty here), so this stays correct across crash-restart.
    _repl_pm_delta = _repl_req.get("plm_messages_delta")
    if _repl_pm_delta is not None:
        _repl_plm_messages.extend(_repl_pm_delta)
    # Reseed the PUBLIC list from the hidden accumulator EVERY cell. The accumulator is
    # the single source of truth — fed by the delta channel (additive) and by a
    # `plm_messages` seed above (replace) — so a seeded/delta'd trajectory persists to
    # later no-input cells, while a cell mutating the public list (append/clear/reassign)
    # can't corrupt the accumulation OR persist a stray rebind into the next cell (K-F7).
    _repl_g["plm_messages"] = _builtins.list(_repl_plm_messages)

    # Guard A: reject a static rebind of a registered policy name BEFORE exec.
    # Pass the immutable-names subset so the audit can produce immutable-specific
    # rejection messages AND can reject `del <immutable>` (a separate AST branch
    # for ast.Delete; mutable `del` stays allowed).
    #
    # Re-import the guard helpers + registry + builtins FRESH from their defining
    # modules EVERY iteration as `_repl_`-prefixed aliases, and call ONLY those.
    # A cell execs in __main__, so the bare PREFIX-injected names (`_audit_cell`,
    # `_post_cell_guard`, `_PLM_POLICIES`, even `_builtins`) are plain __main__
    # globals a cell can rebind — e.g. `_audit_cell = lambda *a, **k: None` (which
    # Guard A does NOT flag: it only rejects POLICY-name rebinds), or a fake
    # `_builtins` whose `.set` returns an empty set to fool the audit. Either path
    # SILENTLY disables Guard A/C + the immutability seal for the rest of the
    # session. Re-importing each iteration overwrites any such rebind BEFORE the
    # guard runs, so the loop never calls a cell-supplied guard or feeds it a
    # cell-supplied builtin. (Reassigning the MODULE attribute —
    # `plm.policy._audit_cell = ...` — or poisoning sys.modules remains the
    # documented irreducible Python-no-privacy boundary, like
    # `_PLM_POLICIES.__setitem__` and `_llm_infra._BLESSED_CALLERS` reassignment.)
    # Mirrors the existing per-iteration `_IMMUTABLE_POLICIES` re-import. (#R5-1)
    import builtins as _repl_builtins
    from plm.policy import (
        _audit_cell as _repl_audit_cell,
        _PLM_POLICIES as _repl_plm_store,
    )
    from plm.policy.registry import _IMMUTABLE_POLICIES as _repl_imm
    _repl_immset_for_audit = _repl_builtins.set(_repl_imm)
    _repl_audit_err = _repl_audit_cell(
        _repl_code, _repl_builtins.set(_repl_plm_store), _repl_immset_for_audit
    )
    if _repl_audit_err is not None:
        _repl_stderr_buf.write(_repl_audit_err + "\n")
        _repl_write_frame({
            "type": "result", "vars_blob": _repl_collect_vars(),
            "stdout": "", "stderr": _repl_stderr_buf.getvalue(),
        })
        continue

    _repl_terminal_etype = None
    _repl_return_value = None

    # Cell stdout is redirected to `_repl_stdout_buf` (reset per cell at boot), so prints
    # emitted BEFORE an error are PRESERVED: they stay in that buffer while the traceback
    # below goes to stderr, and the result frame ships BOTH. A print-then-error cell
    # surfaces its output AND the traceback — the same guarantee exec_ns gives (which was
    # modeled on this loop).
    try:
        _builtins.exec(_builtins.compile(_repl_code, _repl_cell_file, "exec"), _repl_g, _repl_g)
    except BaseException as _repl_exc:
        # Classify the terminal with a FRESHLY re-imported builtins, not the bare
        # `_builtins` the cell just had a chance to rebind: a fake `_builtins` whose
        # `.type`/`.getattr` lie could otherwise mask the cell's own exception as a
        # silent RETURN (or hide its traceback). Guard C must still run below. (#H10)
        import builtins as _repl_builtins
        _repl_exc_name = _repl_builtins.type(_repl_exc).__name__
        if _repl_exc_name == "_REPLReturn":
            _repl_terminal_etype = "return"
            _repl_return_value = _repl_builtins.getattr(_repl_exc, "value", None)
        else:
            # Write the traceback STRAIGHT to the buffer (a cell rebinding `sys.stderr`
            # defeats the per-cell redirect, so `print_exc()` could vanish), dropping
            # THIS exec frame so the model sees only its own source. (K-F1/K-F6)
            _repl_tb_obj = _repl_exc.__traceback__
            _repl_stderr_buf.write("".join(_traceback.format_exception(
                _repl_builtins.type(_repl_exc), _repl_exc,
                _repl_tb_obj.tb_next if _repl_tb_obj else None)))

    # Re-establish a TRUSTED handle to the REAL __main__ globals before Guard C.
    # A cell can rebind the bare `_repl_g` (or `_builtins`) __main__ name during its
    # exec (e.g. `_repl_g = {}`); trusting the pre-exec `_repl_g` would make Guard C
    # "restore" immutable defaults into a DECOY dict, leaving the real __main__
    # hijack live for the next cell (#H9). `builtins.globals()` is frame-derived —
    # it returns THIS loop's real __main__ dict regardless of what the `_repl_g` key
    # was rebound to — and the fresh `import` defeats a rebound `_repl_builtins`.
    import builtins as _repl_builtins
    _repl_g = _repl_builtins.globals()
    # Guard C: restore any policy whose __main__ binding drifted; clean up del'd.
    # Re-import FRESH (see the Guard A note): the cell may have rebound the bare
    # `_post_cell_guard` __main__ global, so the fresh module alias is what we call.
    from plm.policy import _post_cell_guard as _repl_post_cell_guard
    _repl_post_cell_guard(_repl_g, _repl_stderr_buf)

    # Guard C+: the same protection for the non-policy INJECTED helpers (callables:
    # policy ops, parallel, exec_ns, the constraint surface). Restore any the cell
    # rebound or deleted so a granted helper can't be clobbered for the next cell.
    # (Uses the trusted `_repl_g` established above; `_REPL_INJECTED_CANON` holds the
    # boot-canonical callables — never `plm_messages`/registries, so live state is safe.)
    _repl_reverted = [
        _repl_k for _repl_k, _repl_v in _REPL_INJECTED_CANON.items()
        if _repl_g.get(_repl_k) is not _repl_v
    ]
    for _repl_k in _repl_reverted:
        _repl_g[_repl_k] = _REPL_INJECTED_CANON[_repl_k]
    if _repl_reverted:
        _repl_stderr_buf.write(
            "\n[repl guard] kernel-injected name(s) reverted: "
            + ", ".join(sorted(_repl_reverted))
            + " — granted helpers; rebinding them is reverted each cell.\n"
        )

    if _repl_terminal_etype == "return":
        try:
            _repl_return_blob = _dill.dumps(_repl_return_value)
        except Exception as _repl_pickle_err:
            _repl_return_blob = None
            _repl_stderr_buf.write(
                "RETURN: could not serialize return value: "
                + _builtins.repr(_repl_pickle_err) + "\n"
            )
        _repl_write_frame({
            "type":        "return",
            "return_blob": _repl_return_blob,
            "vars_blob":   _repl_collect_vars(),
            "stdout":      _repl_stdout_buf.getvalue(),
            "stderr":      _repl_stderr_buf.getvalue(),
        })
    else:
        _repl_write_frame({
            "type":      "result",
            "vars_blob": _repl_collect_vars(),
            "stdout":    _repl_stdout_buf.getvalue(),
            "stderr":    _repl_stderr_buf.getvalue(),
        })
'''
