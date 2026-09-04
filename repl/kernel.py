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
import copy as _repl_copy
import os as _repl_os, struct as _repl_struct, pickle as _repl_pickle, socket as _repl_socket

# Parent-death detector (Linux): if the parent dies abruptly, deliver SIGTERM so this
# kernel doesn't linger ORPHANED mid-cell — start_new_session put us in our own session,
# so we wouldn't otherwise be reaped with the parent. Best-effort.
try:
    import ctypes as _repl_ctypes, signal as _repl_signal
    _repl_ctypes.CDLL(None).prctl(1, _repl_signal.SIGTERM)   # PR_SET_PDEATHSIG=1
except Exception:
    pass

# SIGINT policy — ONE rule for the whole kernel. The parent sends SIGINT for exactly one purpose:
# abort a cell BODY that's running too long (a clean-boundary cell timeout). So SIGINT is IGNORED
# kernel-wide by default, and ARMED (-> KeyboardInterrupt) ONLY around the cell exec in KERNEL_LOOP.
# A late / mis-timed SIGINT — landing in setup, the post-exec guards, the snapshot/finalize, the
# idle read, rehydrate, or boot — is then a no-op, instead of escaping into a spurious `boot_error`
# / false "did NOT execute" / torn result frame. (Robust import: the prctl block above swallows an
# ImportError, which could leave its `signal` alias unbound — bind our own here.)
import signal as _repl_signal_mod
_repl_signal_mod.signal(_repl_signal_mod.SIGINT, _repl_signal_mod.SIG_IGN)

# TRUSTED signal handles, captured at boot BEFORE any cell runs — immune to a later
# `sys.modules['signal']` poison (the per-cell `import signal` would return the fake). Arm/disarm
# go through these, each SWALLOWING any failure: a thrown arm/disarm must NEVER escape and turn an
# executed cell into a spurious boot_error / false "did NOT execute" (M4).
_REPL_SIG_SET = _repl_signal_mod.signal
_REPL_SIGINT = _repl_signal_mod.SIGINT
_REPL_SIG_IGN = _repl_signal_mod.SIG_IGN
_REPL_SIG_DFL = _repl_signal_mod.default_int_handler


def _repl_arm_sigint():
    try:
        _REPL_SIG_SET(_REPL_SIGINT, _REPL_SIG_DFL)
    except Exception:
        pass                                              # arm failed -> body just isn't SIGINT-abortable


def _repl_disarm_sigint():
    try:
        _REPL_SIG_SET(_REPL_SIGINT, _REPL_SIG_IGN)
    except Exception:
        pass                                              # disarm failed -> never let it escape the cell

_repl_sock = _repl_socket.socket(_repl_socket.AF_UNIX, _repl_socket.SOCK_STREAM)
_repl_sock.connect(_repl_os.environ["_REPL_SOCK_PATH"])
_repl_sock_io = _repl_sock.makefile("rwb", buffering=0)
_repl_req_in  = _repl_sock_io
_repl_resp_out = _repl_sock_io

# a 4-byte length header names up to ~4 GiB; on a desync/corruption that would drive an
# unbounded read. Real frames are tiny, so a length past this cap is a corrupt header -> bail
# (the parent then respawns). MUST match session.py's _MAX_FRAME_BYTES.
_REPL_MAX_FRAME = 2 * 1024 ** 3


def _repl_read_frame():
    hdr = b""
    while _builtins.len(hdr) < 4:                       # unbuffered SocketIO.read() may return
        _chunk = _repl_req_in.read(4 - _builtins.len(hdr))   # 1-3 bytes for the 4-byte header
        if not _chunk:                                  # under fragmentation; loop until full
            raise EOFError("parent closed request socket")   # (only b'' is a genuine close)
        hdr += _chunk
    (_n,) = _repl_struct.unpack(">I", hdr)
    if _n > _REPL_MAX_FRAME:                            # corrupt/desynced length -> bail (parent respawns)
        raise EOFError(f"frame length {_n} exceeds cap {_REPL_MAX_FRAME} (corrupt header)")
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


def _repl_clip(_s, _n=240):
    # Bound ONE diagnostic piece (an exception repr, a name list) so a single huge object can't eat
    # the whole rehydrate-note budget — every failure then gets a fair, short mention.
    _s = _builtins.str(_s)
    return _s if _builtins.len(_s) <= _n else (_s[:_n] + "…[+" + _builtins.str(_builtins.len(_s) - _n) + " chars]")


def _repl_reset_buffers():
    global _repl_stdout_buf, _repl_stderr_buf
    _repl_stdout_buf = _io.StringIO()
    _repl_stderr_buf = _io.StringIO()
    # Re-derive the routing proxies + set_main_streams FRESH from the trusted module each cell — NOT
    # from the `_repl_`-prefixed __main__ aliases PREFIX bound. Those aliases are excluded from the
    # Guard C+ canon (`_REPL_INJECTED` filters `_repl`/`_REPL`), so a cell rebinding one (e.g.
    # `_repl_out_proxy = 0`) would otherwise propagate here and PERMANENTLY brick stdout/stderr
    # capture — unlike `_repl_stdout_buf`, which is recreated each cell and self-heals. A fresh
    # import resolves through sys.modules, ignoring any cell rebind (the R5-1 pattern the loop uses
    # to re-import plm.repl._kernel_state). `_repl_bs` is a function-local, not a __main__ name, so
    # there is nothing for a cell to clobber. The kernel still READS `_repl_stdout_buf` for output.
    import plm._branch_state as _repl_bs
    _sys.stdout = _repl_bs._OUT_PROXY
    _sys.stderr = _repl_bs._ERR_PROXY
    _repl_bs.set_main_streams(_repl_stdout_buf, _repl_stderr_buf)
'''


KERNEL_LOOP = r'''
# `boot_stderr`: soft EXTRA-policy install tracebacks captured during PREFIX (a
# failed DEFAULT is a hard boot_error and never reaches here). The parent
# surfaces it once on the next cell so it isn't lost to the buffer reset.
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
        # DEFENSIVE BACKSTOP — under the normal invariant this cannot fire. SIGINT is armed
        # (-> KeyboardInterrupt) ONLY around the cell-body exec in KERNEL_LOOP and disarmed the
        # instant it returns; everywhere else, including THIS idle top-level read, SIGINT is
        # SIG_IGN'd kernel-wide (see KERNEL_BOOTSTRAP), so a stray/late timeout SIGINT here is
        # simply ignored and never raises. This handler is kept only for the off-nominal case
        # where an arm/disarm somehow left SIGINT armed across the idle read: swallow + re-enter
        # rather than let it propagate to the outer boot_error wrapper (which would poison the
        # stream, kill the kernel, and silently drop the next cell). The cell a stray SIGINT was
        # meant for has already completed and reported, so there is nothing to abort.
        continue

    _repl_etype = _repl_req.get("type")

    if _repl_etype == "shutdown":
        break

    if _repl_etype == "rehydrate":
        _repl_rehydrate_error = None
        try:
            _repl_blob = _repl_req.get("vars_blob") or b""
            # HYBRID restore. The snapshot is a wrapper: a COMBINED blob (fast path — dill memoization
            # preserves cross-object identity, e.g. a var aliasing a policy restores to the SAME object)
            # PLUS per-object blobs (recovery). The wrapper holds only bytes/str/list, so it always
            # loads. Try the combined blob; if ONE un-loadable object would sink the WHOLE restore (a
            # by-reference __main__ class, a foreign model, an un-rebuildable recipe), fall back to
            # loading each object on its own — drop only the failures, keep the rest. `_repl_restored`
            # ends up _snap-shaped either way, so the apply logic below is unchanged.
            _repl_wrapper = _dill.loads(_repl_blob) if _repl_blob else {}
            _repl_dropped_load = []
            if _builtins.isinstance(_repl_wrapper, dict) and "_combined" in _repl_wrapper:
                try:
                    _repl_restored = _dill.loads(_repl_wrapper.get("_combined") or b"")
                except BaseException as _repl_combined_exc:
                    # Combined blob poisoned by some un-loadable object -> recover each object alone.
                    _repl_rehydrate_error = (_repl_rehydrate_error or "") + (
                        "[rehydrate] combined snapshot could not load ("
                        + _repl_clip(_builtins.repr(_repl_combined_exc))
                        + "); recovering objects individually. ")
                    _repl_restored = {}
                    _repl_reg2 = {}
                    _repl_rec2 = {}
                    _repl_psrc2 = _repl_wrapper.get("policy_sources") or {}
                    for _repl_vn, _repl_vb in (_repl_wrapper.get("vars") or {}).items():
                        try:
                            _repl_restored[_repl_vn] = _dill.loads(_repl_vb)
                        except BaseException:
                            _repl_dropped_load.append("var " + _repl_vn)
                    for _repl_pn2, _repl_pb in (_repl_wrapper.get("policies") or {}).items():
                        try:
                            _repl_reg2[_repl_pn2] = _dill.loads(_repl_pb)
                        except BaseException:
                            if _repl_pn2 not in _repl_psrc2:           # a source fallback may still cover it
                                _repl_dropped_load.append("policy " + _repl_pn2)
                    for _repl_rn, _repl_rb in (_repl_wrapper.get("recipes") or {}).items():
                        try:
                            _repl_rec2[_repl_rn] = _dill.loads(_repl_rb)
                        except BaseException:
                            _repl_dropped_load.append("constraint " + _repl_rn)
                    _repl_restored["_PLM_POLICIES"] = _repl_reg2
                    _repl_restored["_PLM_POLICY_SOURCES"] = _repl_psrc2
                    _repl_restored["_CONSTRAINT_RECIPES"] = _repl_rec2
                    _repl_restored["_CONSTRAINT_DROPPED"] = _repl_wrapper.get("dropped") or []
            else:
                # Empty / legacy / foreign shape -> treat directly (back-compat: b"" -> {}).
                _repl_restored = _repl_wrapper if _builtins.isinstance(_repl_wrapper, dict) else {}
            if _repl_dropped_load:
                _repl_rehydrate_error = (_repl_rehydrate_error or "") + (
                    "[rehydrate] " + _builtins.str(_builtins.len(_repl_dropped_load))
                    + " object(s) failed to load and were DROPPED (everything else was restored): "
                    + _repl_clip("; ".join(_builtins.sorted(_repl_dropped_load))) + ". ")
            # Vars that couldn't be SNAPSHOTTED at all (unpicklable -> dropped at dump time) are named
            # here too, so a vanished variable is never silent.
            _repl_var_dropped = ((_repl_wrapper.get("var_dropped") or [])
                                 if _builtins.isinstance(_repl_wrapper, dict) else [])
            if _repl_var_dropped:
                _repl_rehydrate_error = (_repl_rehydrate_error or "") + (
                    "[rehydrate] " + _builtins.str(_builtins.len(_repl_var_dropped))
                    + " variable(s) could not be snapshotted (unpicklable), so they did NOT survive "
                    + "restart: " + _repl_clip(_builtins.sorted(_repl_var_dropped)) + ". ")

            # Pop the source-fallback registry (policies that couldn't snapshot by
            # value — e.g. class policies) BEFORE globals().update so it never leaks
            # into __main__; it's re-installed from source after the reconcile.
            _repl_pol_sources = _repl_restored.pop("_PLM_POLICY_SOURCES", None) or {}
            # CALL-built constraints (Constraint.field) that couldn't dill-pickle
            # were snapshotted as RECIPES — popped here so they don't reach globals().update;
            # replayed back into __main__ after the reconcile (mirrors the policy-source path).
            _repl_constraint_recipes = _repl_restored.pop("_CONSTRAINT_RECIPES", None) or {}
            # Constraints the SNAPSHOT had to drop (un-recipe-able / unpicklable recipe) — recorded
            # at snapshot time so we can name them now, else a downstream cell hits a surprise
            # NameError with no cause. (the snapshot-side counterpart of the replay note.)
            _repl_constraint_dropped = _repl_restored.pop("_CONSTRAINT_DROPPED", None) or []
            if _repl_constraint_dropped:
                _repl_rehydrate_error = (_repl_rehydrate_error or "") + (
                    "[rehydrate] " + _builtins.str(_builtins.len(_repl_constraint_dropped))
                    + " constraint(s) could not be snapshotted (un-recipe-able or unpicklable), so "
                    + "they did NOT survive restart: "
                    + _repl_clip(_builtins.sorted(_repl_constraint_dropped)) + "; ")

            # === Phase 2: subscript-poisoning closure ===
            # Discard the snapshot's copies of immutable LLM defaults and keep
            # PREFIX's freshly-installed v0. Guard A skips `ast.Subscript` targets,
            # so a previous-session cell could have written
            # `_PLM_POLICIES["llm"] = evil_proxy` (the dict is in
            # __main__ via `from plm.policy import (...)`); Guard C treats the
            # registry as canonical, so the poisoned proxy round-trips through
            # the snapshot. We keep the existing "snapshot is canonical for
            # mutables" semantic (PLM-deleted extras stay deleted) BUT force the
            # LLM defaults to come from on-disk source.
            from plm.policy.defaults import _bless_llm_callers as _repl_bless_after
            from plm.policy.registry import _unsealed as _repl_unsealed, _SEALED_POLICIES as _repl_sealed, _store_writable as _repl_store_writable

            # Snapshot the FRESH-PREFIX boot policies (name -> proxy) BEFORE the
            # registry reset below, so orphaned __main__ bindings can be reaped
            # after reconciliation (see the drop loop further down,).
            _repl_preboot = {
                _repl_pn: _PLM_POLICIES[_repl_pn]
                for _repl_pn in _builtins.list(_PLM_POLICIES)
            }

            if "_PLM_POLICIES" in _repl_restored:
                # Force-restore EVERY sealed policy (the LLM defaults AND a metaparam's sealed extras
                # — `_SEALED_POLICIES`, not just the blessed `_LLM_DEFAULT_POLICIES`) to its fresh
                # PREFIX v0. A sealed policy must never come from the snapshot: a prior cell could have
                # poisoned `_PLM_POLICIES["<sealed>"] = evil` via a Subscript target Guard A skips, and
                # that copy round-trips through dill. Restricting this to the LLM defaults left sealed
                # extras poisonable across crash-restart (NR3-7).
                _repl_saved_sealed = {
                    n: _PLM_POLICIES[n] for n in _repl_sealed if n in _PLM_POLICIES
                }
                # The registry protects sealed entries during normal cell exec; this is a
                # harness-owned full reset, so drop the seal for it. `_unsealed` keeps this path
                # explicit + force-restores the freshly-installed v0 regardless of the snapshot.
                with _repl_unsealed(), _repl_store_writable():   # harness rehydrate -> authorize registry writes
                    _PLM_POLICIES.clear()
                    _PLM_POLICIES.update(_repl_restored.pop("_PLM_POLICIES"))
                    _PLM_POLICIES.update(_repl_saved_sealed)    # force-overwrite back to v0

            _builtins.globals().update(_repl_restored)

            # Force-rebind every SEALED name in __main__ to the registry's v0 proxy. REQUIRED because
            # the snapshot carries sealed policies (`llm`/`react_auto`/a sealed extra) as
            # __main__ globals (they're NOT in _REPL_INJECTED), so globals().update(...) above would
            # otherwise overwrite PREFIX's v0 binding with the snapshot's OLD/poisoned proxy — leaving
            # _PLM_POLICIES (v0) inconsistent with __main__ (OLD). All `<sealed>(...)` calls from cells
            # look up __main__["<sealed>"], so this rebind is what guarantees PLM hits v0. (Known
            # limitation, documented in the plan: a cell-level alias like `f = llm` rehydrates
            # pointing at the OLD proxy via dill identity-preservation; PLM should look up via the
            # global at call time, not cache the reference.)
            _repl_g = _builtins.globals()
            for _repl_dn in _repl_sealed:
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
            # leaves it untouched (no clobbering a legitimate cell variable).
            for _repl_pn, _repl_proxy in _repl_preboot.items():
                if _repl_pn not in _PLM_POLICIES and _repl_g.get(_repl_pn) is _repl_proxy:
                    _repl_g.pop(_repl_pn, None)

            # Re-install policies that could NOT be snapshotted by value (a class
            # policy: its depth-cap __call__ wrap closes over the _POLICY_CALL_DEPTH
            # ContextVar, so dill pickles the __main__ class by-reference —
            # unresolvable on the fresh respawn). The snapshot carried their
            # `_p_source` instead; re-exec `@policy` on it via the normal install
            # path so an AUTHORED class policy SURVIVES a hard respawn rather than
            # being silently lost. Skip a name a by-value entry already
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
                        # Surface, don't swallow: an authored class policy whose
                        # body fails to re-exec on respawn must NOT vanish silently.
                        _repl_rehydrate_error = (_repl_rehydrate_error or "") + (
                            "[rehydrate] policy " + _repl_pn + " not replayed: "
                            + _repl_clip(_builtins.repr(_repl_reinstall_exc)) + "; ")

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
                            + _repl_clip(_builtins.repr(_repl_crec_exc)) + "; ")
            elif _repl_constraint_recipes:
                # Recipes exist but the constraint surface never imported (no pydantic in this
                # kernel) -> they'd be SILENTLY dropped and downstream cells would hit surprise
                # NameErrors. Surface which names were lost so the cause is visible.
                _repl_rehydrate_error = (_repl_rehydrate_error or "") + (
                    "[rehydrate] " + _builtins.str(_builtins.len(_repl_constraint_recipes))
                    + " constraint(s) not replayed (constraint surface unavailable): "
                    + _repl_clip(_builtins.sorted(_repl_constraint_recipes)) + "; ")

            _rebuild_linecache_from_policies()

            # Refresh `_BLESSED_CALLERS` against the current registry's code
            # objects. Defensively uniform: for the LLM defaults we just
            # kept-from-PREFIX it's effectively a no-op (same code objects
            # PREFIX already blessed), but any future immutable default that
            # legitimately rehydrates would get correctly blessed here.
            _repl_bless_after()
        except Exception as _repl_rehydrate_exc:
            _repl_rehydrate_error = (_repl_rehydrate_error or "") + _repl_clip(_builtins.repr(_repl_rehydrate_exc))
        if _repl_rehydrate_error and _builtins.len(_repl_rehydrate_error) > 4096:   #  cap so a
            _repl_rehydrate_error = ("[...rehydrate notes truncated...]\n"           # pathological respawn
                                     + _repl_rehydrate_error[-4096:])                # can't balloon to MB
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
    # Reset the capture buffers via the side module (re-imported fresh), so a cell that rebound the
    # __main__ `_repl_reset_buffers` to a no-op can't leak stdout across cells. NO __main__ fallback:
    # the `or _repl_reset_buffers` fallback re-opened that exact hole (a cell nulling both the side
    # attr and the __main__ name). `_repl_ks.reset_buffers` is set once at boot (prefix.py). (NR3-3)
    import plm.repl._kernel_state as _repl_ks
    _repl_ks.reset_buffers()
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
    # accumulator and rebind the PUBLIC `plm_messages` to a fresh DEEP copy of it,
    # so a cell mutating `plm_messages` — append/clear/reassign OR editing a turn
    # in place — can't corrupt the accumulation (LEAK-PROOF). After a respawn the
    # parent resends the full trajectory (the accumulator is empty here), so this
    # stays correct across crash-restart.
    _repl_pm_delta = _repl_req.get("plm_messages_delta")
    if _repl_pm_delta is not None:
        _repl_plm_messages.extend(_repl_pm_delta)
    # Reseed the PUBLIC list from the hidden accumulator EVERY cell. The accumulator is
    # the single source of truth — fed by the delta channel (additive) and by a
    # `plm_messages` seed above (replace). A DEEP copy makes the public view leak-proof:
    # no cell mutation (list-level OR an in-place turn edit) reaches the accumulator, and
    # no stray rebind persists into the next cell. Robust: fall back to a shallow list if a
    # woven turn can't deepcopy, so the reseed never crashes a cell.
    try:
        _repl_g["plm_messages"] = _repl_copy.deepcopy(_repl_plm_messages)
    except _builtins.Exception:
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
    # Mirrors the existing per-iteration `_SEALED_POLICIES` re-import.
    import builtins as _repl_builtins
    from plm.policy import (
        _audit_cell as _repl_audit_cell,
        _PLM_POLICIES as _repl_plm_store,
    )
    from plm.policy.registry import _SEALED_POLICIES as _repl_imm
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
    # ARM SIGINT for the cell BODY ONLY: a timeout SIGINT lands here -> KeyboardInterrupt ->
    # caught below as an interrupted-result (graceful abort). It is IGNORED in every other phase
    # (see the kernel-wide SIG_IGN in KERNEL_BOOTSTRAP), so it can never escape into a spurious
    # boot_error / false "did NOT execute". The armed window is EXACTLY the exec: the inner
    # `finally` DISARMs the instant exec returns/raises — BEFORE the classification/traceback block
    # below — so a late, mis-timed timeout SIGINT can't tear the traceback formatting. Arm and
    # disarm go through the trusted-ref helpers, which can't throw even if a cell poisoned
    # `sys.modules['signal']`.
    _repl_arm_sigint()
    try:
        try:
            _builtins.exec(_builtins.compile(_repl_code, _repl_cell_file, "exec"), _repl_g, _repl_g)
        finally:
            _repl_disarm_sigint()
    except BaseException as _repl_exc:
        # Classify the terminal with a FRESHLY re-imported builtins, not the bare
        # `_builtins` the cell just had a chance to rebind: a fake `_builtins` whose
        # `.type`/`.getattr` lie could otherwise mask the cell's own exception as a
        # silent RETURN (or hide its traceback). Guard C must still run below. SIGINT is
        # already DISARMED here (inner finally), so this block can't be torn by a late SIGINT.
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
    # hijack live for the next cell. `builtins.globals()` is frame-derived —
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
    # The canon is read from the SIDE MODULE (re-imported fresh), NOT the __main__
    # `_REPL_INJECTED_CANON` a cell could rebind to `{}` (revert nothing -> a clobbered
    # `_dill` persists -> snapshot corruption,) or to a non-dict (`.items()` raises,
    # ). Wrapped so a broken canon can never crash the loop. (Uses the trusted `_repl_g`.)
    import plm.repl._kernel_state as _repl_ks
    try:
        _repl_canon = _repl_ks.CANON
        _repl_reverted = [
            _repl_k for _repl_k, _repl_v in _repl_canon.items()
            if _repl_g.get(_repl_k) is not _repl_v
        ]
        for _repl_k in _repl_reverted:
            _repl_g[_repl_k] = _repl_canon[_repl_k]
    except Exception as _repl_canon_exc:
        _repl_reverted = []
        _repl_stderr_buf.write("\n[repl guard] Guard C+ skipped (canon unreadable): "
                               + _builtins.repr(_repl_canon_exc) + "\n")
    if _repl_reverted:
        _repl_stderr_buf.write(
            "\n[repl guard] kernel-injected name(s) reverted: "
            + ", ".join(_builtins.sorted(_repl_reverted))   # _builtins.: a cell var named `sorted` must
                                                            # not break the loop (all-builtins-via-_builtins)
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
