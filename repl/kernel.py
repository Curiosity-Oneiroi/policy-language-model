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

_repl_sock = _repl_socket.socket(_repl_socket.AF_UNIX, _repl_socket.SOCK_STREAM)
_repl_sock.connect(_repl_os.environ["_REPL_SOCK_PATH"])
_repl_sock_io = _repl_sock.makefile("rwb", buffering=0)
_repl_req_in  = _repl_sock_io
_repl_resp_out = _repl_sock_io


def _repl_read_frame():
    hdr = _repl_req_in.read(4)
    if not hdr or _builtins.len(hdr) < 4:
        raise EOFError("parent closed request socket")
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
_repl_write_frame({"type": "ready"})

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

            if "_PLM_POLICIES" in _repl_restored:
                _repl_saved_defaults = {
                    n: _PLM_POLICIES[n] for n in _repl_undup_defs if n in _PLM_POLICIES
                }
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

            _rebuild_linecache_from_policies()

            # Refresh `_BLESSED_CALLERS` against the current registry's code
            # objects. Defensively uniform: for the LLM defaults we just
            # kept-from-PREFIX it's effectively a no-op (same code objects
            # PREFIX already blessed), but any future immutable default that
            # legitimately rehydrates would get correctly blessed here.
            _repl_bless_after()
        except Exception as _repl_rehydrate_exc:
            _repl_rehydrate_error = _builtins.repr(_repl_rehydrate_exc)
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

    # Guard A: reject a static rebind of a registered policy name BEFORE exec.
    # Pass the immutable-names subset so the audit can produce immutable-specific
    # rejection messages AND can reject `del <immutable>` (a separate AST branch
    # for ast.Delete; mutable `del` stays allowed).
    from plm.policy.registry import _IMMUTABLE_POLICIES as _repl_imm
    _repl_immset_for_audit = _builtins.set(_repl_imm)
    _repl_audit_err = _audit_cell(
        _repl_code, _builtins.set(_PLM_POLICIES), _repl_immset_for_audit
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

    try:
        _builtins.exec(_builtins.compile(_repl_code, _repl_cell_file, "exec"), _repl_g, _repl_g)
    except BaseException as _repl_exc:
        # Capture the terminal type only — Guard C must still run below.
        _repl_exc_name = _builtins.type(_repl_exc).__name__
        if _repl_exc_name == "_REPLReturn":
            _repl_terminal_etype = "return"
            _repl_return_value = _builtins.getattr(_repl_exc, "value", None)
        else:
            _traceback.print_exc()

    # Guard C: restore any policy whose __main__ binding drifted; clean up del'd.
    _post_cell_guard(_repl_g, _repl_stderr_buf)

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
