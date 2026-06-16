"""Tests for the @policy layer (plm/policy/) and its kernel integration.

In-process tests construct policies directly (no subprocess). `@policy` injects
into `sys.modules["__main__"]` and uses the shared `_PLM_POLICIES` registry, so
the autouse `_isolate` fixture clears both between tests. inspect.getsource reads
this very file from linecache, so policies are defined inside the test functions
(their source is dedented by the extractor). Class policies use UNIQUE names —
class source is recovered by FIRST same-named ClassDef across the file.

Integration tests drive a real `PythonReplSession` (subprocess venv kernel) via a
module-scoped fixture; they skip if a session can't be built.
"""

from __future__ import annotations

import sys
import warnings

import pytest

from plm.policy import (
    policy, list_policies, get_policy, read_policy, rewrite_policy,
    edit_policy, delete_policy, _PLM_POLICIES, _FunctionPolicy,
    _audit_cell, _post_cell_guard,
)
from plm.policy.proxy import _policy_note  # noqa: F401


@pytest.fixture(autouse=True)
def _isolate():
    """Clear registry + __main__ policy bindings around every in-process test."""
    main = sys.modules["__main__"].__dict__
    before = set(main)
    yield
    for n in list(_PLM_POLICIES):
        main.pop(n, None)
    from plm.policy.registry import _store_writable
    with _store_writable():                          # harness reset -> authorize the registry write
        _PLM_POLICIES.clear()
    # also drop any stray names @policy injected that weren't in the registry
    for n in set(main) - before:
        main.pop(n, None)


# ============================ function policies ============================

def test_call_read_comment_version():
    @policy
    def predict(s):
        # a tuned comment
        return "v1"
    assert predict("x") == "v1"
    src = read_policy("predict")
    assert src.startswith("def predict")
    assert "# a tuned comment" in src        # comment preserved verbatim
    assert get_policy("predict")._p_version == 0
    assert callable(predict) and not __import__("inspect").isfunction(predict)


def test_rewrite_swaps_inner_bumps_version():
    @policy
    def predict(s):
        return 1
    predict._rewrite("def predict(s):\n    return 2\n")
    assert predict("x") == 2
    assert predict._p_version == 1


def test_edit_first_vs_all():
    @policy
    def predict(s):
        return ["a", "a"]
    predict._edit('"a"', '"b"')                 # first only
    assert predict("x") == ["b", "a"]
    predict._edit('"a"', '"b"', replace_all=True)
    assert predict("x") == ["b", "b"]


def test_insert_and_delete_lines():
    @policy
    def predict(s):
        return 1
    predict._insert(1, "    return 99\n")        # inserted return (line 2) executes first
    assert predict("x") == 99
    predict._delete_lines(2)                       # remove it -> back to return 1
    assert predict("x") == 1


def test_noop_edits_dont_bump_version():
    @policy
    def predict(s):
        return 1
    v = predict._p_version
    predict._edit("not-found", "x")
    predict._edit("", "X")                       # empty old -> no-op (not splice-every-char)
    predict._delete_lines(0)                      # start<1 -> no-op (not text-duplicating)
    predict._insert(-1, "junk")                   # after_line<0 -> no-op
    predict._insert(0, "")                        # empty content -> no-op
    assert predict._p_version == v
    assert predict("x") == 1


def test_syntax_error_notes_unchanged_no_raise():
    @policy
    def predict(s):
        return 1
    predict._rewrite("def predict(s:\n  return 1")            # parse SyntaxError
    predict._edit("def predict(s):", "def predict(s):\n    await foo")  # compile-only error
    assert predict("x") == 1                       # unchanged, never raised


def test_shape_rules_note_unchanged():
    @policy
    def predict(s):
        return 1
    predict._rewrite("def predict(s): return 1\ndef other(): return 2\n")  # two defs
    predict._rewrite("THRESHOLD = 0.5\ndef predict(s): return THRESHOLD\n")  # sibling
    predict._rewrite("class predict:\n    pass\n")             # kind-swap to class
    assert predict("x") == 1


def test_rewrite_strips_policy_decorator():
    @policy
    def predict(s):
        return 1
    predict._rewrite("@policy\ndef predict(s):\n    return 7\n")  # strips @policy, no recursion
    assert predict("x") == 7
    predict._rewrite(predict._p_source)            # round-trip safe
    assert predict("x") == 7


def test_liveness_all_refs_share_proxy():
    @policy
    def predict(s):
        return 'v1'
    g = predict
    m = {"p": predict}
    predict._edit("'v1'", "'v2'")
    assert predict("x") == g("x") == m["p"]("x") == "v2"


def test_rename_rekeys_and_no_stale_slot_collision():
    import inspect as _inspect
    @policy
    def predict(s):
        return "P1"
    predict._edit("def predict", "def forecast")
    assert "forecast" in _PLM_POLICIES and "predict" not in _PLM_POLICIES
    forecast = sys.modules["__main__"].__dict__["forecast"]
    assert forecast("x") == "P1"
    assert forecast._p_filename == "<policy-forecast>"
    # a NEW predict gets its own slot; getsource of both is correct
    @policy
    def predict(s):
        return "P2"
    assert "P1" in _inspect.getsource(forecast._inner)
    assert "P2" in _inspect.getsource(predict._inner)


def test_getsource_and_signature_and_repr():
    import inspect as _inspect
    @policy
    def predict(state, k=3):
        return state
    assert "def predict(state, k=3)" in _inspect.getsource(predict._inner)
    assert list(_inspect.signature(predict).parameters) == ["state", "k"]
    assert repr(predict) == "<policy predict v0>"
    predict._rewrite("def predict(state, k=3):\n    return state * 2\n")
    assert "return state * 2" in _inspect.getsource(predict._inner)  # linecache synced


def test_argcount_change_and_async_rewrite_refused():
    """A rewrite may change the signature (argcount), but the repl is SYNCHRONOUS, so a rewrite
    to `async def` is REFUSED — the old sync policy is kept intact."""
    @policy
    def predict(s):
        return s
    predict._rewrite("def predict(s, n):\n    return s * n\n")         # sync argcount change works
    assert predict("a", 3) == "aaa"
    predict._rewrite("async def predict(s, n):\n    return s * n\n")   # async rewrite SOFT-refused (NP3-3)
    assert predict("a", 3) == "aaa"                                    # unchanged + still sync


def test_closure_warning_then_nameerror(capsys):
    threshold = 0.5
    @policy
    def predict(x):
        return x > threshold           # closes over test-function local
    err = capsys.readouterr().err
    assert "captures enclosing-function local" in err
    assert "threshold" in err
    with pytest.raises(NameError):
        predict(1)                      # closure lost on re-exec


# ============================ class policies ============================

def test_class_instantiate_type_getsource():
    import inspect as _inspect
    @policy
    class NetA:
        def m(self):
            return 1
    a = NetA()
    assert a.m() == 1
    assert type(NetA) is type and _inspect.isclass(NetA)
    assert "def m(self)" in _inspect.getsource(NetA.m)
    NetA._rewrite("class NetA:\n    def m(self):\n        return 2\n")
    assert "return 2" in _inspect.getsource(NetA.m)   # method getsource post-rewrite


def test_class_existing_instance_sees_new_method():
    @policy
    class NetB:
        def m(self):
            return 1
    a = NetB()
    NetB._rewrite("class NetB:\n    def m(self):\n        return 2\n    def extra(self):\n        return 9\n")
    assert a.m() == 2 and a.extra() == 9
    assert isinstance(a, NetB)


def test_class_super_survives_inplace_edit():
    class Base:
        def greet(self):
            return "base"
    # In the kernel, a policy's base lives in __main__ (defined in a cell). Mirror
    # that here so the under-__main__ re-exec of `class NetC(Base)` resolves Base.
    sys.modules["__main__"].__dict__["Base"] = Base
    @policy
    class NetC(Base):
        def greet(self):
            return "net+" + super().greet()
    a = NetC()
    assert a.greet() == "net+base"
    NetC._rewrite("class NetC(Base):\n    def greet(self):\n        return 'NET2+' + super().greet()\n")
    assert a.greet() == "NET2+base"               # zero-arg super() still resolves


def test_class_property_and_classmethod_add():
    @policy
    class NetD:
        def __init__(self):
            self._x = 5
    NetD._rewrite(
        "class NetD:\n"
        "    def __init__(self):\n        self._x = 5\n"
        "    @property\n    def x(self):\n        return self._x\n"
        "    @classmethod\n    def make(cls):\n        return cls()\n"
    )
    a = NetD.make()
    assert a.x == 5


def test_class_docstring_updates():
    @policy
    class NetE:
        "old doc"
        def m(self):
            return 1
    NetE._rewrite('class NetE:\n    "new doc"\n    def m(self):\n        return 1\n')
    assert NetE.__doc__ == "new doc"


def test_slotted_class_inplace_method_edit_keeps_slot():
    @policy
    class NetF:
        __slots__ = ("x",)
        def m(self):
            return "v1"
    a = NetF()
    a.x = 99
    NetF._rewrite("class NetF:\n    __slots__ = ('x',)\n    def m(self):\n        return 'v2'\n")
    assert a.x == 99                              # slot member-descriptor NOT clobbered
    assert a.m() == "v2"                          # method updated


def test_subclass_of_policy_class():
    @policy
    class NetG:
        def m(self):
            return 1
    class Big(NetG):
        pass
    b = Big()
    NetG._rewrite("class NetG:\n    def m(self):\n        return 42\n")
    assert b.m() == 42                            # edit propagates to subclass instances


def test_structural_fallback_slots_introduced(capsys):
    @policy
    class NetH:
        def m(self):
            return 1
    old = NetH
    NetH._rewrite("class NetH:\n    __slots__ = ('y',)\n    def m(self):\n        return 2\n")
    new = sys.modules["__main__"].__dict__["NetH"]
    assert new is not old                         # class replaced
    assert new().m() == 2
    assert "replaced the class" in capsys.readouterr().err
    assert new._p_source and callable(new._rewrite)  # replacement is a full policy


def test_class_call_with_wrapped_attr_still_depth_capped_after_rewrite():
    """#12: editing a class policy to a `__call__` that carries `__wrapped__` (as
    `@functools.wraps`/`@lru_cache` would set) must STILL install the depth-cap
    wrapper — the re-wrap guard keys on our private `_plm_depth_wrapped` marker,
    not the generic `__wrapped__` (which the old check mistook for 'already
    ours' and skipped, leaving __call__ uncapped)."""
    @policy
    class CallerC:
        def __call__(self, x):
            return x + 1
    src = (
        "class CallerC:\n"
        "    def __call__(self, x):\n"
        "        return x + 5\n"
        "    __call__.__wrapped__ = __call__   # mimics @functools.wraps setting __wrapped__\n"
    )
    CallerC._rewrite(src)
    cc = sys.modules["__main__"].__dict__["CallerC"]
    installed = cc.__dict__["__call__"]
    assert getattr(installed, "_plm_depth_wrapped", False) is True   # cap wrapper IS installed
    assert cc()(10) == 15                                            # behavior preserved


# ============================ decorator / guard / registry ============================

def test_decorator_type_and_value_errors():
    with pytest.raises(TypeError):
        policy(42)
    with pytest.raises(ValueError):
        policy(lambda x: x)                       # unnamed
    # reserved name
    src = "def policy(): pass"
    ns = {}
    exec(src, ns)
    with pytest.raises(ValueError):
        policy(ns["policy"])
    # kernel-internal prefix
    def _repl_x():
        return 1
    with pytest.raises(ValueError):
        policy(_repl_x)


def test_duplicate_policy_reserved_name_gentle_refusal(capsys):
    """C26: duplicate_policy must refuse a kernel-reserved / internal-prefix
    new_name the SAME way @policy would — with a gentle `_policy_note`, NOT a
    raw ValueError traceback from deep in the install path."""
    from plm.policy import duplicate_policy

    @policy
    def src_pol(s):
        return 1

    # Names @policy rejects: a reserved helper name, and the kernel-internal
    # prefixes (__ / _repl / _REPL). Each must come back as None + a note.
    # #11: a NON-STR new_name (e.g. 123) must ALSO be a gentle refusal, not a raw
    # AttributeError from `.isidentifier()` deep in the shared name rule.
    for bad in ("list_policies", "__x", "_repl_foo", "_REPLthing", 123, None):
        capsys.readouterr()                               # clear
        result = duplicate_policy("src_pol", bad)
        assert not result, f"{bad!r} should be refused"   # falsy PolicyResult on refusal
        err = capsys.readouterr().err
        assert "[policy] duplicate:" in err
        # the original is untouched and no broken policy was installed
        assert bad not in _PLM_POLICIES
    # sanity: a clean name still works. The successful path re-execs
    # "@policy\ndef ..." in __main__, which needs the `policy` NAME bound there
    # (the kernel PREFIX injects it; in-process we bind it for this step).
    sys.modules["__main__"].__dict__["policy"] = policy
    dup = duplicate_policy("src_pol", "src_pol_copy")
    assert dup is not None and "src_pol_copy" in _PLM_POLICIES


def test_guard_a_rejects_static_rebind():
    @policy
    def predict(s):
        return 1
    names = set(_PLM_POLICIES)
    assert _audit_cell("predict = 5\n", names)
    assert _audit_cell("for predict in range(3): pass\n", names)
    assert _audit_cell("import predict\n", names)
    assert _audit_cell("predict += 1\n", names)
    assert _audit_cell("a, *predict = [1, 2, 3]\n", names)
    # allowed:
    assert _audit_cell("x = predict\n", names) is None
    assert _audit_cell("predict: int\n", names) is None        # bare annotation binds nothing
    assert _audit_cell("del predict\n", names) is None
    assert _audit_cell("@policy\ndef predict(): pass\n", names) is None  # re-decoration ok


def test_guard_c_restores_rebind_and_cleans_del():
    import io
    @policy
    def predict(s):
        return 1
    g = sys.modules["__main__"].__dict__
    canonical = _PLM_POLICIES["predict"]
    g["predict"] = 999                            # dynamic rebind (audit can't see it)
    buf = io.StringIO()
    _post_cell_guard(g, buf)
    assert g["predict"] is canonical
    assert "[policy guard]" in buf.getvalue()
    # del cleanup
    del g["predict"]
    _post_cell_guard(g, io.StringIO())
    assert "predict" not in _PLM_POLICIES


def test_sync_leaves_rebind_for_guard_c():
    """A by-name helper (which calls _sync) after a dynamic rebind must NOT evict
    the policy — Guard C restores it post-cell."""
    import io
    @policy
    def predict(s):
        return 1
    g = sys.modules["__main__"].__dict__
    g["predict"] = 5                              # rebind
    assert "predict" in list_policies()           # _sync did NOT drop it
    _post_cell_guard(g, io.StringIO())            # Guard C restores
    assert g["predict"] is _PLM_POLICIES["predict"]


def test_redecoration_kind_change_replaces():
    @policy
    def predict(s):
        return 1
    assert isinstance(_PLM_POLICIES["predict"], _FunctionPolicy)
    @policy
    class predict:                                # noqa: N801 - same name, now a class
        def m(self):
            return 2
    assert isinstance(_PLM_POLICIES["predict"], type)
    assert _PLM_POLICIES["predict"]().m() == 2


# ============================ single-blob identity (dill) ============================

def test_single_blob_preserves_identity():
    dill = pytest.importorskip("dill")
    @policy
    def predict(s):
        return 1
    snap = {"predict": predict, "m": {"p": predict}, "lst": [predict],
            "_PLM_POLICIES": dict(_PLM_POLICIES)}
    restored = dill.loads(dill.dumps(snap))
    p = restored["predict"]
    assert restored["m"]["p"] is p
    assert restored["lst"][0] is p
    assert restored["_PLM_POLICIES"]["predict"] is p
    assert p("x") == 1


def test_class_policy_survives_dill_roundtrip():
    dill = pytest.importorskip("dill")
    @policy
    class NetI:
        def m(self):
            return 7
    n = NetI()
    snap = {"NetI": NetI, "inst": n}
    restored = dill.loads(dill.dumps(snap))
    R = restored["NetI"]
    assert type(restored["inst"]) is R            # identity preserved
    assert restored["inst"].m() == 7
    assert callable(R._rewrite)                    # edit API survived (named-fn classmethods)


def test_rewrite_add_nested_helper():
    @policy
    def predict(s):
        return s
    predict._rewrite(
        "def predict(s):\n"
        "    def helper(x):\n        return x * 2\n"   # nested helper (allowed: inside body)
        "    return helper(s)\n"
    )
    assert predict(5) == 10


def test_class_method_removed():
    @policy
    class NetJ:
        def m(self):
            return 1
        def gone(self):
            return 2
    a = NetJ()
    assert a.gone() == 2
    NetJ._rewrite("class NetJ:\n    def m(self):\n        return 1\n")  # 'gone' removed
    assert not hasattr(NetJ, "gone")
    assert a.m() == 1


def test_class_compatible_base_change():
    class BaseX:
        def who(self):
            return "X"
    class BaseY:
        def who(self):
            return "Y"
    main = sys.modules["__main__"].__dict__
    main["BaseX"] = BaseX
    main["BaseY"] = BaseY
    @policy
    class NetK(BaseX):
        pass
    a = NetK()
    assert a.who() == "X"
    NetK._rewrite("class NetK(BaseY):\n    pass\n")   # compatible base swap, in place
    assert a.who() == "Y"


def test_get_source_raises_on_missing():
    from plm.policy.decorator import _get_source_for_obj
    class Foo:
        pass
    with pytest.raises(RuntimeError):
        _get_source_for_obj(Foo, "<no-such-cell>")    # class, empty linecache -> RuntimeError


def test_del_via_helper_excludes_from_list():
    @policy
    def predict(s):
        return 1
    assert "predict" in list_policies()
    delete_policy("predict")
    assert "predict" not in list_policies()
    assert "predict" not in sys.modules["__main__"].__dict__


def test_class_policy_staticmethod_call_not_depth_wrapped():
    """#R5-6: a class policy whose __call__ is a @staticmethod must NOT be wrapped
    by the depth-cap (which reads cls.__dict__['__call__'] as a plain method and
    calls it `_orig_call(self, ...)` — mis-forwarding self for a staticmethod and
    raising TypeError). The wrap is skipped for non-FunctionType descriptors; the
    call works and the descriptor is left intact (the LLM-depth gate is enforced
    independently by the blessed-caller checks, not this advisory cap)."""
    @policy
    class CallSM:
        @staticmethod
        def __call__():
            return "ok-static"

    assert CallSM()() == "ok-static"                     # no mis-forwarded-self TypeError
    assert isinstance(CallSM.__dict__["__call__"], staticmethod)   # left a staticmethod (unwrapped)

    # And the in-place re-wrap path (rewrite) is guarded too: rewriting to another
    # staticmethod __call__ must not break the call.
    CallSM._rewrite(
        "class CallSM:\n    @staticmethod\n    def __call__():\n        return 'ok2'\n"
    )
    assert CallSM()() == "ok2"
    assert isinstance(CallSM.__dict__["__call__"], staticmethod)


def test_root_loop_coerces_non_dict_tool_args(monkeypatch):
    """#R5-3: a `python` tool call whose `arguments` is valid JSON but NOT an object
    ('null'/'[]'/'123') must not crash PLM.__call__ with a raw AttributeError from
    `targs.get('code', ...)`. The root loop coerces non-dict targs to {} (mirroring
    both inner loops), runs an empty cell, and exhausts the budget GRACEFULLY
    (PLMTaskFailure). A fake REPL keeps this in-process — no subprocess kernel."""
    import asyncio
    from plm.plm import PLM, PLMMetaParameters, PLMTaskFailure

    class _FakeRepl:                                     # the shape PLM.__call__ needs
        kernel_epoch = 0
        def __init__(self, **kw): pass
        def execute_cell(self, code, _seed, _delta):
            return {"type": "result", "stdout": "", "stderr": ""}
        def close(self): pass

    monkeypatch.setattr("plm.plm.PythonReplSession", _FakeRepl)

    class _BadArgsBackend:                               # always emits non-object tool args
        model = "stub"
        def __init__(self): self.calls = 0
        async def generate(self, messages=None, tools=None, **kw):
            self.calls += 1
            return {
                "content": "",
                "tool_calls": [{"id": f"t{self.calls}", "type": "function",
                                "function": {"name": "python", "arguments": "null"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

    backend = _BadArgsBackend()
    plm = PLM(model_backend=backend,
              metaparams=PLMMetaParameters(system_prompt="solve it"),
              max_turns=2, return_budget=0)
    with pytest.raises(PLMTaskFailure):                  # graceful exhaustion, NOT AttributeError
        asyncio.run(plm([{"role": "user", "content": "go"}]))
    assert backend.calls >= 1


# ============================ integration (subprocess kernel) ============================

@pytest.fixture(scope="module")
def repl():
    try:
        from plm.repl import PythonReplSession
    except Exception as e:                         # pragma: no cover
        pytest.skip(f"repl import failed: {e}")
    try:
        s = PythonReplSession(workspace=None, preinstall=("dill",),
                              cell_timeout=10.0, sigint_grace=2.0)
    except Exception as e:                         # pragma: no cover
        pytest.skip(f"could not start kernel session: {e}")
    yield s
    s.close()


def test_int_liveness(repl):
    repl.execute_cell("@policy\ndef predict(s): return 'v1'\n")
    repl.execute_cell("g = predict\nmy = {'p': predict}\n")
    repl.execute_cell("predict._edit(\"'v1'\", \"'v2'\")")
    r = repl.execute_cell("print(predict('x'), g('x'), my['p']('x'))")
    assert r["stdout"].strip() == "v2 v2 v2", r


def test_int_injected_helpers_protected(repl):
    """Kernel-injected NON-policy helpers (exec_ns, parallel, policy ops) are restored
    if a cell rebinds OR deletes them — granted helpers can't be clobbered for the next
    cell, the same protection immutable policies get. (`plm_messages` is NOT in the
    protected callable set; it is instead reseeded from the trajectory accumulator each
    cell — see test_int_plm_messages_seed_channel.)"""
    r0 = repl.execute_cell("exec_ns = 5\ndel parallel")    # clobber + delete callable helpers
    assert "reverted" in r0["stderr"], r0                  # the guard noted the revert
    r = repl.execute_cell("print(callable(exec_ns), callable(parallel))")
    assert r["stdout"].strip() == "True True", r           # both restored next cell
    # K-F7: a cell mutating `plm_messages` does NOT persist (reseeded from the accumulator)
    repl.execute_cell("plm_messages.append('stray')")
    r2 = repl.execute_cell("print('stray' in plm_messages)")
    assert r2["stdout"].strip() == "False", r2


def test_int_constraint_surface_ambient_with_pydantic():
    """The prefix best-effort-injects the WHOLE constraint surface when the kernel has
    pydantic (standard kernels do, via DEFAULT_PREINSTALL): a cell uses `Constraint`,
    `ConstraintViolation`, and the `@constraint` decorator with NO import. The default
    `repl` fixture is dill-only (no pydantic), so it can't show this — boot a pydantic
    kernel here. This is the end-to-end check that the guarded warmup actually fires."""
    try:
        from plm.repl import PythonReplSession
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"repl import failed: {e}")
    try:
        s = PythonReplSession(workspace=None, preinstall=("dill", "pydantic"),
                              cell_timeout=15.0, sigint_grace=2.0)
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"could not start pydantic kernel: {e}")
    try:
        r = s.execute_cell(
            "ok = (Constraint.__name__ == 'Constraint' and hasattr(Constraint, 'field') "
            "and ConstraintViolation.__name__ == 'ConstraintViolation' and callable(constraint))\n"
            "print('AMBIENT_OK' if ok else 'AMBIENT_FAIL')"
        )
        assert r["stdout"].strip() == "AMBIENT_OK", r
    finally:
        s.close()


def test_int_crash_restart_constraints_survive():
    """D5 fix — CALL-built constraints survive a hard respawn. A Constraint.field(...)
    and a composite (A & B) are rebuilt from their snapshot RECIPES; a structural
    subclass survives by value. All three validate (accept + reject) correctly after the
    crash. Needs a pydantic kernel (the constraint surface)."""
    try:
        from plm.repl import PythonReplSession
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"repl import failed: {e}")
    try:
        s = PythonReplSession(workspace=None, preinstall=("dill", "pydantic"),
                              cell_timeout=15.0, sigint_grace=2.0)
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"could not start pydantic kernel: {e}")
    try:
        # Constraint is AMBIENT in cells. A field + composite (recipes) AND a STRUCTURAL
        # class with a default, a Field bound, and a validator (reconstructed from fields).
        s.execute_cell(
            "from pydantic import Field, model_validator\n"
            "field_c = Constraint.field(type=object, is_instance_of=int, predicate=lambda x: x > 100)\n"
            "comp_c = Constraint.field(type=object, is_instance_of=int) & "
            "Constraint.field(type=object, is_instance_of=int, predicate=lambda x: x > 0)\n"
            "class StructC(Constraint):\n"
            "    name: str = 'anon'\n"
            "    n: int = Field(ge=0)\n"
            "    @model_validator(mode='after')\n"
            "    def _chk(self):\n"
            "        if self.name == 'BAD':\n"
            "            raise ValueError('bad')\n"
            "        return self\n")
        ep = s.kernel_epoch
        s.execute_cell("import os as _o\n_o._exit(0)")  # hard crash -> respawn + rehydrate
        assert s.kernel_epoch > ep, "expected a respawn"
        # all three survived and validate the GOOD values (StructC keeps its 'anon' default)
        r = s.execute_cell(
            "print(field_c.validate(200), comp_c.validate(5), StructC.validate({'n': 3}).name)")
        assert r["stdout"].strip() == "200 5 anon", r
        # ... and all REJECT bad ones — predicates, operators, the Field bound, AND the validator
        r2 = s.execute_cell(
            "def _rej(c, v):\n"
            "    try:\n"
            "        c.validate(v); return False\n"
            "    except Exception:\n"
            "        return True\n"
            "print(_rej(field_c, 5), _rej(comp_c, -1), "
            "_rej(StructC, {'n': -1}), _rej(StructC, {'name': 'BAD', 'n': 0}))")
        assert r2["stdout"].strip() == "True True True True", r2
    finally:
        s.close()


def test_int_crash_restart_nested_constraints_survive():
    """H1 — a Constraint NESTED inside another survives a hard respawn. Before the fix the
    recipe kept the inner class live: `addr: Address` (a struct field typed as another
    structural Constraint) pickled BY-REFERENCE and made the rehydrate `dill.loads` fail,
    SINKING THE WHOLE SNAPSHOT (sibling vars + all policies gone); a `Constraint.field(...)`-
    typed field (the README's Pattern-2) and a `list_of=Constraint.field(...)` factory kwarg
    were silently dropped. Now to_recipe recurses into nested Constraints."""
    try:
        from plm.repl import PythonReplSession
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"repl import failed: {e}")
    try:
        s = PythonReplSession(workspace=None, preinstall=("dill", "pydantic"),
                              cell_timeout=15.0, sigint_grace=2.0)
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"could not start pydantic kernel: {e}")
    try:
        s.execute_cell(
            "from typing import Literal\n"
            "class Address(Constraint):\n    city: str\n"
            "class Person(Constraint):\n    name: str\n    addr: Address\n"
            "class Order(Constraint):\n"
            "    currency: Constraint.field(type=Literal['USD','EUR'])\n"
            "    qty: Constraint.field(type=int, int_range=(1,100))\n"
            "nested = Constraint.field(type=list[Constraint.field(type=int, int_range=(1,5))])\n"
            "some_var = 42\n")                          # a plain sibling: must NOT be sunk
        ep = s.kernel_epoch
        r0 = s.execute_cell("import os as _o\n_o._exit(0)")   # hard crash -> respawn + rehydrate
        assert s.kernel_epoch > ep, "expected a respawn"
        assert not r0.get("rehydrate_error"), r0.get("rehydrate_error")  # snapshot must NOT be sunk
        r = s.execute_cell(
            "print(globals().get('some_var'), "
            "Order.validate({'currency':'USD','qty':10}).qty, "
            "nested.validate([1,2,3]), "
            "Person.validate({'name':'x','addr':{'city':'NYC'}}).addr.city)")
        assert r["stdout"].strip() == "42 10 [1, 2, 3] NYC", r
        # nested rules still ENFORCED after the respawn
        r2 = s.execute_cell(
            "def _rej(c, v):\n"
            "    try:\n"
            "        c.validate(v); return False\n"
            "    except Exception:\n"
            "        return True\n"
            "print(_rej(Order, {'currency':'GBP','qty':10}), _rej(nested, [1,99]), "
            "_rej(Person, {'name':'x','addr':{'city':123}}))")
        assert r2["stdout"].strip() == "True True True", r2
    finally:
        s.close()


def test_int_kernel_internals_cell_rebind_proof():
    """NR-1/1b/2/4: a cell rebinding the kernel's own __main__ internals must NOT disable the
    guard mechanism — the KERNEL_LOOP reads them from the re-imported side module
    `plm.repl._kernel_state`, not the cell-rebindable __main__ globals. (Within the trust model;
    defends an accidental rebind of an obscure internal, not deliberate sys.modules poisoning.)"""
    try:
        from plm.repl import PythonReplSession
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"repl import failed: {e}")
    try:
        s = PythonReplSession(workspace=None, preinstall=("dill",), cell_timeout=15.0, sigint_grace=2.0)
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"could not start kernel: {e}")
    try:
        # NR-1: rebind Guard C+'s canon to {} AND clobber an injected helper -> still reverted
        # next cell (Guard C+ reads the side module's CANON, not the cell's {}).
        s.execute_cell("_REPL_INJECTED_CANON = {}\nparallel = 'EVIL'")
        assert s.execute_cell("print(type(parallel).__name__)")["stdout"].strip() == "function"
        # NR-1b: rebind the canon to a non-dict -> Guard C+ must not crash the loop.
        s.execute_cell("_REPL_INJECTED_CANON = None")
        assert s.execute_cell("print('alive')")["stdout"].strip() == "alive"
        # NR-4: rebind the buffer reset to a no-op -> stdout must not leak into the next cell.
        s.execute_cell("_repl_reset_buffers = lambda: None\nprint('cellA')")
        assert s.execute_cell("print('cellB')")["stdout"].strip() == "cellB"
        # NR-2: empty the snapshot blocklist -> crash-restart still works (anchor survives).
        s.execute_cell("_REPL_INJECTED = set()\nanchor = 'KEEP'")
        ep = s.kernel_epoch
        s.execute_cell("import os as _o\n_o._exit(0)")
        assert s.kernel_epoch > ep
        assert s.execute_cell("print(globals().get('anchor'))")["stdout"].strip() == "KEEP"
    finally:
        s.close()


def test_int_close_does_not_resurrect_session():
    """O5: a worker thread's post-close auto-respawn (its blocked read wakes EOF when close()
    kills the kernel) must NOT resurrect a torn-down session — no orphan kernel, no 30s
    accept() hang. Normal crash-restart (with _closed False) is completely unaffected."""
    try:
        from plm.repl import PythonReplSession
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"repl import failed: {e}")
    try:
        s = PythonReplSession(workspace=None, preinstall=("dill",), cell_timeout=15.0, sigint_grace=2.0)
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"could not start kernel: {e}")
    try:
        # Normal crash-restart still works (the _closed guards are inert when not closed).
        s.execute_cell("x = 1")
        ep = s.kernel_epoch
        s.execute_cell("import os as _o\n_o._exit(0)")
        assert s.kernel_epoch > ep
        assert s.execute_cell("print(globals().get('x'))")["stdout"].strip() == "1"
    finally:
        s.close()
    # After close, what the in-flight worker would do must spawn NOTHING (no orphan kernel).
    assert s._closed
    ep2 = s.kernel_epoch
    s._kill_and_respawn()
    assert s.kernel_epoch == ep2
    s._spawn_kernel()                                  # direct spawn is also a no-op when closed
    assert s.kernel_epoch == ep2


def test_compute_child_pythonpath_merges_and_dedupes():
    """NR-3: the child PYTHONPATH PREPENDS the plm dir to the inherited/caller PYTHONPATH and
    PRESERVES it (doesn't clobber), deduping PER DIRECTORY — so the kernel can import `plm`
    while a caller-supplied `env=` PYTHONPATH (which beats os.environ in the constructor's env
    merge) survives into the child."""
    import os
    from plm.repl.session import _compute_child_pythonpath as f
    sep = os.pathsep
    plm_dir = f("").split(sep)[0]
    assert f("") == plm_dir                                          # empty -> just the plm dir
    assert f(f"/my/libs{sep}/other").split(sep) == [plm_dir, "/my/libs", "/other"]   # plm first, preserved
    assert f(f"{plm_dir}{sep}/my/libs{sep}/my/libs").split(sep) == [plm_dir, "/my/libs"]  # per-dir dedupe


def test_frame_length_cap_rejects_oversize():
    """NR-5: a frame-length header past the cap is a corrupt/desynced protocol error -> the
    parent raises _FrameDecodeError (caught -> respawn) instead of an unbounded read; a normal
    length still decodes. The kernel-side cap (_REPL_MAX_FRAME) matches the parent's."""
    import io, struct, pickle, types
    from plm.repl.session import PythonReplSession, _MAX_FRAME_BYTES, _FrameDecodeError
    from plm.repl.kernel import KERNEL_BOOTSTRAP

    class FakeResp:                                       # minimal stand-in for the response socket
        def __init__(self, data): self._b = io.BytesIO(data)
        def read(self, n): return self._b.read(n)
        def fileno(self): return 0

    over = struct.pack(">I", _MAX_FRAME_BYTES + 1)        # oversized length header
    with pytest.raises(_FrameDecodeError):
        PythonReplSession._read_frame_with_timeout(types.SimpleNamespace(_resp_r=FakeResp(over)), None)

    body = pickle.dumps({"type": "ok"})                   # a normal frame still decodes
    frame = struct.pack(">I", len(body)) + body
    got = PythonReplSession._read_frame_with_timeout(types.SimpleNamespace(_resp_r=FakeResp(frame)), None)
    assert got == {"type": "ok"}

    assert _MAX_FRAME_BYTES == 2 * 1024 ** 3 < 2 ** 32                    # sane: below the uint32 header max
    assert "_REPL_MAX_FRAME = 2 * 1024 ** 3" in KERNEL_BOOTSTRAP          # parent + kernel caps agree


def test_nr31_respawn_result_signals_not_executed():
    """NR3-1: every kill+respawn path returns ONE consistent envelope (built by `_respawn_result`)
    carrying executed=False + a RE-RUN note — so a caller can detect a non-run, and the paths can't
    drift in shape again (previously only the EOF path set the flag)."""
    from plm.repl import PythonReplSession
    # unit: the single builder always sets the flag
    env = PythonReplSession._respawn_result("PRE", "[note]\n")
    assert env["executed"] is False and env["stderr"] == "PRE[note]\n" and env["return_obj"] is None
    # integration: a kernel that exits before returning a result -> respawn path sets it
    s = PythonReplSession(workspace=None, preinstall=("dill",), cell_timeout=10.0, sigint_grace=2.0)
    try:
        r = s.execute_cell("import os as _o\n_o._exit(0)")
        assert r.get("executed") is False and "RE-RUN" in r.get("stderr", "")
        assert s.execute_cell("print(2 + 2)")["stdout"].strip() == "4"     # respawned + usable
    finally:
        s.close()


def test_f4_root_loop_survives_malformed_tool_calls():
    """F4 + PLM1 + B1: PLM's root loop must not CRASH on a non-conformant backend — it mirrors
    react_llm's guards (normalize the whole RESPONSE to a dict, normalize tool_calls shape,
    isinstance(tc, dict), isinstance(function, dict), isinstance(code, str), reply a user-turn for a
    non-dict tc). Inputs that used to crash PLM.__call__ raw (the outer try has no `except`) now
    re-prompt and exhaust the budget gracefully (PLMTaskFailure), no TypeError/KeyError/
    AttributeError. `["junk"]` hit the bare `tool_call["function"]` subscript (F4); a non-string
    `code` arg hit `_strip_code_fences` (PLM1); a non-dict response hit `response.get(...)` (B1)."""
    import asyncio
    from plm.plm import PLM, PLMMetaParameters, PLMTaskFailure

    # each entry is a full backend RESPONSE (B1 covers non-dict responses; F4/PLM1 cover bad tool_calls)
    responses = [
        {"content": "", "tool_calls": ["junk"]},                                          # F4: non-dict tc
        {"content": "", "tool_calls": [{"id": "1", "function": {"name": "python",
                                                                "arguments": '{"code": 123}'}}]},  # PLM1: non-str code
        None,                                                                             # B1: non-dict response
        ["not", "a", "dict"],                                                             # B1: non-dict response
    ]
    for resp in responses:
        class _FakeBackend:
            model = "fake-model"
            def __init__(self, r): self._r = r
            async def generate(self, messages=None, tools=None, **kw):
                return self._r                                   # malformed every round

        plm = PLM(model_backend=_FakeBackend(resp),
                  metaparams=PLMMetaParameters(system_prompt="sys"), max_turns=1, return_budget=1)
        with pytest.raises(PLMTaskFailure):                      # graceful budget-exhaust, NOT a crash
            asyncio.run(plm([{"role": "user", "content": "go"}]))


def test_int_generator_policies_survive_crash_restart():
    """M3: EVERY generator-policy shape survives a hard respawn AND stays depth-correct after —
    a function generator, a class generator __call__, a class generator method, and nesting
    (normal->gen, gen->gen via `yield from`, gen->normal). Policies re-exec from `_p_source` on
    respawn and the proxy re-applies the depth-tracking wrap. (Bodies re-import the depth
    ContextVar so the probe survives — an unpicklable global ref would be lost equally for a
    normal policy.)"""
    try:
        from plm.repl import PythonReplSession
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"repl import failed: {e}")
    try:
        s = PythonReplSession(workspace=None, preinstall=("dill",), cell_timeout=20.0, sigint_grace=2.0)
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"could not start kernel: {e}")
    try:
        s.execute_cell(
            "@policy\ndef rep():\n    from plm.policy.proxy import _POLICY_CALL_DEPTH as _D\n    return _D.get()\n"
            "@policy\ndef genrep():\n    from plm.policy.proxy import _POLICY_CALL_DEPTH as _D\n    yield _D.get()\n"
            "@policy\ndef norm_calls_gen():\n    return list(genrep())\n"
            "@policy\ndef gen_yieldfrom():\n    yield from genrep()\n"
            "@policy\ndef gen_calls_norm():\n    yield rep()\n"
            "@policy\nclass GCall:\n    def __call__(self):\n"
            "        from plm.policy.proxy import _POLICY_CALL_DEPTH as _D\n        yield _D.get()\n"
            "@policy\nclass GMeth:\n    def stream(self):\n        for i in range(3):\n            yield i*i\n"
            "    def total(self):\n        return sum(self.stream())\n")
        probe = ("print(list(genrep()), norm_calls_gen(), list(gen_yieldfrom()), "
                 "list(gen_calls_norm()), list(GCall()()), GMeth().total())")
        expected = "[1] [2] [2] [2] [1] 5"
        assert s.execute_cell(probe)["stdout"].strip() == expected      # before crash
        ep = s.kernel_epoch
        s.execute_cell("import os as _o\n_o._exit(0)")
        assert s.kernel_epoch > ep
        r = s.execute_cell(probe)
        assert r["stdout"].strip() == expected, r                       # depth-correct AFTER respawn
    finally:
        s.close()


def test_int_class_hotswap(repl):
    repl.execute_cell("@policy\nclass Netz:\n    def m(self): return 1\n")
    repl.execute_cell("a = Netz()")
    repl.execute_cell("Netz._rewrite('class Netz:\\n    def m(self): return 2\\n    def n(self): return 9\\n')")
    r = repl.execute_cell("print(a.m(), a.n(), isinstance(a, Netz))")
    assert r["stdout"].strip() == "2 9 True", r


def test_int_class_redecoration_structural_is_canonical(repl):
    """#10: re-decorating a class via `@policy` with a STRUCTURAL change (new
    __slots__) forces _remove_and_recreate. The decorator must bind/return the
    NEW canonical class — `__main__`'s name, the registry, and instances must all
    agree (no clobber-back to the dead old class). Cells are separate sources, so
    the second cell's __slots__ version is recovered correctly (unlike one file)."""
    repl.execute_cell("@policy\nclass NetRedec:\n    def m(self): return 1\n")
    repl.execute_cell("@policy\nclass NetRedec:\n    __slots__ = ('z',)\n    def m(self): return 2\n")
    r = repl.execute_cell(
        "print(NetRedec().m(), _PLM_POLICIES['NetRedec'] is NetRedec)")
    assert r["stdout"].strip() == "2 True", r          # new behavior + registry==__main__


def test_int_guard_rejects_rebind(repl):
    repl.execute_cell("@policy\ndef pol(s): return 1\n")
    r = repl.execute_cell("pol = 5")
    assert "rebind policy" in r["stderr"]
    r2 = repl.execute_cell("print(pol('x'))")
    assert r2["stdout"].strip() == "1", r2          # still callable


def test_int_return_and_policy(repl):
    repl.execute_cell("@policy\ndef predict(s): return s * 2\n")
    r = repl.execute_cell("RETURN(predict('ab'))")
    assert r["type"] == "return" and r["return_obj"] == "abab", r


def test_int_crash_restart_identity(repl):
    repl.execute_cell("@policy\ndef predict(s): return 'before'\n")
    repl.execute_cell("box = {'p': predict}")
    # force a hard timeout -> SIGKILL -> respawn from cached snapshot
    repl.execute_cell("import time as _t\n_t.sleep(999)")
    r = repl.execute_cell("print(box['p'] is predict, predict('x'))")
    assert r["stdout"].strip() == "True before", r
    # post-respawn edit still propagates through the same proxy
    repl.execute_cell("predict._edit(\"'before'\", \"'after'\")")
    r2 = repl.execute_cell("print(predict('x'), box['p']('x'))")
    assert r2["stdout"].strip() == "after after", r2


def test_int_policy_creates_policy(repl):
    repl.execute_cell(
        "@policy\n"
        "def parent():\n"
        "    @policy\n"
        "    def child():\n"
        "        return 'child!'\n"
        "    return child\n"
    )
    repl.execute_cell("parent()")
    r = repl.execute_cell("print('child' in list_policies(), child())")
    assert r["stdout"].strip() == "True child!", r


def test_int_cross_policy_call_and_liveness(repl):
    repl.execute_cell("@policy\ndef helper(x): return x * 10\n")
    repl.execute_cell("@policy\ndef predict(s): return helper(s) + 1\n")   # resolves helper via __main__
    r = repl.execute_cell("print(predict(5))")
    assert r["stdout"].strip() == "51", r
    repl.execute_cell("helper._edit('* 10', '* 100')")                      # edit helper...
    r2 = repl.execute_cell("print(predict(5))")
    assert r2["stdout"].strip() == "501", r2                                # ...predict sees it live


def test_int_builtin_shadow_doesnt_brick(repl):
    repl.execute_cell("@policy\ndef p(s): return 1\n")
    # A cell shadows builtins the kernel loop/audit/collect use in __main__.
    repl.execute_cell("set = 5\nlen = 'x'\ntype = 3\ncompile = None\n")
    # Next cells must still work: audit uses _builtins.set, collect uses _builtins.len,
    # exec uses _builtins.compile — all immune to the __main__ shadows.
    r = repl.execute_cell("print(p('x'))")
    assert r["stdout"].strip() == "1", r
    r2 = repl.execute_cell("p._rewrite('def p(s): return 2'); print(p('x'))")
    assert r2["stdout"].strip() == "2", r2


def test_int_guard_survives_cell_rebinding_guard_helpers(repl):
    """#R5-1: a cell that rebinds the guard helpers in __main__ via bare-name
    assigns (which Guard A does NOT flag — it only rejects POLICY-name rebinds)
    must NOT disable Guard A/C. The loop re-imports the helpers fresh from their
    module every iteration, so a SUBSEQUENT attempt to hijack an immutable
    default's __main__ binding is still rejected and the real proxy preserved."""
    # Disable the guards the easy way — plain __main__ rebinds Guard A won't catch.
    repl.execute_cell(
        "_audit_cell = lambda *a, **k: None\n"
        "_post_cell_guard = lambda *a, **k: None\n"
    )
    # Now try to hijack an immutable default. Fresh-imported Guard A must reject it.
    r = repl.execute_cell("natural_llm = lambda *a, **k: 'PWNED'\n")
    assert "rebind policy 'natural_llm'" in r["stderr"], r
    # natural_llm is STILL the real immutable proxy, not the hijack lambda — the
    # __main__ binding equals the registry's canonical object (identity check; no
    # `type()`/builtins, since the shared session permanently shadows them).
    r2 = repl.execute_cell(
        "print('natural_llm' in list_policies(), _PLM_POLICIES['natural_llm'] is natural_llm)"
    )
    assert r2["stdout"].strip() == "True True", r2


def test_int_immutable_redecoration_skips_closure_warning(repl):
    """#R5-7: re-decorating an immutable default is refused BEFORE the
    closure-capture check runs, so the misleading 'will NameError at call' warning
    is not emitted for a body that is discarded — only the 'ignored' note fires."""
    r = repl.execute_cell(
        "def _outer():\n"
        "    captured = 5\n"
        "    @policy\n"
        "    def natural_llm():\n"
        "        return captured\n"          # a real enclosing-local capture (free var)
        "    return natural_llm\n"
        "_outer()\n"
    )
    assert "re-decoration ignored" in r["stderr"], r              # the refusal note fired
    assert "captures enclosing-function local" not in r["stderr"], r   # NOT the closure warning


# ===================== seed channel + plm_messages =====================


def test_strip_code_fences_unwraps_only_structural_fences():
    """_strip_code_fences unwraps a genuinely fence-wrapped arg but NEVER touches
    backticks embedded in valid Python (finding C00/C10)."""
    from plm._react_helper import _strip_code_fences

    # genuine wrappers -> unwrapped
    assert _strip_code_fences("```python\nprint('hi')\n```") == "print('hi')"
    assert _strip_code_fences("```\nx = 1\n```") == "x = 1"
    assert _strip_code_fences("```py3\n\nprint(1)\n\n```") == "print(1)"

    # valid Python with ``` inside a docstring / string / comment -> UNCHANGED
    doc = 'def f():\n    """ex:\n    ```\n    f()\n    ```\n    """\n    return 1'
    assert _strip_code_fences(doc) == doc
    assert _strip_code_fences("s = '```'\nprint(s)") == "s = '```'\nprint(s)"
    multiline = 'x = """```\nnot a fence\n```"""\nprint(x)'
    assert _strip_code_fences(multiline) == multiline

    # no fence at all -> unchanged
    assert _strip_code_fences("a = 1\nb = 2") == "a = 1\nb = 2"

    # code AFTER a fenced block is NOT silently dropped (old bug took only 'A'):
    out = _strip_code_fences("```python\nA\n```\nprint('after')")
    assert "after" in out and "```" in out


def test_strip_code_fences_bad_language_raises():
    """A genuine fence wrapper with a non-Python language tag raises SyntaxError."""
    import pytest
    from plm._react_helper import _strip_code_fences
    with pytest.raises(SyntaxError):
        _strip_code_fences("```sql\nSELECT 1\n```")


def test_compute_insert_no_glue_when_prefix_lacks_newline():
    """_compute_insert must not glue `content` onto a prefix line that lacks a
    trailing newline (reachable when a prior _edit stripped the final '\\n')."""
    from plm.policy.edits import _compute_insert
    # last line has NO trailing newline -> insert after it must start a new line
    assert _compute_insert("a = 1\nb = 2", 2, "c = 3\n") == "a = 1\nb = 2\nc = 3\n"
    # normal case (prefix already ends in '\n') unchanged
    assert _compute_insert("a = 1\nb = 2\n", 2, "c = 3\n") == "a = 1\nb = 2\nc = 3\n"
    # mid-file insert still works
    assert _compute_insert("x\ny\n", 1, "INS\n") == "x\nINS\ny\n"
    # #16: content lacking a trailing newline on a MID-source insert must not
    # glue onto the following line (symmetric to the prefix guard).
    assert _compute_insert("a\nb\nc\n", 1, "X") == "a\nX\nb\nc\n"
    # content with no trailing '\n' inserted at end-of-source needs no extra '\n'
    assert _compute_insert("a\nb\n", 2, "X") == "a\nb\nX"


def test_public_trajectory_strips_internal_keys():
    """public_trajectory() drops _-prefixed harness keys, keeps the real
    conversation fields, and deep-copies (mutating the result can't touch the
    source trajectory)."""
    from plm._react_helper import public_trajectory

    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "hi", "reasoning": "think",
         "tool_calls": [{"id": "t1", "type": "function",
                         "function": {"name": "python", "arguments": "{}"}}],
         "_timestamp": 123.0, "_structured": False},
        {"role": "tool", "tool_call_id": "t1", "content": "out", "_timestamp": 124.0},
        {"role": "user", "content": "r", "_plm_meta": "final_round_return_reminder"},
    ]
    pub = public_trajectory(msgs)
    assert all(not any(k.startswith("_") for k in m) for m in pub)   # no _-keys
    assert pub[1]["reasoning"] == "think"                            # real fields kept
    assert pub[1]["tool_calls"][0]["id"] == "t1"
    assert pub[2]["tool_call_id"] == "t1"
    pub[1]["tool_calls"][0]["id"] = "MUT"                            # deep copy
    assert msgs[1]["tool_calls"][0]["id"] == "t1"


def test_int_plm_messages_additive_accumulation():
    """plm_messages_delta APPENDS to the kernel accumulator; the public
    plm_messages is rebound to a fresh copy each cell, so a cell mutating it
    cannot corrupt the accumulation (the additive-seed optimization, C21)."""
    from plm.repl import PythonReplSession
    try:
        s = PythonReplSession(workspace=None, preinstall=("dill",),
                              cell_timeout=10.0, sigint_grace=2.0)
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"could not start kernel session: {e}")
    try:
        def run(code, delta=None):
            return s.execute_cell(code, None, delta)["stdout"].strip()
        assert run("print(len(plm_messages))", [{"role": "user", "content": "a"}]) == "1"
        assert run("print(len(plm_messages))", [{"role": "assistant", "content": "b"}]) == "2"
        assert run("print([m['role'] for m in plm_messages])") == "['user', 'assistant']"  # no delta -> unchanged
        assert run("plm_messages.clear(); print(len(plm_messages))") == "0"               # public copy cleared
        assert run("print(len(plm_messages))", [{"role": "tool", "content": "c"}]) == "3"  # accumulator intact -> 3
    finally:
        s.close()


def test_int_plm_messages_seed_channel():
    """Real kernel, end-to-end for the seed channel + plm_messages. Uses its OWN
    session (not the shared module fixture) so it is isolated from cross-test
    kernel state (e.g. another test shadowing `len` in __main__)."""
    from plm.repl import PythonReplSession
    try:
        s = PythonReplSession(workspace=None, preinstall=("dill",),
                              cell_timeout=10.0, sigint_grace=2.0)
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"could not start kernel session: {e}")
    try:
        # fresh kernel: plm_messages exists and is empty (no NameError pre-seed)
        assert s.execute_cell("print(plm_messages)")["stdout"].strip() == "[]"
        # seed → bound and readable inside the cell
        seed1 = [{"role": "user", "content": "hi"},
                 {"role": "assistant", "content": "yo", "tool_calls": None}]
        r = s.execute_cell(
            "print(len(plm_messages), plm_messages[0]['role'], plm_messages[1]['content'])",
            seed={"plm_messages": seed1})
        assert r["stdout"].strip() == "2 user yo", r
        # re-seed → updates
        r2 = s.execute_cell("print(len(plm_messages))",
                            seed={"plm_messages": seed1 + [{"role": "tool", "content": "x"}]})
        assert r2["stdout"].strip() == "3", r2
        # no seed → previous value untouched
        assert s.execute_cell("print(len(plm_messages))")["stdout"].strip() == "3"
        # general channel: binds ANY name, not just plm_messages
        r3 = s.execute_cell("print(my_obj['k'])", seed={"my_obj": {"k": 42}})
        assert r3["stdout"].strip() == "42", r3
        # protected name: @policy refuses to shadow it (frozen kernel-internal name)
        r4 = s.execute_cell("@policy\ndef plm_messages():\n    return 1\n")
        assert "shadow" in (r4["stderr"] or "").lower(), r4
        # robustness: seeding binds via _repl_g[name]=obj and never touches __main__
        # builtins, so it works even if a prior cell shadowed one (read without len).
        # (MUST be last — it poisons `len` in this session.)
        s.execute_cell("len = 'x'")
        r5 = s.execute_cell("print(plm_messages[0]['role'], plm_messages[1]['content'])",
                            seed={"plm_messages": seed1})
        assert r5["stdout"].strip() == "user yo", r5
    finally:
        s.close()


def test_int_policy_depth_cap_fires_in_kernel(repl):
    """In a REAL kernel (recursion limit raised in PREFIX), a runaway recursive
    @policy trips the UNIFORM policy-depth cap with the clear message — not an
    opaque native 'maximum recursion depth' error. This is the production path
    C13 was about (the cap never fired at the default limit)."""
    from plm.policy.proxy import POLICY_CALL_DEPTH_CAP
    repl.execute_cell("@policy\ndef _deep(n):\n    return _deep(n - 1) if n > 0 else 0\n")
    r = repl.execute_cell(f"_deep({POLICY_CALL_DEPTH_CAP + 50})")
    err = r["stderr"] or ""
    assert "Policy call depth" in err, err[-300:]
    assert "maximum recursion depth" not in err, err[-300:]


# =============================================================================
# Frame transport robustness + answer-render guard
# =============================================================================

def _bare_session():
    """A PythonReplSession built WITHOUT spawning a kernel — for testing the
    execute_cell timeout/desync DECISION logic in isolation. Records SIGINTs
    and respawns; `_read_frame_with_timeout`/`_write_frame` are stubbed per-test."""
    import threading
    from plm.repl.session import PythonReplSession
    s = object.__new__(PythonReplSession)
    s.execution_count = 0
    s._io_lock = threading.Lock()
    s._cell_timeout = 1.0
    s._sigint_grace = 0.5
    s.cached_vars_blob = b""
    s.last_rehydrate_error = None
    s._pending_boot_stderr = None
    s.respawns = 0
    s.signals = []

    class _Proc:
        def send_signal(_inner, sig):
            s.signals.append(sig)
    s._proc = _Proc()
    s._kill_and_respawn = lambda: setattr(s, "respawns", s.respawns + 1)
    s._write_frame = lambda frame: None
    return s


def test_frame4_partial_timeout_respawns_without_retry():
    """#4: a timeout MID-FRAME (stream desynced) must respawn directly — NO
    SIGINT, NO second read on the misaligned socket."""
    from plm.repl.session import _CellTimeout
    s = _bare_session()
    calls = []
    def _read(timeout):
        calls.append(timeout)
        raise _CellTimeout(partial=True)
    s._read_frame_with_timeout = _read
    out = s.execute_cell("x = 1")
    assert s.respawns == 1
    assert s.signals == []                 # no naive SIGINT-retry
    assert len(calls) == 1                  # no second read on a desynced socket
    assert "timed out" in out["stderr"]


def test_frame4_clean_timeout_sigint_then_grace_survives():
    """#4: a timeout at a CLEAN frame boundary tries a graceful SIGINT; if the
    kernel writes a clean result, the SESSION SURVIVES (no respawn)."""
    import signal as _sig
    from plm.repl.session import _CellTimeout
    s = _bare_session()
    seq = [_CellTimeout(partial=False),
           {"type": "result", "stdout": "ok", "vars_blob": b"", "stderr": ""}]
    def _read(timeout):
        item = seq.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item
    s._read_frame_with_timeout = _read
    out = s.execute_cell("slow()")
    assert s.signals == [_sig.SIGINT]      # graceful interrupt attempted
    assert s.respawns == 0                 # session survived
    assert out["stdout"] == "ok"


def test_frame4_decode_error_respawns():
    """#4: a frame decode error (corrupt/misaligned body) respawns instead of
    escaping uncaught."""
    from plm.repl.session import _FrameDecodeError
    s = _bare_session()
    s._read_frame_with_timeout = lambda timeout: (_ for _ in ()).throw(_FrameDecodeError("corrupt"))
    out = s.execute_cell("x = 1")
    assert s.respawns == 1
    assert "preserved" in out["stderr"]


def test_sigint_during_post_exec_snapshot_yields_real_result():
    """Unified SIGINT model (real kernel): SIGINT is armed (-> KeyboardInterrupt) ONLY around the
    cell body, and SIG_IGN in every other phase. So a SIGINT landing AFTER the body finished —
    here fired from inside the snapshot dump itself, deterministically in the P4 finalize window,
    not racing a timeout — is a no-op: the kernel finishes finalizing and delivers the cell's REAL
    RETURN value, with NO respawn and NO false "did NOT execute". (Pre-fix, the KeyboardInterrupt
    escaped `_repl_collect_vars` -> boot_error -> respawn — finding C1, and the whole class of
    P1/P3/P4/loop-back/rehydrate escapes with it.)"""
    try:
        from plm.repl import PythonReplSession
        s = PythonReplSession(workspace=None, preinstall=("dill", "pydantic"),
                              cell_timeout=10.0, sigint_grace=2.0)
    except Exception as e:                                  # pragma: no cover
        pytest.skip(f"could not start kernel session: {e}")
    try:
        out = s.execute_cell(
            "import os, signal\n"
            "class _SigOnDump:\n"
            "    def __reduce__(self):\n"
            "        os.kill(os.getpid(), signal.SIGINT)   # SIGINT DURING the post-exec snapshot\n"
            "        return (dict, ())\n"
            "sig_obj = _SigOnDump()\n"                      # a global -> dumped in P4 -> fires the signal
            "RETURN('real-result')\n")
        assert out.get("type") == "return", out
        assert out.get("return_obj") == "real-result", out
        assert "RE-RUN" not in (out.get("stderr") or ""), out          # no spurious respawn
        # the kernel survived and still serves cells (signal was ignored, not fatal)
        alive = s.execute_cell("print('alive-after-snapshot-sigint')")
        assert "alive-after-snapshot-sigint" in (alive.get("stdout") or ""), alive
    finally:
        s.close()


def test_sigint_armed_only_during_cell_body():
    """During the cell body, SIGINT is armed (`default_int_handler`) so a clean-boundary timeout can
    abort a runaway cell; the kernel-wide default is SIG_IGN. We observe the armed half from inside
    a running cell (the cell body IS the armed phase)."""
    try:
        from plm.repl import PythonReplSession
        s = PythonReplSession(workspace=None, preinstall=("dill",),
                              cell_timeout=10.0, sigint_grace=2.0)
    except Exception as e:                                  # pragma: no cover
        pytest.skip(f"could not start kernel session: {e}")
    try:
        out = s.execute_cell(
            "import signal\n"
            "print(getattr(signal.getsignal(signal.SIGINT), '__name__', '?'))")
        assert "default_int_handler" in (out.get("stdout") or ""), out
    finally:
        s.close()


def _fake_resp_stream(data: bytes):
    class FakeStream:
        def __init__(self, d): self.buf = d
        def fileno(self): return -1            # never used when deadline is None
        def read(self, k):
            chunk = self.buf[:k]; self.buf = self.buf[k:]; return chunk
    return FakeStream(data)


def test_frame4_timeout_none_waits_and_decodes():
    """#4: timeout=None must NOT synthesize a timeout — it blocks and decodes a
    full frame (PLM runs unbounded circuits; 'no timeout' = wait forever)."""
    import pickle, struct
    from plm.repl.session import PythonReplSession
    s = object.__new__(PythonReplSession)
    payload = {"type": "result", "stdout": "hi"}
    body = pickle.dumps(payload)
    s._resp_r = _fake_resp_stream(struct.pack(">I", len(body)) + body)
    assert s._read_frame_with_timeout(None) == payload


def test_frame4_corrupt_body_raises_decode_error():
    """#4: a corrupt body surfaces as `_FrameDecodeError`, never a raw
    unpickling error."""
    import struct
    from plm.repl.session import PythonReplSession, _FrameDecodeError
    s = object.__new__(PythonReplSession)
    body = b"not-a-valid-pickle"
    s._resp_r = _fake_resp_stream(struct.pack(">I", len(body)) + body)
    with pytest.raises(_FrameDecodeError):
        s._read_frame_with_timeout(None)


def test_render_answer_never_raises():
    """#5: `_render_answer` is cosmetic and must never raise — a value whose
    model_dump_json()/str() blows up falls back to a safe placeholder, so a
    validated RETURN is never lost to a display bug."""
    from plm._react_helper import _render_answer

    class StrBoom:
        def __str__(self): raise RuntimeError("boom")

    class DumpBoom:
        def model_dump_json(self): raise ValueError("nope")

    class GoodModel:
        def model_dump_json(self): return '{"ok": true}'

    assert _render_answer(None) == ""
    assert _render_answer(42) == "42"
    assert _render_answer(GoodModel()) == '{"ok": true}'
    assert _render_answer(StrBoom()) == "<unserializable answer: StrBoom>"
    assert _render_answer(DumpBoom()) == "<unserializable answer: DumpBoom>"


# =============================================================================
# Misc batch: budget coercion, edit int-guards, Guard A match/case
#, rename clobber-protection
# =============================================================================

def test_coerce_budget_normalizes():
    """#18: model-supplied budgets coerce to a non-negative int."""
    from plm._react_helper import _coerce_budget
    assert _coerce_budget(None, 5) == 5         # None -> default
    assert _coerce_budget("x", 5) == 5          # non-int -> default
    assert _coerce_budget(True, 5) == 5         # bool excluded -> default
    assert _coerce_budget(-3, 5) == 0           # negative -> clamp to 0
    assert _coerce_budget(0, 5) == 0
    assert _coerce_budget(7, 5) == 7


def test_compute_edit_non_int_line_is_noop():
    """#24: a non-int line arg is a no-op (None), not a raw slice TypeError."""
    from plm.policy.edits import _compute_insert, _compute_delete
    assert _compute_insert("a\nb\n", 2.5, "X\n") is None     # float after_line
    assert _compute_insert("a\nb\n", None, "X\n") is None
    assert _compute_delete("a\nb\n", 1.5, None) is None       # float start
    assert _compute_delete("a\nb\n", 1, 2.5) is None          # float end
    # sanity: valid ints still work
    assert _compute_insert("a\nb\n", 1, "X\n") == "a\nX\nb\n"
    assert _compute_delete("a\nb\n", 1, 1) == "b\n"


def test_guard_a_descends_into_match_case():
    """#15: Guard A must reject a policy rebind hidden inside a match/case body."""
    @policy
    def predict(s):
        return 1
    names = set(_PLM_POLICIES)
    rebind_in_case = "match 1:\n    case 1:\n        predict = 5\n"
    assert _audit_cell(rebind_in_case, names)          # flagged (truthy error string)
    # sanity: a benign match (no policy rebind) is NOT flagged
    assert not _audit_cell("match 1:\n    case _:\n        y = 2\n", names)


def test_rename_refuses_clobbering_plain_global(capsys):
    """#13: renaming a policy onto an existing plain __main__ global is refused
    (note, no rename), so the variable isn't silently overwritten."""
    @policy
    def predict(s):
        return 1
    main = sys.modules["__main__"].__dict__
    main["existing_var"] = 123
    try:
        capsys.readouterr()
        predict._edit("def predict", "def existing_var")    # rename via edit -> target exists
        err = capsys.readouterr().err
        assert "refusing to rename" in err
        assert "predict" in _PLM_POLICIES and "existing_var" not in _PLM_POLICIES
        assert main["existing_var"] == 123                   # NOT clobbered
    finally:
        main.pop("existing_var", None)


def test_int_extra_policy_failure_soft_and_surfaced():
    """#26: a broken EXTRA policy must NOT brick boot (stays soft) and its
    traceback is surfaced once on the first cell's stderr (not lost to the reset
    boot buffer). A broken DEFAULT, by contrast, is a hard boot_error -> the
    parent raises (can't be exercised here without breaking a bundled default)."""
    import json
    try:
        from plm.repl import PythonReplSession
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"repl import failed: {e}")
    bad = {"bad_extra": "@policy\ndef __bad_extra():\n    return 1\n"}   # reserved __ prefix -> @policy raises
    try:
        s = PythonReplSession(workspace=None, preinstall=("dill",), cell_timeout=10.0,
                              env={"_PLM_EXTRA_POLICIES": json.dumps(bad)})
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"could not start kernel session: {e}")
    try:
        r = s.execute_cell("print('booted', 'react_llm' in list_policies())")
        assert "booted True" in r["stdout"], r          # defaults present; broken extra didn't brick boot
        assert "boot: extra-policy install warning" in r["stderr"], r   # surfaced once
    finally:
        s.close()


def test_d3_extra_colliding_with_sealed_default_rejected():
    """D3: an extra policy named like a SEALED default (react_llm/...) is REJECTED at
    boot — it can't re-decorate the default BEFORE the seal loop and get sealed as
    operator code. The collision is surfaced and the canonical default body is kept."""
    import json
    try:
        from plm.repl import PythonReplSession
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"repl import failed: {e}")
    bad = {"react_llm": "@policy\ndef react_llm(*a, **k):\n    return 'HIJACKED'\n"}
    try:
        s = PythonReplSession(workspace=None, preinstall=("dill",), cell_timeout=10.0,
                              env={"_PLM_EXTRA_POLICIES": json.dumps(bad)})
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"could not start kernel session: {e}")
    try:
        r = s.execute_cell("print('HIJACK' in read_policy('react_llm'))")
        assert r["stdout"].strip() == "False", r        # the extra's body did NOT replace the default
        assert "collides with a sealed default" in r["stderr"], r   # rejection surfaced
    finally:
        s.close()


def test_sec3_agent_depth_env_poison_ignored():
    """WF Sec#3: the root depth is seeded ONCE at boot, so a cell reassigning
    os.environ['AGENT_DEPTH'] mid-session can't raise the depth ceiling."""
    try:
        from plm.repl import PythonReplSession
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"repl import failed: {e}")
    try:
        s = PythonReplSession(workspace=None, preinstall=("dill",), cell_timeout=10.0,
                              env={"AGENT_DEPTH": "2"})
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"could not start kernel session: {e}")
    try:
        r = s.execute_cell(
            "import os\n"
            "from plm.policy.defaults import _llm_infra\n"
            "os.environ['AGENT_DEPTH'] = '999'\n"        # try to poison the ceiling
            "print(_llm_infra._remaining())")
        assert r["stdout"].strip() == "2", r            # still the boot value, NOT 999
    finally:
        s.close()


def test_safe_describe_never_raises():
    """#6: _safe_describe falls back to a placeholder instead of letting a
    describe() failure escape the error/prompt/stderr formatting paths (which
    would abort the task, violating the never-escape contract)."""
    from plm._react_helper import _safe_describe

    class Boom:
        def describe(self):
            raise RuntimeError("describe blew up")

    class Good:
        def describe(self):
            return "ok-desc"

    assert _safe_describe(Boom()) == "<constraint description unavailable>"
    assert _safe_describe(Good()) == "ok-desc"


def test_plmmetaparameters_rejects_non_dict_extra_policies():
    """#2 (parent fail-fast): a non-dict extra_policies is caught at construction
    with a clear TypeError, not shipped to the kernel to brick boot."""
    from plm.plm import PLMMetaParameters
    PLMMetaParameters(system_prompt="x")                       # None ok
    PLMMetaParameters(system_prompt="x", extra_policies={"n": "@policy\ndef n(): ...\n"})  # dict[str,str] ok
    for bad in (["a"], "foo", 123, {"n": 1}):                  # list/str/int/dict[str,int]
        with pytest.raises(TypeError):
            PLMMetaParameters(system_prompt="x", extra_policies=bad)


def test_metaparams_from_dir_loads_sealed_and_mutable():
    """Folder metaparam: from_dir reads system_prompt.md + policies/sealed/*.py (-> sealed_policies)
    + policies/mutable/*.py (-> extra_policies), keyed by file stem."""
    import plm
    import pathlib
    from plm.plm import PLMMetaParameters
    base = pathlib.Path(plm.__path__[0]) / "metaparams" / "example"
    if not base.is_dir():
        pytest.skip("example metaparam folder not present")
    mp = PLMMetaParameters.from_dir(base)
    assert "example" in mp.system_prompt
    assert sorted(mp.extra_policies or {}) == ["editable_helper"]      # mutable bucket
    assert sorted(mp.sealed_policies or {}) == ["locked_helper"]       # sealed bucket
    assert "@policy" in mp.sealed_policies["locked_helper"]


def test_metaparams_name_in_both_buckets_rejected():
    """A name in BOTH extra_policies (mutable) and sealed_policies (immutable) is ambiguous -> reject."""
    from plm.plm import PLMMetaParameters
    PLMMetaParameters(system_prompt="x", sealed_policies={"s": "@policy\ndef s(): ...\n"})   # sealed-only ok
    with pytest.raises(ValueError, match="both|exactly one"):
        PLMMetaParameters(system_prompt="x",
                          extra_policies={"foo": "@policy\ndef foo(): ...\n"},
                          sealed_policies={"foo": "@policy\ndef foo(): ...\n"})


def test_metaparams_name_comes_from_def_not_key():
    """NP3-4 (API side): a policy's name is the @policy def/class name parsed from its source,
    NEVER the dict key / file stem. PLMMetaParameters re-keys both buckets to the real name,
    detects two same-named policies, and rejects a source that isn't exactly one top-level def."""
    from plm.plm import PLMMetaParameters as MP
    mp = MP(system_prompt="x", sealed_policies={"auth": "@policy\ndef authenticate(x):\n    return x\n"})
    assert list(mp.sealed_policies) == ["authenticate"]            # key 'auth' re-keyed to def name
    mp2 = MP(system_prompt="x",
             extra_policies={"k": "@policy\nclass Doubler:\n    def __call__(self, x):\n        return x * 2\n"})
    assert list(mp2.extra_policies) == ["Doubler"]                 # class name too
    with pytest.raises(ValueError, match="exactly ONE top-level"):
        MP(system_prompt="x", extra_policies={"x": "y = 5\n"})     # no def/class
    with pytest.raises(ValueError, match="both define"):           # two files, same def name
        MP(system_prompt="x", extra_policies={"a": "@policy\ndef h(x):\n    return x\n",
                                              "b": "@policy\ndef h(x):\n    return x\n"})


def test_int_sealed_extra_policy_locked_unduplicable_unblessed():
    """metaparams policies/sealed/: a sealed extra installs immutable + un-duplicable + NOT blessed
    (no raw LLM-primitive access), survives crash-restart still sealed; a mutable extra stays
    editable. Driven directly via the env vars PLM's `from_dir` populates."""
    try:
        from plm.repl import PythonReplSession
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"repl import failed: {e}")
    import json
    env = {
        "_PLM_EXTRA_POLICIES": json.dumps({"editable": "@policy\ndef editable(x):\n    return x + 1\n"}),
        "_PLM_SEALED_EXTRA_POLICIES": json.dumps({
            "locked": "@policy\ndef locked(x):\n    return x * 2\n",
            "peeker": ("@policy\ndef peeker():\n"
                       "    from plm.policy.defaults._llm_infra import descend\n"
                       "    descend()\n    return 'leaked'\n"),
        }),
    }
    try:
        s = PythonReplSession(workspace=None, preinstall=("dill",), cell_timeout=15.0,
                              sigint_grace=2.0, env=env)
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"could not start kernel: {e}")
    try:
        assert s.execute_cell("print(locked(5), editable(5))")["stdout"].strip() == "10 6"
        s.execute_cell("locked._rewrite('def locked(x):\\n    return 0\\n')")           # immutable
        assert s.execute_cell("print(locked(5))")["stdout"].strip() == "10"
        r = s.execute_cell("duplicate_policy('locked', 'forked'); print('forked' in list_policies())")
        assert r["stdout"].strip() == "False"                                            # un-duplicable
        r2 = s.execute_cell("print(peeker())")                                           # NOT blessed
        assert "leaked" not in r2["stdout"] and r2["stderr"].strip() != ""
        s.execute_cell("editable._rewrite('def editable(x):\\n    return x * 100\\n')")  # mutable -> editable
        assert s.execute_cell("print(editable(5))")["stdout"].strip() == "500"
        ep = s.kernel_epoch                                                              # survives crash-restart
        s.execute_cell("import os as _o\n_o._exit(0)")
        assert s.kernel_epoch > ep
        assert s.execute_cell("print(locked(5))")["stdout"].strip() == "10"
        s.execute_cell("locked._rewrite('def locked(x):\\n    return 0\\n')")
        assert s.execute_cell("print(locked(5))")["stdout"].strip() == "10"              # still immutable
    finally:
        s.close()


def test_int_sealed_class_extra_resists_flag_flip():
    """NP3-1: a sealed CLASS extra is a raw type whose `_p_immutable` is freely writable (unlike a
    function proxy, which write-locks it). The seal gates + `_is_default` must anchor to the
    canonical registry (`_is_sealed_obj`, identity-based), so flipping the flag cannot un-seal it —
    rewrite, remove, AND duplicate stay refused. Normal class policies remain fully mutable."""
    try:
        from plm.repl import PythonReplSession
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"repl import failed: {e}")
    import json
    env = {"_PLM_SEALED_EXTRA_POLICIES": json.dumps({
        "Locked": "@policy\nclass Locked:\n    def __call__(self, x):\n        return x * 2\n"})}
    try:
        s = PythonReplSession(workspace=None, preinstall=("dill",), cell_timeout=15.0,
                              sigint_grace=2.0, env=env)
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"could not start kernel: {e}")
    try:
        assert s.execute_cell("print(Locked()(5))")["stdout"].strip() == "10"
        s.execute_cell("Locked._p_immutable = False")                    # the attack: flip the writable flag
        assert s.execute_cell("print(Locked._p_immutable)")["stdout"].strip() == "False"   # flag DID flip...
        s.execute_cell("Locked._rewrite('class Locked:\\n    def __call__(self, x):\\n        return 999\\n')")
        assert s.execute_cell("print(Locked()(5))")["stdout"].strip() == "10"              # ...but rewrite blocked
        r = s.execute_cell("Locked._remove(); print('Locked' in list_policies())")
        assert r["stdout"].strip() == "True"                                               # remove blocked
        r2 = s.execute_cell("duplicate_policy('Locked','Fork'); print('Fork' in list_policies())")
        assert r2["stdout"].strip() == "False"                                             # duplicate blocked
        # R4: RE-DECORATION (@policy on the same name, the kind-change path) is also blocked even with
        # the flag flipped — the decorator pre-check now uses _is_default (the _is_sealed_obj anchor).
        s.execute_cell("@policy\nclass Locked:\n    def __call__(self, x):\n        return 777\n")
        assert s.execute_cell("print(Locked()(5))")["stdout"].strip() == "10"              # re-decoration ignored
        # no false positive: a NORMAL class policy is still mutable
        s.execute_cell("@policy\nclass Free:\n    def __call__(self, x):\n        return x + 1")
        s.execute_cell("Free._rewrite('class Free:\\n    def __call__(self, x):\\n        return x + 100\\n')")
        assert s.execute_cell("print(Free()(5))")["stdout"].strip() == "105"
    finally:
        s.close()


def test_int_sealed_extra_seals_real_name_not_stem():
    """NP3-4: from_dir keys sealed_policies by FILE STEM, but @policy registers by the def-name.
    The kernel seals the names ACTUALLY installed (the install delta), not the transport key — so a
    file `auth.py` defining `def authenticate(...)` seals `authenticate`, never the phantom `auth`."""
    try:
        from plm.repl import PythonReplSession
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"repl import failed: {e}")
    import json
    env = {"_PLM_SEALED_EXTRA_POLICIES": json.dumps({       # key 'auth' != def-name 'authenticate'
        "auth": "@policy\ndef authenticate(x):\n    return x * 2\n"})}
    try:
        s = PythonReplSession(workspace=None, preinstall=("dill",), cell_timeout=15.0,
                              sigint_grace=2.0, env=env)
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"could not start kernel: {e}")
    try:
        assert s.execute_cell("print('authenticate' in list_policies())")["stdout"].strip() == "True"
        assert s.execute_cell("print('auth' in list_policies())")["stdout"].strip() == "False"   # no phantom
        s.execute_cell("authenticate._rewrite('def authenticate(x):\\n    return 999\\n')")
        assert s.execute_cell("print(authenticate(5))")["stdout"].strip() == "10"                 # real name sealed
        r = s.execute_cell("duplicate_policy('authenticate','f'); print('f' in list_policies())")
        assert r["stdout"].strip() == "False"                                                     # un-duplicable
    finally:
        s.close()


def test_int_sealed_extra_poison_force_restored_on_rehydrate():
    """NR3-7: the rehydrate subscript-poisoning defense force-restores EVERY sealed policy from its
    fresh-PREFIX v0 — `_SEALED_POLICIES`, not just `_LLM_DEFAULT_POLICIES`. A sealed EXTRA whose
    registry entry was poisoned (via `dict.__setitem__`, which bypasses the store's protection and
    Guard A) must come back as v0 after crash-restart."""
    try:
        from plm.repl import PythonReplSession
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"repl import failed: {e}")
    import json
    env = {"_PLM_SEALED_EXTRA_POLICIES": json.dumps({
        "locked": "@policy\ndef locked(x):\n    return x * 2\n"})}
    try:
        s = PythonReplSession(workspace=None, preinstall=("dill",), cell_timeout=15.0,
                              sigint_grace=2.0, env=env)
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"could not start kernel: {e}")
    try:
        assert s.execute_cell("print(locked(5))")["stdout"].strip() == "10"
        s.execute_cell("dict.__setitem__(_PLM_POLICIES, 'locked', lambda x: 999)")    # poison
        assert s.execute_cell("print(_PLM_POLICIES['locked'](5))")["stdout"].strip() == "999"
        ep = s.kernel_epoch
        s.execute_cell("import os as _o\n_o._exit(0)")                                # crash
        assert s.kernel_epoch > ep
        assert s.execute_cell("print(_PLM_POLICIES['locked'](5))")["stdout"].strip() == "10"  # v0 restored
    finally:
        s.close()


def test_int_extra_policies_non_dict_payload_is_soft():
    """#2 (kernel soft-guard): a valid-JSON-but-not-object _PLM_EXTRA_POLICIES
    (set directly in env, bypassing the parent check) must NOT brick boot — it's
    ignored and noted via boot_stderr."""
    try:
        from plm.repl import PythonReplSession
        s = PythonReplSession(workspace=None, preinstall=("dill",), cell_timeout=10.0,
                              env={"_PLM_EXTRA_POLICIES": "[]"})
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"could not start kernel session: {e}")
    try:
        r = s.execute_cell("print('booted', 'react_llm' in list_policies())")
        assert "booted True" in r["stdout"], r                 # did not brick boot
        assert "not an object" in r["stderr"], r               # ignored + noted (#26 surface)
    finally:
        s.close()


def test_int_extra_policies_invalid_json_is_soft_and_noted():
    """NR-9: a MALFORMED-JSON _PLM_EXTRA_POLICIES (set directly in env) must NOT brick boot — like
    the not-an-object sibling it's ignored and surfaced via boot_stderr (was swallowed silently)."""
    try:
        from plm.repl import PythonReplSession
        s = PythonReplSession(workspace=None, preinstall=("dill",), cell_timeout=10.0,
                              env={"_PLM_EXTRA_POLICIES": "{not valid json"})
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"could not start kernel session: {e}")
    try:
        r = s.execute_cell("print('booted', 'react_llm' in list_policies())")
        assert "booted True" in r["stdout"], r                 # did not brick boot
        assert "not valid JSON" in r["stderr"], r              # ignored + noted
    finally:
        s.close()


def test_int_refused_redecoration_no_linecache_poison(repl):
    """#4: a refused re-decoration of an immutable default must NOT write the
    rejected source into the <policy-name> linecache slot (getsource/traceback
    fidelity)."""
    repl.execute_cell("@policy\ndef natural_llm():\n    return 'HACKED_SENTINEL'\n")  # refused (immutable)
    r = repl.execute_cell(
        "import linecache; _e = linecache.cache.get('<policy-natural_llm>'); "
        "print('HACKED_SENTINEL' in ''.join(_e[2]) if _e else 'NOSLOT')"
    )
    assert r["stdout"].strip() in ("False", "NOSLOT"), r       # slot absent or not poisoned


def test_redecoration_function_getsource_still_synced():
    """#4 regression guard: re-decorating a mutable FUNCTION policy via @policy
    (same-kind in-place) keeps linecache synced — getsource shows the NEW source.
    After deferring the top-level linecache write, this path relies SOLELY on
    _rewrite syncing it (proxy.py:257)."""
    import inspect

    @policy
    def foo():
        return 1

    @policy
    def foo():
        return 2

    src = inspect.getsource(foo._inner)
    assert "return 2" in src and "return 1" not in src      # NEW source, not stale
    assert foo() == 2


def test_frame_respawn_surfaces_pending_boot_stderr():
    """#10: a pending boot/rehydrate diagnostic must be surfaced on a RESPAWN
    early-return, not lost when _kill_and_respawn overwrites the field."""
    from plm.repl.session import _CellTimeout
    s = _bare_session()
    s._pending_boot_stderr = "EXTRA_POLICY_BOOM"
    s._read_frame_with_timeout = lambda timeout: (_ for _ in ()).throw(_CellTimeout(partial=True))
    out = s.execute_cell("x = 1")
    assert s.respawns == 1
    assert "EXTRA_POLICY_BOOM" in out["stderr"]      # surfaced on the respawn path
    assert s._pending_boot_stderr is None            # captured + cleared (no double-surface)


def test_compute_insert_respects_unicode_line_boundaries():
    """#11: a line ending in a non-\\n/\\r splitlines boundary (\\u2028) must NOT
    get a spurious '\\n' appended by the newline guard."""
    from plm.policy.edits import _compute_insert
    # prefix already ends on a  boundary -> no extra '\n' before content
    assert _compute_insert("a\u2028b\u2028", 1, "X\n") == "a\u2028X\nb\u2028"
    # content ending on a  boundary -> no extra '\n' before the tail
    assert _compute_insert("p\nq\n", 1, "X\u2028") == "p\nX\u2028q\n"


def test_int_close_resets_pid_box():
    """#R4-6: after close() reaps the child, the pid box is cleared so the
    GC/atexit finalizer can't SIGKILL a recycled PID."""
    try:
        from plm.repl import PythonReplSession
        s = PythonReplSession(workspace=None, preinstall=("dill",), cell_timeout=10.0)
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"kernel unavailable: {e}")
    assert s._child_pid_box[0] is not None              # set on spawn
    s.close()
    assert s._child_pid_box[0] is None                  # cleared after reap


def test_int_respawn_drops_deleted_boot_policy_orphan():
    """#R5-2: deleting a MUTABLE boot policy (base_verifier) then surviving a hard
    respawn must leave it deleted in BOTH the registry AND __main__. The fresh
    PREFIX re-installs every default into __main__ on respawn; rehydrate reconciles
    only the registry, so without the orphan-reap the 'deleted' policy reappears in
    __main__ as a live callable global invisible to list/get/read/edit_policy and
    Guard C. The reap is identity-checked, so it never clobbers an immutable
    default that was force-restored, nor a user value re-bound to that name."""
    try:
        from plm.repl import PythonReplSession
        s = PythonReplSession(workspace=None, preinstall=("dill",), cell_timeout=10.0)
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"could not start kernel session: {e}")
    try:
        r0 = s.execute_cell("print('base_verifier' in list_policies())")
        if r0["stdout"].strip() != "True":              # pragma: no cover
            pytest.skip(f"base_verifier default not present: {r0}")
        s.execute_cell("delete_policy('base_verifier')")
        r1 = s.execute_cell(
            "import sys as _s; _g = _s.modules['__main__'].__dict__; "
            "print('base_verifier' in list_policies(), 'base_verifier' in _g)"
        )
        assert r1["stdout"].strip() == "False False", r1   # gone from both, pre-respawn
        # Hard-kill the kernel -> EOF -> SIGKILL + respawn + rehydrate from the
        # cached snapshot (which lacks base_verifier). os._exit gives a deterministic
        # respawn (a SIGINT-interruptible sleep could recover WITHOUT respawning).
        s.execute_cell("import os as _o9\n_o9._exit(0)")
        r2 = s.execute_cell(
            "import sys as _s; _g = _s.modules['__main__'].__dict__; "
            "print('base_verifier' in list_policies(), 'base_verifier' in _g, "
            "'natural_llm' in list_policies(), 'natural_llm' in _g)"
        )
        # base_verifier reaped from BOTH; the immutable natural_llm survives in both
        # (force-restored, identity-distinct from base_verifier -> not over-reaped).
        assert r2["stdout"].strip() == "False False True True", r2
    finally:
        s.close()


def test_int_late_sigint_at_idle_read_does_not_drop_next_cell():
    """#R6-6: a late SIGINT delivered while the kernel is IDLE at the top-level
    read (the parent sent it on a clean-boundary timeout, but the cell had already
    finished and reported) must be swallowed and the read re-entered — NOT turned
    into a boot_error that kills the kernel and silently drops the next cell."""
    import os
    import signal
    import time
    try:
        from plm.repl import PythonReplSession
        s = PythonReplSession(workspace=None, preinstall=("dill",), cell_timeout=10.0)
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"could not start kernel session: {e}")
    try:
        s.execute_cell("x = 41")                        # kernel now idle at the top-level read
        pid = s._child_pid_box[0]
        ep = s.kernel_epoch                             # respawns bump kernel_epoch (+ the pid)
        assert pid is not None
        os.kill(pid, signal.SIGINT)                     # late SIGINT while idle
        time.sleep(0.2)                                 # let it deliver + be swallowed
        r = s.execute_cell("print(x + 1)")              # the NEXT cell must still run
        assert r["stdout"].strip() == "42", r           # not dropped (with the bug: "")
        assert s.kernel_epoch == ep, r                  # no respawn (no spurious boot_error)
        assert s._child_pid_box[0] == pid, r            # same live kernel process
    finally:
        s.close()


@pytest.mark.parametrize("bad_args", [
    "1" + "0" * 5001,                       # huge int literal -> ValueError (int_max_str_digits)
    "[" * 100000 + "]" * 100000,            # deeply-nested JSON -> RecursionError
])
def test_root_loop_survives_unparseable_tool_args(monkeypatch, bad_args):
    """#H13/#H14: tool-args that raise a NON-JSONDecodeError on json.loads (a huge
    integer literal -> ValueError; deeply-nested JSON -> RecursionError) must coerce
    to {} and exhaust GRACEFULLY (PLMTaskFailure), not escape PLM.__call__ as a raw
    exception that aborts the task. A fake REPL keeps this in-process."""
    import asyncio
    from plm.plm import PLM, PLMMetaParameters, PLMTaskFailure

    class _FakeRepl:
        kernel_epoch = 0
        def __init__(self, **kw): pass
        def execute_cell(self, code, _seed, _delta):
            return {"type": "result", "stdout": "", "stderr": ""}
        def close(self): pass

    monkeypatch.setattr("plm.plm.PythonReplSession", _FakeRepl)

    class _BadArgsBackend:
        model = "stub"
        def __init__(self): self.calls = 0
        async def generate(self, messages=None, tools=None, **kw):
            self.calls += 1
            return {
                "content": "",
                "tool_calls": [{"id": f"t{self.calls}", "type": "function",
                                "function": {"name": "python", "arguments": bad_args}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

    backend = _BadArgsBackend()
    plm = PLM(model_backend=backend, metaparams=PLMMetaParameters(system_prompt="s"),
              max_turns=2, return_budget=0)
    with pytest.raises(PLMTaskFailure):                  # graceful, NOT raw ValueError/RecursionError
        asyncio.run(plm([{"role": "user", "content": "go"}]))
    assert backend.calls >= 1


def test_int_guard_c_survives_repl_g_rebind():
    """#H9: a cell rebinding the bare `_repl_g` __main__ name must NOT make Guard C
    restore into a decoy dict — it re-derives the REAL globals from a trusted
    builtins, so an in-cell hijack of an immutable default is still reverted."""
    try:
        from plm.repl import PythonReplSession
        s = PythonReplSession(workspace=None, preinstall=("dill",), cell_timeout=10.0)
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"could not start kernel session: {e}")
    try:
        # Hijack natural_llm via subscript (Guard A skips ast.Subscript) AND rebind
        # the loop's `_repl_g` to a decoy, in the SAME cell.
        s.execute_cell("globals()['natural_llm'] = 'HIJACKED'\n_repl_g = {}\n")
        r = s.execute_cell(
            "print(_PLM_POLICIES['natural_llm'] is natural_llm, 'natural_llm' in list_policies())")
        assert r["stdout"].strip() == "True True", r     # restored to the real proxy (not 'HIJACKED')
    finally:
        s.close()


def test_int_post_exec_classify_ignores_cell_builtins_fake():
    """#H10: the post-exec terminal classifier uses a FRESHLY re-imported builtins,
    so a cell rebinding `_builtins` to a fake whose `.type` lies (to mask its own
    exception as a silent RETURN) cannot hide the real error."""
    try:
        from plm.repl import PythonReplSession
        s = PythonReplSession(workspace=None, preinstall=("dill",), cell_timeout=10.0)
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"could not start kernel session: {e}")
    try:
        cell = (
            "import builtins as _rb\n"
            "class _R(BaseException): pass\n"
            "_R.__name__ = '_REPLReturn'\n"                              # fake exc type named as a RETURN
            "class _Fake:\n"
            "    def __getattr__(self, n): return getattr(_rb, n)\n"     # proxy everything real...
            "    def type(self, x): return _R if isinstance(x, BaseException) else _rb.type(x)\n"  # ...but lie
            "_builtins = _Fake()\n"
            "raise ZeroDivisionError('real-error')\n"
        )
        r = s.execute_cell(cell)
        assert r.get("type") != "return", r              # NOT masked into a silent RETURN
        assert "ZeroDivisionError" in r["stderr"] and "real-error" in r["stderr"], r
        r2 = s.execute_cell("print('alive')")
        assert r2["stdout"].strip() == "alive", r2       # kernel still healthy
    finally:
        s.close()


def test_int_class_policy_survives_respawn_and_deleted_stays_deleted():
    """#H11/#H12: a hard respawn must (a) keep an authored CLASS policy alive (via
    source-rebuild — it can't pickle by value), (b) keep an authored function
    policy, and (c) keep a deleted mutable default deleted. Previously a single
    unpicklable class policy sank the WHOLE registry snapshot, losing everything and
    resurrecting the deleted default."""
    try:
        from plm.repl import PythonReplSession
        s = PythonReplSession(workspace=None, preinstall=("dill",), cell_timeout=15.0)
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"could not start kernel session: {e}")
    try:
        s.execute_cell(
            "@policy\nclass MyVerifier:\n    def __call__(self, messages):\n        return 'CLASS-OK'\n")
        s.execute_cell("@policy\ndef helper(x):\n    return x * 2\n")
        s.execute_cell("delete_policy('base_verifier')")
        s.execute_cell("import os as _o\n_o._exit(0)")    # hard kill -> respawn + rehydrate
        r = s.execute_cell(
            "import sys; g = sys.modules['__main__'].__dict__\n"
            "print(MyVerifier()(['m']), helper(5), "
            "'base_verifier' not in _PLM_POLICIES, 'base_verifier' not in g, "
            "'natural_llm' in _PLM_POLICIES)\n"
        )
        # class survives (CLASS-OK) + function survives (10) + deleted stays gone + immutable kept
        assert r["stdout"].strip() == "CLASS-OK 10 True True True", r
    finally:
        s.close()


def test_policyresult_enum_outcomes_on_edit_and_duplicate():
    """The edit/duplicate API ALWAYS returns a result object so PLM writes ONE handling pattern: an
    in-place op returns PolicyResult (OK on success, a descriptive failure status otherwise);
    duplicate ALWAYS returns PolicyValueResult (.value = the new policy on success). `if r: ... else:
    handle(r.status)` reads the same everywhere; the `[policy]` note still fires as the backstop on a
    problem. This lets cell code HANDLE a failure (try-retry) — impossible with the note-only path."""
    from plm.repl import PythonReplSession
    s = PythonReplSession(workspace=None, preinstall=("dill",), cell_timeout=15.0, sigint_grace=2.0)
    try:
        s.execute_cell("@policy\ndef greet(x):\n    return 'hi ' + x\n")
        # no-match: falsy PolicyResult(NO_MATCH) + note backstop + source UNCHANGED
        r = s.execute_cell("res = greet._edit('NOPE','X')\n"
                           "print(res.status.name, bool(res), repr(greet('bob')))")
        assert r["stdout"].split()[:2] == ["NO_MATCH", "False"]
        assert "'hi bob'" in r["stdout"]                      # source unchanged (edit did NOT apply)
        assert "not found" in r["stderr"]                     # note backstop fired
        # the try-retry pattern the enum unlocks (impossible with the old note-only path)
        r = s.execute_cell("for old in ['Hello ', 'hi ']:\n"
                           "    if greet._edit(old, 'hey '):\n        break\n"
                           "print(greet('bob'))")
        assert r["stdout"].strip() == "hey bob"
        # clean in-place success -> PolicyResult(OK), truthy (NOT None — uniform shape)
        assert s.execute_cell("r = greet._edit('hey ','HEY ')\n"
                              "print(r.status.name, bool(r))")["stdout"].strip() == "OK True"
        # descriptive failure statuses
        for code, want in [
            ("greet._edit(None, 'x')",                          "BAD_ARGS"),
            ("greet._rewrite('def greet(x): return (')",        "SYNTAX_ERROR"),
            ("greet._rewrite('x=1\\ndef greet(x): return x')",  "NOT_ONE_DEF"),
            ("natural_llm._rewrite('def natural_llm(): pass')", "IMMUTABLE"),
            ("duplicate_policy('nope', 'z')",                   "NOT_FOUND"),
            ("duplicate_policy('greet', 'list_policies')",      "NAME_INVALID"),
        ]:
            out = s.execute_cell(f"print(({code}).status.name)")["stdout"].strip()
            assert out == want, f"{code} -> {out!r} (want {want})"
        # clean duplicate success -> PolicyValueResult(OK); .value is the new policy
        r = s.execute_cell("r2 = duplicate_policy('greet', 'greet2')\n"
                           "print(r2.value('z'), bool(r2), r2.status.name)")
        assert r["stdout"].strip() == "HEY z True OK"         # .value is the policy; OK + truthy
    finally:
        s.close()


def test_policyresult_uniform_across_class_and_by_name():
    """A1/A2: the PolicyResult edit contract must be IDENTICAL across EVERY policy shape — function
    methods, class-policy methods, AND the by-name helpers — because they all route through the one
    shared edit core (_do_edit/_do_insert/_do_delete). Regression: a class-policy edit used to raise a
    raw TypeError on bad args, silently swallow a no-match (no note), and discard the rewrite result;
    the by-name helpers always returned None and raised KeyError on an unknown name."""
    from plm.repl import PythonReplSession
    s = PythonReplSession(workspace=None, preinstall=("dill",), cell_timeout=15.0, sigint_grace=2.0)

    def out(c):
        return s.execute_cell(c)["stdout"].strip()

    try:
        s.execute_cell("@policy\nclass Net:\n    def __call__(self, x):\n        return x * 2\n")
        s.execute_cell("@policy\ndef greet(x):\n    return 'hi'\n")
        # CLASS-policy methods now mirror the function contract exactly
        assert out("print(Net._edit(123, 'y').status.name)") == "BAD_ARGS"        # was a raw TypeError
        r = s.execute_cell("print(Net._edit('NOPE', 'y').status.name)")
        assert r["stdout"].strip() == "NO_MATCH" and "not found" in r["stderr"]   # was silent None, no note
        assert out("print(Net._insert(0, 999).status.name)") == "BAD_ARGS"        # non-str content
        assert out("print(Net._edit('return x * 2', 'return x * 3').status.name)") == "OK"   # success -> OK (truthy)
        # BY-NAME helpers now propagate the result + map an unknown name -> NOT_FOUND (not KeyError)
        assert out("print(edit_policy('greet', 'NOPE', 'y').status.name)") == "NO_MATCH"
        assert out("print(edit_policy('does_not_exist', 'a', 'b').status.name)") == "NOT_FOUND"
        assert out("print(insert_into('greet', 0, 123).status.name)") == "BAD_ARGS"
        assert out("print(bool(edit_policy('greet', \"'hi'\", \"'yo'\")), greet(0))") == "True yo"   # OK + applied
    finally:
        s.close()


def test_registry_read_only_to_model_main_loop_and_branch():
    """H1+: the policy registry is READ-ONLY to cell/model code. A bare `_PLM_POLICIES.pop()/.clear()/
    [x]=` is refused in the main loop AND inside a `parallel()` branch (where it previously silently
    wiped the SHARED registry). Deletion/authoring go through the sanctioned API only, which still
    works. Reads (`list_policies`, `_PLM_POLICIES[x]`) stay free."""
    try:
        from plm.repl import PythonReplSession
        s = PythonReplSession(workspace=None, preinstall=("dill", "pydantic"),
                              cell_timeout=10.0, sigint_grace=2.0)
    except Exception as e:                                  # pragma: no cover
        pytest.skip(f"could not start kernel session: {e}")
    try:
        s.execute_cell("@policy\ndef alpha(x): return x")
        out = s.execute_cell(
            "try:\n"
            "    _PLM_POLICIES.pop('alpha')\n"
            "    direct = 'ALLOWED'\n"
            "except TypeError:\n"
            "    direct = 'refused'\n"
            "r = parallel(lambda: _PLM_POLICIES.pop('alpha', 'M'), lambda: _PLM_POLICIES.clear())\n"
            "branch_refused = all(isinstance(x, BaseException) for x in r)\n"
            "print(direct, branch_refused, 'alpha' in list_policies())")
        assert out.get("stdout", "").strip() == "refused True True", out
        # sanctioned API still works
        out = s.execute_cell("print(bool(delete_policy('alpha')), 'alpha' not in list_policies())")
        assert out.get("stdout", "").strip() == "True True", out
    finally:
        s.close()


def test_with_edit_grants_work_for_class_policies():
    """H2 + M1 (root: class policies now carry the uniform `_p_name`): a CLASS policy can be granted
    an IN-PLACE edit in a `parallel()` branch — `with_edit` no longer rejects classes (policy-hood is
    checked kind-agnostically via `_p_source`/`_rewrite`), and `_rename_guard` no longer mis-fires on
    a class in-place edit. An ungranted class edit is still refused; a rename is still refused."""
    try:
        from plm.repl import PythonReplSession
        s = PythonReplSession(workspace=None, preinstall=("dill", "pydantic"),
                              cell_timeout=10.0, sigint_grace=2.0)
    except Exception as e:                                  # pragma: no cover
        pytest.skip(f"could not start kernel session: {e}")
    try:
        s.execute_cell("@policy\nclass Net:\n    def __call__(self, x):\n        return x + 1")
        out = s.execute_cell(
            "has_name = (getattr(Net, '_p_name', None) == 'Net')\n"
            "g, _ = parallel(with_edit(lambda: Net._edit('x + 1', 'x + 100'), Net), lambda: 0)\n"
            "granted = (not isinstance(g, BaseException)) and bool(g) and Net()(5) == 105\n"
            "u = parallel(lambda: Net._edit('x + 100', 'x + 9'))\n"
            "ungranted_refused = type(u[0]).__name__ == 'ParallelMutationError'\n"
            "rn = parallel(with_edit(lambda: Net._rewrite("
            "    'class R:\\n    def __call__(self, x):\\n        return x'), Net))\n"
            "rename_refused = type(rn[0]).__name__ == 'ParallelMutationError' and 'Net' in list_policies()\n"
            "print(has_name, granted, ungranted_refused, rename_refused)")
        assert out.get("stdout", "").strip() == "True True True True", out
        # main-loop class rename keeps the uniform _p_name in sync with __name__
        out = s.execute_cell(
            "Net._rewrite('class Net2:\\n    def __call__(self, x):\\n        return x')\n"
            "print(_PLM_POLICIES['Net2']._p_name == 'Net2')")
        assert out.get("stdout", "").strip() == "True", out
    finally:
        s.close()


def test_proxy_global_rebind_does_not_brick_capture():
    """H3 + M10: a cell rebinding the `_repl_`-prefixed stream-proxy / set_main_streams globals must
    NOT brick or silently lose output capture — `_repl_reset_buffers` re-derives them FRESH from
    `plm._branch_state` each cell (the R5-1 re-import pattern), so a clobber in __main__ is ignored.
    (Pre-fix: clobbering `_repl_out_proxy` permanently bricked stdout for the rest of the session.)"""
    try:
        from plm.repl import PythonReplSession
        s = PythonReplSession(workspace=None, preinstall=("dill",),
                              cell_timeout=10.0, sigint_grace=2.0)
    except Exception as e:                                  # pragma: no cover
        pytest.skip(f"could not start kernel session: {e}")
    try:
        s.execute_cell("_repl_out_proxy = 12345; _repl_err_proxy = None")    # H3: clobber the proxy alias
        out = s.execute_cell("print('after-proxy-rebind')")
        assert out.get("stdout", "").strip() == "after-proxy-rebind", out
        assert "AttributeError" not in (out.get("stderr") or ""), out
        s.execute_cell("_repl_set_main_streams = lambda *a, **k: None")      # M10: clobber set_main_streams
        out = s.execute_cell("print('after-sms-rebind')")
        assert out.get("stdout", "").strip() == "after-sms-rebind", out      # output not silently lost
    finally:
        s.close()
