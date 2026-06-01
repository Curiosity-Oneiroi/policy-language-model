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


def test_async_and_argcount_change():
    import asyncio
    @policy
    def predict(s):
        return s
    predict._rewrite("async def predict(s, n):\n    return s * n\n")   # sync -> async, +arg
    assert asyncio.run(predict("a", 3)) == "aaa"


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


def test_int_class_hotswap(repl):
    repl.execute_cell("@policy\nclass Netz:\n    def m(self): return 1\n")
    repl.execute_cell("a = Netz()")
    repl.execute_cell("Netz._rewrite('class Netz:\\n    def m(self): return 2\\n    def n(self): return 9\\n')")
    r = repl.execute_cell("print(a.m(), a.n(), isinstance(a, Netz))")
    assert r["stdout"].strip() == "2 9 True", r


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
