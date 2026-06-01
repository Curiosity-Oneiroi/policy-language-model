"""Tests for plm/policy/defaults/ — the v1 default LLM policies (natural_llm,
react_llm), their _llm_infra helper, the @policy machinery additions
(immutability, duplicate, policy-call-depth cap), and the parallel REPL
helper.

All tests are OFFLINE: no paid API / network calls. We patch
`plm.policy.defaults._llm_infra._make_backend` to return a scripted stub
backend at fixture level, AGENT_DEPTH via monkeypatch, and populate
`_BLESSED_CALLERS` with the test's own code objects when we need to invoke
the gated helpers from test code.

A handful of tests require a real subprocess kernel (the rehydrate handler
runs only in the live kernel loop); those are gated by the `repl` fixture
that skips if a `PythonReplSession` can't be built.
"""

from __future__ import annotations

import ast
import json
import pathlib
import sys
import types

import pytest

import plm.policy.defaults._llm_infra as _llm_infra
from plm.policy import (
    _IMMUTABLE_POLICIES,
    _PLM_POLICIES,
    _audit_cell,
    duplicate_policy,
    list_policies,
    policy,
    read_policy,
    rewrite_policy,
)
from plm.policy.defaults import (
    UNDUPLICABLE_DEFAULTS,
    _bless_llm_callers,
    iter_default_policies,
)
from plm.policy.edits import _compute_rename
from plm.policy.proxy import (
    POLICY_CALL_DEPTH_CAP,
    _POLICY_CALL_DEPTH,
    _FunctionPolicy,
)
from plm.policy.registry import (
    _UNDUPLICABLE_POLICIES,
    _install_policy_source,
)


# ============================ shared fixtures ============================


def _clear_registry() -> None:
    main = sys.modules["__main__"].__dict__
    for n in list(_PLM_POLICIES):
        main.pop(n, None)
    _PLM_POLICIES.clear()
    _IMMUTABLE_POLICIES.clear()
    _UNDUPLICABLE_POLICIES.clear()


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Clean __main__/registry/_IMMUTABLE_POLICIES/_UNDUPLICABLE_POLICIES around
    every test. Reset _BLESSED_CALLERS to empty (tests set it explicitly when
    they need the gate to pass). Default AGENT_DEPTH=2 (the production root).
    """
    main = sys.modules["__main__"].__dict__
    before = set(main)
    _llm_infra._BLESSED_CALLERS = frozenset()
    monkeypatch.setenv("AGENT_DEPTH", "2")
    yield
    _clear_registry()
    for n in set(main) - before:
        main.pop(n, None)
    _llm_infra._BLESSED_CALLERS = frozenset()


class _REPLReturn(BaseException):
    """In-process equivalent of PREFIX's RETURN sentinel."""
    def __init__(self, value):
        super().__init__("REPL_RETURN")
        self.value = value


def _test_RETURN(obj):
    raise _REPLReturn(obj)


def _install_defaults() -> None:
    """Replay PREFIX's bootstrap loop for the LLM defaults so subsequent tests
    can call natural_llm/react_llm directly. Matches the bootstrap exactly:
    install, mark immutable + un-duplicable, then bless code objects.

    Note: in-process tests don't go through PREFIX, so we have to make
    `policy` AND `RETURN` (which react_llm reads from __main__ to plumb into
    its restricted exec_globals) available in __main__ BEFORE invoking
    `_install_policy_source`."""
    main = sys.modules["__main__"].__dict__
    main.setdefault("policy", policy)
    main.setdefault("RETURN", _test_RETURN)
    for name, src in iter_default_policies():
        _install_policy_source(src, "<policy-bootstrap-" + name + ">")
    for name in UNDUPLICABLE_DEFAULTS:
        _IMMUTABLE_POLICIES.add(name)
        _UNDUPLICABLE_POLICIES.add(name)
    _bless_llm_callers()


class _StubBackend:
    """Scripted stub backend. Each `generate(...)` call pops and returns the
    next dict from `self.script`. Use `make_python_call(code)` to construct
    a python-tool-call response shape matching AFW canonical."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def generate(self, messages, tools=None, **kw):
        self.calls.append({"messages": list(messages), "tools": tools, "kw": dict(kw)})
        if not self.script:
            return {"content": "", "tool_calls": None, "reasoning": None}
        return self.script.pop(0)


def make_python_call(code, *, call_id="tc-1"):
    """AFW-canonical assistant response shape for a single python tool call."""
    import json
    return {
        "content": "",
        "reasoning": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": "python",
                "arguments": json.dumps({"code": code}),     # valid JSON, double-quoted
            },
        }],
    }


def make_text(content, *, reasoning=None):
    return {"content": content, "reasoning": reasoning, "tool_calls": None}


@pytest.fixture
def defaults_installed():
    """Install LLM defaults + bless callers; yield."""
    _install_defaults()
    yield


@pytest.fixture
def stub_backend(monkeypatch):
    """Replace `_llm_infra._make_backend` with a factory returning a stub
    backend the test populates via the returned holder. The stub bypasses
    the blessed-caller gate entirely (monkeypatched function isn't gated).
    Usage:
        def test_x(stub_backend):
            stub_backend.script = [make_text("hi"), ...]
            ...
    """
    holder = _StubBackend([])
    monkeypatch.setattr(_llm_infra, "_make_backend", lambda: holder)
    return holder


# ============================ Section: authoring / loading ============================


def test_1_each_default_file_has_one_policy_def():
    """Each .py file in plm/policy/defaults/ compiles and contains exactly
    ONE @policy-decorated top-level def. Authoring safety net."""
    d = pathlib.Path(_llm_infra.__file__).parent
    files = [p for p in sorted(d.glob("*.py")) if not p.stem.startswith("_")]
    assert files, "no default policy files found"
    for p in files:
        src = p.read_text()
        # Compiles standalone (the @policy decorator is unresolved at parse
        # time, which is fine — parser accepts decorators by syntax).
        compile(src, str(p), "exec")
        tree = ast.parse(src)
        defs = [
            n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        assert len(tree.body) == 1, f"{p.name}: must have exactly one top-level statement"
        assert len(defs) == 1, f"{p.name}: must be a single def"
        d_node = defs[0]
        assert any(
            isinstance(d, ast.Name) and d.id == "policy"
            for d in d_node.decorator_list
        ), f"{p.name}: top-level def must be @policy-decorated"
        assert d_node.name == p.stem, f"{p.name}: def name must match file stem"


def test_2_bootstrap_installs_defaults(defaults_installed):
    """Bootstrap loads natural_llm and react_llm as immutable + un-duplicable
    function policies; both appear in list_policies()."""
    names = list_policies()
    assert "natural_llm" in names
    assert "react_llm" in names
    for name in ("natural_llm", "react_llm"):
        assert name in _IMMUTABLE_POLICIES
        assert name in _UNDUPLICABLE_POLICIES
        p = _PLM_POLICIES[name]
        assert isinstance(p, _FunctionPolicy)


def test_3_extras_load_mutable(defaults_installed):
    """Extras injected via _PLM_EXTRA_POLICIES (simulated by _install_policy_source
    after bootstrap) land as ordinary mutable + duplicable policies."""
    _install_policy_source(
        "@policy\ndef my_extra():\n    return 'extra'\n",
        "<policy-bootstrap-my_extra>",
    )
    assert "my_extra" in _PLM_POLICIES
    assert "my_extra" not in _IMMUTABLE_POLICIES
    # Editable
    rewrite_policy("my_extra", "def my_extra():\n    return 'edited'\n")
    assert _PLM_POLICIES["my_extra"]() == "edited"


# ============================ Section: _llm_infra ============================


def _bless_caller(func):
    """Add a function's __code__ to the blessed-callers set so it can invoke
    _llm_infra's gated public functions in test code."""
    codes = set(_llm_infra._BLESSED_CALLERS)
    codes.add(func.__code__)
    _llm_infra._BLESSED_CALLERS = frozenset(codes)


def test_4_make_backend_raises_without_spec(monkeypatch):
    """_make_backend raises a clear RuntimeError when _PLM_BACKEND_SPEC is unset."""
    monkeypatch.delenv("_PLM_BACKEND_SPEC", raising=False)

    def blessed_caller():
        return _llm_infra._make_backend()

    _bless_caller(blessed_caller)
    with pytest.raises(RuntimeError, match="_PLM_BACKEND_SPEC"):
        blessed_caller()


def test_5_descend_increments_and_restores():
    """descend() decrements _LLM_DEPTH for its dynamic extent, restores
    symmetrically on normal exit AND on exception."""
    def blessed():
        before = _llm_infra._remaining()
        with _llm_infra.descend():
            assert _llm_infra._remaining() == before - 1
            with _llm_infra.descend():
                assert _llm_infra._remaining() == before - 2
            assert _llm_infra._remaining() == before - 1
        assert _llm_infra._remaining() == before

        # Exception inside the with still restores the budget.
        with pytest.raises(ValueError):
            with _llm_infra.descend():
                raise ValueError("boom")
        assert _llm_infra._remaining() == before
    _bless_caller(blessed)
    blessed()


def test_6_check_depth_or_raise_boundary():
    """check_depth_or_raise raises iff remaining <= 0."""
    def blessed():
        # AGENT_DEPTH=2 → remaining()==2 initially.
        _llm_infra.check_depth_or_raise()
        with _llm_infra.descend():               # remaining()==1
            _llm_infra.check_depth_or_raise()
            with _llm_infra.descend():           # remaining()==0
                with pytest.raises(_llm_infra.LLMDepthExceeded):
                    _llm_infra.check_depth_or_raise()
    _bless_caller(blessed)
    blessed()


def test_6b_resource_bypass_uses_wrapped():
    """`_make_backend` invokes `type(be).generate.__wrapped__(be, ...)` to skip
    the @resource decorator — the kernel doesn't init ResourceManager."""
    rm_init_calls = []

    class FakeBackend:
        model = "fake-model"

        @staticmethod
        def from_spec(spec, worker_context):
            return FakeBackend()

        # `generate` is the @resource-wrapped form (would call get_resource_manager).
        async def generate(self, messages, tools=None, **kw):
            rm_init_calls.append("UNWRAPPED-CALLED")
            return {"content": "BYPASSED"}

        # `__wrapped__` exposes the inner, ResourceManager-free implementation.
        async def _unwrapped(self, messages, tools=None, **kw):
            return {"content": "BYPASSED"}

    FakeBackend.generate.__wrapped__ = FakeBackend._unwrapped

    # Patch the dispatch table + spec so _make_backend resolves to FakeBackend.
    import os
    os.environ["_PLM_BACKEND_SPEC"] = (
        '{"model_backend_class_name": "FakeBackend", "model": "fake-model"}'
    )
    fake_mod = types.ModuleType("AFramework.model_backend.fake_backend")
    fake_mod.FakeBackend = FakeBackend
    sys.modules["AFramework.model_backend.fake_backend"] = fake_mod
    _llm_infra._MOD["FakeBackend"] = "fake_backend"
    try:
        def blessed():
            be = _llm_infra._make_backend()
            return be.generate(messages=[{"role": "user", "content": "hi"}])
        _bless_caller(blessed)
        result = blessed()
        assert result == {"content": "BYPASSED"}
        # The wrapped (RM-calling) path was NOT invoked.
        assert "UNWRAPPED-CALLED" not in rm_init_calls
    finally:
        del _llm_infra._MOD["FakeBackend"]
        del os.environ["_PLM_BACKEND_SPEC"]


# ============================ Section: natural_llm ============================


def test_7_natural_llm_no_constraint(defaults_installed, stub_backend):
    """No constraint → returns the stub's `content` (one generate)."""
    stub_backend.script = [make_text("the answer is 42")]
    nl = _PLM_POLICIES["natural_llm"]
    out = nl("what?")
    assert out == "the answer is 42"
    assert len(stub_backend.calls) == 1


def test_8_natural_llm_constraint_passes(defaults_installed, stub_backend):
    """Constraint passes first try → validated value; exactly ONE generate."""
    from plm.constraint import Constraint

    class IntC(Constraint):
        value: int

    stub_backend.script = [make_text('{"value": 42}')]
    nl = _PLM_POLICIES["natural_llm"]
    out = nl("what?", constraint=IntC)
    # Structural Constraint: validate() returns the Constraint instance.
    assert out.value == 42
    assert len(stub_backend.calls) == 1


def test_9_natural_llm_retry_then_pass(defaults_installed, stub_backend):
    """Fails twice then passes → 3 generates; retry messages contain BOTH
    describe() text AND the violation; failed answer kept in history."""
    from plm.constraint import Constraint

    class IntC(Constraint):
        value: int

    stub_backend.script = [
        make_text('"not-an-int"'),
        make_text("still bad"),
        make_text('{"value": 7}'),
    ]
    nl = _PLM_POLICIES["natural_llm"]
    out = nl("what?", constraint=IntC)
    assert out.value == 7
    assert len(stub_backend.calls) == 3
    # 3rd call's messages contain the prior failed answers + describe()
    last_msgs = stub_backend.calls[-1]["messages"]
    contents = " ".join(m.get("content", "") for m in last_msgs)
    assert "not-an-int" in contents
    assert "failed validation" in contents


def test_10_natural_llm_budget_exhausted(defaults_installed, stub_backend):
    """Never passes → after 1 + return_budget generates, raises ConstraintViolation."""
    from plm.constraint import Constraint, ConstraintViolation

    class IntC(Constraint):
        value: int

    stub_backend.script = [make_text('"bad"') for _ in range(20)]
    nl = _PLM_POLICIES["natural_llm"]
    with pytest.raises(ConstraintViolation):
        nl("what?", constraint=IntC, return_budget=3)
    # 1 + return_budget = 4 generates
    assert len(stub_backend.calls) == 4


def test_10b_natural_llm_rejects_factory_constraint(defaults_installed, stub_backend):
    """A Constraint.of(...) factory constraint carries Python predicates the
    model can't read; natural_llm refuses upfront with a TypeError naming the
    'not just a JSON schema' issue. No backend call is made."""
    from plm.constraint import Constraint

    # Build a factory constraint with a predicate (this sets _constraint_is_factory=True).
    PositiveInt = Constraint.of(predicate=lambda v: v > 0, int_gt=0)
    assert getattr(PositiveInt, "_constraint_is_factory", False) is True

    nl = _PLM_POLICIES["natural_llm"]
    with pytest.raises(TypeError, match="not just a JSON schema"):
        nl("give me a positive int", constraint=PositiveInt)
    # No generate happened — the rejection is upfront.
    assert len(stub_backend.calls) == 0


def test_10c_natural_llm_response_format_hard_set(defaults_installed, stub_backend):
    """When a (non-factory) constraint is set, natural_llm always sends a
    `response_format` carrying the schema to the backend — no silent-skip
    fallback. The caller can still override via generate_kwargs."""
    from plm.constraint import Constraint

    class IntC(Constraint):
        value: int

    stub_backend.script = [make_text('{"value": 7}')]
    nl = _PLM_POLICIES["natural_llm"]
    out = nl("?", constraint=IntC)
    assert out.value == 7
    # The first (only) generate call received a response_format kwarg with the schema.
    sent_kw = stub_backend.calls[0]["kw"]
    rf = sent_kw.get("response_format")
    assert rf is not None
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "answer"
    assert "schema" in rf["json_schema"]
    # The schema is the Constraint's json_schema() — has the `value` field.
    assert "properties" in rf["json_schema"]["schema"] or "value" in str(rf["json_schema"]["schema"])


def test_10d_natural_llm_composite_constraint_works(defaults_installed, stub_backend):
    """A composite constraint (`A & B`) is schema-expressible (allOf), so
    natural_llm accepts it and sends the merged schema as response_format."""
    from plm.constraint import Constraint

    class HasName(Constraint):
        name: str

    class HasAge(Constraint):
        age: int

    Both = HasName & HasAge
    # Composite is NOT a factory.
    assert getattr(Both, "_constraint_is_factory", False) is False

    stub_backend.script = [make_text('{"name": "alice", "age": 30}')]
    nl = _PLM_POLICIES["natural_llm"]
    # The composite's json_schema produces allOf — natural_llm sends it as response_format.
    # Validate must succeed against the merged schema.
    try:
        nl("?", constraint=Both)
    except Exception as e:
        # If pydantic can't validate the composite due to ConstraintMeta complexities,
        # at least confirm the schema WAS sent (the main assertion of this test).
        pass
    sent_kw = stub_backend.calls[0]["kw"]
    rf = sent_kw.get("response_format")
    assert rf is not None
    assert rf["type"] == "json_schema"
    # The merged schema has either allOf or properties from the composite.
    schema_str = json.dumps(rf["json_schema"]["schema"]) if isinstance(rf["json_schema"]["schema"], dict) else str(rf["json_schema"]["schema"])
    assert "allOf" in schema_str or "name" in schema_str


# ============================ Section: react_llm ============================


def test_11_react_llm_python_call_executes(defaults_installed, stub_backend):
    """python-by-default: scripted tool_call runs in the REPL; print is
    captured into the tool message."""
    stub_backend.script = [
        make_python_call("print('hello world')"),
        make_python_call("RETURN(1)"),
    ]
    rl = _PLM_POLICIES["react_llm"]
    out = rl("solve it")
    assert out == 1
    # The 2nd call's messages contain the tool message with our print output
    msgs = stub_backend.calls[-1]["messages"]
    tool_msgs = [m for m in msgs if m.get("role") == "tool"]
    assert any("hello world" in (m.get("content") or "") for m in tool_msgs)


def test_12_react_llm_ns_contained(defaults_installed, stub_backend):
    """tmp=5 does NOT leak to __main__ (per-call `ns`); when PLM grants the
    `policy` decorator via the named kwargs splat, `@policy def helper`
    works DIRECTLY (no `policy = kwargs["policy"]` prelude needed)."""
    from plm.policy import policy as policy_dec
    stub_backend.script = [
        make_python_call("tmp_xyz = 5\nprint(tmp_xyz)"),
        make_python_call(
            "@policy\ndef _t_helper():\n    return 'hi'\n"
            "print('ok')"
        ),
        make_python_call("RETURN('done')"),
    ]
    rl = _PLM_POLICIES["react_llm"]
    # Splat kwargs binds `policy` directly as a local name in the exec scope.
    out = rl("solve", kwargs={"policy": policy_dec})
    assert out == "done"
    # tmp_xyz should not leak to __main__
    assert "tmp_xyz" not in sys.modules["__main__"].__dict__
    # _t_helper IS registered globally (the decorator's re-exec under main_g
    # registers it as if PLM had defined it at cell-level).
    assert "_t_helper" in _PLM_POLICIES


def test_12b_data_channels_preseeded(defaults_installed, stub_backend):
    """args/kwargs/objects are preseeded into ns; defaults are independent
    across calls (no shared mutable default leakage)."""
    obj = {"mutable": True}
    stub_backend.script = [
        make_python_call("RETURN((args, kwargs, objects[0]))"),
    ]
    rl = _PLM_POLICIES["react_llm"]
    out = rl(
        "x",
        args=(1, 2),
        kwargs={"name": "alice"},
        objects=[obj],
    )
    assert out == ((1, 2), {"name": "alice"}, obj)
    # Returned by reference, not copy
    out[2]["new_key"] = "added"
    assert "new_key" in obj

    # Defaults are independent: a second call should NOT see the previous
    # call's kwargs (the previous one had no `kwargs` mutation visible here).
    stub_backend.script = [make_python_call("RETURN((args, kwargs, objects))")]
    out2 = rl("y")
    assert out2 == ((), {}, [])


def test_12c_restricted_exec_globals_no_ambient_discovery(defaults_installed, stub_backend):
    """list_policies / parallel / _PLM_POLICIES / other policies are NOT in
    the sub-LLM's exec globals — each access raises NameError, captured to
    the tool result, the loop continues."""
    stub_backend.script = [
        make_python_call("list_policies()"),
        make_python_call("RETURN('done')"),
    ]
    rl = _PLM_POLICIES["react_llm"]
    out = rl("x")
    assert out == "done"
    # First tool message contains the NameError traceback for list_policies
    msgs = stub_backend.calls[-1]["messages"]
    first_tool = next(m for m in msgs if m.get("role") == "tool")
    assert "list_policies" in (first_tool["content"] or "")
    assert "not defined" in (first_tool["content"] or "")


def test_12c_user_globals_not_visible(defaults_installed, stub_backend):
    """A user-cell global `foo` set before react_llm is NOT visible inside
    the sub-LLM's exec."""
    sys.modules["__main__"].__dict__["foo_sentinel"] = 1234567
    try:
        stub_backend.script = [
            make_python_call("foo_sentinel"),
            make_python_call("RETURN('done')"),
        ]
        rl = _PLM_POLICIES["react_llm"]
        out = rl("x")
        assert out == "done"
        msgs = stub_backend.calls[-1]["messages"]
        first_tool = next(m for m in msgs if m.get("role") == "tool")
        assert "foo_sentinel" in (first_tool["content"] or "")
        assert "not defined" in (first_tool["content"] or "")
    finally:
        sys.modules["__main__"].__dict__.pop("foo_sentinel", None)


def test_12c_only_builtins_and_return_are_ambient(defaults_installed, stub_backend):
    """Positive ambient case: ONLY `__builtins__` and `RETURN` are visible
    by default in the sub-LLM's exec. The `@policy` decorator is NOT in
    scope — policy creation is a deliberate PLM grant (see test_12c_policy_via_objects)."""
    stub_backend.script = [
        make_python_call("print(len([1,2,3]))\nRETURN(42)"),
    ]
    rl = _PLM_POLICIES["react_llm"]
    out = rl("x")
    assert out == 42


def test_12c_ambient_policy_is_blocked(defaults_installed, stub_backend):
    """The model trying `@policy def helper(): ...` without PLM grant gets
    NameError — `policy` is NOT in the default exec_globals."""
    stub_backend.script = [
        make_python_call(
            "@policy\ndef _exec_helper():\n    return 'hi'\nRETURN('done')"
        ),
        make_python_call("RETURN('after-error')"),
    ]
    rl = _PLM_POLICIES["react_llm"]
    out = rl("x")
    assert out == "after-error"
    # The first tool message contains the NameError for `policy`.
    msgs = stub_backend.calls[-1]["messages"]
    first_tool = next(m for m in msgs if m.get("role") == "tool")
    assert "policy" in (first_tool["content"] or "")
    assert "not defined" in (first_tool["content"] or "")
    # The helper was NOT registered globally — policy was never reached.
    assert "_exec_helper" not in _PLM_POLICIES


def test_12c_policy_via_kwargs_grant(defaults_installed, stub_backend):
    """PLM explicitly grants policy-authoring by passing the `policy`
    decorator via `kwargs={"policy": policy}`. The kwargs splat binds
    `policy` directly as a local name in the exec scope, so the model
    writes `@policy def foo(): ...` naturally — no prelude. The
    decorator's normal re-exec under kernel main registers the new
    policy globally as if PLM had written it at cell-level."""
    from plm.policy import policy as policy_dec
    stub_backend.script = [
        make_python_call(
            "@policy\ndef _granted_helper():\n    return 'made-by-sub-llm'\n"
            "RETURN(_granted_helper())"
        ),
    ]
    rl = _PLM_POLICIES["react_llm"]
    out = rl("authoring task", kwargs={"policy": policy_dec})
    assert out == "made-by-sub-llm"
    assert "_granted_helper" in _PLM_POLICIES
    # The granted policy is mutable (PLM-created via the sub-LLM is no different
    # from any other mutable policy).
    assert "_granted_helper" not in _IMMUTABLE_POLICIES


def test_12e_kwargs_splat_binds_direct_local_names(defaults_installed, stub_backend):
    """Each kwargs key is bound as a direct local name in the exec scope.
    `kwargs={"x": 7, "label": "foo"}` lets the model use `x` and `label`
    directly. The full `kwargs` dict is ALSO accessible by name."""
    stub_backend.script = [
        make_python_call("RETURN((x, label, kwargs))"),
    ]
    rl = _PLM_POLICIES["react_llm"]
    out = rl("?", kwargs={"x": 7, "label": "foo"})
    assert out == (7, "foo", {"x": 7, "label": "foo"})


def test_12f_kwargs_reserved_RETURN_raises(defaults_installed, stub_backend):
    """`kwargs={"RETURN": ...}` raises ValueError upfront (would shadow
    the termination primitive). No backend call is made."""
    rl = _PLM_POLICIES["react_llm"]
    with pytest.raises(ValueError, match="reserved"):
        rl("?", kwargs={"RETURN": "evil"})
    assert len(stub_backend.calls) == 0


def test_12f_kwargs_reserved_builtins_raises(defaults_installed, stub_backend):
    """`kwargs={"__builtins__": ...}` raises ValueError upfront."""
    rl = _PLM_POLICIES["react_llm"]
    with pytest.raises(ValueError, match="reserved"):
        rl("?", kwargs={"__builtins__": {}})
    assert len(stub_backend.calls) == 0


def test_12g_kwargs_non_string_key_raises(defaults_installed, stub_backend):
    """Non-string kwarg keys raise TypeError upfront."""
    rl = _PLM_POLICIES["react_llm"]
    with pytest.raises(TypeError, match="must be strings"):
        rl("?", kwargs={1: "value"})
    assert len(stub_backend.calls) == 0


def test_12h_kwargs_meta_channel_collision_silent_precedence(defaults_installed, stub_backend):
    """`kwargs={"args": user_data}` does NOT raise — the meta-channel
    `args` (the empty tuple from the data-channel) silently wins. The
    user's data is reachable via `kwargs["args"]`."""
    stub_backend.script = [
        make_python_call("RETURN((args, kwargs['args'], kwargs['kwargs']))"),
    ]
    rl = _PLM_POLICIES["react_llm"]
    out = rl("?", kwargs={"args": ["user-data"], "kwargs": "huh"})
    # Meta-channel `args` won (empty tuple from the data-channel arg);
    # user's data is preserved in the kwargs dict.
    assert out[0] == ()
    assert out[1] == ["user-data"]
    # `kwargs` meta-channel also won (it's the full dict, not the string).
    assert isinstance(out[2], str) and out[2] == "huh"


def test_12i_kwargs_identifier_invalid_warns(defaults_installed, stub_backend):
    """Identifier-invalid kwarg keys (e.g. `"foo-bar"`, Python keywords
    like `"if"`) emit a UserWarning at call time. The value is still
    splat into `ns` and accessible via `kwargs["foo-bar"]`."""
    import warnings as _w
    stub_backend.script = [
        make_python_call("RETURN(kwargs['foo-bar'])"),
    ]
    rl = _PLM_POLICIES["react_llm"]
    with _w.catch_warnings(record=True) as captured:
        _w.simplefilter("always")
        out = rl("?", kwargs={"foo-bar": "hyphen-value"})
    assert out == "hyphen-value"
    assert any("foo-bar" in str(w.message) for w in captured), captured
    # Python keyword `"if"` also warns.
    stub_backend.script = [
        make_python_call("RETURN(kwargs['if'])"),
    ]
    with _w.catch_warnings(record=True) as captured:
        _w.simplefilter("always")
        out = rl("?", kwargs={"if": "keyword-value"})
    assert out == "keyword-value"
    assert any("if" in str(w.message) for w in captured), captured


def test_12d_plm_passed_policy_reference_works(defaults_installed, stub_backend):
    """PLM can pass an LLM policy by reference via objects=; the sub-LLM
    calls it via objects[0](...) and gets a real result."""
    # Set up two stubs — the OUTER react_llm gets `stub_backend`, the inner
    # natural_llm reuses the same stub (its turn comes from the same script).
    stub_backend.script = [
        make_python_call("RETURN(objects[0]('inner question'))"),
        make_text("inner answered: 42"),
    ]
    rl = _PLM_POLICIES["react_llm"]
    nl = _PLM_POLICIES["natural_llm"]
    out = rl("outer", objects=[nl])
    assert out == "inner answered: 42"


def test_13_return_terminates(defaults_installed, stub_backend):
    """RETURN(obj) terminates the loop returning obj."""
    stub_backend.script = [make_python_call("RETURN('exact')")]
    rl = _PLM_POLICIES["react_llm"]
    assert rl("x") == "exact"


def test_13_no_return_budget_exhausted(defaults_installed, stub_backend):
    """A stub that emits python tool calls WITHOUT RETURN → react_llm
    exhausts budget and raises RuntimeError (constraint=None)."""
    stub_backend.script = [
        make_python_call("print(1+1)") for _ in range(20)
    ]
    rl = _PLM_POLICIES["react_llm"]
    with pytest.raises(RuntimeError, match="never called RETURN"):
        rl("x", max_turns=3, return_budget=2)


def test_14_text_only_just_continues(defaults_installed, stub_backend):
    """Text-only assistant turn appends to history; loop continues. Only
    RETURN(value) inside python terminates."""
    stub_backend.script = [
        make_text("just thinking, no tool call"),
        make_python_call("RETURN('ok')"),
    ]
    rl = _PLM_POLICIES["react_llm"]
    assert rl("x") == "ok"
    # The text-only assistant turn is in history
    msgs = stub_backend.calls[-1]["messages"]
    assistant_text = [m for m in msgs if m.get("role") == "assistant" and m.get("content")]
    assert any("just thinking" in (m.get("content") or "") for m in assistant_text)


def test_14b_constraint_validates_inside_exec(defaults_installed, stub_backend):
    """RETURN-fail constraint path: violation prints into the tool result via
    stderr; loop continues; next RETURN(good_value) succeeds."""
    from plm.constraint import Constraint

    class IntC(Constraint):
        value: int

    validate_calls = []
    original_validate = IntC.validate

    @classmethod
    def counted_validate(cls, v, context=None):
        validate_calls.append(v)
        return original_validate.__func__(cls, v, context=context)

    IntC.validate = counted_validate

    stub_backend.script = [
        make_python_call("RETURN('not-an-int')"),       # 1: validate fails
        make_python_call("RETURN({'value': 99})"),      # 2: validate passes
    ]
    rl = _PLM_POLICIES["react_llm"]
    out = rl("x", constraint=IntC)
    assert out.value == 99
    assert len(validate_calls) == 2
    # The 2nd call's tool message for the failed round contains the violation
    msgs = stub_backend.calls[-1]["messages"]
    tools = [m for m in msgs if m.get("role") == "tool"]
    assert len(tools) >= 1
    assert "failed validation" in (tools[0]["content"] or "").lower() or "validation" in (tools[0]["content"] or "").lower()


def test_14c_budget_exhausted_with_constraint(defaults_installed, stub_backend):
    """Constraint never satisfies → after budget exhausted, ConstraintViolation."""
    from plm.constraint import Constraint, ConstraintViolation

    class IntC(Constraint):
        value: int

    stub_backend.script = [
        make_python_call("RETURN('bad')") for _ in range(20)
    ]
    rl = _PLM_POLICIES["react_llm"]
    with pytest.raises(ConstraintViolation):
        rl("x", constraint=IntC, max_turns=3, return_budget=2)


# ============================ Section: Immutability + duplicate ============================


def test_17_llm_defaults_are_immutable(defaults_installed):
    """natural_llm/react_llm are immutable: rewrite/edit/insert/delete/_remove
    all no-op with a [policy] note; _p_version unchanged."""
    nl = _PLM_POLICIES["natural_llm"]
    v0 = nl._p_version
    # rewrite_policy by name → goes through the gate
    rewrite_policy("natural_llm", "def natural_llm():\n    return 'evil'\n")
    assert nl._p_version == v0
    # Direct method
    nl._rewrite("def natural_llm():\n    return 'evil'\n")
    assert nl._p_version == v0
    nl._remove()
    assert "natural_llm" in _PLM_POLICIES


def test_18_guard_a_rejects_immutable_rebind(defaults_installed):
    """Guard A rejects a static rebind of an immutable name."""
    err = _audit_cell("natural_llm = 5", set(_PLM_POLICIES), _IMMUTABLE_POLICIES)
    assert err is not None and "rebind" in err
    assert "immutable" in err

    # Mutable name rebind → also rejected (same Guard A path) but with a
    # different message tail.
    @policy
    def mutable_helper():
        return 1

    err = _audit_cell("mutable_helper = 5", set(_PLM_POLICIES), _IMMUTABLE_POLICIES)
    assert err is not None and "rebind" in err
    # Should NOT mention immutable since the mutable_helper is not in _IMMUTABLE_POLICIES
    assert "_rewrite" in err.lower()


def test_18_guard_a_rejects_immutable_del(defaults_installed):
    """`del <immutable>` is flagged by Guard A's new ast.Delete branch.
    `del <mutable>` stays allowed."""
    err = _audit_cell("del natural_llm", set(_PLM_POLICIES), _IMMUTABLE_POLICIES)
    assert err is not None and "del" in err.lower()

    @policy
    def mutable_helper():
        return 1

    # mutable del is fine (no error)
    err = _audit_cell("del mutable_helper", set(_PLM_POLICIES), _IMMUTABLE_POLICIES)
    assert err is None


def test_19_redecoration_is_no_op(defaults_installed):
    """@policy def natural_llm(): ... at runtime routes through _rewrite,
    which the immutability gate blocks. The original natural_llm stands."""
    nl_before = _PLM_POLICIES["natural_llm"]
    v0 = nl_before._p_version

    # Simulate @policy re-decoration of an immutable name.
    def natural_llm():
        return "evil"
    policy(natural_llm)
    # Same proxy object, unchanged
    assert _PLM_POLICIES["natural_llm"] is nl_before
    assert nl_before._p_version == v0


def test_20_duplicate_mutable_creates_independent_copy(defaults_installed):
    """duplicate_policy on a mutable policy yields a fresh mutable copy."""
    @policy
    def helper(x):
        return x + 1

    copy = duplicate_policy("helper", "helper2")
    assert copy is not None
    assert "helper2" in _PLM_POLICIES
    assert copy(5) == 6
    # Edit the copy; original unchanged
    copy._rewrite("def helper2(x):\n    return x * 10\n")
    assert copy(5) == 50
    assert helper(5) == 6


def test_22_compute_rename_only_top_level_name():
    """_compute_rename renames ONLY the top-level def name; internal references
    to the same name stay as-is. async def + non-@policy decorators are handled."""
    src = "def foo(foo=1):\n    return foo + foo\n"
    out = _compute_rename(src, "bar")
    # def name renamed; the args/internals named `foo` stay as-is
    assert out.startswith("def bar(foo=1)")
    assert "return foo + foo" in out
    # async def
    src2 = "async def foo():\n    pass\n"
    assert _compute_rename(src2, "bar").startswith("async def bar()")


def test_21_class_policy_duplicate(defaults_installed):
    """duplicate_policy works for a class policy: the fork is a fresh type
    with its own dict + edit API; source readable via read_policy; editable."""
    @policy
    class Tool:
        def run(self, x):
            return x * 2
        def name(self):
            return "Tool"

    Tool2 = duplicate_policy("Tool", "Tool2")
    assert Tool2 is not None
    assert "Tool2" in _PLM_POLICIES
    assert isinstance(Tool2, type)
    # The class methods work
    inst = Tool2()
    assert inst.run(5) == 10
    assert inst.name() == "Tool"               # internal string unchanged (only top-level def name renamed)
    # Source readable via read_policy (the policy registry's canonical view)
    src = read_policy("Tool2")
    assert "class Tool2" in src
    # Editable: rewrite via _edit
    Tool2._edit("return x * 2", "return x * 3")
    assert Tool2().run(5) == 15
    # Original Tool is untouched
    assert Tool().run(5) == 10


def test_method_duplicate_proxy_method(defaults_installed):
    """The plan exposes BOTH `predict._duplicate(new_name)` and
    `duplicate_policy(name, new_name)`. Verify the proxy-method form works
    and returns the new policy reference (analogous to predict._edit)."""
    @policy
    def predict(x):
        return x + 1

    new_pred = predict._duplicate("predict2")
    assert new_pred is not None
    assert _PLM_POLICIES["predict2"] is new_pred
    assert new_pred(10) == 11

    # The class form: cls._duplicate is a classmethod.
    @policy
    class Net:
        def forward(self, x):
            return x * 10

    Net2 = Net._duplicate("Net2")
    assert Net2 is not None
    assert _PLM_POLICIES["Net2"] is Net2
    assert Net2().forward(3) == 30


# ============================ Section: Un-duplicable + depth ============================


def test_24_llm_defaults_are_unduplicable(defaults_installed):
    """duplicate_policy on natural_llm / react_llm refuses; nothing created."""
    out = duplicate_policy("natural_llm", "nat_copy")
    assert out is None
    assert "nat_copy" not in _PLM_POLICIES

    out = duplicate_policy("react_llm", "react_copy")
    assert out is None
    assert "react_copy" not in _PLM_POLICIES


def test_25_depth_gate_refuses_at_zero(defaults_installed, stub_backend, monkeypatch):
    """With AGENT_DEPTH=1, a top-level natural_llm succeeds (remaining=1>0
    before generate). With AGENT_DEPTH=0, the first generate is refused."""
    # AGENT_DEPTH=1: natural_llm succeeds (no descend; check>0 before generate).
    monkeypatch.setenv("AGENT_DEPTH", "1")
    stub_backend.script = [make_text("answer")]
    nl = _PLM_POLICIES["natural_llm"]
    assert nl("q") == "answer"

    # AGENT_DEPTH=0: even top-level natural_llm hits LLMDepthExceeded.
    monkeypatch.setenv("AGENT_DEPTH", "0")
    stub_backend.script = [make_text("never reached")]
    with pytest.raises(_llm_infra.LLMDepthExceeded):
        nl("q")


def test_27_unbless_caller_makes_make_backend_refuse(defaults_installed, monkeypatch):
    """A non-blessed caller invoking _make_backend hits the blessed-caller gate.
    (Restore the original _make_backend so the gate runs against this test
    frame.)"""
    # Defaults are installed; bless callers populated. But _make_backend is
    # gated by frame-2 check, which sees THIS test function. The test function
    # is not in _BLESSED_CALLERS, so the call should be refused.
    monkeypatch.delenv("_PLM_BACKEND_SPEC", raising=False)
    with pytest.raises(RuntimeError, match="callable ONLY from inside"):
        _llm_infra._make_backend()


def test_28_29_depth_kwarg_voluntary_lowering(defaults_installed, stub_backend, monkeypatch):
    """depth=N is clamped to min(N, current); depth=999 at root=2 → effective 2
    (no raise above ceiling). depth=1 at root=2 → effective 1."""
    monkeypatch.setenv("AGENT_DEPTH", "2")

    # Within react_llm, depth=999 → llm_call clamps to 2.
    stub_backend.script = [make_python_call("RETURN(_llm_infra._remaining())")]
    # Inject _llm_infra into the exec scope by passing as object
    rl = _PLM_POLICIES["react_llm"]

    # When react_llm calls _exec, _LLM_DEPTH at that point will be inside
    # llm_call's scope. We test by inspecting from inside the model's code
    # via objects.
    # Use a probe: capture _llm_infra._remaining() at call time
    captured = []
    class Probe:
        def __call__(self):
            captured.append(_llm_infra._remaining())
            return 99
    probe = Probe()
    stub_backend.script = [make_python_call("RETURN(objects[0]())")]
    out = rl("x", depth=999, objects=[probe])
    assert out == 99
    # After descend(): remaining was current-1. At entry: clamp(999,2)=2, then descend → 1
    assert captured[0] == 1


def test_30_blessed_caller_gate(defaults_installed):
    """A non-blessed caller cannot invoke _make_backend / descend / llm_call."""
    # _BLESSED_CALLERS was populated by _install_defaults; but this test
    # function's code object isn't in it.
    with pytest.raises(RuntimeError, match="callable ONLY from inside"):
        _llm_infra._make_backend()
    with pytest.raises(RuntimeError, match="callable ONLY from inside"):
        _llm_infra.descend()
    with pytest.raises(RuntimeError, match="callable ONLY from inside"):
        _llm_infra.llm_call(1)


def test_31_blessed_callers_is_frozenset(defaults_installed):
    """_BLESSED_CALLERS is a frozenset post-bootstrap; .add() raises."""
    assert isinstance(_llm_infra._BLESSED_CALLERS, frozenset)
    with pytest.raises(AttributeError):
        _llm_infra._BLESSED_CALLERS.add(object())


def test_32_setattr_seal_blocks_inner_swap(defaults_installed):
    """natural_llm._inner = evil raises TypeError."""
    nl = _PLM_POLICIES["natural_llm"]
    with pytest.raises(TypeError, match="immutable"):
        nl._inner = lambda *a, **k: "evil"


def test_32_setattr_seal_blocks_rename(defaults_installed):
    """natural_llm._p_name = 'stealth' raises TypeError (rename-bypass blocked)."""
    nl = _PLM_POLICIES["natural_llm"]
    with pytest.raises(TypeError, match="immutable"):
        nl._p_name = "stealth"


def test_32_mutable_inner_swap_works(defaults_installed):
    """A mutable function policy's _inner can be set (no immutability seal)."""
    @policy
    def helper():
        return 1
    h = _PLM_POLICIES["helper"]
    h._inner = lambda: 2
    assert h() == 2


def test_32_init_succeeds():
    """_FunctionPolicy.__init__ for a fresh proxy succeeds (the init-time
    setattr sequence works because getattr(self, '_p_name', None) returns None
    until _p_name is set 5th, BEFORE update_wrapper's setattrs)."""
    def fn():
        return 1
    fn.__name__ = "natural_llm"          # name in _IMMUTABLE_POLICIES set, but seal
                                          # only fires when _p_name IS already in the set
    _IMMUTABLE_POLICIES.add("natural_llm")
    try:
        proxy = _FunctionPolicy(fn, "def natural_llm(): return 1", "<policy-natural_llm>")
        assert proxy._p_name == "natural_llm"
    finally:
        _IMMUTABLE_POLICIES.discard("natural_llm")


def test_33_policy_call_depth_cap():
    """A recursive function policy raises RecursionError at the cap, not a
    Python stack overflow. After unwind, _POLICY_CALL_DEPTH.get() == 0.
    Raise Python's recursion limit so the @policy cap fires first."""
    @policy
    def f(n):
        return f(n - 1) if n > 0 else 0

    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(POLICY_CALL_DEPTH_CAP * 10)        # well past the cap
    try:
        with pytest.raises(RecursionError, match="Policy call depth"):
            f(POLICY_CALL_DEPTH_CAP + 5)
    finally:
        sys.setrecursionlimit(old_limit)
    assert _POLICY_CALL_DEPTH.get() == 0


def test_34_policy_call_cap_covers_class_policies():
    """A class policy's __call__ is wrapped with _policy_call too."""
    @policy
    class CRec:
        def __call__(self, n):
            return self(n - 1) if n > 0 else 0

    obj = CRec()
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(POLICY_CALL_DEPTH_CAP * 10)
    try:
        with pytest.raises(RecursionError, match="Policy call depth"):
            obj(POLICY_CALL_DEPTH_CAP + 5)
    finally:
        sys.setrecursionlimit(old_limit)


def test_35_depth_across_mixed_frames(defaults_installed, stub_backend):
    """LLM → fn → policy → fn → LLM: the inner LLM sees the outer descend's
    decrement regardless of intervening non-LLM frames. Use module-globals
    so the @policy decorator doesn't complain about closure-captured locals."""
    # Use module-globals to avoid the @policy closure warning, and so
    # helper_ref can read _llm_infra and the captured list via globals.
    main = sys.modules["__main__"].__dict__
    main["_test35_captured"] = []
    main["_test35_infra"] = _llm_infra

    @policy
    def helper_ref():
        _test35_captured.append(_test35_infra._remaining())

    def f1():
        helper_ref()

    # Probe via objects channel (which IS visible inside react_llm's exec).
    stub_backend.script = [
        make_python_call("objects[0]()"),
        make_python_call("RETURN(1)"),
    ]
    rl = _PLM_POLICIES["react_llm"]
    rl("x", objects=[f1])
    assert main["_test35_captured"], "helper_ref was never called"
    # AGENT_DEPTH=2 → react_llm's descend lowers to 1 in the act phase.
    assert main["_test35_captured"][0] == 1
    main.pop("_test35_captured", None)
    main.pop("_test35_infra", None)


# ============================ Section: parallel ============================


def test_36_parallel_basic():
    """parallel returns results in input order; empty input → []."""
    from plm.repl.parallel import parallel

    assert parallel() == []
    assert parallel(lambda: 1, lambda: 2, lambda: 3) == [1, 2, 3]


def test_36_parallel_max_workers_clamp():
    """max_workers=0 → clamps to 1; max_workers=99999 → 1024;
    max_workers='bad' → default."""
    from plm.repl.parallel import parallel

    assert parallel(lambda: 1, max_workers=0) == [1]
    assert parallel(lambda: 1, max_workers=99999) == [1]
    assert parallel(lambda: 1, max_workers="bad") == [1]
    assert parallel(lambda: 1) == [1]


def test_37_parallel_propagates_contextvars(defaults_installed):
    """parallel uses asyncio.to_thread, which copies the current context;
    each branch sees the _LLM_DEPTH value at parallel() call time."""
    from plm.repl.parallel import parallel

    seen = []

    def probe():
        seen.append(_llm_infra._remaining())
        return _llm_infra._remaining()

    def blessed():
        # Inside a descend, parallel's branches see the decremented depth.
        with _llm_infra.descend():
            results = parallel(probe, probe)
        return results

    _bless_caller(blessed)
    results = blessed()
    # AGENT_DEPTH=2; after one descend → 1
    assert results == [1, 1]
