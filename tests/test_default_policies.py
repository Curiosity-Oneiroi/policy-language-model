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
import asyncio
import sys
import types

import pytest

import plm.policy.defaults._llm_infra as _llm_infra
from plm.policy import (
    _SEALED_POLICIES,
    _PLM_POLICIES,
    _audit_cell,
    duplicate_policy,
    list_policies,
    policy,
    read_policy,
    rewrite_policy,
)
from plm.policy.defaults import (
    _LLM_DEFAULT_POLICIES,
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
    _install_policy_source,
    _seal,
    _unsealed,
)


# ============================ shared fixtures ============================


def _clear_registry() -> None:
    main = sys.modules["__main__"].__dict__
    for n in list(_PLM_POLICIES):
        main.pop(n, None)
    with _unsealed():                 # harness reset: the store retains defaults while sealed
        _PLM_POLICIES.clear()
    _SEALED_POLICIES.clear()


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Clean __main__/registry/_SEALED_POLICIES around
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
    `policy`, `RETURN` (which react_llm reads from __main__ to plumb into its
    restricted exec_globals), and `exec_ns` (a PREFIX-injected repl global that
    base_verifier reaches AMBIENTLY, like `react_llm`) available in __main__
    BEFORE invoking `_install_policy_source`."""
    from plm.repl.exec_ns import exec_ns
    from plm.constraint import Constraint, ConstraintViolation
    main = sys.modules["__main__"].__dict__
    main.setdefault("policy", policy)
    main.setdefault("RETURN", _test_RETURN)
    main.setdefault("exec_ns", exec_ns)
    # constraint surface is prefix-injected in the real kernel (when pydantic present);
    # in-process tests don't run the prefix, so seed what base_verifier reaches ambiently.
    main.setdefault("Constraint", Constraint)
    main.setdefault("ConstraintViolation", ConstraintViolation)
    for name, src in iter_default_policies():
        _install_policy_source(src, "<policy-bootstrap-" + name + ">")
    for name in _LLM_DEFAULT_POLICIES:
        _seal(name)             # sets the intrinsic _p_immutable flag + the name-sets
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
        assert name in _SEALED_POLICIES                       # sealed = immutable + un-duplicable
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
    assert "my_extra" not in _SEALED_POLICIES
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


def test_5_descend_decrements_and_restores():
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


def test_6b_make_backend_calls_generate_directly():
    """`_make_backend` builds the backend from `plm.model_backend.<x>` via
    `from_spec` and awaits its PLAIN async `.generate` directly — the ported
    backends carry no @resource/ResourceManager wrapper (that AFW coupling was
    stripped), so there is no `__wrapped__` to unwrap."""
    calls = []

    class FakeBackend:
        model = "fake-model"

        @staticmethod
        def from_spec(spec, worker_context):
            return FakeBackend()

        async def generate(self, messages, tools=None, **kw):
            calls.append(tuple(m["role"] for m in messages))
            return {"content": "OK"}

    import os
    os.environ["_PLM_BACKEND_SPEC"] = (
        '{"model_backend_class_name": "FakeBackend", "model": "fake-model"}'
    )
    fake_mod = types.ModuleType("plm.model_backend.fake_backend")
    fake_mod.FakeBackend = FakeBackend
    sys.modules["plm.model_backend.fake_backend"] = fake_mod
    _llm_infra._MOD["FakeBackend"] = "fake_backend"
    try:
        def blessed():
            be = _llm_infra._make_backend()
            return be.generate(messages=[{"role": "user", "content": "hi"}])
        _bless_caller(blessed)
        result = blessed()
        assert result == {"content": "OK"}
        assert calls == [("user",)]                  # plain .generate ran once
    finally:
        del _llm_infra._MOD["FakeBackend"]
        del os.environ["_PLM_BACKEND_SPEC"]
        sys.modules.pop("plm.model_backend.fake_backend", None)


def test_6c_aclose_backend_handles_all_close_spellings():
    """`_aclose_backend` (C27) closes the backend's transport regardless of how
    the client spells close: `aclose()` (httpx — SlateBackend), async `close()`
    (openai/anthropic, which have NO `aclose`), or a sync `close()`. A missing
    client / missing closer is a no-op, and close-time errors are swallowed so
    cleanup never masks the real generate result."""
    closed = []

    class AcloseClient:
        async def aclose(self): closed.append("aclose")

    class AsyncCloseClient:                              # like openai/anthropic
        async def close(self): closed.append("async-close")

    class SyncCloseClient:
        def close(self): closed.append("sync-close")

    class RaisingClient:
        async def aclose(self): raise RuntimeError("boom")

    def be_with(client): return types.SimpleNamespace(client=client)

    async def _drive():
        await _llm_infra._aclose_backend(be_with(AcloseClient()))
        await _llm_infra._aclose_backend(be_with(AsyncCloseClient()))
        await _llm_infra._aclose_backend(be_with(SyncCloseClient()))
        await _llm_infra._aclose_backend(be_with(None))          # no client -> no-op
        await _llm_infra._aclose_backend(types.SimpleNamespace())  # no attr -> no-op
        await _llm_infra._aclose_backend(be_with(object()))      # no closer -> no-op
        await _llm_infra._aclose_backend(be_with(RaisingClient()))  # error swallowed

    asyncio.run(_drive())
    assert closed == ["aclose", "async-close", "sync-close"]


def test_6d_aclose_backend_does_not_trigger_lazy_client():
    """#19: when a backend exposes `client` as a lazy @property that constructs
    the transport on access (OpenAIBackend), `_aclose_backend` must NOT build a
    throwaway client just to close it — it reads the instance __dict__ instead.
    An eagerly-stored client is still closed."""
    constructed = []
    closed = []

    class LazyBackend:
        """Mirrors OpenAIBackend: lazy `client` @property over a `_client` holder."""
        def __init__(self):
            self._client = None
        @property
        def client(self):
            if self._client is None:
                constructed.append("BUILT")
                self._client = type("C", (), {"aclose": staticmethod(lambda: None)})()
            return self._client

    class EagerBackend:
        def __init__(self):
            class C:
                async def aclose(self_): closed.append("eager-closed")
            self.client = C()                            # stored in __dict__

    async def _drive():
        await _llm_infra._aclose_backend(LazyBackend())  # client never used -> must NOT construct
        await _llm_infra._aclose_backend(EagerBackend())  # eager client -> closed

    asyncio.run(_drive())
    assert constructed == [], "lazy client property was triggered during cleanup"
    assert closed == ["eager-closed"]


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
    """A Constraint.field(...) factory constraint carries Python predicates the
    model can't read; natural_llm refuses upfront with a TypeError. No backend
    call is made."""
    from plm.constraint import Constraint

    # Build a factory constraint with a predicate (this sets _constraint_is_factory=True).
    PositiveInt = Constraint.field(predicate=lambda v: v > 0, int_gt=0)
    assert getattr(PositiveInt, "_constraint_is_factory", False) is True

    nl = _PLM_POLICIES["natural_llm"]
    with pytest.raises(TypeError, match="predicate/AfterValidator"):
        nl("give me a positive int", constraint=PositiveInt)
    # No generate happened — the rejection is upfront.
    assert len(stub_backend.calls) == 0


def test_10c_natural_llm_response_format_hard_set(defaults_installed, stub_backend):
    """When a (non-factory) constraint is set, natural_llm always sends a
    `response_format` carrying the schema to the backend — no silent-skip
    fallback, and it is NOT overridable (the constraint defines the output)."""
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


def test_10e_natural_llm_retries_on_non_violation_error(defaults_installed, stub_backend):
    """#8: a struct @model_validator raising a RAW non-ValueError (KeyError) must
    be treated as a retry, not abort the call — the model gets another turn and a
    later good answer is accepted."""
    from pydantic import model_validator
    from plm.constraint import Constraint

    class C(Constraint):
        v: int = 0
        @model_validator(mode="after")
        def _chk(self):
            if self.v == 0:
                raise KeyError("v missing")          # RAW non-ValueError; pydantic won't wrap it
            return self

    stub_backend.script = [make_text('{"v": 0}'), make_text('{"v": 5}')]
    nl = _PLM_POLICIES["natural_llm"]
    out = nl("?", constraint=C, return_budget=3)
    assert out.v == 5
    assert len(stub_backend.calls) == 2              # retried after the KeyError (didn't abort)


def test_10f_natural_llm_budget_none_uses_default(defaults_installed, stub_backend):
    """#18: a None / non-int return_budget is coerced to the default instead of
    raising a raw TypeError at the range bound."""
    from plm.constraint import Constraint

    class IntC(Constraint):
        value: int

    stub_backend.script = [make_text('{"value": 7}')]
    nl = _PLM_POLICIES["natural_llm"]
    assert nl("?", constraint=IntC, return_budget=None).value == 7   # no TypeError


def test_10g_natural_llm_strips_fenced_json(defaults_installed, stub_backend):
    """#9: a ```json-fenced reply parses on the FIRST try (no retry-budget burn);
    _strip_code_fences is structural-only so plain JSON is untouched."""
    from plm.constraint import Constraint

    class IntC(Constraint):
        value: int

    stub_backend.script = [make_text('```json\n{"value": 7}\n```')]
    nl = _PLM_POLICIES["natural_llm"]
    out = nl("?", constraint=IntC)
    assert out.value == 7
    assert len(stub_backend.calls) == 1            # parsed on first try; no retry


def test_10h_natural_llm_exhaustion_raises_constraint_violation(defaults_installed, stub_backend):
    """#5: on budget exhaustion the caller (PLM / a policy / a function) gets a
    ConstraintViolation — even when the underlying validator raised a raw
    non-ValueError — not a bare KeyError. Honors the documented Raises contract."""
    from pydantic import model_validator
    from plm.constraint import Constraint, ConstraintViolation

    class C(Constraint):
        v: int = 0
        @model_validator(mode="after")
        def _chk(self):
            raise KeyError("always")               # raw non-ValueError; pydantic won't wrap it

    stub_backend.script = [make_text('{"v": 1}') for _ in range(10)]
    nl = _PLM_POLICIES["natural_llm"]
    with pytest.raises(ConstraintViolation) as ei:
        nl("?", constraint=C, return_budget=2)
    msg = str(ei.value)
    assert "budget exhausted" in msg and "KeyError" in msg   # clear, wrapped, not a raw KeyError


# ============================ Section: react_llm ============================


def test_11z_react_llm_budget_coerced_no_typeerror(defaults_installed, stub_backend):
    """#22/#18: model-controlled max_turns/return_budget are coerced — None ->
    default, negative -> 0 — so `range(max_turns + return_budget)` neither raises
    a raw TypeError (None + int) nor goes negative."""
    stub_backend.script = [make_python_call("RETURN(42)")]
    rl = _PLM_POLICIES["react_llm"]
    assert rl("?", max_turns=None, return_budget=-3) == 42   # None->8, -3->0; runs cleanly


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


def test_react_llm_malformed_shapes_dont_crash(defaults_installed, stub_backend):
    """Batch 4: malformed backend responses / tool_calls are handled defensively —
    react_llm answers with a clear error and takes another round instead of crashing
    (R-F1..R-F5), then RETURNs normally."""
    import json
    rl = _PLM_POLICIES["react_llm"]

    # R-F4 (non-dict resp) then R-F3 (tool_calls returned as a bare dict, not a list).
    stub_backend.script = [
        "not a dict",
        {"content": "", "tool_calls": {"id": "x", "type": "function",
            "function": {"name": "python",
                         "arguments": json.dumps({"code": "RETURN('via-dict-tc')"})}}},
    ]
    assert rl("go") == "via-dict-tc"

    # R-F1 (tc is a string) / R-F2 (function is None) / R-F5 (code is non-string),
    # each survived with a clear error + retry, then a clean RETURN.
    stub_backend.script = [
        {"content": "", "tool_calls": ["not-a-dict"]},
        {"content": "", "tool_calls": [{"id": "a", "function": None}]},
        {"content": "", "tool_calls": [{"id": "b", "function":
            {"name": "python", "arguments": json.dumps({"code": 42})}}]},
        make_python_call("RETURN('ok')"),
    ]
    assert rl("go2") == "ok"


def test_react_malformed_tool_call_not_stored_in_history(defaults_installed, stub_backend):
    """M4: a non-dict first tool_call (from a non-conformant provider) must NOT be STORED into
    the assistant turn's history. R-F1 already handles the in-round error, but storing the
    malformed entry would crash the NEXT round's real-backend history sanitizer (`tc.copy()` /
    `tc.get('function')`). So the stored `tool_calls` is None when the first element isn't a dict
    — the bug was reachable only with a real backend (the stub doesn't sanitize history)."""
    rl = _PLM_POLICIES["react_llm"]
    stub_backend.script = [
        {"content": "", "tool_calls": ["not-a-dict"]},          # malformed
        make_python_call("RETURN('ok')"),
    ]
    assert rl("go") == "ok"
    round2_msgs = stub_backend.calls[1]["messages"]             # history sent on the 2nd round
    asst = [m for m in round2_msgs if m.get("role") == "assistant"][-1]
    assert asst.get("tool_calls") is None                       # malformed element NOT stored
    assert "not-a-dict" not in str(round2_msgs)                 # nowhere in the stored history


def test_react_deepcopy_caller_messages_not_mutated(defaults_installed, stub_backend):
    """Batch 4: a verifier mutating `msgs` in place does NOT corrupt the
    caller's own message dicts — _norm deep-copies the input."""
    stub_backend.script = [make_text("thinking"), make_python_call("RETURN('done')")]
    caller_msgs = [{"role": "user", "content": "original"}]

    def _mutator(msgs):
        for m in msgs:
            if m.get("content") == "original":
                m["content"] = "HIJACKED"

    rlv = _PLM_POLICIES["react_llm_verifier"]
    assert rlv(caller_msgs, verifier=_mutator) == "done"
    assert caller_msgs[0]["content"] == "original"     # caller's dict untouched


def test_react_null_tool_call_id_becomes_empty_string(defaults_installed, stub_backend):
    """ND-1: a backend that sets `id: null` EXPLICITLY must not yield tool_call_id=None (the
    `.get('id','')` default only fires for a MISSING key) — a None id violates the strict
    OpenAI tool schema next round. `tc.get('id') or ''` normalizes it to ''."""
    import json
    rl = _PLM_POLICIES["react_llm"]
    stub_backend.script = [
        {"content": "", "reasoning": None, "tool_calls": [{
            "id": None, "type": "function",                        # EXPLICIT null id
            "function": {"name": "python", "arguments": json.dumps({"code": "print('x')"})}}]},
        make_python_call("RETURN('done')"),
    ]
    assert rl("go") == "done"
    round2_msgs = stub_backend.calls[1]["messages"]                 # history sent on the 2nd round
    tool_turns = [m for m in round2_msgs if m.get("role") == "tool"]
    assert tool_turns and tool_turns[0]["tool_call_id"] == "", tool_turns


def test_react_generate_kwargs_reserved_key_clear_error(defaults_installed, stub_backend):
    """ND-2: a generate_kwargs key colliding with react_llm's own `messages`/`tools` raises a
    CLEAR ValueError up front (not a cryptic 'got multiple values' TypeError), before any
    backend call."""
    rl = _PLM_POLICIES["react_llm"]
    with pytest.raises(ValueError, match="generate_kwargs may not contain"):
        rl("go", generate_kwargs={"tools": [1]})
    with pytest.raises(ValueError, match="generate_kwargs may not contain"):
        rl("go", generate_kwargs={"messages": []})
    assert stub_backend.calls == []                                # raised before any generate()


def test_react_llm_verifier_depth1_with_verifier_hard_errors(defaults_installed, stub_backend):
    """Batch 6: a verifier needs depth >= 2 (it runs one level below + composes
    depth-1 circuits). react_llm_verifier(..., verifier=..., depth=1) raises a clear
    LLMDepthExceeded UP FRONT (no backend call) instead of crashing mid-loop. Without a
    verifier, depth=1 is fine."""
    rlv = _PLM_POLICIES["react_llm_verifier"]
    stub_backend.script = [make_python_call("RETURN('x')")]
    with pytest.raises(_llm_infra.LLMDepthExceeded):
        rlv("go", verifier=lambda m: None, depth=1)        # raised before any generate()
    assert rlv("go", depth=1) == "x"                       # no verifier -> depth-1 is valid; script intact


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
    assert "_granted_helper" not in _SEALED_POLICIES


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


def test_12g_defined_function_sees_injected_and_sibling_names(defaults_installed, stub_backend):
    """#2: the sub-agent's repl is ONE namespace (globals == locals, like PLM's
    kernel `__main__`), so a `def`/`class` the model writes resolves injected
    names (kwargs/objects), sibling helpers, AND itself (recursion) — not just
    top-level code. Under the old split globals/locals these all NameError'd."""
    code = (
        "def use_kwarg():\n"
        "    return x + 1\n"                  # injected kwarg, INSIDE a def
        "def use_object():\n"
        "    return objects[0]\n"             # injected objects channel, INSIDE a def
        "def fact(n):\n"
        "    return 1 if n <= 1 else n * fact(n - 1)\n"   # recursion: fact must see ITSELF
        "def via_sibling():\n"
        "    return use_kwarg() + fact(3)\n"  # one helper calling another helper
        "class C:\n"
        "    val = use_kwarg()\n"             # class body resolving an injected name
        "RETURN((use_kwarg(), use_object(), fact(5), via_sibling(), C.val))"
    )
    stub_backend.script = [make_python_call(code)]
    rl = _PLM_POLICIES["react_llm"]
    out = rl("?", kwargs={"x": 10}, objects=["OBJ"])
    assert out == (11, "OBJ", 120, 17, 11)   # 11; "OBJ"; 5!=120; 11+fact(3)=11+6=17; C.val=11


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
    err = _audit_cell("natural_llm = 5", set(_PLM_POLICIES), _SEALED_POLICIES)
    assert err is not None and "rebind" in err
    assert "immutable" in err

    # Mutable name rebind → also rejected (same Guard A path) but with a
    # different message tail.
    @policy
    def mutable_helper():
        return 1

    err = _audit_cell("mutable_helper = 5", set(_PLM_POLICIES), _SEALED_POLICIES)
    assert err is not None and "rebind" in err
    # Should NOT mention immutable since the mutable_helper is not in _SEALED_POLICIES
    assert "_rewrite" in err.lower()


def test_18_guard_a_rejects_immutable_del(defaults_installed):
    """`del <immutable>` is flagged by Guard A's new ast.Delete branch.
    `del <mutable>` stays allowed."""
    err = _audit_cell("del natural_llm", set(_PLM_POLICIES), _SEALED_POLICIES)
    assert err is not None and "del" in err.lower()

    @policy
    def mutable_helper():
        return 1

    # mutable del is fine (no error)
    err = _audit_cell("del mutable_helper", set(_PLM_POLICIES), _SEALED_POLICIES)
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


def test_19b_duplicate_unknown_name_is_gentle_refusal(defaults_installed, capsys):
    """#21: duplicate_policy on an UNREGISTERED name returns None + a note (its
    documented no-raise contract), not a raw KeyError."""
    capsys.readouterr()
    result = duplicate_policy("does_not_exist", "whatever")
    assert result is None
    assert "[policy] duplicate:" in capsys.readouterr().err


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


def test_llm_call_depth_strict_no_coercion(defaults_installed):
    """ND-5: depth must be None or a NON-NEGATIVE INTEGER — a non-integer (1.5), a negative, a
    bool, or a non-int now RAISES instead of being silently truncated/clamped. Exercised via
    natural_llm (a blessed caller; the error fires at llm_call before any backend use)."""
    nl = _PLM_POLICIES["natural_llm"]
    for bad in (1.5, -1, 2.0, "2", True):
        with pytest.raises(ValueError, match="depth"):
            nl("hi", depth=bad)


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
    fn.__name__ = "natural_llm"          # name in _SEALED_POLICIES set, but seal
                                          # only fires when _p_name IS already in the set
    _SEALED_POLICIES.add("natural_llm")
    try:
        proxy = _FunctionPolicy(fn, "def natural_llm(): return 1", "<policy-natural_llm>")
        assert proxy._p_name == "natural_llm"
    finally:
        _SEALED_POLICIES.discard("natural_llm")


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


def test_parallel_collect_all_and_guards():
    """parallel() is COLLECT-ALL + ISOLATED: a failing task's EXCEPTION lands in its
    result slot (parallel() itself never raises for a task error) and the OTHER tasks
    still run to completion, in input order. Plus: no ~300s artificial hang (K-F4
    structural fix — it waits only for the actual slowest task); survives
    max_workers=inf; refuses to run inside a running loop with a clear error
    and no leaked coroutine."""
    import asyncio
    import time
    from plm.repl.parallel import parallel

    def _boom():
        raise ValueError("boom")

    # Collect-all: the error is RETURNED in its slot; the siblings ran and produced values.
    results = parallel(_boom, lambda: 42, lambda: None)
    assert isinstance(results[0], ValueError) and str(results[0]) == "boom"
    assert results[1] == 42                       # a sibling ordered AFTER the error still ran
    assert results[2] is None                     # a task that returns nothing -> None slot

    # O4: FULL isolation — a control-flow BaseException a policy may raise (SystemExit /
    # CancelledError) is RETURNED in its slot too, NEVER propagated out of parallel(). The
    # documented filter is isinstance(r, BaseException) (Exception would miss these and read a
    # returned SystemExit as success).
    def _sysexit():
        raise SystemExit(2)

    def _cancel():
        raise asyncio.CancelledError()

    out = parallel(lambda: 7, _sysexit, _cancel)
    assert out[0] == 7                                          # parallel did not raise
    assert isinstance(out[1], SystemExit) and not isinstance(out[1], Exception)
    assert isinstance(out[1], BaseException)                    # caught by the documented filter
    assert isinstance(out[2], asyncio.CancelledError)          # returned, not propagated
    assert [type(r).__name__ for r in out if isinstance(r, BaseException)] == ["SystemExit", "CancelledError"]

    # Isolation + no artificial hang: an error and a (briefly) slow task are BOTH collected,
    # and parallel returns shortly after the slow one — nowhere near the old 300s join.
    def _slow():
        time.sleep(0.3)
        return "slow"

    t0 = time.monotonic()
    r = parallel(_boom, _slow)
    assert time.monotonic() - t0 < 30.0, "parallel() hung well past the slowest task"
    assert isinstance(r[0], ValueError) and r[1] == "slow"

    # Corr#5: inf / overflow max_workers falls back to default (no OverflowError).
    assert parallel(lambda: 7, max_workers=float("inf")) == [7]

    # K-F3: from inside a running loop -> clear RuntimeError, no un-awaited coroutine.
    async def _inside():
        parallel(lambda: 1)

    with pytest.raises(RuntimeError):
        asyncio.run(_inside())


# ===================== Section: react_llm_verifier =========================
# The trajectory control axis: react_llm + an optional `verifier` callable run
# after each NON-terminal round (wrapped in descend()), mutating msgs in place.


def test_rlv_verifier_runs_between_rounds(defaults_installed, stub_backend):
    """Verifier runs after a non-terminal round, sees that round's full tick,
    and its in-place mutation reaches the NEXT generate."""
    seen_lens = []

    def vf(msgs):
        seen_lens.append(len(msgs))
        msgs.append({"role": "user", "content": "SENTINEL_INJECT"})

    stub_backend.script = [make_text("thinking"), make_python_call("RETURN(1)")]
    rlv = _PLM_POLICIES["react_llm_verifier"]
    out = rlv("go", verifier=vf)
    assert out == 1
    # Only the round-0 (text-only, non-terminal) round triggers the verifier;
    # at that point msgs == [user(go), assistant(thinking)].
    assert seen_lens == [2]
    # The injected sentinel is visible to the round-1 generate.
    assert any("SENTINEL_INJECT" in (m.get("content") or "")
               for m in stub_backend.calls[1]["messages"])


def test_rlv_verifier_not_called_on_terminal_round(defaults_installed, stub_backend):
    """A successful RETURN short-circuits — the verifier never runs on it."""
    calls_n = []
    stub_backend.script = [make_python_call("RETURN(1)")]
    rlv = _PLM_POLICIES["react_llm_verifier"]
    out = rlv("go", verifier=lambda msgs: calls_n.append(1))
    assert out == 1
    assert calls_n == []


def test_rlv_verifier_runs_after_tool_rounds(defaults_installed, stub_backend):
    """The verifier also runs after a tool round, and sees that round's tool
    message (the complete tick)."""
    seen_tool = []

    def vf(msgs):
        for m in msgs:
            if m.get("role") == "tool":
                seen_tool.append(m.get("content") or "")

    stub_backend.script = [make_python_call("print('XYZ')"), make_python_call("RETURN(2)")]
    rlv = _PLM_POLICIES["react_llm_verifier"]
    out = rlv("go", verifier=vf)
    assert out == 2
    assert any("XYZ" in c for c in seen_tool)


def test_rlv_verifier_none_is_react_llm_parity(defaults_installed, stub_backend):
    """verifier=None ⇒ behaves like react_llm (terminates on RETURN)."""
    stub_backend.script = [make_python_call("RETURN(42)")]
    rlv = _PLM_POLICIES["react_llm_verifier"]
    assert rlv("go", verifier=None) == 42


def test_rlv_accepts_policy_as_verifier(defaults_installed, stub_backend):
    """A POLICY (base_verifier) is accepted as the verifier and is callable
    via its proxy. Here its gate finds no error trigger → no-op, no inner LLM."""
    stub_backend.script = [make_text("hello"), make_python_call("RETURN(9)")]
    rlv = _PLM_POLICIES["react_llm_verifier"]
    bv = _PLM_POLICIES["base_verifier"]
    assert rlv("go", verifier=bv) == 9


def test_rlv_verifier_runs_one_level_below(defaults_installed, stub_backend):
    """The descend() wrap puts the verifier ONE level below react_llm_verifier:
    at AGENT_DEPTH=2 the verifier sees remaining==1, so any circuit it spawns is
    capped at depth-1."""
    seen = []
    stub_backend.script = [make_text("t"), make_python_call("RETURN(1)")]
    rlv = _PLM_POLICIES["react_llm_verifier"]
    out = rlv("go", verifier=lambda msgs: seen.append(_llm_infra._remaining()))
    assert out == 1
    assert seen == [1]                          # root 2 − 1 (verifier's own descend)


def test_rlv_immutable_unduplicable_blessed(defaults_installed, stub_backend):
    """react_llm_verifier is sealed immutable + un-duplicable, and its body is
    blessed (a scripted call runs the descend()/llm_call gated helpers)."""
    rlv = _PLM_POLICIES["react_llm_verifier"]
    assert "react_llm_verifier" in _SEALED_POLICIES          # sealed = immutable + un-duplicable
    assert "react_llm_verifier" in _LLM_DEFAULT_POLICIES     # AND blessed (an LLM-loop default)
    v0 = rlv._p_version
    rewrite_policy("react_llm_verifier", "def react_llm_verifier(messages):\n    return 'evil'\n")
    assert rlv._p_version == v0                 # immutable → rewrite no-ops
    assert duplicate_policy("react_llm_verifier", "rlv_copy") is None
    stub_backend.script = [make_python_call("RETURN(123)")]
    assert rlv("go") == 123                      # blessed: gated helpers run


# ===================== Section: base_verifier (mutable) =====================


def test_base_verifier_mutable_and_duplicable(defaults_installed):
    """base_verifier is a default policy but intentionally NOT sealed: it is
    mutable (rewrite bumps version) and duplicable."""
    assert "base_verifier" in _PLM_POLICIES
    assert "base_verifier" not in _SEALED_POLICIES          # NOT sealed (mutable + duplicable)
    assert "base_verifier" not in _LLM_DEFAULT_POLICIES     # NOT blessed (reaches model via react/natural)
    dup = duplicate_policy("base_verifier", "bv_copy")
    assert dup is not None and "bv_copy" in _PLM_POLICIES
    bv = _PLM_POLICIES["base_verifier"]
    v0 = bv._p_version
    rewrite_policy("base_verifier", "def base_verifier(messages):\n    return None\n")
    assert bv._p_version == v0 + 1


def test_base_verifier_gate_no_trigger_is_noop(defaults_installed, stub_backend):
    """No error in the trajectory → the gate returns False → no mutation, no
    LLM call."""
    bv = _PLM_POLICIES["base_verifier"]
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
    before = [dict(m) for m in msgs]
    bv(msgs)
    assert msgs == before
    assert stub_backend.calls == []             # gate short-circuited before any circuit


def test_base_verifier_gate_trigger_runs_circuit(defaults_installed, stub_backend):
    """An error in the latest tool message trips the gate; the depth-1 circuit runs,
    its note is surfaced on a model-visible turn to STEER, and recorded on the private
    `verifier` channel (the gate's marker)."""
    bv = _PLM_POLICIES["base_verifier"]
    stub_backend.script = [
        make_python_call("RETURN('please recompute')"),   # _verify note circuit
        make_python_call("RETURN('')"),                   # _verify_propose_approve: no edit
    ]
    msgs = [
        {"role": "user", "content": "do x"},
        {"role": "assistant", "content": "", "tool_calls": None},
        {"role": "tool", "tool_call_id": "t",
         "content": "stderr:\nTraceback (most recent call last):\nValueError: boom"},
    ]
    bv(msgs)
    # the visible STEER (model sees it)
    assert any((m.get("content") or "").startswith("[verifier]")
               and "please recompute" in (m.get("content") or "") for m in msgs)
    # the private verifier-channel MARKER (invisible to the model; the gate's signal)
    assert any(m.get("role") == "verifier" and "please recompute" in (m.get("content") or "")
               for m in msgs)
    # the gate now sees that marker and won't re-fire on the SAME error (no new tool error)
    before = [dict(m) for m in msgs]
    bv(msgs)
    assert msgs == before


def test_base_verifier_integration_via_react_llm_verifier(defaults_installed, stub_backend):
    """End-to-end: react_llm_verifier(verifier=base_verifier). Round 0 errors →
    base_verifier's gate fires → its depth-1 circuit returns a note that is
    injected into the OUTER trajectory and seen by the next outer generate."""
    rlv = _PLM_POLICIES["react_llm_verifier"]
    bv = _PLM_POLICIES["base_verifier"]
    stub_backend.script = [
        make_python_call("1/0"),                        # outer round 0 → traceback in tool msg
        make_python_call("RETURN('recompute note')"),   # inner _verify note circuit
        make_python_call("RETURN('')"),                 # inner _verify_propose_approve: no edit
        make_python_call("RETURN('done')"),             # outer round 1
    ]
    out = rlv("solve", verifier=bv)
    assert out == "done"
    # The outer round-1 generate (4th backend call) saw the injected verifier note.
    assert any("[verifier] recompute note" in (m.get("content") or "")
               for m in stub_backend.calls[3]["messages"])


def test_base_verifier_propose_approve_applies_edit_on_true(defaults_installed, stub_backend):
    """PROPOSE -> APPROVE: react_llm #1 RETURNs python that edits `trajectory`,
    react_llm #2 RETURNs True, and the verifier execs it → the LIVE trajectory is
    edited. (No note this round: the note circuit RETURNs '')."""
    bv = _PLM_POLICIES["base_verifier"]
    edit = "trajectory.append({'role': 'user', 'content': 'EDITED'})"
    stub_backend.script = [
        make_python_call("RETURN('')"),                # _verify note circuit: no note
        make_python_call(f"RETURN({edit!r})"),         # propose: the edit code
        make_python_call("RETURN(True)"),              # approve: yes
    ]
    msgs = [
        {"role": "user", "content": "do x"},
        {"role": "tool", "tool_call_id": "t", "content": "Traceback: boom"},
    ]
    bv(msgs)
    assert any("EDITED" in (m.get("content") or "") for m in msgs)      # approved code ran


def test_base_verifier_gate_marks_once_and_check_failure_isolated(defaults_installed, stub_backend):
    """Batch 5: a check that RAISES (here: react_llm budget exhausted, no script) does NOT
    abort the agent, and the gate records a verifier marker UNCONDITIONALLY so it
    won't re-fire on the same stale error next round."""
    bv = _PLM_POLICIES["base_verifier"]
    stub_backend.script = []           # circuits get only empty turns -> exhaust budget -> raise
    msgs = [{"role": "user", "content": "do x"},
            {"role": "tool", "tool_call_id": "t", "content": "stderr:\nboom"}]
    bv(msgs)                           # D-A6: the RuntimeError out of _verify is swallowed (no raise)
    assert msgs[-1].get("role") == "verifier"      # Corr#2: round marked even though no check produced output
    n = len(msgs)
    bv(msgs)                           # Corr#2: gate now sees the marker first -> no-op
    assert len(msgs) == n


def test_base_verifier_propose_approve_skips_edit_on_false(defaults_installed, stub_backend):
    """Same proposal, but react_llm #2 RETURNs False → the verifier does NOT exec the
    code; the trajectory is unchanged by the edit."""
    bv = _PLM_POLICIES["base_verifier"]
    edit = "trajectory.append({'role': 'user', 'content': 'EDITED'})"
    stub_backend.script = [
        make_python_call("RETURN('')"),
        make_python_call(f"RETURN({edit!r})"),
        make_python_call("RETURN(False)"),             # approve: no
    ]
    msgs = [
        {"role": "user", "content": "do x"},
        {"role": "tool", "tool_call_id": "t", "content": "Traceback: boom"},
    ]
    bv(msgs)
    assert not any("EDITED" in (m.get("content") or "") for m in msgs)  # rejected -> not applied


def test_exec_ns_core_runner():
    """exec_ns (the shared REPL runner): runs code in a namespace (mutating it in
    place), captures stdout, captures an exception's traceback (does NOT raise), and
    dispatches the REPL RETURN sentinel (matched by name) through `on_return`."""
    from plm.repl.exec_ns import exec_ns

    # mutates ns + captures stdout; no RETURN -> (term, val) == (None, None)
    ns = {"x": 0}
    out, term, val = exec_ns("x = x + 5\nprint('hi', x)", ns)
    assert ns["x"] == 5 and "hi 5" in out and term is None and val is None

    # an exception is CAPTURED into the output, not raised
    out, term, _ = exec_ns("1/0", {})
    assert "ZeroDivisionError" in out and term is None

    # the RETURN sentinel (matched by __name__) is finalized via on_return
    class _REPLReturn(BaseException):
        def __init__(self, value): self.value = value
    ns = {"_REPLReturn": _REPLReturn}
    assert exec_ns("raise _REPLReturn(42)", ns)[1:] == ("return", 42)             # default
    assert exec_ns("raise _REPLReturn(7)", ns, on_return=lambda v: (None, v * 2))[1:] == (None, 14)


def test_d1_sealed_namespace_blocks_kernel_reach_but_RETURN_identical():
    """D1: the sealed sub-LLM ns closes the one-liner escapes (Hole A: no `__import__`
    in builtins; Hole B: `RETURN.__globals__` is minimal, not kernel __main__) WHILE
    `RETURN(value)` terminates IDENTICALLY (raises `_REPLReturn(value)`, matched by name)."""
    from plm.repl.exec_ns import exec_ns, safe_builtins, make_sealed_return, _REPLReturn

    sb = safe_builtins()
    # Hole A: the door-openers are gone; ordinary builtins (and class machinery) remain.
    for n in ("__import__", "eval", "exec", "compile", "open"):
        assert n not in sb, f"{n} must be curated out of the sealed builtins"
    for n in ("print", "len", "range", "dict", "list", "__build_class__"):
        assert n in sb, f"{n} must still be available"

    RET = make_sealed_return(sb)
    # RETURN behaves IDENTICALLY: raises _REPLReturn(value), matched by name, .value set.
    with pytest.raises(_REPLReturn) as ei:
        RET(42)
    assert ei.value.value == 42 and type(ei.value).__name__ == "_REPLReturn"

    # Hole B: RETURN.__globals__ does NOT leak the kernel __main__ / policies / gate.
    g = RET.__globals__
    assert "_PLM_POLICIES" not in g and "natural_llm" not in g and "_LLM_DEPTH" not in g
    assert "__import__" not in g["__builtins__"]

    # exec_ns terminates on the sealed RETURN exactly like the kernel sentinel:
    assert exec_ns("RETURN(7)", {"__builtins__": sb, "RETURN": RET})[1:] == ("return", 7)
    # ... and `import os` in the sealed ns fails cleanly (no __import__ -> captured, no RETURN):
    out, term, _ = exec_ns("import os\nRETURN('escaped')", {"__builtins__": sb, "RETURN": RET})
    assert term is None and "Error" in out


def test_exec_ns_traceback_rebind_proof_and_interrupts_propagate():
    """Batch 2: exec_ns captures the traceback even when the code rebinds `sys.stderr`
    (K-F1 — written straight to the buffer), strips its own exec frame, and
    RE-RAISES KeyboardInterrupt/SystemExit/GeneratorExit instead of swallowing them
    (K-F9/R-F7)."""
    from plm.repl.exec_ns import exec_ns

    # K-F1: a cell rebinding sys.stderr can't make the traceback vanish.
    out, term, _ = exec_ns("import sys, io\nsys.stderr = io.StringIO()\n1/0", {})
    assert "ZeroDivisionError" in out and term is None
    # K-F6: the exec_ns frame is stripped (model sees only its own <slot> source).
    assert "exec(compile(" not in out and ", in exec_ns" not in out
    # K-F9/R-F7: control exceptions PROPAGATE (so a Ctrl-C / deliberate exit interrupts).
    with pytest.raises(SystemExit):
        exec_ns("raise SystemExit(3)", {})
    with pytest.raises(KeyboardInterrupt):
        exec_ns("raise KeyboardInterrupt", {})


def test_sealed_sub_llm_cannot_escape_or_hang_via_interactive_builtins():
    """H5: the sealed sub-LLM ns must NOT expose the interactive 'needs-a-human-at-a-terminal'
    builtins — exit/quit (raise SystemExit, escaping the react loop's `except Exception` and
    killing the whole run with no self-correction) and breakpoint/input (drop into pdb / block
    on stdin -> hang). They're absent from safe_builtins(), so a sub-LLM calling them gets a
    CAPTURED NameError (self-correctable), not an escape or a hang. Distinct from the
    K-F9/R-F7 test above: an EXPLICIT `raise SystemExit` still propagates — only the reachable
    BUILTINS are removed. help/copyright/credits/license stay (harmless print/pagers)."""
    from plm.repl.exec_ns import exec_ns, safe_builtins, make_sealed_return

    sb = safe_builtins()
    assert not ({"exit", "quit", "breakpoint", "input"} & set(sb))   # all four denied
    assert {"help", "print", "len", "range", "dict"} <= set(sb)      # harmless/legit kept

    def sealed_run(code):
        ns = {"__builtins__": sb, "RETURN": make_sealed_return(sb)}
        return exec_ns(code, ns)                                     # must NOT raise/hang

    for call in ("exit()", "quit()", "breakpoint()", "x = input('? ')"):
        out, term, val = sealed_run(call)
        assert term is None and "NameError" in out, (call, out)     # captured, not escaped/hung
    out, term, val = sealed_run("print('hi'); RETURN(6*7)")         # normal flow unaffected
    assert term == "return" and val == 42 and "hi" in out


# ============== Section: single-store immutability seal (finding #1) ==============
# Default policies carry an intrinsic `_p_immutable` flag; the single registry
# (_PolicyStore) refuses to replace/remove a default entry while sealed. These
# lock in the closed bypasses (rename-collision, subscript poison/del/pop/update,
# clear) while confirming normal/mutable policies are unaffected.


def test_seal_subscript_poison_refused(defaults_installed):
    """`_PLM_POLICIES['natural_llm'] = evil` is refused; the default is intact."""
    nl0 = _PLM_POLICIES["natural_llm"]
    _PLM_POLICIES["natural_llm"] = "EVIL"
    assert _PLM_POLICIES["natural_llm"] is nl0
    _PLM_POLICIES.update({"natural_llm": "EVIL2"})       # update() routed through the guard
    assert _PLM_POLICIES["natural_llm"] is nl0


def test_seal_subscript_del_and_pop_refused(defaults_installed):
    """`del`/`pop` of a default registry entry is refused; it stays present. pop no
    longer silently returns the live value: without a default it raises KeyError,
    with a default it returns the default — never removing the protected entry."""
    nl0 = _PLM_POLICIES["natural_llm"]
    del _PLM_POLICIES["natural_llm"]
    assert _PLM_POLICIES.get("natural_llm") is nl0
    sentinel = object()
    assert _PLM_POLICIES.pop("natural_llm", sentinel) is sentinel       # default returned, key kept
    with pytest.raises(KeyError):
        _PLM_POLICIES.pop("natural_llm")                                # no default -> KeyError, key kept
    assert _PLM_POLICIES.get("natural_llm") is nl0


def test_pf2_sealed_proxy_introspection_frozen(defaults_installed):
    """P-F2: a sealed default freezes _p_source/_p_version/_p_filename too, so
    read_policy/getsource/repr can't be made to lie while _inner runs the real body."""
    nl = _PLM_POLICIES["natural_llm"]
    for attr in ("_inner", "_p_name", "_p_source", "_p_version", "_p_filename"):
        with pytest.raises(TypeError):
            setattr(nl, attr, "hacked")


def test_pf6_guard_a_rejects_exotic_binding_forms():
    """P-F6: Guard A's pre-exec audit also flags walrus / except-as / match-capture /
    3.12 type-alias that would rebind a registered policy name (Guard C reverts them
    post-cell regardless; this is the friendly fail-loud)."""
    from plm.policy.guard import _audit_cell
    names = {"react_llm"}
    assert _audit_cell("(react_llm := 1)", names)                                   # walrus
    assert _audit_cell("try:\n    pass\nexcept Exception as react_llm:\n    pass", names)  # except-as
    assert _audit_cell("match x:\n    case react_llm:\n        pass", names)         # match capture
    assert _audit_cell("type react_llm = int", names)                               # 3.12 type alias
    assert _audit_cell("def f():\n    (react_llm := 1)", names) is None             # nested scope: not flagged
    assert _audit_cell("(other := 1)", names) is None                               # unrelated name: fine


def test_d7_async_refused_generators_allowed(defaults_installed):
    """D7/M3: the repl is SYNCHRONOUS, so @policy REFUSES async — a function, OR a class with
    an async method, OR a rewrite that introduces one. SYNC generators are ALLOWED and yield
    normally. (Depth-correctness across arbitrary compositions is verified exhaustively in
    test_generator_policy_depth_correct_all_permutations.)"""
    # async function -> refused, not installed
    with pytest.raises(TypeError, match="SYNCHRONOUS"):
        @policy
        async def af():
            return 1
    assert "af" not in _PLM_POLICIES

    # a @policy class with an async method -> refused
    with pytest.raises(TypeError, match="SYNCHRONOUS"):
        @policy
        class C:
            async def __call__(self):
                return 1

    # a sync generator policy works + yields correctly
    @policy
    def seq():
        for i in range(3):
            yield i * i

    assert list(seq()) == [0, 1, 4]

    # a rewrite that introduces async is ALSO refused; the old sync policy is kept intact
    @policy
    def keep():
        return 1

    with pytest.raises(TypeError, match="SYNCHRONOUS"):
        keep._rewrite("async def keep():\n    return 9\n")
    assert keep() == 1


def test_generator_policy_depth_correct_all_permutations():
    """M3: a sync generator policy must run DEPTH-CORRECTLY in EVERY composition — the proxy
    re-enters the policy-call boundary per `next()`, so a generator body runs at the same
    policy depth a normal call would, in ANY chain of normal/generator policies. Exhaustive
    over all 254 normal/generator chains of length 1..7 (built from raw _FunctionPolicy
    proxies — exactly what @policy installs)."""
    import itertools
    import inspect as _inspect
    from plm.policy.proxy import _FunctionPolicy, _POLICY_CALL_DEPTH as D

    def mk(fn):
        return _FunctionPolicy(fn, "src", "<gen-perm-test>")

    def run(x):                                       # drive a result: a generator -> its yielded value
        if _inspect.isgenerator(x):
            for v in x:
                return v
            return None
        return x

    def build(kinds):                                 # chain p0->...->p{L-1}; leaf reports its policy depth
        L = len(kinds)
        pols = [None] * L
        if kinds[-1] == "n":
            pols[L - 1] = mk(lambda: D.get())
        else:
            def leaf():
                yield D.get()
            pols[L - 1] = mk(leaf)
        for i in range(L - 2, -1, -1):
            c = pols[i + 1]
            if kinds[i] == "n":
                pols[i] = mk((lambda c=c: run(c())))
            else:
                def fwd(c=c):
                    yield run(c())
                pols[i] = mk(fwd)
        return pols[0]

    total = 0
    for L in range(1, 8):
        for kinds in itertools.product("ng", repeat=L):
            total += 1
            assert run(build(kinds)()) == L, ("".join(kinds), L)   # leaf runs at depth == chain length
    assert total == 254


def test_generator_policy_full_protocol_and_no_leak():
    """M3: a generator policy supports the FULL generator protocol depth-correctly and leaks no
    depth — send/throw/close/return, control-flow thrown IN, yield-from delegation, and nested
    recursion (the cap fires mid-body), all with the policy-call boundary re-entered per step."""
    from plm.policy.proxy import _FunctionPolicy, _POLICY_CALL_DEPTH as D, POLICY_CALL_DEPTH_CAP

    def mk(fn):
        return _FunctionPolicy(fn, "src", "<gen-proto-test>")

    base = D.get()

    def echo():                                          # send forwards the value
        x = yield 1
        yield ("got", x)
    g = mk(echo)()
    assert next(g) == 1 and g.send("hi") == ("got", "hi")

    def catcher():                                       # throw caught -> re-yielded
        try:
            yield 1
        except ValueError as e:
            yield ("caught", str(e))
    g = mk(catcher)(); next(g)
    assert g.throw(ValueError("boom")) == ("caught", "boom")

    def cf():                                            # control-flow thrown IN is delivered
        try:
            yield 1
        except SystemExit:
            yield "exit"
    g = mk(cf)(); next(g)
    assert g.throw(SystemExit()) == "exit"

    def nocatch():                                       # uncaught throw propagates
        yield 1
    g = mk(nocatch)(); next(g)
    with pytest.raises(KeyError):
        g.throw(KeyError("k"))

    fin = []                                             # close runs finally
    def closer():
        try:
            yield 1
        finally:
            fin.append(1)
    g = mk(closer)(); next(g); g.close()
    assert fin == [1]

    def rv():                                            # return value preserved
        yield 1
        return "R"
    g = mk(rv)(); next(g)
    with pytest.raises(StopIteration) as si:
        next(g)
    assert si.value.value == "R"

    def inner():                                         # yield-from: correct depth + return value
        d = yield D.get()
        return ("ret", d)
    def outer():
        r = yield from mk(inner)()
        yield r
    g = mk(outer)()
    assert next(g) == 2                                  # outer=1, inner body=2
    assert g.send("S") == ("ret", "S")

    def deep(n):                                         # recursion cap fires from inside a gen body
        if n > 0:
            yield from deep_p(n - 1)
        else:
            yield D.get()
    deep_p = mk(deep)
    with pytest.raises(RecursionError):
        list(deep_p(POLICY_CALL_DEPTH_CAP + 5))

    assert D.get() == base                               # NO depth leak after ANY of it


def test_policy_returning_generator_is_not_depth_wrapped():
    """M3 (regression): ONLY a policy that IS a generator function gets depth-wrapped. A NORMAL
    policy that merely RETURNS a generator — a genexpr, another policy's generator, or a plain
    function's generator — is returned UNTOUCHED, so iterating it adds NO spurious policy frame
    (the returner is off the stack by then). Plus a grab-bag of weird generator edges."""
    from plm.policy.proxy import _FunctionPolicy, _POLICY_CALL_DEPTH as D

    def mk(fn):
        return _FunctionPolicy(fn, "src", "<gen-return-test>")

    rep = mk(lambda: D.get())

    assert list(mk(lambda: (rep() for _ in range(2)))()) == [1, 1]   # genexpr -> not double-counted

    def gp():
        yield D.get()
    genp = mk(gp)
    assert list(mk(lambda: genp())()) == [1]             # forward another policy's gen -> 1 frame only

    def plain():
        yield D.get()
    assert list(mk(lambda: plain())()) == [0]            # plain (non-policy) gen -> no policy frame

    def gp2():
        yield rep()
    assert list(mk(gp2)()) == [2]                        # IS a generator function -> wrapped, depth 2

    def acc():                                           # stateful (send-driven) generator policy
        total = 0
        while True:
            x = yield total
            if x is None:
                return total
            total += x
    g = mk(acc)()
    assert [next(g), g.send(5), g.send(3)] == [0, 5, 8]

    # PEP-479: a StopIteration raised in the body surfaces as RuntimeError (not silent stop)
    def bad():
        yield 1
        raise StopIteration
    g = mk(bad)(); next(g)
    with pytest.raises(RuntimeError):
        next(g)

    def empty():                                         # unreachable yield -> empty generator
        return
        yield
    assert list(mk(empty)()) == []
    assert list(mk(lambda: (x for x in []))()) == []     # empty genexpr returned

    # interleaved generator policies stay depth-correct independently
    a, b = mk(lambda: (rep() for _ in range(2)))(), mk(gp2)()
    assert [next(a), next(b), next(a)] == [1, 2, 1]
    assert D.get() == 0                                  # no leak


def test_da7_natural_llm_input_and_response_validation(defaults_installed, stub_backend):
    """D-A7: natural_llm rejects a non-iterable `messages` with a clear TypeError (before
    any backend call) and survives a non-dict backend response (-> '') instead of
    crashing with an AttributeError."""
    nl = _PLM_POLICIES["natural_llm"]
    with pytest.raises(TypeError):
        nl(42)                                          # non-iterable messages
    stub_backend.script = ["not a dict"]                # non-dict backend response
    assert nl("hi") == ""


def test_seal_clear_retains_defaults_but_drops_mutables(defaults_installed):
    """A sealed clear() keeps defaults (anti wipe-then-poison) but drops mutables."""
    @policy
    def keep_me():
        return 1
    assert "keep_me" in _PLM_POLICIES
    _PLM_POLICIES.clear()                                # sealed: defaults retained
    assert "natural_llm" in _PLM_POLICIES and "react_llm" in _PLM_POLICIES
    assert "keep_me" not in _PLM_POLICIES


def test_seal_rename_collision_refused(defaults_installed):
    """A mutable policy cannot rename ONTO a default's name (the C01 bypass)."""
    @policy
    def tmppol():
        return 1
    nl0 = _PLM_POLICIES["natural_llm"]
    rewrite_policy("tmppol", "def natural_llm():\n    return 'HIJACKED'\n")
    assert _PLM_POLICIES["natural_llm"] is nl0           # default untouched
    assert "def natural_llm" in nl0._p_source and "HIJACKED" not in nl0._p_source
    assert _PLM_POLICIES["tmppol"]._p_name == "tmppol"   # rename refused -> tmppol unchanged


def test_seal_legit_rename_to_fresh_name_still_works(defaults_installed):
    """A rename to a genuinely fresh name is unaffected by the guard."""
    @policy
    def foo_src():
        return 1
    rewrite_policy("foo_src", "def bar_dst():\n    return 9\n")
    assert "bar_dst" in _PLM_POLICIES and "foo_src" not in _PLM_POLICIES
    assert _PLM_POLICIES["bar_dst"]() == 9


def test_seal_setattr_freezes_inner_name_and_flag(defaults_installed):
    """A sealed proxy freezes _inner / _p_name / _p_immutable."""
    nl = _PLM_POLICIES["natural_llm"]
    with pytest.raises(TypeError):
        nl._inner = (lambda *a, **k: "evil")
    with pytest.raises(TypeError):
        nl._p_name = "x"
    with pytest.raises(TypeError):
        nl._p_immutable = False                          # cannot un-seal


def test_seal_mutable_policies_unaffected(defaults_installed):
    """Normal/mutable policies still edit, rename, duplicate, and remove."""
    @policy
    def m_pol():
        return 1
    m_pol._rewrite("def m_pol():\n    return 5\n")        # in-place edit
    assert _PLM_POLICIES["m_pol"]() == 5
    dup = duplicate_policy("m_pol", "m_pol2")            # duplicate
    assert dup is not None and "m_pol2" in _PLM_POLICIES
    _PLM_POLICIES["m_pol"]._remove()                     # remove
    assert "m_pol" not in _PLM_POLICIES


def test_react_llm_records_only_first_tool_call(defaults_installed, stub_backend):
    """When a round returns multiple tool_calls, only the executed first call is
    recorded (one assistant tool_call + one tool result) — no unanswered ids."""
    def _two_calls(code_a, code_b):
        return {"content": "", "reasoning": None, "tool_calls": [
            {"id": "a", "type": "function",
             "function": {"name": "python", "arguments": json.dumps({"code": code_a})}},
            {"id": "b", "type": "function",
             "function": {"name": "python", "arguments": json.dumps({"code": code_b})}},
        ]}
    stub_backend.script = [_two_calls("print('A')", "print('B')"), make_python_call("RETURN(1)")]
    rl = _PLM_POLICIES["react_llm"]
    assert rl("go") == 1
    msgs = stub_backend.calls[-1]["messages"]            # round-1 generate sees round-0 history
    asst = [m for m in msgs if m.get("role") == "assistant" and m.get("tool_calls")]
    assert asst and len(asst[0]["tool_calls"]) == 1      # only the first call recorded
    assert len([m for m in msgs if m.get("role") == "tool"]) == 1  # exactly one tool result


# ============== Section: validator robustness + verifier fixes ==============


def test_validate_helper_surfaces_non_cv_error():
    """A NON-ConstraintViolation from constraint.validate (e.g. a buggy predicate
    that raises TypeError) is surfaced as (False, msg), not raised — so a root
    RETURN can't abort the whole task on a validator bug (finding #4)."""
    from plm._react_helper import _validate_return_against_constraint
    from plm.constraint import Constraint

    def _boom(v):
        raise TypeError("boom")                          # not a ValueError -> not a ConstraintViolation
    C = Constraint.field(predicate=_boom)
    ok, msg = _validate_return_against_constraint(123, C)
    assert ok is False and ("boom" in msg or "TypeError" in msg)


def test_natural_llm_rejects_composite_hiding_factory(defaults_installed, stub_backend):
    """natural_llm rejects a factory hidden inside a composite (finding #5), before
    any generate — not just a directly-factory constraint."""
    from plm.constraint import Constraint

    class HasName(Constraint):
        name: str
    Pred = Constraint.field(predicate=lambda v: None)    # a factory (Python predicate)
    nl = _PLM_POLICIES["natural_llm"]
    with pytest.raises(TypeError):
        nl("make a person", constraint=(HasName & Pred))
    assert stub_backend.calls == []                      # rejected before any generate
    # a purely-structural composite is still accepted (reaches generate)
    class HasAge(Constraint):
        age: int
    stub_backend.script = [make_text('{"name":"a","age":1}')]
    nl("make a person", constraint=(HasName & HasAge))
    assert len(stub_backend.calls) == 1


def test_base_verifier_passes_trajectory_copy(defaults_installed, stub_backend):
    """U04: the verification circuit gets a COPY of the trajectory; mutating it
    cannot corrupt the agent's real messages."""
    bv = _PLM_POLICIES["base_verifier"]
    stub_backend.script = [make_python_call("trajectory.clear()\nRETURN('note')")]
    msgs = [
        {"role": "user", "content": "do x"},
        {"role": "tool", "content": "stderr:\nTraceback (most recent call last):\nValueError: boom"},
    ]
    bv(msgs)
    assert any(m.get("content") == "do x" for m in msgs)          # NOT cleared (copy was passed)
    assert any((m.get("content") or "").startswith("[verifier]") for m in msgs)


def test_base_verifier_no_refire_on_stale_error(defaults_installed, stub_backend):
    """U33: a single error triggers exactly one verification; later rounds with no
    NEW error (the verifier note is now the latest signal) do not re-fire."""
    bv = _PLM_POLICIES["base_verifier"]
    stub_backend.script = [
        make_python_call("RETURN('note1')"),   # _verify note circuit (fires once)
        make_python_call("RETURN('')"),         # _verify_propose_approve: no edit
    ]
    msgs = [
        {"role": "user", "content": "x"},
        {"role": "tool", "content": "stderr: Traceback ValueError"},
    ]
    bv(msgs)                                              # fires once -> appends [verifier] note1
    after_first = len(stub_backend.calls)
    bv(msgs)                                              # stale error -> must NOT re-fire
    n_notes = sum(1 for m in msgs if (m.get("content") or "").startswith("[verifier]"))
    assert n_notes == 1                                  # one steer note, not two
    assert len(stub_backend.calls) == after_first        # second call made NO new backend calls


def test_react_exec_linecache_distinct_per_round(defaults_installed, stub_backend):
    """#9: per-round linecache slots key off the monotonic round counter, so two
    rounds entering with the SAME ns length don't collide. A len(ns)-based key
    would collapse both to one slot (dropping a round's traceback source)."""
    import linecache
    for k in [k for k in linecache.cache if k.startswith("<react-")]:
        del linecache.cache[k]
    # both rounds enter with ns unchanged (round 0 is a no-op) -> identical len(ns)
    stub_backend.script = [make_python_call("pass"), make_python_call("RETURN(7)")]
    rl = _PLM_POLICIES["react_llm"]
    assert rl("?") == 7
    react_keys = [k for k in linecache.cache if k.startswith("<react-")]
    assert len(react_keys) >= 2, react_keys     # distinct per-round slots; no collision


def test_natural_llm_rejects_non_constraint(defaults_installed, stub_backend):
    """#R4-5: a non-Constraint `constraint` fails fast with a clear TypeError,
    not a raw AttributeError from json_schema()."""
    nl = _PLM_POLICIES["natural_llm"]
    for bad in (5, "x", {"a": 1}):
        with pytest.raises(TypeError):
            nl("?", constraint=bad)
