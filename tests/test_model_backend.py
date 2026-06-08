"""Tests for plm.model_backend — the self-contained backends ported from
AFramework with the AFW couplings (@resource / ResourceManager / registries /
AFW Logger) stripped. No network: we only build via `from_spec` and assert the
shape `_make_backend` relies on; we never call `.generate`.
"""

from __future__ import annotations

import importlib
import pathlib

import pytest

from plm.model_backend import ModelBackend

# class name -> module suffix (mirrors _llm_infra._MOD)
_BACKENDS = {
    "OpenAIBackend": "openai_backend",
    "VLLMBackend": "vllm_backend",
    "AnthropicBackend": "anthropic_backend",
    "SlateBackend": "slate_backend",
}


def test_package_import_is_dependency_free():
    """`import plm.model_backend` exposes the ABC without importing any backend
    SDK (openai/anthropic/httpx) — concrete backends are imported lazily."""
    import plm.model_backend as mb
    assert mb.ModelBackend is ModelBackend


@pytest.mark.parametrize("cls_name,mod_name", list(_BACKENDS.items()))
def test_backend_from_spec_constructs(cls_name, mod_name):
    """Each backend builds from a spec exactly as `_make_backend` does, with no
    network and no AFramework — and exposes `.model` + an async `.generate`."""
    cls = getattr(importlib.import_module("plm.model_backend." + mod_name), cls_name)
    spec = {
        "model_backend_class_name": cls_name,
        "model": "test-model",
        "api_key": "EMPTY",
        "base_url": "http://localhost:1234/v1",
        "max_context_length": 4096,
    }
    be = cls.from_spec(spec, None)
    assert isinstance(be, ModelBackend)
    assert be.model == "test-model"
    assert callable(getattr(be, "generate", None))


def test_backends_have_no_afw_couplings():
    """Static guard: no ported file imports AFramework / its engine / registries /
    logger, and no `@resource` decorator survived the strip."""
    import plm.model_backend as mb
    d = pathlib.Path(mb.__path__[0])
    bad = ("from AFramework", "import AFramework", "from ..engine",
           "from ..registries", "from ..logger")
    for f in sorted(d.glob("*.py")):
        for ln in f.read_text().splitlines():
            s = ln.strip()
            assert not s.startswith(bad), f"{f.name}: {ln!r}"
            assert not s.startswith("@resource("), f"{f.name}: {ln!r}"


def test_backend_spec_round_trips_credentials():
    """#R4-1: PLM builds the kernel _PLM_BACKEND_SPEC via a hasattr loop over
    (api_key, base_url, max_context_length); each backend must expose those so a
    custom credential/endpoint survives into the in-cell sub-LLM (not silently
    fall back to OPENAI_API_KEY + api.openai.com)."""
    def build_spec(mb):                       # mirrors plm.py PLM.__call__ spec build
        spec = {"model_backend_class_name": type(mb).__name__, "model": getattr(mb, "model", None)}
        for attr in ("api_key", "base_url", "max_context_length"):
            if hasattr(mb, attr):
                spec[attr] = getattr(mb, attr)
        return spec

    OpenAIBackend = getattr(importlib.import_module("plm.model_backend.openai_backend"), "OpenAIBackend")
    mb = OpenAIBackend(api_key="SECRET", base_url="http://proxy:1234/v1", model="m", max_context_length=4096)
    spec = build_spec(mb)
    assert spec.get("api_key") == "SECRET" and spec.get("base_url") == "http://proxy:1234/v1"
    rebuilt = OpenAIBackend.from_spec(spec, None)
    assert rebuilt._client_kwargs.get("api_key") == "SECRET"
    assert rebuilt._client_kwargs.get("base_url") == "http://proxy:1234/v1"

    AnthropicBackend = getattr(importlib.import_module("plm.model_backend.anthropic_backend"), "AnthropicBackend")
    ab = AnthropicBackend(api_key="AK", model="claude-x", max_context_length=4096)
    assert build_spec(ab).get("api_key") == "AK"


def test_vllm_from_spec_uses_round_tripped_max_context_length():
    """#R5-4: PLM round-trips the parent's RESOLVED context under the
    'max_context_length' key — the same key OpenAI/Slate/Anthropic from_spec read.
    VLLMBackend historically read only the never-emitted 'max_context_override',
    so the value was dropped in-kernel and __init__ fell through to a BLOCKING
    requests.get(base_url+'/models') probe on every generate. from_spec must now
    consume the round-tripped value (no probe, no 150000 fallback); an explicit
    'max_context_override' still wins."""
    VLLMBackend = getattr(importlib.import_module("plm.model_backend.vllm_backend"), "VLLMBackend")
    spec = {
        "model_backend_class_name": "VLLMBackend",
        "model": "Qwen/Qwen3",
        "base_url": "http://vllm.invalid:9999/v1",     # a real /models probe would FAIL here
        "api_key": "EMPTY",
        "max_context_length": 262144,                  # the parent's resolved value
    }
    be = VLLMBackend.from_spec(spec, None)
    assert be.max_context_length == 262144             # round-tripped; not the 150000 fallback
    # an explicit override still takes precedence over the round-tripped value
    be2 = VLLMBackend.from_spec(dict(spec, max_context_override=4096), None)
    assert be2.max_context_length == 4096


def test_vllm_spec_round_trips_api_key():
    """#R6-3: VLLMBackend must expose self.api_key so PLM's spec-builder hasattr
    loop round-trips an authenticated vLLM key — it previously stored only
    base_url, silently dropping the key on the in-kernel from_spec round-trip
    (the credential fix covered OpenAI/Anthropic/Slate but missed VLLM)."""
    VLLMBackend = getattr(importlib.import_module("plm.model_backend.vllm_backend"), "VLLMBackend")
    be = VLLMBackend(base_url="http://vllm.invalid:9999/v1", model="m",
                     api_key="VLLM_SECRET", max_context_override=4096)   # override -> no probe
    assert be.api_key == "VLLM_SECRET"                 # exposed as a plain attr
    spec = {"model_backend_class_name": "VLLMBackend", "model": getattr(be, "model", None)}
    for attr in ("api_key", "base_url", "max_context_length", "use_responses_api"):
        if hasattr(be, attr):
            spec[attr] = getattr(be, attr)
    assert spec.get("api_key") == "VLLM_SECRET"        # emitted by the builder loop
    assert VLLMBackend.from_spec(spec, None).api_key == "VLLM_SECRET"   # preserved in-kernel


def test_openai_spec_round_trips_use_responses_api():
    """#R6-5: an EXPLICIT use_responses_api override must round-trip to the kernel
    sub-LLM. OpenAIBackend.from_spec already reads the key; PLM's spec builder now
    emits it, so an override set against the model-name auto-detection no longer
    silently flips in-kernel. The hasattr guard no-ops it for the other backends."""
    OpenAIBackend = getattr(importlib.import_module("plm.model_backend.openai_backend"), "OpenAIBackend")
    be = OpenAIBackend(model="gpt-4o", api_key="x", use_responses_api=True)  # gpt-4o auto-detects False
    assert be.use_responses_api is True
    spec = {"model_backend_class_name": "OpenAIBackend", "model": getattr(be, "model", None)}
    for attr in ("api_key", "base_url", "max_context_length", "use_responses_api"):
        if hasattr(be, attr):
            spec[attr] = getattr(be, attr)
    assert spec.get("use_responses_api") is True       # emitted by the builder
    rebuilt = OpenAIBackend.from_spec(spec, None)
    assert rebuilt.use_responses_api is True           # preserved (not re-auto-detected to False)


def test_openai_reasoning_classifier_consistent_across_sites():
    """O6: the `use_responses_api` auto-detect and the `_is_reasoning_model` property must NEVER
    disagree, and must cover o2/o4/o5+ + uppercase + a `provider/` prefix. Before, the auto-detect
    used a literal ["o1","o3","gpt-5"] substring list (missed o2/o4, mishandled prefixes) while the
    property used an anchored case-sensitive regex — so an o4-mini/proxied id got routed down the
    wrong API path (temperature -> 400, or a missing reasoning dict)."""
    mod = importlib.import_module("plm.model_backend.openai_backend")
    OpenAIBackend, _is = mod.OpenAIBackend, mod._is_reasoning_model_name
    reasoning = ["o1", "o3", "o4-mini", "o2", "O1-preview", "openai/o3-mini", "azure/gpt-5", "gpt-5"]
    non = ["gpt-4o", "gpt-4", "gpt-3.5-turbo", "claude-3"]
    for m in reasoning:
        b = OpenAIBackend(model=m, base_url="http://x", api_key="x")
        assert b.use_responses_api is True and b._is_reasoning_model is True, m
    for m in non:
        b = OpenAIBackend(model=m, base_url="http://x", api_key="x")
        assert b.use_responses_api is False and b._is_reasoning_model is False, m
    for m in reasoning + non:                          # one shared classifier -> can never disagree
        b = OpenAIBackend(model=m, base_url="http://x", api_key="x")
        assert b.use_responses_api == b._is_reasoning_model == _is(m), m


def test_sanitize_strips_verifier_role_messages():
    """role 'verifier' is a PRIVATE trajectory channel — the shared
    `_sanitize_messages_for_api` drops it entirely so it never reaches the provider
    API (the model can't see verifier notes); all other roles pass through."""
    OpenAIBackend = getattr(importlib.import_module("plm.model_backend.openai_backend"), "OpenAIBackend")
    be = OpenAIBackend(model="m", api_key="x")
    msgs = [
        {"role": "user", "content": "do x"},
        {"role": "verifier", "content": "private verifier note"},
        {"role": "assistant", "content": "ok"},
    ]
    out = be._sanitize_messages_for_api(msgs)
    assert [m["role"] for m in out] == ["user", "assistant"]            # verifier dropped
    assert all("private verifier note" not in (m.get("content") or "") for m in out)


def test_sanitize_history_drops_nondict_tool_call():
    """M4 (defense-in-depth): the shared `_sanitize_message_history` must DROP a malformed
    (non-dict) tool_call instead of crashing on `tc.copy()` — so ANY backend survives a
    non-conformant tool_call that reached history from any source (not only react's storage)."""
    OpenAIBackend = getattr(importlib.import_module("plm.model_backend.openai_backend"), "OpenAIBackend")
    be = OpenAIBackend(model="m", api_key="x")
    msgs = [{"role": "assistant", "content": "",
             "tool_calls": ["BAD", 42, {"id": "1", "function": {"name": "f"}}]}]
    out = be._sanitize_message_history(msgs, {})                        # must not raise
    assert [type(tc).__name__ for tc in out[0]["tool_calls"]] == ["dict"]   # non-dicts dropped, dict kept
