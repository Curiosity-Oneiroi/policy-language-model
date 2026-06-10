from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

from plm.model_backend import ModelBackend
from plm._react_helper import (
    _build_last_round_reminder,
    _format_repl_output,
    _strip_code_fences,
    _track_model_call,
    _validate_return_against_constraint,
    clone_messages,
    public_trajectory,
)
from plm.constraint import Constraint
from plm.metaparams import get_system_prompt
from plm.plm_tools import _tool_python
from plm.repl import PythonReplSession, PREFIX, DEFAULT_PREINSTALL

_PLM_ROOT_DEPTH: int = 2


_INTERNAL_KWARGS: frozenset[str] = frozenset({
    "cumulative_prompt_tokens",
    "original_cumulative_prompt_tokens",
})


_BACKEND_DEPS: dict[str, tuple[str, ...]] = {
    "SlateBackend":     ("httpx",),
    "OpenAIBackend":    ("openai",),
    "AnthropicBackend": ("anthropic",),
    "VLLMBackend":      ("openai",),
}


class PLMTaskFailure(RuntimeError):
    """Raised when PLM exhausts max_turns + return_budget without a successful
    RETURN(obj) call."""
    pass


def _policy_def_name(source: str, where: str) -> str:
    """A policy's NAME is intrinsic to its SOURCE — the single top-level @policy def/class name —
    never a filename or dict key (those are only transport labels). Parse `source` and return that
    name, raising a clear error unless it is exactly one top-level def/class."""
    import ast
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise ValueError(f"PLMMetaParameters: {where}: source is not valid Python ({e})")
    defs = [n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    if len(defs) != 1:
        raise ValueError(
            f"PLMMetaParameters: {where}: a policy is exactly ONE top-level def/class "
            f"(found {len(defs)}); the policy name comes from that def/class, not the filename/key"
        )
    return defs[0].name


@dataclass
class PLMMetaParameters:
    system_prompt: str
    extra_policies: dict[str, str] | None = None         # mutable + duplicable (normal)
    sealed_policies: dict[str, str] | None = None        # immutable + un-duplicable (sealed, NOT blessed)

    def __post_init__(self) -> None:
        if not (self.system_prompt or "").strip():       # NP3-8: empty/whitespace system prompt is
            import warnings                               # almost always a mistake (e.g. an empty
            warnings.warn(                                # system_prompt.md) — PLM would run with NO
                "PLMMetaParameters: system_prompt is empty/whitespace-only — PLM will run with no "
                "system instructions; provide a non-empty system prompt.", stacklevel=2)
        for _attr in ("extra_policies", "sealed_policies"):
            _v = getattr(self, _attr)
            if _v is not None and not (
                isinstance(_v, dict)
                and all(isinstance(k, str) and isinstance(s, str) for k, s in _v.items())
            ):
                raise TypeError(
                    f"PLMMetaParameters.{_attr} must be a dict[str, str] "
                    f"(policy name -> @policy source) or None"
                )
        # Re-key both buckets by the REAL policy name (the @policy def/class name parsed from each
        # source), NEVER the dict key / file stem the caller or from_dir supplied. A policy's name
        # is intrinsic to its def/class; a filename or param key is only a transport label, and
        # trusting it silently mis-registers / mis-seals (this is the API-side half of NP3-4; the
        # kernel seals the install delta as the matching backstop). Also catches two same-name files.
        for _attr in ("extra_policies", "sealed_policies"):
            _bucket = getattr(self, _attr)
            if not _bucket:
                continue
            _canon: dict[str, str] = {}
            for _k, _src in _bucket.items():
                _name = _policy_def_name(_src, f"{_attr}[{_k!r}]")
                if _name in _canon:
                    raise ValueError(
                        f"PLMMetaParameters.{_attr}: two policies both define {_name!r} "
                        f"(the name comes from the def/class, so they collide)"
                    )
                _canon[_name] = _src
            setattr(self, _attr, _canon)
        # A name in BOTH buckets is ambiguous (mutable vs sealed) — reject (REAL names now).
        if self.extra_policies and self.sealed_policies:
            _dup = sorted(set(self.extra_policies) & set(self.sealed_policies))
            if _dup:
                raise ValueError(
                    f"PLMMetaParameters: policy name(s) {_dup} appear in BOTH extra_policies "
                    f"(mutable) and sealed_policies (immutable) — put each in exactly one"
                )

    @classmethod
    def from_dir(cls, path: Any) -> "PLMMetaParameters":
        """Load a metaparam FOLDER:

            <path>/
              system_prompt.md            # (or .txt) — the system prompt (required)
              policies/
                sealed/  <name>.py        # immutable + un-duplicable (NOT blessed)
                mutable/ <name>.py        # mutable + duplicable (normal)

        Each `.py` is one `@policy def/class <name>(...)`; the policy NAME comes from that
        def/class — the filename is only a label (a file `auth.py` defining `authenticate`
        registers `authenticate`). Files starting with `_` are skipped as helpers. Either
        policies subfolder may be absent. Point PLM at one such folder per call — multiple
        folders under a common root act as a library of selectable metaparams.
        """
        import pathlib
        base = pathlib.Path(path)
        if not base.is_dir():
            raise FileNotFoundError(f"PLMMetaParameters.from_dir: not a directory: {base}")
        sp = next((base / f for f in ("system_prompt.md", "system_prompt.txt") if (base / f).is_file()), None)
        if sp is None:
            raise FileNotFoundError(
                f"PLMMetaParameters.from_dir: missing system_prompt.md (or .txt) in {base}"
            )

        def _load(sub: str) -> dict[str, str]:
            out: dict[str, str] = {}
            pdir = base / "policies" / sub
            if pdir.is_dir():
                for f in sorted(pdir.glob("*.py")):
                    if not f.name.startswith("_"):       # skip helper/private files
                        out[f.stem] = f.read_text(encoding="utf-8")
            return out

        return cls(
            system_prompt=sp.read_text(encoding="utf-8"),
            extra_policies=_load("mutable") or None,
            sealed_policies=_load("sealed") or None,
        )


class PLM:
    """PLM meta-reasoner
    """

    def __init__(
        self,
        *,
        model_backend: ModelBackend,
        metaparams: PLMMetaParameters,
        max_turns: int = 100,
        return_budget: int = 5,
        pip_install_packages: str = "",
        tool_timeout: Optional[float] = None,
        session_workspace: Optional[str] = None,
    ):
        self.model_backend = model_backend
        self.metaparams = metaparams
        self.max_turns = max_turns
        self.return_budget = return_budget
        self.pip_install_packages = pip_install_packages
        self.tool_timeout = tool_timeout
        self.session_workspace = session_workspace

    async def __call__(
        self,
        messages: List[Dict[str, Any]],
        *,
        constraint: Optional[Type[Constraint]] = None,
        **generate_kwargs: Any,
    ) -> Dict[str, Any]:
        """Drive the React loop until RETURN or budget exhaustion.

        Returns:
            {
                "answer":   EXACTLY the value PLM passed to RETURN(...) — the
                            Constraint-validated/coerced value when a constraint was
                            set, else whatever object the model returned. (PLM and
                            its sub-policies always finalize via RETURN, so the
                            terminal assistant text is never the answer.)
                "messages": final trajectory (system prompt + all rounds),
                "metrics":  {"main_turns", "model_calls", "tool_calls", "total_tokens"}
            }
        """
        
        messages = list(messages)
        messages.insert(0, {"role": "system", "content": self.metaparams.system_prompt})

        repl_env = {"AGENT_DEPTH": str(_PLM_ROOT_DEPTH)}
        if self.session_workspace:
            repl_env["SESSION_WORKSPACE"] = self.session_workspace

        mb = self.model_backend
        spec: dict[str, Any] = {
            "model_backend_class_name": type(mb).__name__,
            "model": getattr(mb, "model", None),
        }

        # `use_responses_api` round-trips an EXPLICIT API-style override to the
        # kernel sub-LLM. OpenAIBackend.from_spec already reads the key; without it
        # here, an explicit `use_responses_api=` set against the model-name
        # auto-detection silently flips in-kernel. The hasattr guard no-ops it for
        # the other backends (they lack the attr).
        for attr in ("api_key", "base_url", "max_context_length", "use_responses_api"):
            if hasattr(mb, attr):
                spec[attr] = getattr(mb, attr)
        repl_env["_PLM_BACKEND_SPEC"] = json.dumps(spec)

        if getattr(self.metaparams, "extra_policies", None):
            repl_env["_PLM_EXTRA_POLICIES"] = json.dumps(self.metaparams.extra_policies)
        if getattr(self.metaparams, "sealed_policies", None):
            repl_env["_PLM_SEALED_EXTRA_POLICIES"] = json.dumps(self.metaparams.sealed_policies)

        # Append the chosen backend's runtime SDK dep so the kernel's per-session
        # uv venv can `import plm.model_backend.<x>` and call it successfully.
        backend_deps = _BACKEND_DEPS.get(type(mb).__name__, ())
        extra_packages = (
            tuple(self.pip_install_packages.split()) if self.pip_install_packages.strip() else ()
        ) + backend_deps

        repl = PythonReplSession(
            workspace=self.session_workspace,
            env=repl_env,
            cell_timeout=self.tool_timeout,
            code_prefix=PREFIX,
            preinstall=DEFAULT_PREINSTALL + extra_packages,
        )

        tools = [_tool_python()]

        clean_generate_kwargs = {
            k: v for k, v in generate_kwargs.items() if k not in _INTERNAL_KWARGS
        }
        clean_generate_kwargs.setdefault(
            "chat_template_kwargs",
            generate_kwargs.get("chat_template_kwargs", {"enable_thinking": True}),
        )

        plm_ctx: Dict[str, Any] = {"main_turns": 0}
        token_state: Dict[str, Any] = {
            "cumulative_prompt_tokens": generate_kwargs.get("cumulative_prompt_tokens", 0) or 0,
        }
        metrics: Dict[str, Any] = {
            "model_calls": [],
            "tool_calls": [],
            "total_tokens": 0,
            "main_turns": 0,
        }

        constraint_is_set = constraint is not None
        _DONE = object()
        terminal_answer: Any = _DONE
        reasoning: Optional[str] = None

        total_rounds_budget = self.max_turns + self.return_budget

        seeded_count = 0
        seeded_epoch: Optional[int] = None

        # ---- React loop ------------------------------------------------------
        try:
            for _round in range(total_rounds_budget):

                if _round == self.max_turns - 1:
                    messages.append(_build_last_round_reminder(constraint))

                msgs = clone_messages(messages)

                call_start = time.time()
                cpt = token_state.get("cumulative_prompt_tokens", 0) or 0
                
                response = await self.model_backend.generate(
                    messages=msgs,
                    tools=tools,
                    cumulative_prompt_tokens=cpt if cpt > 0 else None,
                    **clean_generate_kwargs,
                )
                
                _track_model_call(response, call_start, metrics, token_state)

                content = response.get("content", "") or ""
                reasoning = response.get("reasoning")
                plm_ctx["main_turns"] = plm_ctx.get("main_turns", 0) + 1

                tool_calls = response.get("tool_calls") or []
                if isinstance(tool_calls, dict):                  # a single call returned bare -> wrap
                    tool_calls = [tool_calls]
                elif not isinstance(tool_calls, list):            # any other malformed shape -> none
                    tool_calls = []

                messages.append({
                    "role": "assistant",
                    "content": content,
                    "reasoning": reasoning,
                    # Store ONLY the executed+answered call, and ONLY if it's a dict — a non-dict
                    # entry would corrupt the next round's history sanitizer. Mirrors react_llm.
                    "tool_calls": ([tool_calls[0]] if tool_calls and isinstance(tool_calls[0], dict) else None),
                    "_timestamp": time.time(),
                })

                if not tool_calls:
                    # Text-only round (the model emitted no python call). This
                    # never terminates the task — only a successful RETURN does.
                    # Whatever it said, just take another round.
                    continue

                tool_call = tool_calls[0]
                # PLM emitted parallel tool_calls; the root loop runs ONLY the first. Surface the
                # nudge in PLM's OWN tool result (its message stream) so PLM self-corrects to one
                # python call per turn — mirrors react_llm's note to its sub-LLM: each model's
                # coaching stays in its own stream, never leaks across levels.
                _parallel_note = (
                    "[plm] you emitted " + str(len(tool_calls)) + " tool_calls; only the FIRST ran "
                    "— send ONE `python` call per turn.\n\n") if len(tool_calls) > 1 else ""

                if not isinstance(tool_call, dict):
                    # Malformed first tool_call from a non-conformant backend: the assistant turn
                    # stored tool_calls=None for it, so reply as a USER turn (NOT an orphan `tool`).
                    # Mirrors react_llm's hardening; the root loop's outer try has no except,
                    # so without this a bare-subscript below would crash PLM.__call__ raw.
                    messages.append({"role": "user",
                                     "content": _parallel_note + f"[error] malformed tool_call "
                                                f"({type(tool_call).__name__}); emit a single `python` tool call"})
                    metrics["tool_calls"].append({
                        "tool_name": None, "tool_status": "malformed_tool_call",
                        "tool_result": f"malformed tool_call ({type(tool_call).__name__})"})
                    continue
                _fn = tool_call.get("function")
                _fn = _fn if isinstance(_fn, dict) else {}        # `function: null`/malformed -> {} (mirror react)
                tname = _fn.get("name")

                raw_args = _fn.get("arguments")
                if isinstance(raw_args, dict):
                    targs = raw_args
                elif isinstance(raw_args, str):
                    try:
                        targs = json.loads(raw_args)
                    except Exception:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.get("id") or "",
                            "content": _parallel_note + f"error: invalid JSON arguments for tool {tname!r}",
                        })
                        metrics["tool_calls"].append({
                            "tool_name": tname, "tool_status": "code_arg_invalid",
                            "tool_result": f"error: invalid JSON arguments for tool {tname!r}",
                        })
                        continue
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id") or "",
                        "content": _parallel_note + f"error: invalid arguments for tool {tname!r}",
                    })
                    metrics["tool_calls"].append({
                        "tool_name": tname, "tool_status": "code_arg_invalid",
                        "tool_result": f"error: invalid arguments for tool {tname!r}",
                    })
                    continue
                

                if tname != "python":
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id") or "",
                        "content": _parallel_note + f"error: tool {tname!r} not supported (only `python`)",
                    })
                    metrics["tool_calls"].append({
                        "tool_name": tname, "tool_status": "code_arg_invalid",
                        "tool_result": f"error: tool {tname!r} not supported (only `python`)",
                    })
                    continue

                if not isinstance(targs, dict):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id") or "",
                        "content": _parallel_note + f"error: arguments for tool {tname!r} must be a JSON object",
                    })
                    metrics["tool_calls"].append({
                        "tool_name": tname, "tool_status": "code_arg_invalid",
                        "tool_result": f"error: arguments for tool {tname!r} must be a JSON object",
                    })
                    continue

                raw_code = targs.get("code", "")
                if not isinstance(raw_code, str):
                    # non-string `code` (null/number/object): clear error + retry, NOT a raw
                    # AttributeError from `_strip_code_fences`
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id") or "",
                        "content": _parallel_note + "error: `code` must be a string of Python source",
                    })
                    metrics["tool_calls"].append({
                        "tool_name": tname, "tool_status": "code_arg_invalid",
                        "tool_result": "error: `code` must be a string of Python source",
                    })
                    continue
                try:
                    code = _strip_code_fences(raw_code)
                except SyntaxError as e:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id") or "",
                        "content": _parallel_note + f"stderr:\n{e}",
                    })
                    metrics["tool_calls"].append({
                        "tool_name": tname, "tool_status": "code_arg_invalid",
                        "tool_result": str(e)[:500],
                    })
                    continue

                if repl.kernel_epoch != seeded_epoch:
                    seeded_count = 0
                    seeded_epoch = repl.kernel_epoch
                delta = public_trajectory(messages[seeded_count:])
                seeded_count = len(messages)
                
                exec_result = await asyncio.to_thread(
                    repl.execute_cell, code, None, delta
                )
                etype = exec_result.get("type", "result")

                if etype == "return":
                    return_obj = exec_result.get("return_obj")
                    ok, parsed_or_err = _validate_return_against_constraint(
                        return_obj, constraint,
                    )
                    if not ok:
                        # Constraint validation failed → surface the error and
                        # let the model retry in the next round.
                        augmented = dict(exec_result)
                        augmented["stderr"] = (
                            (exec_result.get("stderr") or "").rstrip()
                            + "\n\n" + parsed_or_err + "\n"
                        )
                        out = _parallel_note + _format_repl_output(augmented)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.get("id") or "",
                            "content": out,
                        })
                        metrics["tool_calls"].append({
                            "tool_name": tname, "tool_status": "return_invalid",
                            "tool_result": out[:500],
                        })
                    else:
                        terminal_answer = parsed_or_err
                        out = _format_repl_output(exec_result)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.get("id") or "",
                            "content": out + "\n[PLM] RETURN accepted; task terminating.",
                        })
                        metrics["tool_calls"].append({
                            "tool_name": tname, "tool_status": "return_valid",
                            "tool_result": out[:500],
                        })
                else:
                    out = _parallel_note + _format_repl_output(exec_result)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id") or "",
                        "content": out,
                    })
                    metrics["tool_calls"].append({
                        "tool_name": tname, "tool_status": "completed",
                        "tool_result": out[:500],
                    })

                if terminal_answer is not _DONE:
                    break
            else:
                raise PLMTaskFailure(
                    f"PLM exhausted max_turns ({self.max_turns}) + "
                    f"return_budget ({self.return_budget}) without calling "
                    f"RETURN successfully"
                    + (" (an output constraint was configured)" if constraint_is_set else "")
                    + "."
                )
        finally:
            try:
                repl.close()
            except Exception:
                pass

        metrics["main_turns"] = plm_ctx.get("main_turns", 0)

        return {
            "answer": terminal_answer,     # exactly the value PLM passed to RETURN(...)
            "messages": messages,
            "metrics": metrics,
        }


# ---- Quick demo (run as script: python -m plm.plm or `python plm/plm.py`) ----

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run a one-shot PLM task.")
    parser.add_argument("--backend", choices=["openai", "vllm", "slate"], default="openai")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--prompt", required=True, help="The user prompt for PLM to solve")
    args = parser.parse_args()

    from plm.model_backend.openai_backend import OpenAIBackend
    from plm.model_backend.vllm_backend import VLLMBackend
    from plm.model_backend.slate_backend import SlateBackend
    backend_cls = {"openai": OpenAIBackend, "vllm": VLLMBackend, "slate": SlateBackend}[args.backend]
    backend = backend_cls(model=args.model, base_url=args.base_url, api_key=args.api_key)
    metaparams = PLMMetaParameters(system_prompt=get_system_prompt())
    plm = PLM(model_backend=backend, metaparams=metaparams, max_turns=args.max_turns)

    async def _main():
        result = await plm([{"role": "user", "content": args.prompt}])
        print("---- ANSWER ----")
        print(result["answer"])
        print("\n---- METRICS ----")
        print(json.dumps(result["metrics"], indent=2, default=str))

    asyncio.run(_main())
