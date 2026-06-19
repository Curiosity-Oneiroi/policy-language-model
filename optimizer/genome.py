"""Candidate genome — (system_prompt, verifier) ONLY.

The seed CODE — chess `policies/mutable/policyzero.py` + any sealed policies — is
FROZEN and copied verbatim into every candidate. Circuits ARE NOT genome: they
are runtime artifacts that the lead INVENTS during a run.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


# Path to the FROZEN chess seed metaparam dir (every candidate inherits these
# `policies/` verbatim). Resolved at import time so callers don't pass it.
_SEED_DIR_DEFAULT = (Path(__file__).resolve().parent.parent / "metaparams" / "chess")


@dataclass
class Candidate:
    """One point in the (system_prompt, verifier) genome space.

    `id` is a short stable handle for ledger rows and run_dirs.
    `parent_id` is None for seed candidates, else the id of the candidate this
    one was proposed FROM (lets us reconstruct the genealogy tree).
    `lineage_note` is a free-text crumb the proposer leaves explaining the edit
    (used purely for logging / inspection, never read by the loop).
    """

    id: str
    system_prompt: str
    verifier: str                                       # python source — MUST define ONE top-level `def verifier(messages)` function. Runs IN THE POLICY-SPACE KERNEL with the full ambient namespace; mutates the trajectory in place.
    parent_id: Optional[str] = None
    lineage_note: str = ""

    def materialize(self, dest_dir: Path, seed_src_dir: Optional[Path] = None) -> Path:
        """Write this candidate as a PLM metaparam dir at `dest_dir`.

        Layout produced (matches `PLMMetaParameters.from_dir`):
            dest_dir/
              system_prompt.md      # this candidate's prompt
              verifier.py           # this candidate's verifier source
              policies/             # copied VERBATIM from `seed_src_dir`
                mutable/policyzero.py
                sealed/...

        `seed_src_dir` defaults to the chess seed. The seed's `policies/` is
        copied verbatim; if the seed has its own `system_prompt.md` or
        `verifier.py` they are IGNORED (this candidate's strings win).
        """
        seed = Path(seed_src_dir) if seed_src_dir else _SEED_DIR_DEFAULT
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)

        (dest / "system_prompt.md").write_text(self.system_prompt, encoding="utf-8")
        (dest / "verifier.py").write_text(self.verifier, encoding="utf-8")

        # Copy the seed's frozen `policies/` tree verbatim. We rmtree + copytree
        # so a re-materialize is idempotent (drops any stale files an earlier
        # candidate wrote into the same dest_dir).
        seed_policies = seed / "policies"
        dest_policies = dest / "policies"
        if dest_policies.exists():
            shutil.rmtree(dest_policies)
        if seed_policies.is_dir():
            shutil.copytree(seed_policies, dest_policies)
        return dest


# ---------- seed verifiers --------------------------------------------------

# Candidate A: baseline — no-op verifier. Same plumbing path as B (so the
# verifier_calls/errors counters move), but does NOT mutate the trajectory.
#
# The verifier now runs IN THE POLICY-SPACE KERNEL after each non-terminal round
# with the FULL ambient namespace a normal cell/policy sees — `react_llm`,
# `natural_llm`, `parallel`, the policy edit APIs (`read_policy`/`edit_policy`/
# `duplicate_policy`/...), `policyzero`, and every PLM object/artefact — and
# `messages` is the WRITABLE live trajectory (mutate it IN PLACE: inject user or
# system turns, edit, prune). It is SEALED + HIDDEN (not in the policy registry)
# and may do anything a cell can, EXCEPT abort the run or edit a sealed policy.
# These seeds stay deliberately SIMPLE (a no-op + a user-role nudge), but the
# optimizer is free to evolve them into anything within that contract.
_VERIFIER_NOOP = '''\
def verifier(messages):
    """Baseline verifier: pure no-op (control candidate). Confirms the kernel-
    resident hook fires without biasing the trajectory. Runs in the policy space
    with the full ambient namespace but mutates nothing. CONTRACT: a verifier is
    ONE top-level `def verifier(messages)` and nothing else — any helpers/imports/
    constants live INSIDE the body."""
    return
'''

# Candidate B: a deliberately SIMPLE delegate-or-measure nudge. The signal
# we encode here:
#   * Look at the last assistant turn.
#   * If it carries a python tool_call whose `code` is "long" (>= 800 chars)
#     AND does not mention `delegate(` / `react_llm(` / `measure(` / `simulate(`,
#     append a USER-role reminder telling the lead to delegate or measure.
# USER role is deliberate: per the prompt-optimizer research notes, the SUBAGENT
# backend (Slate/opus on `behavior_mode=subagent`) ignores most NEW system turns
# mid-stream — a user-role nudge is the reliable channel for caller corrections.
_VERIFIER_DELEGATE_NUDGE = '''\
def verifier(messages):
    """Delegate-or-measure nudge verifier (kernel-resident, SINGLE self-contained def).

    Runs in the policy space each non-terminal round with the full ambient
    namespace (react_llm / natural_llm / parallel / policy edit APIs / policyzero).
    `messages` is the live, writable trajectory — this seed only APPENDS a user
    turn, but a more powerful evolved verifier could call sub-policies, fork/edit a
    mutable policy, or prune/edit turns. It may NOT abort the run or edit a sealed
    policy. CONTRACT: the verifier is ONE top-level `def verifier(messages)` and
    nothing else — every helper, import, and constant lives INSIDE this body (so
    nothing leaks into the kernel namespace). Cross-round state, if needed, lives
    in the kernel namespace, e.g. `globals().setdefault("_v_state", {})`.

    Heuristic: if the latest assistant turn carried a LONG python tool_call whose
    code does NOT obviously delegate (react_llm / natural_llm) or measure
    (simulate / measure), inject a USER-role reminder to delegate or measure.
    USER role (not SYSTEM) is required because real subagent backends (Slate opus
    on behavior_mode=subagent) drop NEW system messages mid-conversation; user
    turns are the reliable mid-stream injection channel.
    """
    import json

    LONG_CODE_THRESHOLD = 800
    DELEGATION_TOKENS = ("react_llm", "react_verifier_llm", "simulate(", "measure(", "natural_llm")
    NUDGE = (
        "[meta-coach] You just wrote a long inline code block. Before another, "
        "ask: can a sub-policy (react_llm / react_verifier_llm) do this work, "
        "or has any of it been MEASURED (simulate / a benchmark) on the current "
        "policy? Spend the next turn DELEGATING or MEASURING, not coding inline."
    )

    def last_assistant_python_code(msgs):
        for m in reversed(msgs):
            if m.get("role") != "assistant":
                continue
            tcs = m.get("tool_calls") or []
            if not tcs:
                return None
            fn = (tcs[0] or {}).get("function") or {}
            if fn.get("name") != "python":
                return None
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    return None
            if not isinstance(args, dict):
                return None
            code = args.get("code")
            return code if isinstance(code, str) else None
        return None

    code = last_assistant_python_code(messages)
    if not code or len(code) < LONG_CODE_THRESHOLD:
        return
    if any(tok in code for tok in DELEGATION_TOKENS):
        return
    messages.append({"role": "user", "content": NUDGE})
'''


def seed_candidates(seed_dir: Optional[Path] = None) -> List[Candidate]:
    """Return the initial population (gen 0).

    Three seeds, all using the FROZEN chess `system_prompt.md`:
      A — no-op verifier (control: prove the hook fires without affecting behavior).
      B — delegate-or-measure user-role nudge (the "minimal coach" hypothesis).
      C — same prompt + budget-aware prepend; same nudge verifier (lets us
          compare prompt-only vs verifier-only vs both moving).

    These three exercise the (prompt, verifier) cross-product enough for a
    Pareto front to actually have structure after gen 0.
    """
    seed = Path(seed_dir) if seed_dir else _SEED_DIR_DEFAULT
    sp = (seed / "system_prompt.md").read_text(encoding="utf-8")

    sp_budget_aware = (
        "## Budget Pressure (50-round lead budget)\n"
        "You have a SCARCE 50-round lead budget. DO NOT spend rounds on inline\n"
        "object-level chess work; spend them DESIGNING workflows, EVALUATORS, and\n"
        "DELEGATING to sub-policies. If a turn produced no experiment, no accepted\n"
        "artifact, and no decision, you wasted it.\n\n"
    ) + sp

    return [
        Candidate(
            id="seedA_noverifier",
            system_prompt=sp,
            verifier=_VERIFIER_NOOP,
            parent_id=None,
            lineage_note="seed: chess prompt + no-op verifier (control)",
        ),
        Candidate(
            id="seedB_nudge",
            system_prompt=sp,
            verifier=_VERIFIER_DELEGATE_NUDGE,
            parent_id=None,
            lineage_note="seed: chess prompt + delegate-or-measure user-role nudge",
        ),
        Candidate(
            id="seedC_budget_nudge",
            system_prompt=sp_budget_aware,
            verifier=_VERIFIER_DELEGATE_NUDGE,
            parent_id=None,
            lineage_note="seed: budget-pressure prepend + delegate-or-measure nudge",
        ),
    ]
