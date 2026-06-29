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
    """Meta-reasoning GUARD (kernel-resident, SINGLE self-contained def).

    The verifier is the LIVE guard of meta-reasoning — and the lever that matters most, because
    a standing system prompt does not reliably change behavior, but THIS runs in the policy space
    each non-terminal round with the full ambient namespace (react_llm / natural_llm / parallel /
    policy-edit APIs / policyzero) and a WRITABLE trajectory. Its job: catch the lab-lead falling
    into bad behavior — solving the problem BY HAND inline instead of DESIGNING sub-LLM circuits,
    or hand-grinding after a dispatch failed — and steer it back toward DESIGNING AGENTIC META-
    REASONING CIRCUITS (policies that compose several reactors across rounds). It may inject turns,
    call sub-policies, or edit/prune the trajectory; it may NOT abort the run or edit a sealed
    policy. CONTRACT: ONE top-level `def verifier(messages)`; every helper/import/constant lives
    INSIDE this body; cross-round state in globals()['_v_state'].

    Seed heuristic (the optimizer SHOULD evolve this into a far stronger guard): track, across
    rounds, how much the lead hand-grinds vs designs circuits. If the latest round is a LONG inline
    code block that does NOT delegate (react_llm / natural_llm / parallel) or measure (simulate),
    it is solving-by-hand — steer toward designing a circuit, escalating if it keeps happening; and
    if a recent dispatch FAILED and the lead responded by hand-writing, call that out specifically
    (redesign the circuit with more rounds / a pipeline — don't grind). USER role for injections:
    subagent backends drop NEW system turns mid-stream; user turns are the reliable channel.
    """
    import json

    st = globals().setdefault("_v_state", {})
    st["inline_rounds"] = st.get("inline_rounds", 0)
    st["circuit_rounds"] = st.get("circuit_rounds", 0)

    LONG = 800
    DELEGATE = ("react_llm", "react_verifier_llm", "natural_llm", "parallel(")
    MEASURE = ("simulate(", "measure(")

    def last_assistant_code(msgs):
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
            c = args.get("code")
            return c if isinstance(c, str) else None
        return None

    def recent_dispatch_failed(msgs):
        seen = 0
        for m in reversed(msgs):
            if m.get("role") != "tool":
                continue
            seen += 1
            r = m.get("content")
            if isinstance(r, list):
                r = " ".join(str(x) for x in r)
            r = str(r or "")
            if ("budget exhausted" in r) or ("ConstraintViolation" in r) or ("LLMDepthExceeded" in r):
                return True
            if seen >= 2:
                break
        return False

    code = last_assistant_code(messages)
    if not code:
        return
    if any(t in code for t in DELEGATE):
        st["circuit_rounds"] = st["circuit_rounds"] + 1
        return
    if len(code) < LONG:
        return
    if any(t in code for t in MEASURE):
        return

    st["inline_rounds"] = st["inline_rounds"] + 1
    n = st["inline_rounds"]
    if recent_dispatch_failed(messages):
        steer = (
            "[meta-coach] A sub-LLM circuit just FAILED and you are now hand-writing the engine "
            "yourself — that surrenders the leverage. A failure means the circuit was under-built "
            "or under-resourced: REDESIGN it (more rounds/budget, split into a pipeline "
            "proposer->critic->reviser, or a policy that composes several reactors) and re-dispatch "
            "the improved circuit. Do not solve it by hand."
        )
    elif n >= 2:
        steer = (
            "[meta-coach] Round " + str(n) + " of solving by hand inline. You are the lab LEAD, not "
            "the engineer — stop typing the chess machinery yourself. DESIGN an agentic circuit: a "
            "policy that composes react_llm / natural_llm reactors (and a parallel fleet or verifier "
            "where useful) to build and test it for you. Spend this round DESIGNING that circuit."
        )
    else:
        steer = (
            "[meta-coach] You just wrote a long inline code block instead of delegating. Before more "
            "inline work, DESIGN a sub-LLM circuit to do it — a react_llm/natural_llm reactor, a "
            "parallel fleet, or a policy that composes several reactors — and let IT build/test the "
            "machinery; then MEASURE the result. Don't hand-grind."
        )
    messages.append({"role": "user", "content": steer})
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
        "## Budget Pressure (15-round lead budget)\n"
        "You have a SCARCE 15-round lead budget. DO NOT spend rounds on inline\n"
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
