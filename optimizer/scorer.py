"""Run scorer — turns a single run_dir into a 7-component vector + scalar.

The scalar uses the EXACT weights the user committed to:

    score = 0.30*proxy_elo
          + 0.20*benchmark_score
          + 0.35*meta_reasoning_quality      # creative sub-LLM circuits that WORK and are USED — the headline nudge
          + 0.05*budget_efficiency
          + 0.05*stability_and_correctness
          + 0.05*anti_gaming_score

The VECTOR is preserved for Pareto selection in `loop.py`; the SCALAR is only
for logging and tie-breaks.

Which components are REAL today vs STUBBED (clearly flagged so we don't
mistake heuristic noise for signal):

  REAL (parses real artifact fields when present):
    - proxy_elo            : best simulates.jsonl `est_elo` → normalized
    - benchmark_score      : best simulates.jsonl `score`  (already in [0,1])
    - stability_and_correctness: counts PLMTaskFailure, policy_timeouts,
                                 verifier_errors, model_calls anomalies

  HEURISTIC / STUB (transparent proxies until we emit per-round signals):
    - meta_reasoning_quality : QUALITY of sub-LLM-circuit use (subsumes the old
                             artifact_throughput + workflow_invention_quality with a
                             far richer signal). Three sub-scores from the trajectory:
                             circuit_design (creative/diverse/composed circuits —
                             named researcher-policies, fleets, verifiers,
                             constraint-gated dispatches), circuit_success (dispatches
                             RETURN instead of erroring), output_benefit (a successful
                             dispatch's output is USED downstream — applied to
                             policyzero / read to prune/decide). design GATES;
                             success+benefit dominate. ORTHOGONAL to Elo: hand-grinding
                             a strong engine scores LOW, delegating+composing well
                             scores HIGH — the behaviour we want gepa to evolve.
    - budget_efficiency    : 1 − (fraction of rounds with no tool_call OR a
                             FAILED tool call). Until the run emits per-round
                             accept/decision events, "made a tool call that
                             didn't error" is our cheapest non-zero proxy.
    - anti_gaming_score    : penalty for metric-padding patterns we can
                             detect from artifacts alone (e.g. massive
                             text-only assistant churn with no tool calls,
                             repeated identical simulate() calls).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Shaped-objective weights. Sum = 1.0.
SCORE_WEIGHTS: Dict[str, float] = {
    "proxy_elo":                  0.30,
    "benchmark_score":            0.20,
    "meta_reasoning_quality":     0.35,   # creative sub-LLM circuits that WORK and are USED — the headline nudge
    "budget_efficiency":          0.05,
    "stability_and_correctness":  0.05,
    "anti_gaming_score":          0.05,
}
assert abs(sum(SCORE_WEIGHTS.values()) - 1.0) < 1e-9, "SCORE_WEIGHTS must sum to 1.0"


# Documented normalization for proxy_elo: linear clamp from a "novice" floor
# (800) to a "strong" ceiling (2000). The choice is intentional — most policy0
# variants sit between these; a finished optimizer should NOT saturate before
# 2000. Documented here so swapping the map later is a single explicit edit.
_ELO_FLOOR = 800.0
_ELO_CEIL = 2000.0


# Heuristic regexes for the STUB components. Kept as a single block so swapping
# them out for a real artifact-event reader later is a one-call replacement.
_RE_POLICY_CREATED = re.compile(r"\b(?:duplicate_policy|@policy|policy\s*\()\s*\(?['\"]([A-Za-z_]\w*)['\"]?")
_RE_SIMULATE_CALL = re.compile(r"\bsimulate\s*\(")

# meta_reasoning_quality: the sub-LLM-circuit surface + dispatch-failure signatures.
_RE_MR_DISPATCH = re.compile(r"\b(react_llm|natural_llm|react_verifier_llm|parallel)\s*\(")
_RE_MR_ASSIGN = re.compile(r"(\w+)\s*=\s*(?:react_llm|natural_llm|react_verifier_llm|parallel)\s*\(")
_RE_MR_NAMED = re.compile(
    r"@policy\s*\n\s*(?:async\s+)?def\s+(\w+)"
    r"|duplicate_policy\s*\([^,]+,\s*['\"](\w+)['\"]"
    r"|(?<![.\w])policy\s*\(\s*['\"](\w+)['\"]")
_RE_MR_CONSTRAINT = re.compile(r"\bconstraint\s*=")
_RE_MR_FAIL = re.compile(r"budget exhausted|ConstraintViolation|LLMDepthExceeded")


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Read JSON file or return None if missing/malformed (we never crash the
    scorer on a partial run dir — scoring a failed/short run is the WHOLE
    point of the optimizer)."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL file or return [] if missing/malformed (per-line)."""
    if not path.is_file():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


# ---------- per-component scorers (each returns float in [0,1]) -------------

def _proxy_elo_term(simulates: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
    """REAL component. The OFFICIAL `est_elo` -> normalized.

    Prefers the dedicated post-RETURN evaluation row — the one tagged `final_eval`,
    which the harness runs on the DELIVERED `policyzero` under a fixed N-game / fixed-seed
    sample. That is the comparable, un-gameable score (every candidate measured the same
    way on its final deliverable). Only when no final-eval row exists (single runs, older
    runs) does it fall back to the best `est_elo` across the model's self-chosen peeks.

    Returns (score, raw_dict). `raw_dict['no_rated_games']=True` when est_elo is missing
    everywhere (simulate sets est_elo=None unless RATED opponents were played); then we
    return 0 — NOT a small constant — so the optimizer never credits a candidate that
    produced no rated play.
    """
    final_elo: Optional[float] = None              # the official post-RETURN evaluation
    best_peek: Optional[float] = None              # fallback: best of the model's own peeks
    for row in simulates:
        summary = row.get("summary") or row
        e = summary.get("est_elo")
        if not isinstance(e, (int, float)):
            continue
        if row.get("final_eval"):
            final_elo = float(e)                   # keep the LAST tagged row (probe runs last)
        best_peek = float(e) if best_peek is None else max(best_peek, float(e))

    chosen = final_elo if final_elo is not None else best_peek
    if chosen is None:
        return 0.0, {"best_est_elo": None, "no_rated_games": True, "final_eval": False}
    norm = max(0.0, min(1.0, (chosen - _ELO_FLOOR) / (_ELO_CEIL - _ELO_FLOOR)))
    return norm, {
        "best_est_elo": chosen,                    # OFFICIAL elo (final-eval if present)
        "final_eval": final_elo is not None,
        "final_eval_elo": final_elo,
        "best_peek_elo": best_peek,
        "no_rated_games": False,
    }


def _benchmark_score_term(simulates: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
    """REAL component. Best `score` field across simulate rows.

    `score` from `simulate` is already a normalized win/draw/loss ratio in
    [0,1]; we just take the max so a candidate that ONLY beat the weakest
    opponent in one of N suites can still get partial credit.
    """
    best: Optional[float] = None
    for row in simulates:
        summary = row.get("summary") or row
        s = summary.get("score")
        if isinstance(s, (int, float)):
            best = s if best is None else max(best, float(s))
    if best is None:
        return 0.0, {"best_score": None}
    return max(0.0, min(1.0, float(best))), {"best_score": best}


def _has_circuit_policy(code: str) -> bool:
    """True if the trajectory defines a policy / function whose BODY itself dispatches
    a sub-LLM (react_llm / natural_llm / parallel) — i.e. a policy that IS an agentic
    circuit (multiple reactors / rounds doing a job), not a one-shot call. Detected by
    indentation: a `def` / `async def` header followed by an indented block that
    contains a dispatch. The strongest 'agentic, not just dispatching' signal."""
    lines = code.splitlines()
    i, n = 0, len(lines)
    while i < n:
        stripped = lines[i].lstrip()
        if stripped.startswith(("def ", "async def ")):
            header_indent = len(lines[i]) - len(stripped)
            body: List[str] = []
            j = i + 1
            while j < n:
                if not lines[j].strip():
                    j += 1
                    continue
                if (len(lines[j]) - len(lines[j].lstrip())) <= header_indent:
                    break
                body.append(lines[j])
                j += 1
            if _RE_MR_DISPATCH.search("\n".join(body)):
                return True
            i = j
        else:
            i += 1
    return False


def _meta_reasoning_quality_term(
    messages: List[Dict[str, Any]],
) -> Tuple[float, Dict[str, Any]]:
    """NEW (heuristic, from the trajectory). Rewards HIGH-QUALITY use of sub-LLM
    circuits — the meta-reasoning that is the whole point of PLM — along three axes:

      (1) circuit_design  — did it build AGENTIC SYSTEMS, not just fire dispatches?
                            Rewards composition: named researcher-policies, parallel
                            fleets, verifier circuits, multiple construct types, and
                            especially a policy whose BODY itself dispatches (a multi-
                            reactor circuit — what's needed to delegate a COMPLEX
                            problem). Plain react_llm calls cap LOW; the top half
                            REQUIRES real agentic structure.
      (2) circuit_success — do those circuits actually WORK, i.e. RETURN instead of
                            erroring ('budget exhausted' / ConstraintViolation /
                            depth)? Fraction of dispatches that returned cleanly.
      (3) output_benefit  — did a SUCCESSFUL circuit's output get USED downstream:
                            applied to policyzero, read to decide/prune, OR fed into
                            another circuit? A HELPER circuit (diagnostician/critic)
                            that writes no code but whose output STEERS the next build
                            counts — indirect benefit is benefit; chaining circuits
                            into a pipeline earns extra. Unreferenced output = 0.

    Combined so design GATES (no circuits -> ~0) and success+benefit DOMINATE
    (circuits that error or go unused earn almost nothing). Deliberately ORTHOGONAL
    to Elo: a candidate that hand-grinds a strong engine scores LOW here; one that
    delegates + composes well scores HIGH — exactly the behaviour we want gepa to
    evolve toward. Subsumes the old artifact_throughput (named policies -> design)
    and workflow_invention_quality (a working, contributing circuit -> success +
    benefit) with a far richer signal. Heuristic until the kernel emits per-dispatch
    events; the regexes are a single block so the swap to real events is one edit.
    """
    # ordered stream of (kind, text): assistant CODE cells + their tool RESULTS
    stream: List[Tuple[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "assistant":
            code = ""
            for tc in (msg.get("tool_calls") or []):
                if isinstance(tc, dict):
                    a = (tc.get("function") or {}).get("arguments")
                    if isinstance(a, str):
                        try:
                            code += (json.loads(a).get("code", "") or "") + "\n"
                        except Exception:
                            code += a + "\n"
            stream.append(("code", code))
        elif role == "tool":
            c = msg.get("content", "")
            if isinstance(c, list):
                c = " ".join(str(x.get("text", "") if isinstance(x, dict) else x) for x in c)
            stream.append(("result", str(c)))
    allcode = "\n".join(t for k, t in stream if k == "code")
    if not allcode.strip():
        return 0.0, {"no_code": True, "stub": True}

    # (1) circuit_design: reward AGENTIC COMPOSITION, not raw dispatching. Plain
    # react_llm calls (even successful, even many) CAP LOW here — to delegate a genuinely
    # complex problem you must build a SYSTEM: named researcher-policies, parallel fleets,
    # verifier circuits, and especially a policy whose BODY itself dispatches (a multi-
    # reactor circuit). Composition DOMINATES; raw volume is only a minor floor, so "just
    # dispatching" can never reach the top half. This is the "it needs to do agentic" bar.
    types_used = {n for n in ("react_llm", "natural_llm", "react_verifier_llm", "parallel")
                  if re.search(r"\b" + n + r"\s*\(", allcode)}
    named: set[str] = set()
    for mm in _RE_MR_NAMED.finditer(allcode):
        named.add(next(g for g in mm.groups() if g))
    n_constraint = len(_RE_MR_CONSTRAINT.findall(allcode))
    n_dispatch = len(_RE_MR_DISPATCH.findall(allcode))
    has_fleet = "parallel(" in allcode
    has_verifier = "react_verifier_llm(" in allcode
    circuit_policy = _has_circuit_policy(allcode)        # a policy that INTERNALLY dispatches
    # agentic points: named policies + fleet + verifier + a policy-that-dispatches +
    # multi-type. ~5 points (e.g. a circuit-policy + a fleet + two named) -> full.
    agentic = (min(3, len(named)) * 0.8
               + (1.5 if has_fleet else 0.0)
               + (1.5 if has_verifier else 0.0)
               + (2.0 if circuit_policy else 0.0)
               + (1.0 if len(types_used) >= 2 else 0.0))
    diversity = min(1.0, len(types_used) / 3.0)
    volume = min(1.0, n_dispatch / 4.0)
    compose = min(1.0, agentic / 5.0)
    circuit_design = 0.20 * diversity + 0.55 * compose + 0.25 * volume

    # (2) circuit_success: fraction of dispatches whose cell did NOT error out
    n_succ = n_fail = 0
    for i, (k, t) in enumerate(stream):
        if k != "code" or not _RE_MR_DISPATCH.search(t):
            continue
        d = len(_RE_MR_DISPATCH.findall(t))
        result = stream[i + 1][1] if i + 1 < len(stream) and stream[i + 1][0] == "result" else ""
        if _RE_MR_FAIL.search(result):
            n_fail += d
        else:
            n_succ += d
    total = n_succ + n_fail
    circuit_success = (n_succ / total) if total else 0.0

    # (3) output_benefit: a SUCCESSFUL dispatch's output is USED downstream — applied to policyzero,
    # OR read to decide/prune, OR FED INTO ANOTHER circuit. The last case is the key one: a HELPER
    # circuit (e.g. a diagnostician that names weaknesses, or a critic that picks a direction) rarely
    # writes code itself, but the lead USES its output to steer the next build — that INDIRECT benefit
    # IS benefit and is credited here (a referenced result counts; a result feeding another dispatch
    # counts EXTRA, since chaining circuits into a pipeline is the behavior we most want). An output
    # never referenced again earned nothing.
    n_used = n_chained = 0
    for i, (k, t) in enumerate(stream):
        if k != "code":
            continue
        for am in _RE_MR_ASSIGN.finditer(t):
            var = am.group(1)
            result = stream[i + 1][1] if i + 1 < len(stream) and stream[i + 1][0] == "result" else ""
            if _RE_MR_FAIL.search(result):
                continue
            later_cells = [t2 for k2, t2 in stream[i + 1:] if k2 == "code"]
            pat = r"\b" + re.escape(var) + r"\b"
            if any(re.search(pat, c) for c in later_cells):
                n_used += 1
                if any(re.search(pat, c) and _RE_MR_DISPATCH.search(c) for c in later_cells):
                    n_chained += 1                     # output steers a later circuit (a pipeline)
    # chained outputs earn a small bonus (capped) — composing circuits into pipelines is desirable.
    output_benefit = min(1.0, (n_used + 0.25 * n_chained) / n_succ) if n_succ else 0.0

    quality = 0.4 * circuit_success + 0.6 * output_benefit
    score = circuit_design * (0.15 + 0.85 * quality)
    return round(score, 6), {
        "circuit_design": round(circuit_design, 3),
        "circuit_success": round(circuit_success, 3),
        "output_benefit": round(output_benefit, 3),
        "n_dispatch": n_dispatch, "n_success": n_succ, "n_fail": n_fail,
        "n_used_downstream": n_used, "n_chained_to_circuit": n_chained,
        "circuit_types": sorted(types_used),
        "named_policies": sorted(named), "n_constraint": n_constraint,
        "has_fleet": has_fleet, "has_verifier": has_verifier,
        "circuit_policy": circuit_policy, "agentic_points": round(agentic, 2),
        "stub": True,
    }


def _budget_efficiency_term(
    metrics: Dict[str, Any],
    messages: List[Dict[str, Any]],
) -> Tuple[float, Dict[str, Any]]:
    """HEURISTIC / STUB. Spec: penalize rounds with NO experiment / NO accepted
    patch / NO decision. Until those are emitted as events, we use the cheapest
    proxy: a round "wasted" if it produced no successful tool call. Concretely:

        wasted = main_turns − count(tool_status in {"completed","return_valid"})

    A run with 0 main_turns yields score 0 (no evidence of work).
    """
    main_turns = int(metrics.get("main_turns") or 0)
    if main_turns <= 0:
        return 0.0, {"main_turns": 0, "stub": True}
    successes = sum(
        1 for tc in (metrics.get("tool_calls") or [])
        if tc.get("tool_status") in ("completed", "return_valid")
    )
    wasted = max(0, main_turns - successes)
    score = max(0.0, 1.0 - wasted / main_turns)
    return score, {"main_turns": main_turns, "successful_tool_rounds": successes,
                   "wasted_rounds": wasted, "stub": True}


def _stability_and_correctness_term(
    run_status: Optional[str],
    answer: Any,
    metrics: Dict[str, Any],
    simulates: List[Dict[str, Any]],
) -> Tuple[float, Dict[str, Any]]:
    """REAL component (composite of multiple real signals).

    Start at 1.0. Subtract for:
      −0.50 if run_status == 'error' (subprocess crashed or PLMTaskFailure)
      −0.20 per verifier_error (capped −0.40)
      −0.05 per policy_timeouts across all simulate rows (capped −0.30)
      −0.20 if answer is None AND run_status != 'error' (silent failure)
    Floor at 0.0.
    """
    raw: Dict[str, Any] = {
        "run_status": run_status, "verifier_errors": 0,
        "policy_timeouts": 0, "silent_failure": False,
    }
    score = 1.0
    if run_status == "error":
        score -= 0.50
        raw["error_run"] = True
    verr = int(metrics.get("verifier_errors") or 0)
    score -= min(0.40, 0.20 * verr)
    raw["verifier_errors"] = verr
    timeouts_total = 0
    for row in simulates:
        s = row.get("summary") or row
        timeouts_total += int(s.get("policy_timeouts") or 0)
    score -= min(0.30, 0.05 * timeouts_total)
    raw["policy_timeouts"] = timeouts_total
    if answer is None and run_status != "error":
        score -= 0.20
        raw["silent_failure"] = True
    return max(0.0, score), raw


def _per_instance_scores(simulates: List[Dict[str, Any]]) -> Dict[str, float]:
    """REAL component (instance axis). Per-opponent win-rate in [0,1].

    Reads every simulate row's `results_by_opponent` cell — shaped
    `{opponent_name: {"W": int, "L": int, "D": int, "X": int}}` — and folds it
    into a single win-rate per opponent rung across ALL rows:

        win_rate(opp) = (W + 0.5*D) / (W + L + D)         # X (errored) ignored

    Returns `{opponent_name: float_in_0_1}`, EMPTY when no rated opponents were
    played (until RATED opponents are played, `per_instance` is empty, so this is
    `{}` — and instance-Pareto degrades gracefully to a scalar tiebreak, see engines.py).

    This is the vector instance-Pareto selection consumes: each opponent rung is
    its OWN objective, so a candidate that uniquely beats one hard opponent stays
    on the frontier even if its aggregate score is mediocre.
    """
    tally: Dict[str, Dict[str, float]] = {}
    for row in simulates:
        summary = row.get("summary") or row
        by_opp = summary.get("results_by_opponent")
        if not isinstance(by_opp, dict):
            continue
        for opp, cell in by_opp.items():
            if not isinstance(cell, dict):
                continue
            acc = tally.setdefault(opp, {"W": 0.0, "L": 0.0, "D": 0.0})
            acc["W"] += float(cell.get("W", 0) or 0)
            acc["L"] += float(cell.get("L", 0) or 0)
            acc["D"] += float(cell.get("D", 0) or 0)
    out: Dict[str, float] = {}
    for opp, acc in tally.items():
        decided = acc["W"] + acc["L"] + acc["D"]
        if decided <= 0:
            continue
        out[opp] = round((acc["W"] + 0.5 * acc["D"]) / decided, 6)
    return out


def _anti_gaming_term(
    messages: List[Dict[str, Any]],
    tool_calls_meta: List[Dict[str, Any]],
) -> Tuple[float, Dict[str, Any]]:
    """HEURISTIC / STUB. Punish metric-padding patterns observable from
    artifacts. Each detected pattern subtracts from a base of 1.0; floor 0.

      −0.30 if >40% of assistant turns are TEXT-ONLY (no tool_call): the lead
            is producing trajectory bulk without execution. A few text turns
            are fine.
      −0.30 if any single simulate() argument-string appears 3+ times in
            tool_result outputs (re-running the same benchmark for ledger
            inflation).
      −0.20 if `artifact_throughput` raw count exceeds a hard cap (10) — the
            "spam tiny policies" pattern. (Hard cap is intentionally generous.)
    """
    raw: Dict[str, Any] = {"text_only_ratio": 0.0, "repeated_simulate_max": 0, "stub": True}
    asst = [m for m in messages if m.get("role") == "assistant"]
    if asst:
        text_only = sum(1 for m in asst if not (m.get("tool_calls") or []))
        ratio = text_only / len(asst)
        raw["text_only_ratio"] = round(ratio, 3)
    else:
        ratio = 0.0
    score = 1.0
    if ratio > 0.40:
        score -= 0.30
    # Repeated simulate(...) signature pattern.
    sig_counts: Dict[str, int] = {}
    for tc in tool_calls_meta:
        r = tc.get("tool_result") or ""
        if isinstance(r, str) and "simulate(" in r:
            sig_counts[r[:200]] = sig_counts.get(r[:200], 0) + 1
    if sig_counts:
        raw["repeated_simulate_max"] = max(sig_counts.values())
        if max(sig_counts.values()) >= 3:
            score -= 0.30
    # Tiny-policy spam.
    distinct = set()
    for tc in tool_calls_meta:
        r = tc.get("tool_result") or ""
        if isinstance(r, str):
            distinct.update(_RE_POLICY_CREATED.findall(r))
    if len(distinct) > 10:
        score -= 0.20
    return max(0.0, score), raw


# ---------- public entry point ---------------------------------------------

def score_run(run_dir: Path, *, verifier_configured: Optional[bool] = None) -> Dict[str, Any]:
    """Score one PLM run directory.

    Args:
      run_dir: a directory written by `plm.experiments.run_entry` containing
               (at minimum) `run.json`; optionally `result.json`,
               `trajectory.json`, `simulates.jsonl`.
      verifier_configured: did the candidate ship a non-no-op verifier? If
               None, we INFER from `metrics.verifier_calls>0` (close enough for
               the dry-run; the loop passes the real value).

    Returns:
      {
        "vector": {<7 component name>: float in [0,1], ...},
        "scalar": float (weighted sum),
        "raw":    {<per-component diagnostic dict>, "run_dir": str},
      }

    NEVER raises — a partial / corrupted run dir scores 0 on every component
    that depends on the missing artifact. That's by design: the optimizer's
    scoring must be MONOTONIC in evidence, not crash on shortfalls.
    """
    run_dir = Path(run_dir)

    run_json = _read_json(run_dir / "run.json") or {}
    result_json = _read_json(run_dir / "result.json") or {}
    trajectory_json = _read_json(run_dir / "trajectory.json") or {}
    simulates = _read_jsonl(run_dir / "simulates.jsonl")

    run_status: Optional[str] = run_json.get("status")
    metrics: Dict[str, Any] = result_json.get("metrics") or {}
    answer: Any = result_json.get("answer")
    messages: List[Dict[str, Any]] = trajectory_json.get("messages") or []
    tool_calls_meta: List[Dict[str, Any]] = metrics.get("tool_calls") or []

    if verifier_configured is None:
        verifier_configured = (metrics.get("verifier_calls") or 0) > 0

    # Per-component scoring (each returns (score, raw_diag)).
    pe, pe_raw = _proxy_elo_term(simulates)
    bs, bs_raw = _benchmark_score_term(simulates)
    mrq, mrq_raw = _meta_reasoning_quality_term(messages)
    be, be_raw = _budget_efficiency_term(metrics, messages)
    sc, sc_raw = _stability_and_correctness_term(run_status, answer, metrics, simulates)
    ag, ag_raw = _anti_gaming_term(messages, tool_calls_meta)
    per_instance = _per_instance_scores(simulates)

    vector = {
        "proxy_elo": pe,
        "benchmark_score": bs,
        "meta_reasoning_quality": mrq,
        "budget_efficiency": be,
        "stability_and_correctness": sc,
        "anti_gaming_score": ag,
    }
    scalar = sum(SCORE_WEIGHTS[k] * vector[k] for k in vector)

    return {
        "vector": vector,
        "scalar": round(scalar, 6),
        "raw": {
            "run_dir": str(run_dir),
            "run_status": run_status,
            "answer_present": answer is not None,
            "main_turns": metrics.get("main_turns"),
            "verifier_calls": metrics.get("verifier_calls"),
            "verifier_errors": metrics.get("verifier_errors"),
            "num_simulates": len(simulates),
            "proxy_elo": pe_raw,
            "benchmark_score": bs_raw,
            "meta_reasoning_quality": mrq_raw,
            "budget_efficiency": be_raw,
            "stability_and_correctness": sc_raw,
            "anti_gaming_score": ag_raw,
            # Instance-Pareto axis: {opponent_name: win_rate_in_0_1}. EMPTY until
            # RATED opponents are played. Consumed by pareto.instance_vector.
            "per_instance": per_instance,
        },
    }
