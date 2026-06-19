"""Run scorer — turns a single run_dir into a 7-component vector + scalar.

The scalar uses the EXACT weights the user committed to:

    score = 0.30*proxy_elo
          + 0.20*benchmark_score
          + 0.15*artifact_throughput
          + 0.10*workflow_invention_quality
          + 0.10*budget_efficiency
          + 0.10*stability_and_correctness
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
    - artifact_throughput  : distinct policies created via `policy(...)` /
                             `duplicate_policy(...)` mentions in tool outputs
                             (proxy for "accepted artifacts" until the run
                             emits a real artifact-accept event)
    - workflow_invention_quality : 1 iff (a verifier is configured AND
                             verifier_calls>0 AND simulates.jsonl is non-empty)
                             — i.e. the run had a verifier that did fire AND
                             measurement signal was produced. Until we emit
                             per-workflow accept rows, this is the cheapest
                             non-trivial proxy that satisfies the user's
                             "explicit verifier AND contributed signal" rule.
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
    "artifact_throughput":        0.15,
    "workflow_invention_quality": 0.10,
    "budget_efficiency":          0.10,
    "stability_and_correctness":  0.10,
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
    """REAL component. Best `est_elo` across all simulate rows -> normalized.

    Returns (score, raw_dict). `raw_dict['no_rated_games']=True` when est_elo is
    missing on every row (recall: simulate sets est_elo=None unless RATED
    opponents were played). In that case we return 0 — NOT a small constant —
    because the optimizer must NOT credit candidates that never produced rated
    play.
    """
    best: Optional[float] = None
    for row in simulates:
        summary = row.get("summary") or row
        e = summary.get("est_elo")
        if isinstance(e, (int, float)):
            best = e if best is None else max(best, float(e))
    if best is None:
        return 0.0, {"best_est_elo": None, "no_rated_games": True}
    norm = max(0.0, min(1.0, (best - _ELO_FLOOR) / (_ELO_CEIL - _ELO_FLOOR)))
    return norm, {"best_est_elo": best, "no_rated_games": False}


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


def _artifact_throughput_term(
    messages: List[Dict[str, Any]],
    tool_calls_meta: List[Dict[str, Any]],
) -> Tuple[float, Dict[str, Any]]:
    """HEURISTIC / STUB. Until runs emit a real per-artifact accept event, we
    proxy "accepted artifacts" with the count of distinct policy names that
    appear in tool RESULTS (so the kernel actually executed the def — failed
    cells don't count toward this), capped at a `policies_for_full_credit`
    saturation point (default 5: more than that and the optimizer should be
    rewarding QUALITY not COUNT — the anti-gaming term picks that up)."""
    saturate_at = 5
    distinct: set[str] = set()
    for tc in tool_calls_meta:
        if tc.get("tool_status") not in ("completed", "return_valid"):
            continue
        result = tc.get("tool_result") or ""
        if not isinstance(result, str):
            continue
        for m in _RE_POLICY_CREATED.finditer(result):
            distinct.add(m.group(1))
    raw = len(distinct)
    score = min(1.0, raw / saturate_at)
    return score, {"distinct_policies_created": raw, "names": sorted(distinct),
                   "stub": True, "saturate_at": saturate_at}


def _workflow_invention_quality_term(
    metrics: Dict[str, Any],
    simulates: List[Dict[str, Any]],
    verifier_configured: bool,
) -> Tuple[float, Dict[str, Any]]:
    """HEURISTIC / STUB. Spec: rewards a workflow ONLY if it has an explicit
    verifier AND contributed signal. Until we emit per-workflow accept rows,
    "had an explicit verifier" === `verifier_configured`, "verifier did fire"
    === `verifier_calls > 0`, and "contributed signal" === non-empty
    simulates.jsonl. All three required for credit. Treated as binary 0/1
    because partial credit on a stub would just be noise."""
    fired = (metrics.get("verifier_calls") or 0) > 0
    has_signal = len(simulates) > 0
    granted = bool(verifier_configured and fired and has_signal)
    return (1.0 if granted else 0.0), {
        "verifier_configured": verifier_configured,
        "verifier_fired": fired,
        "had_simulate_signal": has_signal,
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
    at, at_raw = _artifact_throughput_term(messages, tool_calls_meta)
    wq, wq_raw = _workflow_invention_quality_term(metrics, simulates, verifier_configured)
    be, be_raw = _budget_efficiency_term(metrics, messages)
    sc, sc_raw = _stability_and_correctness_term(run_status, answer, metrics, simulates)
    ag, ag_raw = _anti_gaming_term(messages, tool_calls_meta)
    per_instance = _per_instance_scores(simulates)

    vector = {
        "proxy_elo": pe,
        "benchmark_score": bs,
        "artifact_throughput": at,
        "workflow_invention_quality": wq,
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
            "artifact_throughput": at_raw,
            "workflow_invention_quality": wq_raw,
            "budget_efficiency": be_raw,
            "stability_and_correctness": sc_raw,
            "anti_gaming_score": ag_raw,
            # Instance-Pareto axis: {opponent_name: win_rate_in_0_1}. EMPTY until
            # RATED opponents are played. Consumed by pareto.instance_vector.
            "per_instance": per_instance,
        },
    }
