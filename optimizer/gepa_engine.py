"""GEPA engine — drive the REAL `gepa` library over the PLM chess genome.

This replaces the from-memory "GEPA-style" loop with `gepa.optimize(...)`.
The integration point is `PLMChessAdapter(GEPAAdapter)`:

  * Candidate (gepa's unit) = `{"system_prompt": str, "verifier": str}` — the
    two text components of our genome (the frozen chess `policies/` ride along
    via `Candidate.materialize`).
  * Instances = rated opponent rungs. trainset == valset == one dict per rung.
    Until RATED opponents are played, `per_instance` is empty, so per-opponent
    scores are all 0.0 until a real run measures them.
  * ONE PLM subprocess per distinct candidate, MEMOIZED on a stable hash of the
    candidate dict, so a 6-instance batch costs ONE run, not six.

reflection_lm is a `Callable[[str], str]` (gepa's `LanguageModel` protocol):
  * `BackendReflectLM` — wraps a real PLM backend over Slate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import gepa
from gepa.core.adapter import EvaluationBatch, GEPAAdapter

from .evaluate import EvalConfig, evaluate
from .genome import Candidate
from .scorer import SCORE_WEIGHTS


# Component names = keys of the gepa candidate dict.
COMPONENT_SYSTEM_PROMPT = "system_prompt"
COMPONENT_VERIFIER = "verifier"

# The rated opponent rungs. Each is its OWN gepa instance (instance-Pareto axis).
OPPONENT_RUNGS: List[str] = [
    "maia-1100",
    "maia-1500",
    "maia-1900",
    "stockfish-1320",
    "stockfish-1800",
    "stockfish-2400",
]


def default_trainset() -> List[Dict[str, str]]:
    """trainset/valset = one instance per rated opponent rung."""
    return [{"opponent": n} for n in OPPONENT_RUNGS]


def candidate_hash(candidate: Dict[str, str]) -> str:
    """Stable short hash of a gepa candidate dict — the memo key AND the
    `Candidate.id` (so the run_dir is reproducible per distinct candidate)."""
    blob = json.dumps(candidate, sort_keys=True, ensure_ascii=False)
    return "gepa_" + hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


# ===========================================================================
# Adapter
# ===========================================================================

class PLMChessAdapter(GEPAAdapter):
    """gepa <-> PLM integration. Evaluates a candidate on opponent-rung
    instances by running ONE memoized PLM subprocess and projecting its
    per-opponent win-rates onto the batch."""

    def __init__(self, eval_cfg: EvalConfig) -> None:
        self.eval_cfg = eval_cfg
        # memo: candidate_hash -> the full evaluate() result dict.
        self._eval_memo: Dict[str, Dict[str, Any]] = {}

    # ---- one memoized PLM run per distinct candidate ----------------------
    def _run_candidate(self, candidate: Dict[str, str]) -> Dict[str, Any]:
        key = candidate_hash(candidate)
        cached = self._eval_memo.get(key)
        if cached is not None:
            return cached
        cand = Candidate(
            id=key,
            system_prompt=candidate.get(COMPONENT_SYSTEM_PROMPT, ""),
            verifier=candidate.get(COMPONENT_VERIFIER, ""),
            parent_id=None,
            lineage_note="gepa-proposed candidate",
        )
        result = evaluate(cand, self.eval_cfg)
        self._eval_memo[key] = result
        return result

    # ---- gepa: evaluate a candidate on a batch ----------------------------
    def evaluate(self, batch, candidate, capture_traces: bool = False) -> EvaluationBatch:
        result = self._run_candidate(candidate)
        score = result.get("score") or {}
        raw = score.get("raw") or {}
        per_instance: Dict[str, float] = raw.get("per_instance") or {}
        metric_vector: Dict[str, float] = score.get("vector") or {}
        run_dir = result.get("run_dir")

        outputs: List[Dict[str, Any]] = []
        scores: List[float] = []
        trajectories: Optional[List[Dict[str, Any]]] = [] if capture_traces else None
        objective_scores: List[Dict[str, float]] = []

        excerpt = self._trajectory_excerpt(run_dir) if capture_traces else ""

        for inst in batch:
            opp = inst.get("opponent", "")
            s = float(per_instance.get(opp, 0.0))
            outputs.append({"run_dir": run_dir, "opponent": opp, "score": s})
            scores.append(s)
            # Each opponent rung's win-rate is ALSO surfaced as the per-example
            # objective map alongside the 7 shaped-metric components, so gepa's
            # objective/cartesian frontiers (if enabled) have signal too.
            obj: Dict[str, float] = {k: float(metric_vector.get(k, 0.0)) for k in SCORE_WEIGHTS}
            obj["opponent_winrate"] = s
            objective_scores.append(obj)
            if trajectories is not None:
                trajectories.append({
                    "trajectory_excerpt": excerpt,
                    "opponent": opp,
                    "opponent_score": s,
                    "metric_vector": metric_vector,
                    "subprocess_tail": (result.get("subprocess") or {}).get("tail", "")[-600:],
                })

        # NOTE: gepa counts metric calls as len(batch) (the per-instance
        # fan-out) — the installed gepa's EvaluationBatch has no
        # `num_metric_calls` field to override that. Because the adapter
        # MEMOIZES, all instances of a distinct candidate are actually ONE PLM
        # subprocess, but gepa's budget still ticks by instance count. We size
        # the minibatch (reflection_minibatch_size) to keep the proposal phase
        # cheap; see run_gepa.
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=objective_scores,
        )

    # ---- gepa: build the reflective dataset -------------------------------
    def make_reflective_dataset(
        self, candidate, eval_batch: EvaluationBatch, components_to_update: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        dataset: Dict[str, List[Dict[str, Any]]] = {}
        trajectories = eval_batch.trajectories or []
        outputs = eval_batch.outputs or []
        scores = eval_batch.scores or []

        for comp in components_to_update:
            current_text = candidate.get(comp, "")
            records: List[Dict[str, Any]] = []
            for i, out in enumerate(outputs):
                opp = (out or {}).get("opponent", "")
                s = float(scores[i]) if i < len(scores) else 0.0
                traj = trajectories[i] if i < len(trajectories) else {}
                excerpt = self._head_tail(str(traj.get("trajectory_excerpt", "")), 1200)
                outcome = self._outcome(s)
                feedback = self._feedback(comp, s, traj)
                rec = {
                    "input": f"Play rated chess vs {opp} (rung). Drive the lab-lead "
                              f"meta-reasoner to improve the policy against this opponent.",
                    "current_component": self._clip(current_text, 1400),
                    "outcome": f"win-rate {s:.3f} ({outcome})",
                    "trajectory_excerpt": excerpt,
                    "feedback": feedback,
                }
                records.append(self._bound_record(rec, max_bytes=3000))
            dataset[comp] = records
        return dataset

    # ---- helpers ----------------------------------------------------------
    @staticmethod
    def _outcome(s: float) -> str:
        if s >= 0.55:
            return "W"
        if s <= 0.45:
            return "L"
        return "D"

    @staticmethod
    def _feedback(component: str, s: float, traj: Dict[str, Any]) -> str:
        mv = traj.get("metric_vector") or {}
        weak = sorted(mv.items(), key=lambda kv: kv[1])[:2]
        weak_str = ", ".join(f"{k}={v:.2f}" for k, v in weak) if weak else "no metric signal"
        if s <= 0.0:
            base = ("No rated win signal produced (until RATED opponents are played, "
                    "`per_instance` is empty — improve the "
                    "component so a real run would MEASURE/DELEGATE more).")
        elif s < 0.5:
            base = "Lost more than won against this rung."
        else:
            base = "Held or beat this rung; reinforce what worked."
        if component == COMPONENT_VERIFIER:
            base += (" The verifier runs IN THE POLICY-SPACE KERNEL each round with the "
                     "full ambient namespace (react_llm / natural_llm / parallel / policy "
                     "edit APIs / policyzero): it may inject USER or SYSTEM turns, edit/prune "
                     "the trajectory, call sub-policies, or fork/edit a MUTABLE policy — but "
                     "must not error, abort the run, or edit a sealed policy. Weakest shaped "
                     "metrics: " + weak_str + ".")
        else:
            base += (" The system prompt should push budget-aware delegation and "
                     "measurement; weakest shaped metrics: " + weak_str + ".")
        return base

    @staticmethod
    def _clip(text: str, n: int) -> str:
        text = text or ""
        return text if len(text) <= n else text[:n] + "\n…[clipped]"

    @staticmethod
    def _head_tail(text: str, n: int) -> str:
        text = text or ""
        if len(text) <= n:
            return text
        half = n // 2
        return text[:half] + "\n…[middle elided]…\n" + text[-half:]

    @staticmethod
    def _bound_record(rec: Dict[str, Any], *, max_bytes: int) -> Dict[str, Any]:
        blob = json.dumps(rec, ensure_ascii=False)
        if len(blob.encode("utf-8")) <= max_bytes:
            return rec
        # Shrink the largest free-text fields first.
        for field in ("trajectory_excerpt", "current_component", "feedback"):
            while len(json.dumps(rec, ensure_ascii=False).encode("utf-8")) > max_bytes and rec.get(field):
                rec[field] = rec[field][: max(0, len(rec[field]) - 256)]
                if not rec[field]:
                    rec[field] = "[elided]"
                    break
        return rec

    @staticmethod
    def _trajectory_excerpt(run_dir: Optional[str]) -> str:
        if not run_dir:
            return ""
        traj_path = Path(run_dir) / "trajectory.json"
        if not traj_path.is_file():
            return ""
        try:
            data = json.loads(traj_path.read_text(encoding="utf-8"))
        except Exception:
            return ""
        msgs = data.get("messages") or []
        parts: List[str] = []
        for m in msgs:
            role = m.get("role", "?")
            content = m.get("content")
            if isinstance(content, list):
                content = " ".join(str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content)
            parts.append(f"[{role}] {str(content)[:300]}")
        return "\n".join(parts)


class BackendReflectLM:
    """REAL reflection LM. Wraps a PLM ModelBackend (duck-typed `async generate`).

    gepa calls this with a single `prompt: str` (the rendered template).
    """

    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def __call__(self, prompt: str) -> str:
        import asyncio
        resp = asyncio.run(
            self.backend.generate(
                messages=[{"role": "user", "content": prompt}], tools=None
            )
        )
        return (resp or {}).get("content", "") or ""


# ===========================================================================
# run_gepa — the public driver
# ===========================================================================

def run_gepa(
    seed_candidate: Dict[str, str],
    *,
    eval_cfg: EvalConfig,
    reflection_lm: Callable[[str], str],
    reflection_prompt_template: Optional[str] = None,
    custom_candidate_proposer: Optional[Any] = None,
    use_merge: bool = False,
    frontier_type: str = "instance",
    max_metric_calls: int = 8,
    run_dir: Optional[str] = None,
    candidate_selection_strategy: str = "pareto",
    module_selector: str = "round_robin",
    reflection_minibatch_size: int = 1,
    seed: int = 0,
    label: Optional[str] = None,
) -> Any:
    """Run the REAL `gepa.optimize` over the PLM chess genome.

    `seed_candidate` is a gepa candidate dict: {"system_prompt", "verifier"}.
    Returns gepa's `GEPAResult`. Persists a ledger + summary into `run_dir`.

    `custom_candidate_proposer` is a gepa `ProposalFn`
    (`(candidate, reflective_dataset, components_to_update) -> dict[str,str]`).
    When provided, gepa uses it INSTEAD of the default reflection proposer, so
    `reflection_prompt_template` is dead for that run (gepa only feeds the
    template to its built-in proposer). This is why "+PE2" composites pass the
    PE2 template INTO the proposer (its `base_template`), not via this kwarg —
    see recipes.py / README.md. We forbid passing BOTH here to avoid the
    silently-ignored-template footgun.
    """
    if custom_candidate_proposer is not None and reflection_prompt_template is not None:
        raise ValueError(
            "run_gepa: reflection_prompt_template is ignored when a "
            "custom_candidate_proposer is set (gepa only feeds the template to "
            "its default proposer). Pass PE2 framing via the proposer's "
            "base_template instead, and leave reflection_prompt_template=None."
        )
    trainset = default_trainset()
    valset = default_trainset()
    adapter = PLMChessAdapter(eval_cfg)

    gepa_run_dir = None
    if run_dir is not None:
        gepa_run_dir = str(Path(run_dir) / "gepa_state")
        Path(gepa_run_dir).mkdir(parents=True, exist_ok=True)

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        candidate_selection_strategy=candidate_selection_strategy,
        frontier_type=frontier_type,
        reflection_prompt_template=reflection_prompt_template,
        custom_candidate_proposer=custom_candidate_proposer,
        module_selector=module_selector,
        reflection_minibatch_size=reflection_minibatch_size,
        use_merge=use_merge,
        max_metric_calls=max_metric_calls,
        run_dir=gepa_run_dir,
        display_progress_bar=False,
        seed=seed,
        raise_on_exception=True,
    )

    if run_dir is not None:
        _persist_ledger(result, adapter, run_dir, label=label)
    return result


def _persist_ledger(result: Any, adapter: PLMChessAdapter, run_dir: str, *, label: Optional[str] = None) -> None:
    """Write a JSON ledger + a small summary of the gepa run."""
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)

    candidates = list(getattr(result, "candidates", []) or [])
    val_scores = list(getattr(result, "val_aggregate_scores", []) or [])
    parents = list(getattr(result, "parents", []) or [])
    best_idx = getattr(result, "best_idx", None)

    ledger = {
        "label": label,
        "num_candidates": len(candidates),
        "best_idx": best_idx,
        "best_candidate": getattr(result, "best_candidate", None),
        "val_aggregate_scores": val_scores,
        "parents": parents,
        "total_metric_calls": getattr(result, "total_metric_calls", None),
        "distinct_plm_runs": len(adapter._eval_memo),
        "candidates": [
            {
                "idx": i,
                "hash": candidate_hash(c),
                "system_prompt_len": len(c.get(COMPONENT_SYSTEM_PROMPT, "")),
                "verifier_len": len(c.get(COMPONENT_VERIFIER, "")),
                "val_aggregate_score": val_scores[i] if i < len(val_scores) else None,
                "parent_idxs": parents[i] if i < len(parents) else None,
            }
            for i, c in enumerate(candidates)
        ],
    }
    (out / "gepa_ledger.json").write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    try:
        result_dict = result.to_dict()
        (out / "gepa_result.json").write_text(
            json.dumps(result_dict, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        (out / "gepa_result.json").write_text(
            json.dumps({"error": f"to_dict failed: {e}"}, indent=2), encoding="utf-8"
        )


__all__ = [
    "PLMChessAdapter",
    "BackendReflectLM",
    "run_gepa",
    "default_trainset",
    "candidate_hash",
    "OPPONENT_RUNGS",
    "COMPONENT_SYSTEM_PROMPT",
    "COMPONENT_VERIFIER",
]
