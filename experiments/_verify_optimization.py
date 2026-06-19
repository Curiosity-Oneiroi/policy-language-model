"""$0 synthetic verification for the optimization experiment plumbing (NO vLLM/LLM).

Builds a fake optimization experiment dir by hand + a fake single-run dir, then
drives RunManager + the app route handlers directly and asserts the contract.

    python -m plm.experiments._verify_optimization
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from plm.experiments.runner import RunManager


def _w(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj) if not isinstance(obj, str) else obj, encoding="utf-8")


def _jsonl(p: Path, rows) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def build_optimization_fixture(runs_dir: Path) -> str:
    run_id = "opt-fixture01"
    d = runs_dir / run_id
    mp = d / "metaparams"
    cr = d / "candidate_runs"

    _w(d / "run.json", {"run_id": run_id, "kind": "optimization", "status": "done",
                        "started": 1.0, "ended": 2.0,
                        "config": {"recipe": "E1", "label": "fixture",
                                   "backend": {"name": "VLLMBackend"}}})

    # 4 candidates across 2 gens forming a small tree.
    cands = [
        # cid, gen, parent, method, accepted, pareto, est_elo, per_instance, scalar
        ("gepa_seed0000", 0, None,            "seed",          True,  True,  1200.0, {"maia-1100": 0.6}, 0.40),
        ("gepa_child001", 1, "gepa_seed0000", "PromptBreeder", True,  True,  1450.0, {"maia-1100": 0.7, "maia-1500": 0.5}, 0.55),
        ("gepa_child002", 1, "gepa_seed0000", "PromptBreeder", True,  False, 1100.0, {"maia-1100": 0.4}, 0.30),
        ("gepa_merge003", 1, "gepa_child001", "merge",         True,  True,  1600.0, {"maia-1500": 0.8}, 0.62),
    ]

    pop_rows = []
    for step, (cid, gen, parent, method, accepted, pareto, elo, per_inst, scalar) in enumerate(cands):
        cmp = mp / cid
        # materialized genome
        _w(cmp / "system_prompt.md", f"# prompt for {cid}\nplay good chess.\n")
        _w(cmp / "verifier.py", f"def verifier(messages):\n    return  # {cid}\n")
        rd = cr / f"alpha_{cid}"
        _w(rd / "_optimizer_config.json", {"metaparam_dir": str(cmp.resolve())})
        _w(rd / "run.json", {"run_id": f"alpha_{cid}", "status": "done"})
        _w(rd / "trajectory.json", {"messages": [
            {"role": "user", "content": f"begin {cid}"},
            {"role": "assistant", "content": f"playing as {cid}"},
        ]})
        # simulate row with per-opponent + est_elo
        by_opp = {opp: {"W": int(round(s * 10)), "L": 10 - int(round(s * 10)), "D": 0, "X": 0}
                  for opp, s in per_inst.items()}
        _jsonl(rd / "simulates.jsonl", [{"summary": {
            "score": scalar, "est_elo": elo, "results_by_opponent": by_opp,
            "policy_timeouts": 0}}])
        # one fake game under games/0/<game_id>.json
        _w(rd / "games" / "0" / "g1.json", {"game_id": "g1", "moves": ["e4", "e5"],
                                            "opponent": list(per_inst)[0]})
        pop_rows.append({
            "step": step, "gen": gen, "candidate_id": cid, "parent_id": parent,
            "method": method, "accepted": accepted, "scalar": scalar,
            "vector": {"proxy_elo": 0.5}, "per_instance": per_inst, "est_elo": elo,
            "on_pareto_front": pareto,
            "genome": {"system_prompt_sha": "x", "verifier_sha": "y"},
            "run_dir": str(rd.resolve()), "metaparam_dir": str(cmp.resolve()),
        })

    _jsonl(d / "population.jsonl", pop_rows)
    _jsonl(d / "pareto_front.jsonl", [
        {"gen": 0, "members": ["gepa_seed0000"]},
        {"gen": 1, "members": ["gepa_seed0000", "gepa_child001", "gepa_merge003"]},
    ])
    _w(d / "optimization_result.json", {
        "best_candidate_id": "gepa_merge003",
        "genome": {"system_prompt": "# prompt for gepa_merge003\nplay good chess.\n",
                   "verifier": "def verifier(messages):\n    return  # gepa_merge003\n"},
        "scalar": 0.62, "est_elo": 1600.0, "vector": {"proxy_elo": 0.5},
        "per_instance": {"maia-1500": 0.8},
        "summary": {"recipe": "E1", "label": "fixture", "num_candidates": 4},
    })
    return run_id


def build_single_fixture(runs_dir: Path) -> str:
    run_id = "run-single01"
    d = runs_dir / run_id
    _w(d / "run.json", {"run_id": run_id, "status": "done", "started": 1.0, "ended": 2.0,
                        "config": {"label": "single-fix", "metaparam_dir": "/x/chess",
                                   "backend": {"name": "VLLMBackend"}}})
    _w(d / "result.json", {"answer": "done", "metrics": {"main_turns": 3}})
    _w(d / "trajectory.json", {"messages": [{"role": "user", "content": "hi"}]})
    _jsonl(d / "simulates.jsonl", [{"summary": {"score": 0.5, "est_elo": 1300}}])
    _w(d / "games" / "0" / "gA.json", {"game_id": "gA"})
    return run_id


class _Pool:                                                   # stand-in (no allocation needed here)
    def status(self): return {}


def main() -> int:
    results = []

    def check(name, cond):
        results.append((name, bool(cond)))

    with tempfile.TemporaryDirectory() as tmp:
        runs_dir = Path(tmp) / "runs"
        runs_dir.mkdir(parents=True)
        rm = RunManager(runs_dir, _Pool())

        opt_id = build_optimization_fixture(runs_dir)
        single_id = build_single_fixture(runs_dir)

        # --- get_run (optimization) ---
        r = rm.get_run(opt_id)
        check("get_run kind=optimization", r.get("kind") == "optimization")
        check("get_run population (4 rows)", len(r.get("population") or []) == 4)
        check("get_run candidates (4)", len(r.get("candidates") or []) == 4)
        check("get_run pareto_front (2 gens)", len(r.get("pareto_front") or []) == 2)
        check("get_run optimization_result best", (r.get("optimization_result") or {}).get("best_candidate_id") == "gepa_merge003")
        cand0 = (r.get("candidates") or [{}])[0]
        check("candidate view has lineage fields",
              {"candidate_id", "parent_id", "method", "scalar", "est_elo",
               "per_instance", "on_pareto_front", "run_dir"}.issubset(cand0.keys()))

        # --- get_candidate (genome + trajectory) ---
        c = rm.get_candidate(opt_id, "gepa_child001")
        check("get_candidate genome system_prompt", "gepa_child001" in c["genome"]["system_prompt"])
        check("get_candidate genome verifier", "def verifier" in c["genome"]["verifier"])
        check("get_candidate parent_id", c["parent_id"] == "gepa_seed0000")
        check("get_candidate method", c["method"] == "PromptBreeder")
        check("get_candidate per_instance", c["per_instance"].get("maia-1500") == 0.5)
        check("get_candidate trajectory msgs", len(c["trajectory"]) == 2)
        check("get_candidate unknown -> None", rm.get_candidate(opt_id, "nope") is None)

        # --- candidate games ---
        games = rm.list_candidate_games(opt_id, "gepa_child001")
        check("list_candidate_games (1 sim row)", len(games) == 1)
        g = rm.get_game(opt_id, 0, "g1", candidate_id="gepa_child001")
        check("get_game candidate resolves", g and g.get("game_id") == "g1")
        check("get_game candidate bad -> None", rm.get_game(opt_id, 0, "zzz", candidate_id="gepa_child001") is None)

        # --- list_runs includes kind ---
        lr = {x["run_id"]: x for x in rm.list_runs()}
        check("list_runs opt kind", lr[opt_id]["kind"] == "optimization")
        check("list_runs opt num_candidates", lr[opt_id]["num_candidates"] == 4)
        check("list_runs opt best_elo", lr[opt_id]["best_elo"] == 1600.0)
        check("list_runs single kind", lr[single_id]["kind"] == "single")

        # --- single-run get_run unchanged (exact key set + no opt keys) ---
        sr = rm.get_run(single_id)
        check("single get_run keys unchanged",
              set(sr.keys()) == {"run", "result", "trajectory", "simulates", "live"})
        check("single get_run no kind key", "kind" not in sr)
        check("single get_run result answer", sr["result"]["answer"] == "done")
        check("single get_game still works",
              (rm.get_game(single_id, 0, "gA") or {}).get("game_id") == "gA")

        # --- app route handlers (call functions directly) ---
        import plm.experiments.app as appmod
        appmod.runs = rm                                       # point handlers at our RunManager

        recs = appmod.get_recipes()
        check("/api/recipes lists 10", len(recs) == 10)
        check("/api/recipes shape {id,title,summary}",
              all({"id", "title", "summary"}.issubset(x) for x in recs))
        check("/api/recipes ordered E1..E10",
              [x["id"] for x in recs] == [f"E{i}" for i in range(1, 11)])

        check("route list_candidates", len(appmod.list_candidates(opt_id)) == 4)
        check("route get_candidate", appmod.get_candidate(opt_id, "gepa_merge003")["method"] == "merge")
        check("route list_candidate_games", len(appmod.list_candidate_games(opt_id, "gepa_child001")) == 1)
        check("route get_candidate_game", appmod.get_candidate_game(opt_id, "gepa_child001", 0, "g1")["game_id"] == "g1")

        # backends expose VLLM with base_url
        bks = {b["name"]: b for b in appmod.get_backends()}
        check("/api/backends VLLM supports_base_url", bks["VLLMBackend"]["supports_base_url"] is True)

    # ---- report ----
    width = max(len(n) for n, _ in results)
    npass = sum(1 for _, ok in results if ok)
    print("\n=== optimization plumbing verification ===")
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name.ljust(width)}")
    print(f"\n{npass}/{len(results)} checks passed")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
