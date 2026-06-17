@policy
def simulate(policy, num_games, *, seed=0, return_summary=True):
    """Play `policy` over `num_games` chess games and return rich trajectories.

    This is your measurement instrument for the chess task: hand it a policy (the
    canonical `policyzero`, or any `duplicate_policy` fork / candidate you author)
    and a number of games. It plays them in parallel against the FIXED opponents
    and time control configured for this experiment, and hands back per-game
    trajectories plus an aggregate summary — so you can see where the policy wins,
    where it blunders, and whether an edit actually helped.

    You control WHAT to test and HOW MUCH:

        policy:     the policy to evaluate. A @policy CLASS (like `policyzero`) gets
                    a FRESH instance per game (isolated + reseeded); an instance /
                    bare callable is accepted too (shared — fine if stateless).
        num_games:  how many games to play (positive int) — more games give a
                    tighter estimate of the win rate.
        seed=0:     master RNG seed; opponents, colours, and per-game seeds derive
                    from it, so a run is exactly reproducible.
        return_summary=True: return the {schema_version, config, games, summary}
                    envelope; False -> just the list of game trajectories.

    You do NOT choose the test conditions. The opponent pool, the per-move / per-game
    time control (the clock that forces `policy` to be FAST — so NO LLM in the move
    loop, or you lose on time), the move cap, and illegal-move handling are FIXED for
    the experiment (set outside the kernel) and echoed back under result["config"].
    That keeps the evaluation honest — your job is to improve the POLICY, not the test.

    Returns (return_summary=True): a dict with
        "games"   — per-game trajectory dicts (moves, result, winner, termination,
                    per-move timing/flags, optional eval, pgn),
        "summary" — aggregate {n_games, policy_wins/losses/draws, win_rate, score,
                    est_elo, results_by_opponent, results_by_color, policy_timeouts,
                    policy_forfeits, termination_counts, ...},
        "config"  — the exact (experimenter-set) conditions used.
    """
    import inspect
    import json
    import os
    import time
    try:
        from game.chess import dev_simulate
    except ImportError as e:
        raise RuntimeError(
            f"simulate: the `game` chess package is not importable ({e}). Install it "
            "(e.g. `uv pip install -e game/`) so the chess environment is available."
        ) from e

    # The evaluation conditions are FIXED by the experiment — set in plm.py and passed
    # in via the `_PLM_SIMULATE_CONFIG` env, NOT chosen here by the model. Built-in
    # defaults apply when it is unset (direct / offline use). Only known keys are read;
    # dev_simulate validates the values.
    cfg = {
        "opponents": ["random", "greedy"],            # always-available engine-free bots
        "clock": "per_move", "per_move_s": 1.0, "game_clock_s": 60.0,
        "max_moves": 300,
        "evaluate": False, "eval_depth": 12, "reference_engine": None,
        "on_illegal": "forfeit", "max_illegal_retries": 3,
        "max_workers": None,
    }
    _raw = os.environ.get("_PLM_SIMULATE_CONFIG")
    if _raw:
        try:
            _loaded = json.loads(_raw)
        except (ValueError, TypeError):
            _loaded = None                            # malformed -> keep defaults
        if isinstance(_loaded, dict):
            cfg.update({k: v for k, v in _loaded.items() if k in cfg})

    opponents = cfg["opponents"]
    if isinstance(opponents, str):                     # a lone bot name -> wrap
        opponents = [opponents]

    # A @policy CLASS -> fresh instance PER GAME via the factory (isolated + reseeded);
    # an instance / bare callable is passed through as-is.
    factory = policy if inspect.isclass(policy) else None

    # Always compute the full envelope so the experiment log can sample games + summary,
    # regardless of what the caller asked to be returned.
    result = dev_simulate(
        policy, opponents, num_games,
        policy_factory=factory,
        executor="thread",                             # PLM policies live in the kernel; never pickled
        clock=cfg["clock"], per_move_s=cfg["per_move_s"], game_clock_s=cfg["game_clock_s"],
        max_moves=cfg["max_moves"],
        evaluate=cfg["evaluate"], eval_depth=cfg["eval_depth"],
        reference_engine=cfg["reference_engine"],
        max_illegal_retries=cfg["max_illegal_retries"], on_illegal=cfg["on_illegal"],
        base_seed=seed, max_workers=cfg["max_workers"],
        return_summary=True,
    )

    # Experiment auto-log: when the run harness sets `_PLM_EXPERIMENT_DIR`, append one
    # record per simulate call (score, est-Elo, the policy's source AT THIS MOMENT, a few
    # sampled games) so the dashboard can chart progress + replay games. This is on the
    # SEALED instrument, so the model cannot disable it. Logging must NEVER break a run.
    _expdir = os.environ.get("_PLM_EXPERIMENT_DIR")
    if _expdir:
        try:
            import pathlib
            import random as _random
            base = pathlib.Path(_expdir)
            logf = base / "simulates.jsonl"
            idx = sum(1 for _ in logf.open()) if logf.exists() else 0

            games = result.get("games", []) or []
            # sample up to 4 games to persist for replay: a win, a loss, a draw if present,
            # then fill with random others (deterministic in `seed`).
            picked, seen = [], set()
            by_outcome = {}
            for g in games:
                by_outcome.setdefault(g.get("winner"), []).append(g)
            for key in ("policy", "opponent", "draw", None):
                grp = by_outcome.get(key)
                if grp and grp[0]["game_id"] not in seen:
                    picked.append(grp[0]); seen.add(grp[0]["game_id"])
            extra = [g for g in games if g["game_id"] not in seen]
            _random.Random(seed).shuffle(extra)
            for g in extra[: max(0, 4 - len(picked))]:
                picked.append(g); seen.add(g["game_id"])

            gdir = base / "games" / str(idx)
            gdir.mkdir(parents=True, exist_ok=True)
            for g in picked:
                (gdir / f"{g['game_id']}.json").write_text(json.dumps(g))

            # the policy's source as it stands NOW (the "policyzero code at this point")
            psource = getattr(policy, "_p_source", None)
            pname = getattr(policy, "__name__", None) or type(policy).__name__
            if psource is None:
                try:
                    psource = read_policy(pname)       # ambient kernel helper
                except Exception:
                    psource = None

            record = {
                "idx": idx,
                "ts": time.time(),
                "policy_name": pname,
                "policy_source": psource,
                "num_games": num_games,
                "opponents": opponents,
                "clock": cfg["clock"], "per_move_s": cfg["per_move_s"],
                "game_clock_s": cfg["game_clock_s"], "max_moves": cfg["max_moves"],
                "summary": result.get("summary"),
                "sampled_games": [g["game_id"] for g in picked],
            }
            with logf.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception:
            pass                                       # logging is best-effort; never break the run

    return result if return_summary else result.get("games", [])
