"""
Circuit cookbook — worked, TESTED meta-reasoning circuits the PLM can adapt for the chess task.

Each is a *policy that internally dispatches* sub-LLM reactors (react_llm / natural_llm / parallel)
to BUILD, EXPLORE, DIAGNOSE, TUNE, IMPROVE and HARDEN the chess engine FOR the lab-lead — so the
lead DESIGNS circuits instead of hand-writing code. Wrap any of these in `@policy` to persist it as
a reusable researcher; compose them (diagnose -> add_feature_kept / tune_eval -> harden -> tournament).

Design principles baked in (this is how a circuit is supposed to be engineered):
  * DECOMPOSE — each circuit owns ONE job a sub-agent can actually finish.
  * RIGHT-SIZE the budget — hard sub-tasks get MORE `max_turns`, NOT a strict multi-field
    `constraint=` they can't satisfy (an over-strict constraint makes every RETURN bounce ->
    budget exhausted -> the sub-agent returns nothing). Brief fully (blank-slate rule); verify later.
  * VERIFY IN THE CIRCUIT — compile the returned source, A/B it with `simulate`, keep only what
    measurably helps. Never trust a sub-agent's "it works".
  * HANDLE failure — `parallel` returns exceptions in-slot (isinstance check); `react_llm` raises on
    exhaustion (try/except) -> re-dispatch a smaller/better-resourced circuit, do NOT hand-grind.
  * HARVEST — every circuit returns a concrete artifact (source / weaknesses / elo-delta) to use next.

Ambient names (present in the kernel policy space): react_llm, natural_llm, parallel, simulate,
duplicate_policy, rewrite_policy, policyzero, read_policy.
"""

# ============================ shared helpers ==============================

def _est_elo(sm):
    """Read est_elo defensively from a simulate(...) result (dict OR object)."""
    for path in (("summary", "est_elo"), ("est_elo",)):
        cur = sm
        for k in path:
            cur = cur.get(k) if isinstance(cur, dict) else getattr(cur, k, None)
            if cur is None:
                break
        if isinstance(cur, (int, float)):
            return float(cur)
    return 0.0


def _compiles(src):
    """Cheap validity gate: does the returned policy source at least parse?"""
    if not isinstance(src, str) or "class policyzero" not in src:
        return False
    try:
        compile(src, "<candidate>", "exec")
        return True
    except Exception:
        return False


def _uniq(stem):
    """A run-unique policy name (avoids NAME_TAKEN across repeated circuit calls)."""
    g = globals().setdefault("_cookbook_seq", {})
    g[stem] = g.get(stem, 0) + 1
    return "%s_%d" % (stem, g[stem])


def _as_policy(stem, src):
    """Materialize a source string into a runnable named policy forked from policyzero.
    Returns (policy_obj, name) or (None, None) if it won't compile/install."""
    if not _compiles(src):
        return None, None
    name = _uniq(stem)
    r = duplicate_policy("policyzero", name)
    if not r:                      # NAME_TAKEN / IMMUTABLE / ...
        return None, None
    rewrite_policy(name, src)
    return r.value, name


def _score(policy, games, seed):
    """est_elo of a policy vs the configured ladder; 0.0 on any failure (never raise)."""
    try:
        return _est_elo(simulate(policy, games, seed=seed))
    except Exception:
        return 0.0


# ============================ the circuits ================================

def tournament(n, task, build_kwargs=None, games=12, seed=12345):
    """EXPLORE-AND-SELECT. One ideator proposes N distinct approaches; N builders (same JOB,
    different APPROACH in the user prompt) write a full policy each IN PARALLEL; the circuit
    materializes + scores each vs the ladder and RETURNS the best source. 'A search wants a
    parallel tournament.'  -> {source, est_elo, n_built}."""
    build_kwargs = build_kwargs or {}

    # 1) ideator -> N distinct approaches
    try:
        ideas = react_llm(
            [{"role": "user", "content":
              ("You are a chess-engine strategist. Propose EXACTLY %d DISTINCT, concrete approaches "
               "to: %s. Vary the core idea (aggressive-tactical / solid-positional / endgame-first / "
               "deeper-pruned-search / better-eval-weights ...). RETURN a python list of %d short "
               "strings." % (n, task, n))}],
            max_turns=4, return_budget=3)
    except Exception:
        ideas = None
    if not isinstance(ideas, (list, tuple)) or not ideas:
        ideas = ["distinct approach %d" % (i + 1) for i in range(n)]
    ideas = [str(x) for x in list(ideas)[:n]]

    # 2) builders in PARALLEL — same job, different approach
    def _builder(idea):
        def _go():
            return react_llm(
                [{"role": "user", "content":
                  ("You are a chess-engine engineer. JOB: %s. Use THIS approach: %s. `chess` "
                   "(python-chess) is bound. Write the COMPLETE `policyzero` class source "
                   "(__init__/set_seed/call/__call__; call(state)->a uci in state['legal_uci']; pure "
                   "fast python under the per-move budget; never crash; random.choice fallback). "
                   "SELF-TEST on the start position before finishing. RETURN the full class source "
                   "as a STRING." % (task, idea))}],
                kwargs=build_kwargs, max_turns=16, return_budget=4)
        return _go
    results = parallel(*[_builder(i) for i in ideas])

    # 3) tournament: score each, keep the best (failed builders are skipped, NOT hand-fixed)
    best_src, best_elo, built = None, float("-inf"), 0
    for src in results:
        if isinstance(src, BaseException):
            continue
        pol, _ = _as_policy("tourney", src)
        if pol is None:
            continue
        built += 1
        elo = _score(pol, games, seed)
        if elo > best_elo:
            best_src, best_elo = src, elo
    return {"source": best_src, "est_elo": (best_elo if best_src else None), "n_built": built}


def policy_coder(current_source, change_spec, build_kwargs=None, games=8, seed=777, tries=2):
    """BUILD/EDIT WITH SELF-TEST + CIRCUIT-SIDE VERIFY. Hand a sub-agent the current source + a
    SINGLE focused change; it writes it, self-tests, returns source; the circuit then VERIFIES
    (compiles + plays a few games without crashing) and re-dispatches once with the failure detail
    if it didn't take. -> {source, ok, est_elo}.  (This is the safe 'give it the policy + the change'
    coder you asked for — note: ONE change at a time, generous budget, verify outside.)"""
    build_kwargs = dict(build_kwargs or {})
    build_kwargs["current_source"] = current_source
    feedback = ""
    last = None
    for _ in range(max(1, tries)):
        brief = ("You are a chess-engine engineer. You have the current policy in "
                 "`current_source` (read it first) and `chess` bound. MAKE EXACTLY THIS CHANGE: %s. "
                 "Keep everything else working; do not crash; keep the random fallback. SELF-TEST the "
                 "edited class on a few positions before finishing. RETURN the FULL edited class "
                 "source as a STRING." % change_spec)
        if feedback:
            brief += " PREVIOUS ATTEMPT FAILED VERIFICATION: %s. Fix that." % feedback
        try:
            src = react_llm([{"role": "user", "content": brief}],
                            kwargs=build_kwargs, max_turns=14, return_budget=4)
        except Exception as e:
            feedback = "the build dispatch errored (%s); the task may be too big — keep it minimal" % type(e).__name__
            continue
        last = src
        pol, _ = _as_policy("coded", src)
        if pol is None:
            feedback = "returned source did not compile or was not a full policyzero class"
            continue
        elo = _score(pol, games, seed)
        if elo > 0.0:                      # ran real rated games without dying
            return {"source": src, "ok": True, "est_elo": elo}
        feedback = "the edited policy produced no rated games (likely crashing / illegal moves)"
    return {"source": last, "ok": False, "est_elo": 0.0}


def diagnose(policy=None, games=30, seed=2024):
    """HELPER CIRCUIT (indirect benefit). Measure the policy, hand the LOSING/DRAWN game summaries
    to an analyst reactor, and RETURN a RANKED list of concrete weaknesses to attack next (king
    safety, hangs pieces in tactics, no endgame technique, times out, weak openings ...). The lead
    USES this to decide WHICH circuit to build — that downstream use is the benefit."""
    if policy is None:
        policy = policyzero
    try:
        sm = simulate(policy, games, seed=seed)
    except Exception:
        return {"weaknesses": [], "note": "simulate failed"}
    summary = sm.get("summary") if isinstance(sm, dict) else getattr(sm, "summary", {})
    summary = summary or {}
    try:
        weaknesses = react_llm(
            [{"role": "user", "content":
              ("You are a chess-engine analyst. Here is a measured policy's aggregate result: %r. "
               "Identify the TOP failure modes (each: name + why + a CONCRETE engine fix) and RETURN "
               "a python list of short strings, MOST IMPACTFUL FIRST. Be specific (e.g. 'no king-"
               "safety term -> gets mated; add a pawn-shield/attacker-count penalty')." % (summary,))}],
            max_turns=5, return_budget=3)
    except Exception:
        weaknesses = []
    if not isinstance(weaknesses, (list, tuple)):
        weaknesses = []
    return {"weaknesses": [str(w) for w in weaknesses], "summary": summary}


def _ab_keep(base_source, new_source, games, seed):
    """A/B two source strings vs the ladder (same seed); return the better one + the elo delta."""
    base_pol, _ = _as_policy("ab_base", base_source)
    new_pol, _ = _as_policy("ab_new", new_source)
    base_elo = _score(base_pol, games, seed) if base_pol else 0.0
    new_elo = _score(new_pol, games, seed) if new_pol else float("-inf")
    if new_pol is not None and new_elo > base_elo:
        return {"source": new_source, "kept": True, "delta": new_elo - base_elo, "est_elo": new_elo}
    return {"source": base_source, "kept": False, "delta": (new_elo - base_elo) if new_pol else 0.0,
            "est_elo": base_elo}


def add_feature_kept(current_source, feature, build_kwargs=None, games=12, seed=314):
    """INCREMENTAL IMPROVEMENT + ABLATION. Implement ONE named engine feature (e.g. 'a transposition
    table', 'null-move pruning', 'a king-safety eval term', 'tapered mid/endgame eval') via
    policy_coder, then A/B it against the current engine and KEEP it only if est_elo actually
    improved. -> {source, kept, delta, est_elo}.  Compose per feature to climb toward 1800+."""
    built = policy_coder(current_source, "add %s to the engine" % feature,
                         build_kwargs=build_kwargs, games=max(6, games // 2), seed=seed)
    if not built.get("ok"):
        return {"source": current_source, "kept": False, "delta": 0.0, "feature": feature,
                "note": "feature build failed verification"}
    out = _ab_keep(current_source, built["source"], games, seed)
    out["feature"] = feature
    return out


def tune_eval(current_source, build_kwargs=None, rounds=3, games=12, seed=99):
    """CLOSED-LOOP TUNING. Iteratively let a reactor propose evaluation-weight changes (piece values,
    PST scale, king-safety / mobility / pawn-structure weights), apply each via policy_coder, MEASURE
    with simulate, and feed the elo back so the next proposal learns. Keeps the best. -> {source,
    est_elo, history}.  Eval tuning is one of the biggest Elo levers once the search works."""
    best_src = current_source
    base_pol, _ = _as_policy("tune_base", current_source)
    best_elo = _score(base_pol, games, seed) if base_pol else 0.0
    history = [("baseline", best_elo)]
    for _ in range(max(1, rounds)):
        try:
            proposal = react_llm(
                [{"role": "user", "content":
                  ("You are tuning a chess engine's EVALUATION weights. Current best est_elo=%.0f. "
                   "History: %r. Propose ONE concrete weight change likely to raise strength (state "
                   "exactly which constant to change and to what). RETURN it as a short string." %
                   (best_elo, history[-3:]))}],
                max_turns=4, return_budget=3)
        except Exception:
            break
        coded = policy_coder(best_src, "apply this eval-weight change: %s" % proposal,
                             build_kwargs=build_kwargs, games=max(6, games // 2), seed=seed)
        if not coded.get("ok"):
            history.append((str(proposal)[:40], None))
            continue
        cand_pol, _ = _as_policy("tune_cand", coded["source"])
        elo = _score(cand_pol, games, seed) if cand_pol else 0.0
        history.append((str(proposal)[:40], elo))
        if elo > best_elo:
            best_src, best_elo = coded["source"], elo
    return {"source": best_src, "est_elo": best_elo, "history": history}


def harden(source, build_kwargs=None, games=10, seed=55):
    """ROBUSTNESS GATE. An adversarial critic reactor hunts for ways the engine LOSES for free —
    illegal moves, crashes, per-move TIMEOUTS, eval blunders, slow paths — then a fixer patches them;
    the circuit verifies the patched source still runs. Low-Elo policies bleed points to crashes /
    timeouts, so this is pure upside. -> {source, ok, report}."""
    build_kwargs = dict(build_kwargs or {})
    build_kwargs["current_source"] = source
    try:
        report = react_llm(
            [{"role": "user", "content":
              ("You are an adversarial chess-engine reviewer. The current source is in "
               "`current_source`. Find every way it can LOSE FOR FREE: illegal/None moves, "
               "exceptions, per-move TIMEOUTS (too-deep search with no time guard), eval sign bugs, "
               "pathological slow positions. RETURN a python list of concrete defects, worst first.")}],
            kwargs=build_kwargs, max_turns=6, return_budget=3)
    except Exception:
        report = []
    if not isinstance(report, (list, tuple)) or not report:
        return {"source": source, "ok": True, "report": []}
    fixed = policy_coder(source, "fix ALL of these robustness defects (add a time guard / "
                                 "iterative-deepening cutoff, legal-move + fallback safety): %r" % list(report),
                         build_kwargs=build_kwargs, games=games, seed=seed)
    return {"source": fixed.get("source", source), "ok": bool(fixed.get("ok")),
            "report": [str(x) for x in report]}
