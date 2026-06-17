@policy
class policyzero:
    """The chess policy you (the PLM) must make as strong as possible. THIS is your
    main deliverable — the canonical policy the harness reads as your answer.

    Contract — `call(self, state) -> uci_move_str`:

        A `policyzero` instance is handed the rich, base-types `state` dict for one
        position and must return ONE legal move as a UCI string (e.g. "e2e4",
        "e7e8q"). Pick from `state["legal_uci"]`; `state["legal_moves"]` carries a
        rich dict per move (uci/from/to/piece/is_capture/captured_piece/gives_check/
        promotion/is_castling/...). Other useful keys: `material`, `hanging`,
        `attacked_by_opp`, `attacked_by_stm`, `is_check`, `side_to_move`, `fen`,
        `board_ascii`, `move_history`. The dict is yours to mutate freely.

    Hard rule — NO LLM IN THE MOVE LOOP. Every move is on a wall-clock timeout
    (see `simulate(..., clock=..., per_move_s=...)`): exceed it and you LOSE the
    game on time. An LLM call is far too slow and will flag every move — so `call`
    must be pure, fast Python: heuristics, lookup tables, and (if you want depth)
    your OWN search — e.g. rebuild the position with `import chess;
    chess.Board(state["fen"])`, push candidate moves, and evaluate. Fast code only.

    How to work on this (your loop):

        * Evaluate the current policy: `simulate(policyzero, num_games, opponents=[...])`
          returns per-game trajectories (and, later, win/Elo metrics vs the chosen
          opponents). Read the trajectories to see WHERE it goes wrong — blunders,
          hung pieces, timeouts, the move just before a losing swing.
        * Form a hypothesis, then improve `policyzero` directly with the edit API
          (`policyzero._edit(...)` / `edit_policy(...)` / `policyzero._rewrite(...)`),
          and re-`simulate` to check the win rate moved the right way.
        * You are NOT limited to one policy. `duplicate_policy("policyzero", "cand_x")`
          to spin up experimental variants, A/B them with `simulate`, and fold what
          wins back into `policyzero`. You may also author small helper policies to
          test a hypothesis or compute something. But `policyzero` is the one that
          counts — keep your best, verified strategy here.

    The body below is a deliberately simple SEED: greedy on material, take free
    checks/promotions, shy away from moving onto a square the opponent attacks, a
    nudge toward the centre, random legal move as the floor. It beats random and
    little else — that is the point. Make it better.
    """

    # centipawn-ish piece values (÷100) for the greedy material heuristic
    _VALUE = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0}

    def __init__(self):
        import random
        self._rng = random.Random()

    def set_seed(self, seed):
        # simulate() reseeds each game so the random tie-break is reproducible per game
        self._rng.seed(seed)

    def call(self, state):
        legal = state["legal_moves"]
        if not legal:                                  # game already over; nothing to play
            return ""
        attacked = set(state.get("attacked_by_opp", ()))
        best, best_score = [], None
        for m in legal:
            score = 0.0
            cap = m.get("captured_piece")              # winning material dominates the score
            if cap:
                score += 10.0 * self._VALUE.get(cap.upper(), 0)
            promo = m.get("promotion")
            if promo:
                score += 8.0 * self._VALUE.get(promo.upper(), 0)
            if m.get("gives_check"):
                score += 0.5
            to_sq = m.get("to")
            if to_sq in attacked:                      # rough safety: don't step into a capture
                score -= self._VALUE.get((m.get("piece") or "").upper(), 0)
            if to_sq and to_sq[0] in "cdef" and to_sq[1] in "3456":
                score += 0.1                           # mild central pull
            if best_score is None or score > best_score:
                best, best_score = [m["uci"]], score
            elif score == best_score:
                best.append(m["uci"])
        return self._rng.choice(best)                  # tie-break (seeded per game)

    def __call__(self, state):
        # also a bare callable, so it works wherever a `policy(state)` is expected
        return self.call(state)
