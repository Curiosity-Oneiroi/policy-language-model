"""The termination protocol, published as constants (FULL only).

An author who wants to reproduce `react_auto`'s exact protocol over raw
`react()` calls can use these verbatim. There is deliberately NO session
registry here: under state weaving the caller owns both containers — the
message list and the state dict — so there is nothing for the harness to
keep, key, evict, or tombstone.

Both constants belong to `react_auto`: a sub-agent that must RETURN before its
rounds run out is told so. Raw `react` has NO constants here because it injects
NOTHING — the capability narrates nothing about itself (author ruling); every
word in a stepped weave is either the author's or the model's.

Trajectory grants (the documented recipes — reviewer round 12, A5). Showing a
sub-agent its own trajectory is a namespace grant like any other; choose the
exposure deliberately:

    namespace["messages"] = messages               # LIVE + WRITABLE: it can
                                                   # rewrite its own past
    namespace["trace"] = copy.deepcopy(messages)   # frozen snapshot

    for _ in range(k):                             # refreshed-per-round copy
        namespace["trace"] = copy.deepcopy(messages)
        react(messages, namespace, rounds=1)       # (the old expose_messages)

There is NO live read-only list view (rejected as complexity); if a live grant
corrupts the weave's round pairing, that is a POLICY failure (A5f).
"""

AUTO_BUDGET_BRIEFING = ("You have {max_turns} rounds from now. Terminate with "
                        "RETURN(value) before they run out.")
AUTO_FINAL_ROUND = ("[final round] This is your last round. Call "
                    "RETURN(value) now with your best answer.")
