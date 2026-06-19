"""ContraPrompt — a gepa ``ProposalFn`` built faithfully from the paper.

Source of truth
---------------
This module is reconstructed STRICTLY from the vendored paper text, NOT from
memory and NOT from any reference code (the paper ships no code repo):

    optimizer/vendor/papers/contraprompt_2604.17937.txt
    "ContraPrompt: Contrastive Prompt Optimization" (arXiv 2604.17937)

The pieces we ground in the paper (exact line ranges in that file):

* Algorithm 1 (Appendix A, "FULL ALGORITHM"), lines 705-733. The five phases:
    Phase 1  Instrumented agentic retry loop  (A attempts, default A=3, error
             feedback appended to later attempts) -> attempt records A.
    Phase 2  Contrastive pair mining: per example, delta_i = c+_i - c-_i where
             c+ = best attempt score, c- = worst attempt score; discard pairs
             with delta_i < delta_min = 0.02; rank remaining by delta_i.
             (Section 4.2, lines 340-350.)
    Phase 3  Rule extraction (dyadic trace analysis) over each (tau-, tau+)
             pair + AggregatedFailureAnalysis over the all-fail bucket.
             (Sections 4.3-4.4, lines 356-377.)
    Phase 4  Input-aware TreeMerge: cluster failing inputs by observable
             features phi(x); emit an <always> section + <branch condition=...>
             sections; two levels of nesting max. (Section 4.5, lines 379-411.)
    Phase 5  InjectTree + Evaluate + patience-checkpoint (we do not evaluate
             inside a ProposalFn; gepa owns Evaluate/checkpointing).

* The RULE-EXTRACTION prompt (Section 4.3, lines 356-364). The paper states
  the prompt "presents both traces together with their appended feedback
  contexts and instructs the extractor to focus on changes in reasoning
  approach rather than changes in conditioning, asking: 'What did the improved
  reasoning do differently in its chain of thought? Extract a general rule that
  captures the reasoning pattern.'" Output template (line 359-360):
      "When [input pattern], [strategy] because [causal justification]."

* The DECISION-TREE construction prompt (Section 4.5, lines 379-411). The tree
  is "constructed by a language model that clusters failing inputs by
  structural groupings and assigns rules to relevant clusters"; the "when"
  clause becomes the branch condition and the "because" clause may be condensed;
  branch conditions are "restricted to features directly observable from the
  input"; output uses <always>/<rule> and <branch condition="...">/<rule>
  (worked example, lines 388-407).

Honest adaptation of "the retry loop" to our setting
-----------------------------------------------------
A gepa ``ProposalFn`` runs OFFLINE on an already-collected ``reflective_dataset``;
there is no live model we can call to re-attempt a position, so we CANNOT run
Algorithm 1's Phase 1 (SolveWithRetries) literally. The paper's whole value,
though, comes from Phase 2's contrastive primitive: a (tau-, tau+) pair that is
"a higher-quality and a lower-quality trace on the same input" (Section 9,
lines 636-637; and the formal s(tau+,y) > s(tau-,y) framing, lines 244 & 257).

Our reflective_dataset records ARE exactly such traces — each record is one
per-opponent / per-position outcome with a numeric ``outcome`` (score + W/L/D),
a ``trajectory_excerpt`` and a ``feedback`` string. We therefore SUBSTITUTE the
live retry loop with a SAME-CONTEXT PAIRING over these static records:

  - We reuse Phase 2 verbatim in spirit: split records into low-outcome (tau-)
    and high-outcome (tau+) buckets by a numeric score parsed from ``outcome``.
  - "Same input" in the paper means the identical task instance solved twice.
    We have no repeated instance, so we pair on COMPARABLE inputs instead:
    a lost/weak record (tau-) is matched with a won/strong record (tau+) whose
    input is the nearest comparable (same/adjacent opponent family, lost-vs-won
    game). This is the documented relaxation — we hold "the task family" fixed
    rather than "the exact instance", because the dataset gives us no exact
    re-attempt. delta = score(tau+) - score(tau-); pairs with delta < delta_min
    (=0.02) are dropped and the rest are ranked by delta, EXACTLY as in 4.2.
  - Records with no higher-outcome partner fall into the all-fail bucket and go
    through AggregatedFailureAnalysis (Phase 3, Section 4.4), matching the
    paper's fallback for the "all attempts fail" regime.

Everything downstream — the dyadic rule-extraction prompt, the three-part
"when / do / because" template, and the input-aware <branch> tree — is run
exactly as written in the paper. The only relaxation is the *origin* of the
pairs (static comparable records vs. a live A-attempt retry loop), which is the
one step the paper backs with prose + Algorithm 1 but no code.

gepa ProposalFn protocol (optimizer/../gepa/core/adapter.py)::

    def __call__(self, candidate, reflective_dataset, components_to_update) -> dict[str,str]

candidate = {"system_prompt": str, "verifier": str}
reflective_dataset[component] = list of {input, current_component, outcome,
                                         trajectory_excerpt, feedback}
return {component: new_text}

This module makes NO network / LLM calls; the reflection model is an injected
``Callable[[str], str]`` (a real reflection LM over Slate). It returns a valid
three-part rule on extraction calls and a <branch> tree on tree calls (it
distinguishes the two by markers in the prompt).
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping, Sequence
from typing import Any, Callable

__all__ = [
    "ContraPromptProposer",
    "Rule",
]

# Paper constant (Section 4.2, line 348; config line 785): delta_min = 0.02.
DELTA_MIN = 0.02
# Paper default A=3 (Section 4.1, line 317; config line 785).
DEFAULT_ATTEMPTS = 3

# Markers we embed in the two prompts so a reflection LM can
# tell rule-extraction calls from tree-construction calls.
_RULE_MARKER = "[[CONTRAPROMPT::EXTRACT_RULE]]"
_TREE_MARKER = "[[CONTRAPROMPT::TREE_MERGE]]"


# --------------------------------------------------------------------------- #
# Data: a parsed three-part rule.                                             #
# --------------------------------------------------------------------------- #
class Rule:
    """A parsed three-part contrastive rule.

    Template (paper, line 359-360):
        "When [input pattern], [strategy] because [causal justification]."
    """

    __slots__ = ("when", "do", "because", "raw")

    def __init__(self, when: str, do: str, because: str, raw: str = ""):
        self.when = when.strip()
        self.do = do.strip()
        self.because = because.strip()
        self.raw = raw.strip() or self.render()

    def render(self) -> str:
        return f"When {self.when}, {self.do} because {self.because}."

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return f"Rule(when={self.when!r}, do={self.do!r}, because={self.because!r})"


# Parses "When X, Y because Z." (case-insensitive on the keywords). We split on
# the FIRST comma after "When" and the LAST " because " so commas inside the
# strategy survive.
_WHEN_RE = re.compile(r"\bwhen\b\s*(.*)", re.IGNORECASE | re.DOTALL)


def parse_three_part_rule(text: str) -> Rule | None:
    """Parse one three-part 'when / do / because' rule from ``text``.

    Returns None if the text does not match the template.
    """
    if not text:
        return None
    # Grab the first line that contains all three keyword anchors.
    for line in _candidate_rule_lines(text):
        m = _WHEN_RE.search(line)
        if not m:
            continue
        rest = m.group(1)
        # Need a comma (end of "when" clause) and a " because " (start of cause).
        if "," not in rest or " because " not in rest.lower():
            continue
        when_part, after = rest.split(",", 1)
        # Split off the LAST 'because' so a 'because' inside the strategy is kept.
        idx = after.lower().rfind(" because ")
        do_part = after[:idx]
        because_part = after[idx + len(" because "):]
        because_part = because_part.rstrip().rstrip(".")
        if when_part.strip() and do_part.strip() and because_part.strip():
            return Rule(when_part, do_part, because_part, raw=line)
    return None


def _candidate_rule_lines(text: str):
    # Try whole-string first (multi-line rules), then line by line.
    yield " ".join(text.split())
    for line in text.splitlines():
        line = line.strip().lstrip("-*0123456789.) ").strip()
        if line:
            yield line


# --------------------------------------------------------------------------- #
# Paper prompts (built verbatim from Sections 4.3 and 4.5).                    #
# --------------------------------------------------------------------------- #
def build_rule_extraction_prompt(neg: Mapping[str, Any], pos: Mapping[str, Any]) -> str:
    """Dyadic trace-analysis prompt (paper Section 4.3, lines 356-364).

    Presents BOTH traces with their appended feedback context, instructs the
    extractor to focus on the CHANGE IN REASONING APPROACH rather than the
    change in conditioning, and asks for a rule in the three-part template.
    """
    neg_traj = str(neg.get("trajectory_excerpt", "")).strip()
    pos_traj = str(pos.get("trajectory_excerpt", "")).strip()
    neg_fb = str(neg.get("feedback", "")).strip()
    pos_fb = str(pos.get("feedback", "")).strip()
    return f"""{_RULE_MARKER}
You are performing dyadic reasoning-trace analysis. Below are TWO complete
chain-of-thought traces produced on the same / a comparable input under the
same base prompt: one LOWER-QUALITY (failed) trace and one HIGHER-QUALITY
(successful) trace, each shown with the error/outcome feedback context that was
appended to it.

[tau-] FAILED / LOWER-OUTCOME TRACE
  input    : {str(neg.get('input', '')).strip()}
  outcome  : {str(neg.get('outcome', '')).strip()}
  reasoning: {neg_traj}
  feedback : {neg_fb}

[tau+] SUCCESSFUL / HIGHER-OUTCOME TRACE
  input    : {str(pos.get('input', '')).strip()}
  outcome  : {str(pos.get('outcome', '')).strip()}
  reasoning: {pos_traj}
  feedback : {pos_fb}

IMPORTANT: Focus on the CHANGE IN REASONING APPROACH between the two chains of
thought, NOT on the change in conditioning (do not credit the rule to the mere
presence of appended error feedback). Identify a specific reasoning step that is
PRESENT in the successful trace and ABSENT in the failed trace.

What did the improved reasoning do differently in its chain of thought? Extract
a single GENERAL, TRANSFERABLE rule that captures that reasoning pattern, using
EXACTLY this template (one sentence):

  "When [input pattern], [strategy] because [causal justification]."
"""


def build_failure_analysis_prompt(records: Sequence[Mapping[str, Any]]) -> str:
    """Aggregated failure analysis prompt (paper Section 4.4, lines 370-377).

    For inputs where no higher-outcome partner exists (the all-fail bucket),
    analyze the group collectively to surface what is structurally absent. The
    paper keeps the output in the SAME three-part template so it can feed the
    tree merge, so we ask for the same shape.
    """
    bullets = "\n".join(
        f"  - input: {str(r.get('input', '')).strip()} | outcome: "
        f"{str(r.get('outcome', '')).strip()} | feedback: {str(r.get('feedback', '')).strip()}"
        for r in records
    )
    return f"""{_RULE_MARKER}
These training inputs ALL failed across every attempt (no successful retry, so
no contrastive partner exists). Analyze the group COLLECTIVELY to identify the
systematic pattern of what is structurally ABSENT when the model cannot succeed:

{bullets}

Emit ONE general corrective rule for this failure group, using EXACTLY this
template (one sentence):

  "When [input pattern], [strategy] because [causal justification]."
"""


def build_tree_merge_prompt(rules: Sequence[Rule], failing_inputs: Sequence[str]) -> str:
    """Input-aware tree-construction prompt (paper Section 4.5, lines 379-411).

    The LM clusters failing inputs by observable features phi(x), puts
    universally-applicable rules in <always>, and routes the rest into
    <branch condition="..."> sections (conditions = features observable from the
    input text). The "when" clause becomes the branch condition; the "because"
    clause may be condensed. Two levels of nesting max.
    """
    rule_lines = "\n".join(f"  - {r.render()}" for r in rules)
    inputs_lines = "\n".join(f"  - {i}" for i in failing_inputs if str(i).strip())
    return f"""{_TREE_MARKER}
Construct an INPUT-AWARE decision tree from the extracted rules below. Cluster
the failing inputs by STRUCTURAL groupings (features directly observable from
the input text) and assign each rule to the cluster(s) it serves.

Extracted rules (three-part 'when / do / because' template):
{rule_lines}

Failing inputs to cluster by observable features phi(x):
{inputs_lines}

Output format (XML; AT MOST two levels of nesting):
  - Put rules that apply to EVERY input inside a single <always> ... </always>
    block, one <rule>...</rule> per rule.
  - For each cluster, emit <branch condition="<observable input feature>"> ...
    </branch> containing the <rule>...</rule> entries for that cluster.
  - The 'when' clause of a rule becomes the branch condition; the 'because'
    clause may be condensed to keep the strategy actionable but short.
  - Branch conditions MUST be deterministically checkable from the input text.

Return ONLY the XML tree.
"""


# --------------------------------------------------------------------------- #
# Proposer                                                                     #
# --------------------------------------------------------------------------- #
class ContraPromptProposer:
    """ContraPrompt as a gepa ``ProposalFn``.

    Usage::

        proposer = ContraPromptProposer(reflection_lm=my_llm)
        new_texts = proposer(candidate, reflective_dataset, ["system_prompt"])

    ``__call__`` and ``propose`` are aliases (gepa calls the instance).
    """

    def __init__(self, reflection_lm: Callable[[str], str], attempts: int = DEFAULT_ATTEMPTS,
                 base_template: str | None = None):
        if not callable(reflection_lm):
            raise TypeError("reflection_lm must be callable: (prompt: str) -> str")
        self.reflection_lm = reflection_lm
        self.attempts = attempts  # paper A (default 3); kept for fidelity/diagnostics.
        # Optional PE2 framing. When set, the tree-merge (Phase 4) prompt is
        # substituted into PE2's diagnose->edit grammar (<curr_param>=current
        # component, <side_info>=the contrastive-rule tree-merge instruction) so
        # the composite is "PE2 grammar + ContraPrompt rule mining" — both real.
        self.base_template = base_template
        # Diagnostics populated on each propose() call (handy for the self-test).
        self.last_rules: list[Rule] = []
        self.last_tree: str = ""
        self.last_pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []

    # gepa invokes the instance directly.
    def __call__(
        self,
        candidate: Mapping[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        return self.propose(candidate, reflective_dataset, components_to_update)

    def propose(
        self,
        candidate: Mapping[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        out: dict[str, str] = {}
        for component in components_to_update:
            records = list(reflective_dataset.get(component, []) or [])
            parent_text = str(candidate.get(component, ""))
            if not records:
                out[component] = parent_text  # nothing to learn from; keep parent.
                continue

            # Phase 2: contrastive pair mining over the static records.
            pairs, all_fail = self._mine_contrastive_pairs(records)
            self.last_pairs = pairs

            # Phase 3: dyadic rule extraction + aggregated failure analysis.
            rules: list[Rule] = []
            for neg, pos in pairs:
                prompt = build_rule_extraction_prompt(neg, pos)
                rule = parse_three_part_rule(self._safe_lm(prompt))
                if rule is not None:
                    rules.append(rule)
            if all_fail:
                fa_prompt = build_failure_analysis_prompt(all_fail)
                fa_rule = parse_three_part_rule(self._safe_lm(fa_prompt))
                if fa_rule is not None:
                    rules.append(fa_rule)
            rules = _dedupe_rules(rules)
            self.last_rules = rules

            if not rules:
                out[component] = parent_text  # no signal -> fall back to parent.
                continue

            # Phase 4: input-aware tree merge.
            failing_inputs = [str(r.get("input", "")) for r in records]
            tree_prompt = build_tree_merge_prompt(rules, failing_inputs)
            if self.base_template:
                from .pe2_prompt import frame_with_pe2
                tree_prompt = frame_with_pe2(
                    self.base_template, curr_param=parent_text, side_info=tree_prompt
                )
            tree = self._safe_lm(tree_prompt).strip()
            if "<branch" not in tree and "<always" not in tree:
                tree = _fallback_tree(rules)  # deterministic guarantee.
            self.last_tree = tree

            # Phase 5 (partial): inject. gepa owns Evaluate/checkpoint.
            if component == "verifier":
                out[component] = self._inject_into_verifier(parent_text, rules, tree)
            else:
                out[component] = self._inject_into_prompt(parent_text, tree)
        return out

    # ------------------------------------------------------------------ #
    # Phase 2 — contrastive pair mining (paper Section 4.2, lines 340-350)
    # ------------------------------------------------------------------ #
    def _mine_contrastive_pairs(
        self, records: Sequence[Mapping[str, Any]]
    ) -> tuple[list[tuple[Mapping[str, Any], Mapping[str, Any]]], list[Mapping[str, Any]]]:
        """Pair low-outcome (tau-) with high-outcome (tau+) comparable records.

        Returns (ranked_pairs, all_fail_records). Pairs with
        delta = score(tau+) - score(tau-) < DELTA_MIN are discarded; the rest
        are ranked by delta descending (paper 4.2).
        """
        scored = [(r, _outcome_score(r)) for r in records]
        if not scored:
            return [], []
        scores = [s for _, s in scored]
        mid = (max(scores) + min(scores)) / 2.0
        # tau- bucket = below midpoint (failures/weak); tau+ = at/above midpoint.
        negs = [(r, s) for (r, s) in scored if s < mid]
        poss = [(r, s) for (r, s) in scored if s >= mid]

        # Degenerate split (all equal): treat strictly-min as tau-, strictly-max as tau+.
        if not negs or not poss:
            lo, hi = min(scores), max(scores)
            if hi - lo < DELTA_MIN:
                return [], list(records)  # no contrast -> everything is "all-fail".
            negs = [(r, s) for (r, s) in scored if s == lo]
            poss = [(r, s) for (r, s) in scored if s == hi]

        poss_sorted = sorted(poss, key=lambda t: t[1], reverse=True)
        pairs: list[tuple[float, Mapping[str, Any], Mapping[str, Any]]] = []
        paired_negs: set[int] = set()
        for nr, ns in negs:
            # "Same input" relaxed to "nearest comparable input" (see docstring).
            best = max(
                poss_sorted,
                key=lambda pt: (_input_affinity(nr, pt[0]), pt[1]),
            )
            pr, ps = best
            delta = ps - ns
            if delta >= DELTA_MIN:
                pairs.append((delta, nr, pr))
                paired_negs.add(id(nr))

        pairs.sort(key=lambda t: t[0], reverse=True)  # rank by delta (paper 4.2).
        ranked = [(nr, pr) for (_, nr, pr) in pairs]

        # All-fail bucket: tau- records that never found a qualifying partner.
        all_fail = [nr for (nr, _) in negs if id(nr) not in paired_negs]
        return ranked, all_fail

    # ------------------------------------------------------------------ #
    # Injection                                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _inject_into_prompt(parent: str, tree: str) -> str:
        section = (
            "\n\n## Input-aware routing rules\n"
            "Rules below were mined from contrastive (failed vs. succeeded) traces and\n"
            "organized into an input-aware decision tree. Apply <always> rules every\n"
            "turn; apply a <branch> rule only when its condition matches the input.\n\n"
            f"{tree.strip()}\n"
        )
        return parent.rstrip() + section

    def _inject_into_verifier(self, parent: str, rules: Sequence[Rule], tree: str) -> str:
        """Inject rules into the verifier as a comment block.

        The verifier output MUST still parse to exactly one top-level
        ``def verifier(messages)`` function. We prepend the tree/rules as a comment header and
        validate with ``ast``; on any failure we return the parent verifier
        unchanged (paper provides no verifier component — this is our contract,
        documented in the module docstring).
        """
        if not _has_single_verifier_def(parent):
            # Parent itself is unexpected; do not risk producing worse output.
            return parent
        header_lines = [
            "# == ContraPrompt input-aware routing rules (mined from contrastive traces) ==",
            "# Apply <always> rules every turn; <branch> rules only when condition matches.",
        ]
        for line in tree.splitlines():
            header_lines.append("# " + line.rstrip())
        header_lines.append(
            "# These rules guide the kernel-resident verifier's trajectory-control"
        )
        header_lines.append(
            "# decisions below (it runs in the policy space with the full ambient"
        )
        header_lines.append(
            "# namespace and mutates `messages` in place); they do not alter the"
        )
        header_lines.append("# top-level `def verifier(messages)` signature or contract.")
        candidate_src = "\n".join(header_lines) + "\n\n" + parent.lstrip("\n")
        if _has_single_verifier_def(candidate_src):
            return candidate_src
        return parent  # ast fallback.

    # ------------------------------------------------------------------ #
    def _safe_lm(self, prompt: str) -> str:
        # raise-for-all: a non-working reflection LM fails loudly (no swallowing,
        # consistent with TextGrad/EvoPrompt/PromptBreeder). Exceptions propagate;
        # a non-str/None return is rejected here rather than silently degrading.
        res = self.reflection_lm(prompt)
        if not isinstance(res, str):
            raise TypeError(
                f"reflection_lm returned {type(res).__name__}, expected str"
            )
        return res


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


def _outcome_score(record: Mapping[str, Any]) -> float:
    """Parse a numeric score from a record's ``outcome``.

    The paper scores traces via s(tau, y) (Section 3.1, lines 243-257). Our
    records carry ``outcome`` like "score=0.0 (L)" / "score=1.0 (W)". We first
    read an explicit numeric, then fall back to W/D/L sentiment.
    """
    outcome = str(record.get("outcome", ""))
    m = re.search(r"score\s*[=:]\s*([-+]?\d*\.?\d+)", outcome, re.IGNORECASE)
    if m:
        return float(m.group(1))
    nums = _NUM_RE.findall(outcome)
    if nums:
        return float(nums[0])
    low = outcome.lower()
    if re.search(r"\b(win|won|\(w\)|success)\b", low):
        return 1.0
    if re.search(r"\b(draw|\(d\)|tie)\b", low):
        return 0.5
    if re.search(r"\b(loss|lost|lose|\(l\)|fail)\b", low):
        return 0.0
    return 0.0


_WORD_RE = re.compile(r"[a-z0-9]+")


def _input_affinity(a: Mapping[str, Any], b: Mapping[str, Any]) -> int:
    """Token-overlap affinity between two records' inputs.

    Implements the documented relaxation of "same input" -> "nearest comparable
    input": more shared tokens (opponent family, position descriptors) = more
    comparable.
    """
    ta = set(_WORD_RE.findall(str(a.get("input", "")).lower()))
    tb = set(_WORD_RE.findall(str(b.get("input", "")).lower()))
    return len(ta & tb)


def _dedupe_rules(rules: Sequence[Rule]) -> list[Rule]:
    seen: set[str] = set()
    out: list[Rule] = []
    for r in rules:
        key = (r.when.lower(), r.do.lower())
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _fallback_tree(rules: Sequence[Rule]) -> str:
    """Deterministic <always>/<branch> tree if the LM produced no XML."""
    lines = ["<always>"]
    branches: list[str] = []
    for r in rules:
        # A short 'when' clause -> universally-ish; else route into a branch.
        cond = r.when
        rule_xml = f"  <rule>{r.do} because {r.because}.</rule>"
        if len(cond) <= 24:
            lines.append(rule_xml)
        else:
            branches.append(f'<branch condition="{cond}">\n{rule_xml}\n</branch>')
    lines.append("</always>")
    if not branches and rules:
        # Guarantee at least one <branch> so the tree is genuinely input-aware.
        r0 = rules[0]
        branches.append(
            f'<branch condition="{r0.when}">\n  <rule>{r0.do} because {r0.because}.</rule>\n</branch>'
        )
    return "\n".join(lines + branches)


def _has_single_verifier_def(src: str) -> bool:
    """True iff ``src`` is EXACTLY ONE top-level ``def verifier(messages)`` function
    and NOTHING else (no other top-level defs/classes/imports/assignments; not
    async) — helpers/imports/constants live INSIDE the body. The verifier runs in
    the kernel policy space (full ambient namespace) and mutates the trajectory in
    place."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    body = tree.body
    return len(body) == 1 and isinstance(body[0], ast.FunctionDef) and body[0].name == "verifier"
