"""Real EvoPrompt evolutionary operators (GA crossover + mutation, DE) exposed
as gepa-compatible callables.

GROUNDING — every meta-prompt below is copied VERBATIM from the EvoPrompt source
vendored at ``optimizer/vendor/evoprompt/`` (NOT reconstructed from memory). The
extraction / substitution behaviour replicates the real driver in
``vendor/evoprompt/evoluter.py``:

  * GA crossover+mutation template  — ``vendor/evoprompt/data/template_ga.py``
        ``templates_2["cls"]`` (lines 2-17). The REUSABLE block (the part the
        driver actually fills) is the trailing instruction at lines 11-17, with
        ``<prompt1>`` / ``<prompt2>`` placeholders. Substitution in
        ``GAEvoluter.evolute`` is (evoluter.py:404-405):
            request_content = template.replace("<prompt1>", cand_a).replace("<prompt2>", cand_b)

  * DE template — ``vendor/evoprompt/data/template_de.py`` ``v1["sim"]``
        (lines 3-45). The REUSABLE trailing block is lines 35-45, with
        ``<prompt0>`` (basic/current), ``<prompt1>``, ``<prompt2>`` (two parents)
        and ``<prompt3>`` (donor) placeholders. Substitution in
        ``DEEvoluter.evolute`` is (evoluter.py:570-575):
            request_content = (template.replace("<prompt0>", old_prompt)
                .replace("<prompt1>", a).replace("<prompt2>", b).replace("<prompt3>", c))

  * MUTATION (paraphrase) template — ``vendor/evoprompt/llm_client.py``
        ``paraphrase`` (line 149):
            f"Generate a variation of the following instruction while keeping the semantic meaning.\nInput:{sentence}\nOutput:"

  * Final-prompt extraction — ``vendor/evoprompt/utils.py`` ``get_final_prompt``
        (lines 84-93): split on ``<prompt>`` / ``</prompt>``; else strip a single
        pair of surrounding double-quotes.

GEPA mapping note: gepa's reflection ``ProposalFn`` receives ONE candidate
(``(candidate, reflective_dataset, components_to_update) -> dict[str,str]``), so
only a single-parent operator fits it directly — we expose EvoPrompt's MUTATION
as that ProposalFn (``EvoPromptProposer``). EvoPrompt's two-parent GA crossover
and three-/four-parent DE operators do NOT fit the single-candidate ProposalFn;
in gepa those roles are played by ``use_merge`` / explicit crossover. Here we
expose the REAL EvoPrompt operators as standalone functions
(``evoprompt_crossover`` / ``evoprompt_de``) for the consolidation pass (E2/E8).
"""
from __future__ import annotations

import ast
from typing import Any, Callable, Dict, List, Optional


# ===========================================================================
# VERBATIM EvoPrompt templates (with file:line provenance — see module docstring)
# ===========================================================================

# vendor/evoprompt/data/template_ga.py  templates_2["cls"] lines 11-17 (the
# REUSABLE trailing block the GAEvoluter actually fills via <prompt1>/<prompt2>).
# Reproduced byte-for-byte (including the trailing "1. " priming the model to
# continue the numbered scaffold exactly as the real driver does).
GA_TEMPLATE = """Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts and generate a new prompt:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1. """

# vendor/evoprompt/data/template_de.py  v1["sim"] lines 35-45 (the REUSABLE
# trailing block the DEEvoluter fills via <prompt0..3>). Verbatim.
DE_TEMPLATE = """Please follow the instruction step-by-step to generate a better prompt.
1. Identify the different parts between the Prompt 1 and Prompt 2:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Randomly mutate the different parts
3. Combine the different parts with Prompt 3, selectively replace it with the different parts in step2 and generate a new prompt.
Prompt 3: <prompt3>
4. Crossover the prompt in the step3 with the following basic prompt and generate a final prompt bracketed with <prompt> and </prompt>:
Basic Prompt: <prompt0>

1. """

# vendor/evoprompt/llm_client.py  paraphrase() line 149. Verbatim format string
# (the ``{sentence}`` field becomes our <prompt1> slot so we can reuse the same
# substitution machinery; the literal text is unchanged).
MUTATION_TEMPLATE = (
    "Generate a variation of the following instruction while keeping the "
    "semantic meaning.\nInput:<prompt1>\nOutput:"
)


# ===========================================================================
# Real EvoPrompt extraction — vendor/evoprompt/utils.py get_final_prompt (84-93)
# ===========================================================================

def get_final_prompt(text: str) -> str:
    """Verbatim re-implementation of vendor/evoprompt/utils.py:get_final_prompt.

    Splits on ``<prompt>`` and takes the text before the first ``</prompt>``;
    otherwise strips a single pair of surrounding double-quotes.
    """
    parts = text.split("<prompt>")
    if len(parts) > 1:
        prompt = parts[-1].split("</prompt>")[0]
        prompt = prompt.strip()
        return prompt
    else:
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        return text


# ===========================================================================
# Verifier validation (PLM genome contract: exactly one top-level
# `def verifier(messages)` function)
# ===========================================================================

# Appended to the mutation/crossover request for the `verifier` component so the
# LLM emits a single self-contained def (matches the strict gate).
_VERIFIER_CONTRACT_NOTE = (
    "\n\nHARD REQUIREMENT — the `verifier` MUST be EXACTLY ONE top-level "
    "`def verifier(messages)` and NOTHING else: define every helper function, "
    "import, and constant INSIDE the function body (no top-level helpers / imports "
    "/ classes; not async). Reply with only that one function."
)


def _is_valid_verifier(source: str) -> bool:
    """True iff `source` is EXACTLY ONE top-level `def verifier(messages)` function
    and NOTHING else (no other top-level defs/classes/imports/assignments; not
    async) — every helper/import/constant lives INSIDE the body. The verifier runs
    IN THE POLICY SPACE (kernel) with the full ambient namespace — it may call
    react_llm/natural_llm/parallel, edit mutable policies (e.g. policyzero), create
    functions, and mutate the trajectory in place — but cannot abort the run or
    edit a sealed policy. Mirrors the gate in `PLMMetaParameters.__post_init__`."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    body = tree.body
    if not (len(body) == 1 and isinstance(body[0], ast.FunctionDef)
            and body[0].name == "verifier"):
        return False
    fn = body[0]
    # Must accept the single `messages` positional (allow extra-defaulted args).
    posargs = fn.args.posonlyargs + fn.args.args
    return len(posargs) >= 1


# ===========================================================================
# Core substitution helpers (mirror evoluter.py .replace() chains)
# ===========================================================================

def _ga_request(parent_a: str, parent_b: str) -> str:
    # evoluter.py:404-405
    return GA_TEMPLATE.replace("<prompt1>", parent_a).replace("<prompt2>", parent_b)


def _de_request(old_prompt: str, a: str, b: str, c: str) -> str:
    # evoluter.py:570-575
    return (
        DE_TEMPLATE.replace("<prompt0>", old_prompt)
        .replace("<prompt1>", a)
        .replace("<prompt2>", b)
        .replace("<prompt3>", c)
    )


def _mutation_request(prompt: str) -> str:
    # llm_client.py:149 (paraphrase)
    return MUTATION_TEMPLATE.replace("<prompt1>", prompt)


def _safe_lm(reflection_lm: Callable[[str], str], request: str) -> str:
    """Call the reflection LM. raise-for-all: a non-working LM (exception or
    non-str/None return) fails loudly (consistent with the other proposers);
    only a valid-but-unparseable response falls back to the parent downstream."""
    raw = reflection_lm(request)
    if not isinstance(raw, str):
        raise TypeError(f"reflection_lm returned {type(raw).__name__}, expected str")
    return raw


# ===========================================================================
# Standalone two-/multi-parent operators (consolidation pass — E2/E8)
# ===========================================================================

def evoprompt_crossover(
    parent_a: Dict[str, str],
    parent_b: Dict[str, str],
    *,
    reflection_lm: Callable[[str], str],
    component: str,
) -> str:
    """EvoPrompt GA crossover+mutation (REAL templates_2 template).

    Two-parent operator. In gepa this role is `use_merge` / explicit crossover;
    here we expose the real EvoPrompt operator directly.

    Returns the child text for `component`. For the `verifier` component the
    output MUST parse to exactly one top-level `def verifier(...)`; on failure we
    fall back to parent_a's component (the real driver has no such constraint —
    this is the PLM genome contract).
    """
    a = parent_a.get(component, "")
    b = parent_b.get(component, "")
    request = _ga_request(a, b)
    if component == "verifier":
        request += _VERIFIER_CONTRACT_NOTE
    raw = _safe_lm(reflection_lm, request)
    child = get_final_prompt(raw)
    if component == "verifier" and not _is_valid_verifier(child):
        return a  # fall back to parent on invalid verifier
    if not child.strip():
        return a
    return child


def evoprompt_de(
    parent_a: Dict[str, str],
    parent_b: Dict[str, str],
    *,
    reflection_lm: Callable[[str], str],
    component: str,
    current: Optional[Dict[str, str]] = None,
    donor: Optional[Dict[str, str]] = None,
) -> str:
    """EvoPrompt Differential Evolution (REAL v1 template).

    DE in EvoPrompt uses four slots: the current prompt (`<prompt0>` / Basic),
    two sampled parents (`<prompt1>`, `<prompt2>`), and a donor (`<prompt3>`,
    which the real driver sets to the current-best when `donor_random` is off —
    DEEvoluter.evolute, evoluter.py:567-569). We map:
        prompt0/Basic = `current` (defaults to parent_a),
        prompt1       = parent_a,
        prompt2       = parent_b,
        prompt3 donor = `donor`  (defaults to parent_b).

    Same verifier validation / parent fallback as `evoprompt_crossover`.
    """
    a = parent_a.get(component, "")
    b = parent_b.get(component, "")
    old = (current or parent_a).get(component, "")
    c = (donor or parent_b).get(component, "")
    request = _de_request(old, a, b, c)
    if component == "verifier":
        request += _VERIFIER_CONTRACT_NOTE
    raw = _safe_lm(reflection_lm, request)
    child = get_final_prompt(raw)
    if component == "verifier" and not _is_valid_verifier(child):
        return a
    if not child.strip():
        return a
    return child


# ===========================================================================
# EvoPromptProposer — gepa ProposalFn (single-candidate MUTATION operator)
# ===========================================================================

class EvoPromptProposer:
    """gepa-compatible custom candidate proposer wrapping EvoPrompt's MUTATION.

    gepa calls a ProposalFn as
        __call__(candidate, reflective_dataset, components_to_update) -> dict[str,str]
    (one candidate in, one candidate out). The only EvoPrompt operator that fits
    the single-candidate shape is MUTATION (paraphrase); GA crossover and DE are
    two-/multi-parent and are exposed separately (`evoprompt_crossover`,
    `evoprompt_de`).

    For `mode="ga"` we still run a single-parent MUTATION here (the GA template's
    crossover step needs two parents, so the ProposalFn-shaped op is mutation);
    `mode="de"` behaves the same single-parent way. The two-parent / DE templates
    are reachable via the module-level functions for the consolidation pass.

    For the `verifier` component, the proposed text MUST parse to exactly one
    top-level `def verifier(...)`; otherwise we keep the parent's verifier.
    """

    def __init__(self, reflection_lm: Callable[[str], str], mode: str = "ga",
                 base_template: Optional[str] = None) -> None:
        if mode not in ("ga", "de"):
            raise ValueError(f"mode must be 'ga' or 'de', got {mode!r}")
        self.reflection_lm = reflection_lm
        self.mode = mode
        # Optional PE2 framing. When set, the EvoPrompt mutation request is
        # substituted into PE2's diagnose->edit grammar (<curr_param>=parent,
        # <side_info>=the EvoPrompt mutation/paraphrase instruction) so the
        # composite is "PE2 grammar + EvoPrompt mutation" — both real.
        self.base_template = base_template

    # gepa ProposalFn entrypoint -------------------------------------------
    def __call__(
        self,
        candidate: Dict[str, str],
        reflective_dataset: Dict[str, List[Dict[str, Any]]],
        components_to_update: List[str],
    ) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for comp in components_to_update:
            out[comp] = self.propose(candidate, comp)
        return out

    # explicit single-component mutation (used by the proposer + self-test) --
    def propose(self, candidate: Dict[str, str], component: str) -> str:
        """Apply EvoPrompt MUTATION to one component; return the new text.

        Falls back to the parent component if the model returns an empty result
        or (for `verifier`) an invalid `def verifier`."""
        parent_text = candidate.get(component, "")
        request = _mutation_request(parent_text)
        if self.base_template:
            from .pe2_prompt import frame_with_pe2
            request = frame_with_pe2(
                self.base_template, curr_param=parent_text, side_info=request
            )
        if component == "verifier":
            request += _VERIFIER_CONTRACT_NOTE
        raw = _safe_lm(self.reflection_lm, request)
        child = get_final_prompt(raw)
        if component == "verifier" and not _is_valid_verifier(child):
            return parent_text
        if not child.strip():
            return parent_text
        return child


__all__ = [
    "GA_TEMPLATE",
    "DE_TEMPLATE",
    "MUTATION_TEMPLATE",
    "get_final_prompt",
    "evoprompt_crossover",
    "evoprompt_de",
    "EvoPromptProposer",
]
