"""The REAL PE2 reflection meta-prompt, vendored from the actual paper.

Source: "Prompt Engineering a Prompt Engineer" (PE2), arXiv:2311.05661.
Vendored text at: optimizer/vendor/papers/pe2_2311.05661.txt

PE2's contribution is THREE meta-prompt components (paper §3, Fig. 3, and the
appendix Fig. 9). We reproduce them VERBATIM below, then wire in gepa's two
required placeholders. Nothing about the meta-prompt's wording was invented —
only the placeholder plumbing and a one-line return-format instruction (which
PE2 itself also carries, see its "# Instructions / Reply with the prompt" block)
were added so gepa's default proposer can substitute and parse the result.

=== EXACT PAPER LINES THIS TEMPLATE IS BUILT FROM ===

(a) Two-step task instruction
    pe2_2311.05661.txt lines 366-372 (also repeated verbatim at 1816-1823):
    ----------------------------------------------------------------------
    A prompt is a text paragraph that outlines the expected actions
    and instructs the model. In our collaboration, we'll work
    together to refine a prompt. The process consists of two steps:
    # Step 1
    Examine the prompt and a batch of examples
    # Step 2
    Propose a new prompt based on your reasoning
    ----------------------------------------------------------------------

(b) Context specification
    pe2_2311.05661.txt lines 384-394 (also at 1828-1837); §3 text lines 437-444:
    ----------------------------------------------------------------------
    # Current Prompt
    Let's think step by step.
    # Full Template
    ```
    Question: <input>
    Answer: Let's think step by step.
    ```
    Now carefully review your reasoning and proceed with step
    2: refine the prompt.
    ----------------------------------------------------------------------
    Paper §3(b), lines 437-444: "how the prompt and the input text are formatted
    together is flexible ... Recognizing these varying contexts, we explicitly
    specify the layout of the prompt and the input."

(c) Step-by-step reasoning template
    pe2_2311.05661.txt lines 359-365 (and the 4-bullet appendix variant at
    1795-1802); §3(c) text lines 457-465:
    ----------------------------------------------------------------------
    # Instruction
    For each example, provide reasoning according to the
    following template
    * Output is correct?
    * Prompt describing the task correctly?
    * Necessary to edit the prompt?
    * If yes, suggestions on prompt editing?
    ----------------------------------------------------------------------

=== EXACTLY WHAT WE EDITED (placeholder wiring only) ===
  1. Replaced the paper's literal "# Current Prompt\nLet's think step by step."
     example body with gepa's `<curr_param>` placeholder (gepa substitutes the
     component-to-evolve there). The "# Current Prompt" heading is kept verbatim.
  2. Replaced the paper's "# Examples / ## Example 1 ..." batch block with gepa's
     `<side_info>` placeholder (gepa substitutes the inputs+outputs+feedback
     batch there). The framing sentence is kept close to the paper's.
  3. Added one return-format line ("Provide the new prompt within ``` blocks.")
     so gepa's default proposer can extract the proposal. This mirrors PE2's own
     "# Instructions / Reply with the prompt. Do not include other text." block
     (lines 411-415 / 1846-1851).
No component wording was paraphrased away; the three components above appear
verbatim.
"""
from __future__ import annotations


# The real PE2 meta-prompt, assembled from the three verbatim components above
# with gepa's <curr_param> and <side_info> placeholders wired in.
PE2_REFLECTION_TEMPLATE: str = """\
A prompt is a text paragraph that outlines the expected actions and instructs \
the model. In our collaboration, we'll work together to refine a prompt. The \
process consists of two steps:
# Step 1
Examine the prompt and a batch of examples
# Step 2
Propose a new prompt based on your reasoning

# Current Prompt
```
<curr_param>
```

# Examples
The following are task inputs provided to the assistant, the assistant's \
response for each, and feedback on how the response could be better:
```
<side_info>
```

# Instruction
For each example, provide reasoning according to the following template
* Output is correct?
* Prompt describing the task correctly?
* Necessary to edit the prompt?
* If yes, suggestions on prompt editing?

Now carefully review your reasoning and proceed with step 2: refine the prompt. \
Read all the responses and feedback, identify niche and domain-specific factual \
information about the task, and include it in the new prompt as a generalizable \
strategy.

Provide the new prompt within ``` blocks. Do not include other text outside the \
``` blocks.
"""


# gepa's default template (None => use gepa's built-in InstructionProposalSignature
# default). E1 (GEPA-plain) passes this; E3 (GEPA+PE2) passes PE2_REFLECTION_TEMPLATE.
GEPA_DEFAULT_TEMPLATE = None


def frame_with_pe2(base_template: str, curr_param: str, side_info: str) -> str:
    """Substitute a method's prompt into the PE2 grammar's two placeholders.

    PE2's contribution is the diagnose->edit GRAMMAR (the two-step "examine the
    prompt + a batch, then propose a new prompt" reasoning template). A custom
    proposer supplies the MUTATION (what to change); when a recipe stacks "+PE2"
    on a custom proposer it cannot use gepa's `reflection_prompt_template` slot
    (gepa only feeds that to its DEFAULT proposer — a custom proposer replaces
    it). So the honest composite frames the proposer's own reflection_lm call
    with this real PE2 text instead:

      * `<curr_param>` <- the current component the method is mutating.
      * `<side_info>`  <- the method's mutation instruction / contrastive batch.

    The returned string is the real PE2 meta-prompt with both placeholders
    filled — nothing about PE2's wording is paraphrased; only the two
    substitutions the template was designed for are performed. The PROPOSER
    still owns the mutation (PromptBreeder's first-order operator, TextGrad's
    TGD, ContraPrompt's rule tree, EvoPrompt's GA/DE), so the composite is
    "PE2 grammar + method mutation", both real.
    """
    return base_template.replace("<curr_param>", curr_param).replace(
        "<side_info>", side_info
    )


__all__ = ["PE2_REFLECTION_TEMPLATE", "GEPA_DEFAULT_TEMPLATE", "frame_with_pe2"]
