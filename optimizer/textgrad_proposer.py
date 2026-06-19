"""TextGrad "Textual Gradient Descent" (TGD) as a real gepa ProposalFn.

This implements TextGrad's two-step TGD loop -- (1) a *backward* pass that
computes a *textual gradient* (critique/feedback) for a variable given a
downstream loss, and (2) a *step/update* pass that rewrites the variable using
that gradient -- exactly as the real TextGrad library does.

The prompt templates below are the REAL TextGrad templates, INLINED VERBATIM from
the upstream source (zou-group/textgrad) — self-contained, no vendored files:

    textgrad/autograd/llm_backward_prompts.py   (backward/gradient)
    textgrad/optimizer/optimizer_prompts.py     (step/update)

Provenance of the assembly we replicate (file:line ranges in the vendored tree):

  Backward / gradient pass (textgrad `LLMCall._backward_through_llm_base`):
    - autograd/llm_ops.py:166-171  `_construct_llm_base_backward_prompt`:
          conversation = CONVERSATION_TEMPLATE.format(**backward_info)
          backward_prompt  = CONVERSATION_START_INSTRUCTION_BASE.format(conversation=conversation, **backward_info)
          backward_prompt += OBJECTIVE_INSTRUCTION_BASE.format(**backward_info)
          backward_prompt += EVALUATE_VARIABLE_INSTRUCTION.format(**backward_info)
    - autograd/llm_ops.py:210  gradient_value = backward_engine(backward_prompt, system_prompt=BACKWARD_SYSTEM_PROMPT)
    - templates: autograd/llm_backward_prompts.py
          BACKWARD_SYSTEM_PROMPT                  (lines 13-21)
          CONVERSATION_TEMPLATE                   (lines 24-28)
          CONVERSATION_START_INSTRUCTION_BASE     (lines 42-46)
          OBJECTIVE_INSTRUCTION_BASE              (lines 48-51)
          EVALUATE_VARIABLE_INSTRUCTION           (lines 55-62)

  Step / update pass (textgrad `TextualGradientDescent._update_prompt` + `.step`):
    - optimizer/optimizer.py:146-167  `_update_prompt` -> construct_tgd_prompt(...)
    - optimizer/optimizer.py:116/172  system prompt = OPTIMIZER_SYSTEM_PROMPT.format(new_variable_*_tag=...)
    - optimizer/optimizer.py:181      new_value = new_text.split(START)[1].split(END)[0].strip()
    - optimizer/optimizer.py:111      new_variable_tags = ["<IMPROVED_VARIABLE>", "</IMPROVED_VARIABLE>"]
    - construct_tgd_prompt: optimizer/optimizer_prompts.py:68-111
          (str gradient -> TGD_PROMPT_PREFIX (26-32) + TGD_PROMPT_SUFFIX (45-49))
    - templates: optimizer/optimizer_prompts.py
          GLOSSARY_TEXT             (lines 1-9)
          OPTIMIZER_SYSTEM_PROMPT   (lines 14-23)
          TGD_PROMPT_PREFIX         (lines 26-32)
          TGD_PROMPT_SUFFIX         (lines 45-49)
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from typing import Any, Callable

# --------------------------------------------------------------------------- #
# The REAL textgrad template strings, inlined VERBATIM from upstream
# (zou-group/textgrad) — self-contained, no vendored files. Upstream file:line
# provenance noted on each block below.
# --------------------------------------------------------------------------- #

# Verbatim from textgrad/autograd/llm_backward_prompts.py (zou-group/textgrad):13-62
BACKWARD_SYSTEM_PROMPT = (
    "You are part of an optimization system that improves a given text (i.e. the variable). You are the gradient (feedback) engine. "
    "Your only responsibility is to give intelligent and creative feedback and constructive criticism to variables, given an objective specified in <OBJECTIVE_FUNCTION> </OBJECTIVE_FUNCTION> tags. "
    "The variables may be solutions to problems, prompts to language models, code, or any other text-based variable. "
    "Pay attention to the role description of the variable, and the context in which it is used. You should assume that the variable will be used in a similar context in the future. "
    "Only provide strategies, explanations, and methods to change in the variable. DO NOT propose a new version of the variable, that will be the job of the optimizer. Your only job is to send feedback and criticism (compute 'gradients'). "
    "For instance, feedback can be in the form of 'Since language models have the X failure mode...', 'Adding X can fix this error because...', 'Removing X can improve the objective function because...', 'Changing X to Y would fix the mistake ...', that gets at the downstream objective.\n"
    "If a variable is already working well (e.g. the objective function is perfect, an evaluation shows the response is accurate), you should not give feedback.\n"
)
CONVERSATION_TEMPLATE = (
    "<LM_SYSTEM_PROMPT> {system_prompt} </LM_SYSTEM_PROMPT>\n\n"
    "<LM_INPUT> {prompt} </LM_INPUT>\n\n"
    "<LM_OUTPUT> {response_value} </LM_OUTPUT>\n\n"
)
CONVERSATION_START_INSTRUCTION_BASE = (
    "You will give feedback to a variable with the following role: <ROLE> {variable_desc} </ROLE>. "
    "Here is an evaluation of the variable using a language model:\n\n"
    "{conversation}"
)
OBJECTIVE_INSTRUCTION_BASE = (
    "<OBJECTIVE_FUNCTION>Your goal is to give feedback and criticism to the variable given the above evaluation output. "
    "Our only goal is to improve the above metric, and nothing else. </OBJECTIVE_FUNCTION>\n\n"
)
EVALUATE_VARIABLE_INSTRUCTION = (
    "We are interested in giving feedback to the {variable_desc} "
    "for this conversation. Specifically, give feedback to the following span "
    "of text:\n\n<VARIABLE> "
    "{variable_short} </VARIABLE>\n\n"
    "Given the above history, describe how the {variable_desc} "
    "could be improved to improve the <OBJECTIVE_FUNCTION>. Be very creative, critical, and intelligent.\n\n"
)

# Verbatim from textgrad/optimizer/optimizer_prompts.py (zou-group/textgrad):14-49
_GLOSSARY_TEXT = """
### Glossary of tags that will be sent to you:
# - <LM_SYSTEM_PROMPT>: The system prompt for the language model.
# - <LM_INPUT>: The input to the language model.
# - <LM_OUTPUT>: The output of the language model.
# - <FEEDBACK>: The feedback to the variable.
# - <CONVERSATION>: The conversation history.
# - <FOCUS>: The focus of the optimization.
# - <ROLE>: The role description of the variable."""
OPTIMIZER_SYSTEM_PROMPT = (
    "You are part of an optimization system that improves text (i.e., variable). "
    "You will be asked to creatively and critically improve prompts, solutions to problems, code, or any other text-based variable. "
    "You will receive some feedback, and use the feedback to improve the variable. "
    "The feedback may be noisy, identify what is important and what is correct. "
    "Pay attention to the role description of the variable, and the context in which it is used. "
    "This is very important: You MUST give your response by sending the improved variable between {new_variable_start_tag} {{improved variable}} {new_variable_end_tag} tags. "
    "The text you send between the tags will directly replace the variable.\n\n"
    f"{_GLOSSARY_TEXT}"
)
TGD_PROMPT_PREFIX = (
    "Here is the role of the variable you will improve: <ROLE>{variable_desc}</ROLE>.\n\n"
    "The variable is the text within the following span: <VARIABLE> {variable_short} </VARIABLE>\n\n"
    "Here is the context and feedback we got for the variable:\n\n"
    "<CONTEXT>{variable_grad}</CONTEXT>\n\n"
    "Improve the variable ({variable_desc}) using the feedback provided in <FEEDBACK> tags.\n"
)
TGD_PROMPT_SUFFIX = (
    "Send the improved variable "
    "in the following format:\n\n{new_variable_start_tag}{{the improved variable}}{new_variable_end_tag}\n\n"
    "Send ONLY the improved variable between the <IMPROVED_VARIABLE> tags, and nothing else."
)


# textgrad's default new-variable tags (optimizer/optimizer.py:111)
_NEW_VAR_START = "<IMPROVED_VARIABLE>"
_NEW_VAR_END = "</IMPROVED_VARIABLE>"


# Per-component "role description" of the variable (textgrad Variable.role).
_ROLE_DESCRIPTIONS = {
    "system_prompt": "system prompt that guides a policy agent's behavior",
    "verifier": "python source that is EXACTLY ONE top-level `def verifier(messages)` and NOTHING else — define every helper function, import, and constant INSIDE the function body (no top-level helpers/imports/classes; not async). It runs INSIDE the policy-space kernel after each non-terminal round with the full ambient namespace (react_llm / natural_llm / parallel / policy edit APIs / policyzero); it may mutate the trajectory in place (inject user/system turns, edit, prune) and do anything a cell can, but cannot abort the run or edit a sealed policy",
}
# Per-component downstream "response description" (textgrad response_desc).
_RESPONSE_DESCRIPTIONS = {
    "system_prompt": "the input that steers the policy agent's rollout, which is then scored",
    "verifier": "the kernel-resident trajectory controller that runs each round in the policy space (full ambient namespace) and steers the agent's rollout, which is then scored",
}


def _stringify_reflective_record(record: Mapping[str, Any]) -> tuple[str, str]:
    """Project one reflective-dataset record into textgrad's two slots:
    a `conversation` body (the LM input/output context) and the downstream
    `feedback` (the textual loss). Returns (input_blob, feedback)."""
    parts = []
    inp = record.get("input")
    if inp is not None:
        parts.append(str(inp))
    traj = record.get("trajectory_excerpt")
    if traj:
        parts.append(str(traj))
    outcome = record.get("outcome")
    if outcome is not None:
        parts.append("Outcome: " + str(outcome))
    input_blob = "\n\n".join(parts) if parts else "(no input recorded)"
    feedback = str(record.get("feedback", "")) or "(no feedback recorded)"
    return input_blob, feedback


def _aggregate_reflective(reflective_records: Sequence[Mapping[str, Any]]):
    """Aggregate the reflective dataset for one component into the fields
    textgrad's backward pass needs: a combined conversation blob, the current
    component value, and the combined downstream feedback (the 'loss')."""
    inputs, feedbacks, current = [], [], None
    for rec in reflective_records:
        i, f = _stringify_reflective_record(rec)
        inputs.append(i)
        feedbacks.append(f)
        if current is None and rec.get("current_component") is not None:
            current = str(rec.get("current_component"))
    combined_input = "\n\n---\n\n".join(inputs) if inputs else "(no inputs)"
    combined_feedback = "\n".join(f"- {f}" for f in feedbacks) if feedbacks else "(no feedback)"
    return combined_input, combined_feedback, current


class TextGradProposer:
    """gepa ProposalFn implementing TextGrad's Textual Gradient Descent.

    For each requested component it runs textgrad's real two-step loop:
      1. backward  -> textual gradient (critique) via BACKWARD_SYSTEM_PROMPT
      2. step      -> rewritten component via OPTIMIZER_SYSTEM_PROMPT + TGD prompt

    The `verifier` component is validated with `ast`: the proposed text must
    parse and contain exactly one top-level `def verifier(...)`; otherwise the
    parent value is returned unchanged.
    """

    def __init__(self, reflection_lm: Callable[[str], str], base_template: str | None = None):
        self.reflection_lm = reflection_lm
        # Optional PE2 framing. When set, the TGD update prompt is substituted
        # into PE2's diagnose->edit grammar (<curr_param>=current value,
        # <side_info>=the textual-gradient-driven update instruction) so the
        # composite is "PE2 grammar + TextGrad TGD" — both real.
        self.base_template = base_template

    # ---- step 1: backward / gradient ------------------------------------- #
    def _build_backward_prompt(
        self, component: str, current_value: str, conversation_blob: str, downstream_feedback: str
    ) -> str:
        variable_desc = _ROLE_DESCRIPTIONS.get(component, f"the {component} variable")
        # textgrad's _construct_llm_base_backward_prompt (llm_ops.py:166-171)
        conversation = CONVERSATION_TEMPLATE.format(
            system_prompt="(component under optimization is the variable below)",
            prompt=conversation_blob,
            response_value=current_value,
        )
        prompt = CONVERSATION_START_INSTRUCTION_BASE.format(
            variable_desc=variable_desc, conversation=conversation
        )
        prompt += OBJECTIVE_INSTRUCTION_BASE.format()
        prompt += EVALUATE_VARIABLE_INSTRUCTION.format(
            variable_desc=variable_desc, variable_short=current_value
        )
        # The downstream loss textgrad would carry on the output. The base
        # backward path's OBJECTIVE_INSTRUCTION_BASE has no slot for it, so we
        # append it as the evaluation that drives the gradient (matches the
        # role <FEEDBACK> plays in the chain path / GRADIENT_TEMPLATE).
        prompt += (
            "Here is the downstream feedback (the loss) for this variable:\n\n"
            f"<FEEDBACK>{downstream_feedback}</FEEDBACK>\n\n"
        )
        return prompt

    def _compute_gradient(
        self, component: str, current_value: str, conversation_blob: str, downstream_feedback: str
    ) -> str:
        prompt = self._build_backward_prompt(
            component, current_value, conversation_blob, downstream_feedback
        )
        # textgrad: backward_engine(backward_prompt, system_prompt=BACKWARD_SYSTEM_PROMPT)
        # raise-for-all: a non-working reflection LM (exception or non-str/None
        # return) fails loudly rather than silently yielding an empty gradient.
        raw = self.reflection_lm(f"{BACKWARD_SYSTEM_PROMPT}\n\n{prompt}")
        if not isinstance(raw, str):
            raise TypeError(f"reflection_lm returned {type(raw).__name__}, expected str")
        return raw

    # ---- step 2: step / update ------------------------------------------- #
    def _build_update_prompt(self, component: str, current_value: str, gradient: str) -> tuple[str, str]:
        variable_desc = _ROLE_DESCRIPTIONS.get(component, f"the {component} variable")
        # textgrad construct_tgd_prompt (str gradient branch): PREFIX + SUFFIX
        prompt = TGD_PROMPT_PREFIX.format(
            variable_desc=variable_desc,
            variable_short=current_value,
            variable_grad=gradient,
        )
        prompt += TGD_PROMPT_SUFFIX.format(
            new_variable_start_tag=_NEW_VAR_START,
            new_variable_end_tag=_NEW_VAR_END,
        )
        # textgrad: optimizer_system_prompt.format(new_variable_*_tag=...)
        system = OPTIMIZER_SYSTEM_PROMPT.format(
            new_variable_start_tag=_NEW_VAR_START,
            new_variable_end_tag=_NEW_VAR_END,
        )
        return system, prompt

    def _run_update(self, component: str, current_value: str, gradient: str) -> str | None:
        system, prompt = self._build_update_prompt(component, current_value, gradient)
        full = f"{system}\n\n{prompt}"
        if self.base_template:
            from .pe2_prompt import frame_with_pe2
            full = frame_with_pe2(
                self.base_template, curr_param=current_value, side_info=full
            )
        # raise-for-all: a non-working reflection LM fails loudly; only a valid-
        # but-unparseable response (no <IMPROVED_VARIABLE> tags) falls back to parent.
        raw = self.reflection_lm(full)
        if not isinstance(raw, str):
            raise TypeError(f"reflection_lm returned {type(raw).__name__}, expected str")
        # textgrad parse: split on <IMPROVED_VARIABLE> ... </IMPROVED_VARIABLE>
        try:
            return raw.split(_NEW_VAR_START, 1)[1].split(_NEW_VAR_END, 1)[0].strip()
        except IndexError:
            return None

    # ---- verifier validation --------------------------------------------- #
    @staticmethod
    def _valid_verifier(source: str) -> bool:
        """Accept EXACTLY one top-level ``def verifier(messages)`` function and
        NOTHING else (no other top-level defs/classes/imports/assignments; not
        async) — helpers/imports/constants live INSIDE the body. The verifier runs
        in the kernel policy space with the full ambient namespace and mutates the
        trajectory in place."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return False
        body = tree.body
        return len(body) == 1 and isinstance(body[0], ast.FunctionDef) and body[0].name == "verifier"

    # ---- gepa ProposalFn entrypoints ------------------------------------- #
    def propose(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        out: dict[str, str] = {}
        for component in components_to_update:
            parent_value = candidate.get(component, "")
            records = reflective_dataset.get(component, [])
            conversation_blob, downstream_feedback, current_from_records = _aggregate_reflective(records)
            current_value = current_from_records if current_from_records is not None else parent_value

            gradient = self._compute_gradient(
                component, current_value, conversation_blob, downstream_feedback
            )
            new_value = self._run_update(component, current_value, gradient)

            if new_value is None or new_value == "" or new_value == parent_value:
                out[component] = parent_value
                continue

            if component == "verifier" and not self._valid_verifier(new_value):
                # On failure return parent unchanged.
                out[component] = parent_value
                continue

            out[component] = new_value
        return out

    # gepa ProposalFn protocol: callable with (candidate, reflective_dataset, components)
    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        return self.propose(candidate, reflective_dataset, components_to_update)
