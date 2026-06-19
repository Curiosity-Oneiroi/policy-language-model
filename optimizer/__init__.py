"""plm.optimizer — drive the REAL `gepa` library over the (system_prompt,
verifier) genome of a PLM chess metaparam dir.

A candidate is the two-component dict gepa evolves: {"system_prompt", "verifier"}
(the frozen chess `policies/` ride along via `Candidate.materialize`). The seed
CODE is FROZEN. Ten named `RECIPES` ('E1'..'E10') map to `gepa.optimize`
configurations — gepa kwargs (`frontier_type`/`use_merge`/template) plus which
REAL prompt-optimization method (PE2, PromptBreeder, TextGrad, ContraPrompt,
EvoPrompt) acts as the custom proposer. `run_experiment` drives the loop on the
real Slate backend with a real reflection LM (`BackendReflectLM`).
"""
# Genome + evaluation + scoring.
from .genome import Candidate, seed_candidates
from .scorer import score_run, SCORE_WEIGHTS
from .evaluate import EvalConfig, evaluate

# The REAL gepa engine + reflection LM.
from .gepa_engine import (
    run_gepa,
    PLMChessAdapter,
    BackendReflectLM,
    default_trainset,
    OPPONENT_RUNGS,
)

# The PE2 meta-prompt + its custom-proposer framing helper.
from .pe2_prompt import PE2_REFLECTION_TEMPLATE, frame_with_pe2

# The four REAL prompt-optimization proposers (gepa ProposalFns) + EvoPrompt's
# two-parent operators.
from .promptbreeder_proposer import PromptBreederProposer
from .textgrad_proposer import TextGradProposer
from .contraprompt_proposer import ContraPromptProposer
from .evoprompt_ops import EvoPromptProposer, evoprompt_crossover, evoprompt_de

# Island engine (E10) + recipes + driver.
from .island_engine import run_island
from .recipes import RECIPES, build_recipe
from .experiment import run_experiment

__all__ = [
    # genome / eval / score
    "Candidate", "seed_candidates", "EvalConfig", "evaluate",
    "score_run", "SCORE_WEIGHTS",
    # gepa engine + reflection LM
    "run_gepa", "PLMChessAdapter", "BackendReflectLM",
    "default_trainset", "OPPONENT_RUNGS",
    # PE2
    "PE2_REFLECTION_TEMPLATE", "frame_with_pe2",
    # proposers / operators
    "PromptBreederProposer", "TextGradProposer", "ContraPromptProposer",
    "EvoPromptProposer", "evoprompt_crossover", "evoprompt_de",
    # island / recipes / driver
    "run_island", "RECIPES", "build_recipe", "run_experiment",
]
