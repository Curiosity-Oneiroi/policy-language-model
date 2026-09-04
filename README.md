# Policy Language Model (PLM)

A meta-reasoning harness: instead of reasoning inline on one growing trajectory, the model authors its computation as **policies** in a persistent Python REPL and reasons onward from their results. The theory, the definitions, and the honest experimental record are in the write-up: **[Policy Language Models](https://ddi-droid.github.io/blog/2026/policy-language-models/)**.

This is research code for a paper in progress. The API is stable at the contracts described below; everything else may move.

## What is in here

```
plm.py               the root loop: PLM class, PLMMetaParameters, RETURN protocol
policy/              the @policy layer: editable, hot-swappable function/class policies
policy/defaults/     the three LLM primitives: llm, react_auto, react
constraint/          typed constraints, compiled into the decoding schema of a call
repl/                the persistent kernel: sessions, namespaces, weaving, parallel()
model_backend/       OpenAI, Anthropic, vLLM, Slate, Gemini backends (self-contained)
metaparams/          system prompt + example policies handed to the model
tests/               pytest suite
```

## Setup

Python ≥ 3.12. **Clone into a directory named `plm`** — the REPL kernel imports `plm` as live source from the checkout's parent, so the directory name is load-bearing:

```bash
git clone https://github.com/Curiosity-Oneiroi/policy-language-model.git plm
cd plm
pip install -e .        # or: uv pip install -e .
export OPENAI_API_KEY=...   # or ANTHROPIC_API_KEY, per backend
```

## Quickstart

```python
import asyncio
from plm.plm import PLM, PLMMetaParameters
from plm.model_backend import OpenAIBackend

plm = PLM(
    model_backend=OpenAIBackend(model="gpt-5-mini"),
    metaparams=PLMMetaParameters(
        system_prompt=open("metaparams/example/system_prompt.md").read(),
    ),
    max_turns=50,
)

result = asyncio.run(plm([{"role": "user", "content": "your task"}]))
print(result["answer"])     # exactly the value the model passed to RETURN(...)
```

Each round the model thinks, then runs one cell of Python in the persistent REPL; `print(...)` is what surfaces, and `RETURN(obj)` ends the task with `obj` as the answer. The trajectory holds three things: the reasoning, the code, and the code's output.

## The three LLM primitives

Inside the REPL the model builds neuro-symbolic policies out of three primitives. A sub-LLM call begins empty: its prompt, its inputs, and every capability it will use are handed to it explicitly; nothing is ambient.

```python
llm(messages, *, constraint=None, return_budget=5, generate_kwargs=None, model=None)
```
One generate. With a `constraint`, the constraint's JSON schema is sent as `response_format` and the answer is validated with retry.

```python
react_auto(messages, namespace=None, *, max_turns, constraint=None,
           return_budget=5, generate_kwargs=None, model=None)
```
A conventional sub-agent: model + python tool, looped, terminating only on `RETURN(value)`. `max_turns` is required — deciding how many rounds the work needs is part of briefing it. Granting a capability = putting a key in `namespace`.

```python
react(messages, namespace, *, rounds, generate_kwargs=None, model=None)
```
The raw capability with the termination protocol removed: exactly `rounds` think+code rounds, then stop, still open. Returns `None`, because everything it produced is already in the two containers the caller owns.

## Weaving

The two products of a computation are ordinary objects. **Messages pass by reference**: a `react` run over a list leaves its turns on that list, and the list can be edited, pruned, forked, or handed to another policy. **State passes by ownership**: a sub-agent's namespace is a dict the caller holds, so everything the sub-computation defined stays readable and callable after the call. There is no session object; a session is the pair of containers the author is holding.

## Policies

The `@policy` decorator turns a function or a class into a named artifact that persists across rounds, lives in global scope, and is identified by its source. The model edits policies by source: `read_policy`, `rewrite_policy`, `edit_policy`, `insert_into`, `delete_lines`, `duplicate_policy`, `delete_policy`. Sealed policies (`sealed_policies` in `PLMMetaParameters`) are immutable and cannot be duplicated.

## Tests

Run from the checkout's **parent** directory (running from the repo root shadows the `plm` package with `plm.py`):

```bash
python -m pytest --import-mode=importlib plm/tests -q
```

Tests that need a live model read the usual API-key environment variables and are skipped without them.

## Background

This grew out of engaging with [Recursive Language Models](https://arxiv.org/abs/2512.24601) (Zhang, Kraska, Khattab). The blog post covers the relationship, the formal definition of meta-reasoning as trajectory sufficiency, and the experimental program, including its negative results.
