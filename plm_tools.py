from __future__ import annotations

from typing import Any


def _tool_python() -> dict[str, Any]:
    # Responses-API flat shape: name/description/parameters live at the top
    # level, not nested under a "function" key. We standardise on this form
    # because the openai backend uses the Responses API; _sanitize_and_cache_tools
    # passes non-Chat-Completions-shaped tools through unchanged, so this dict
    # reaches the API as-is.
    #
    # Used by BOTH PLM's top-level loop AND react_llm's inner loop. The
    # description below is intentionally framed for either context: call
    # `RETURN(value)` to finalize, author `@policy def helper(): ...` to
    # create reusable named policies that persist across cells.
    return {
        "type": "function",
        "name": "python",
        "description": (
            "Execute Python in a persistent REPL. Use print statements or "
            "any variable to output observations. To FINALIZE the task, call "
            "`RETURN(value)` from inside the python code — this terminates "
            "the agent loop and returns `value`. You can also author new "
            "policies with `@policy def helper(...): ...` (registered "
            "globally, persisting across cells). When called from inside "
            "`react_llm`, your code runs with restricted globals: only "
            "`__builtins__`, `RETURN`, and `policy` are ambient — anything "
            "else needed must be passed via the call's `args` / `kwargs` / "
            "`objects` channels (preseeded as local names)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source for one cell (statements + optional trailing expr).",
                },
            },
            "required": ["code"],
        },
    }
