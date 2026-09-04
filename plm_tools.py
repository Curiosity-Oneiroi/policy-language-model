from __future__ import annotations

from typing import Any


def _tool_python() -> dict[str, Any]:
    # Responses-API flat shape: name/description/parameters live at the top
    # level, not nested under a "function" key. We standardise on this form
    # because the openai backend uses the Responses API; _sanitize_and_cache_tools
    # passes non-Chat-Completions-shaped tools through unchanged, so this dict
    # reaches the API as-is.
    #
    # Used by BOTH PLM's top-level loop AND react_auto's inner loop, so the
    # description is intentionally GENERIC — just the python-REPL mechanics plus
    # the universal `RETURN(value)` finalize primitive.
    return {
        "type": "function",
        "name": "python",
        "description": (
            "Execute Python in a persistent REPL — state (variables, imports, "
            "definitions) persists across calls. Use `print(...)` to surface "
            "observations (cells run in exec mode, so a bare trailing expression "
            "is NOT printed). To finalize, call `RETURN(value)` from inside the "
            "code: this ends the agent loop and returns `value`."
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
