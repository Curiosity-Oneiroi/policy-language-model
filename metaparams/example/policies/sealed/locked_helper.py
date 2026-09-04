@policy
def locked_helper(x):
    """Example SEALED policy (immutable + un-duplicable, NOT blessed).

    A metaparam ships sealed policies in `policies/sealed/`. They are installed +
    sealed at kernel boot: PLM can CALL them but cannot rewrite, remove, OR duplicate
    them — they are locked, exactly like the LLM-loop defaults. Like any policy it
    reaches a sub-LLM only via `llm`/`react_auto` — sealing does not grant raw
    access (it is NOT blessed).
    """
    return x * 2
