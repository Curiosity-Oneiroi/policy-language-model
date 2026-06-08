@policy
def editable_helper(x):
    """Example NORMAL policy (mutable + duplicable).

    A metaparam ships normal policies in `policies/mutable/`. They are installed at
    kernel boot as ordinary policies — PLM may rewrite/extend/remove them like any
    policy it authors itself.
    """
    return x + 1
