"""bonsai — keep long AI conversations usable, and get back anything it collapsed.

`Session` is the one thing you need:

    from bonsai import Session

    s = Session.open("my-project")
    ctx = s.build_context(budget=50_000)

Everything else is internal and may change.
"""

from bonsai.session import Session

__all__ = ["Session"]
