"""Deterministic reporting cadence for iterative training loops.

Long training loops should not emit one line per epoch in the default view, so
:func:`progress_indices` selects a small, evenly spaced set of iterations to
report. The set is computed once, before the loop starts, from the iteration
count alone: it consumes no randomness, touches no model or optimiser state and
only decides whether an already-computed value is emitted.

The verbosity tier is read from the ``repro_verbosity`` attribute that the
repository logging policy publishes on the root logger. That keeps this module
dependent on nothing but the standard library -- it never imports the shared
logging package and never modifies ``sys.path``.
"""

from __future__ import annotations

import logging

__all__ = ["progress_indices", "reporting_verbosity"]


def reporting_verbosity() -> int:
    """Return the active verbosity tier (0 normal, 1 verbose, 2 debug).

    Defaults to 0 when logging has not been configured by an entry point.
    """
    return int(getattr(logging.getLogger(), "repro_verbosity", 0) or 0)


def progress_indices(total: int, *, updates: int = 10) -> set[int]:
    """Return the 1-based iteration indices at which to report progress.

    The result always includes the first iteration (1) and the last (*total*),
    plus roughly *updates* evenly spaced points in between. Under ``--verbose``
    or ``--debug`` every iteration is reported.
    """
    if total <= 0:
        return set()
    if reporting_verbosity() >= 1 or total <= updates + 1:
        return set(range(1, total + 1))

    marks = {1, total}
    step = max(1, total // updates)
    marks.update(range(step, total, step))
    return marks
