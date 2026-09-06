"""Shared argument-validation helpers (integer acceptance rules).

Every public entry point that takes an integer accepts any
:class:`numbers.Integral` (Python ``int``, ``numpy.int64``, ...) but never a
``bool``; the value is normalised to a plain ``int`` before use.
"""

from __future__ import annotations

import numbers
from typing import Any


def is_integer(value: Any) -> bool:
    """True for integral values (``int``, numpy integer scalars), never ``bool``."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def as_int(name: str, value: Any) -> int:
    """Return ``value`` as a plain ``int``; ``ValueError`` if it is not integral."""
    if not is_integer(value):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(value)


def as_positive_int(name: str, value: Any) -> int:
    """Return ``value`` as a plain ``int``; ``ValueError`` unless integral and > 0."""
    if not is_integer(value) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)
