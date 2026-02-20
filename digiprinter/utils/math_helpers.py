"""Utility math functions for the digiprinter simulation."""

from __future__ import annotations

import numpy as np


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp *value* to the interval [*min_val*, *max_val*]."""
    return float(np.clip(value, min_val, max_val))


def lerp(a: float, b: float, t: float) -> float:
    """Linearly interpolate between *a* and *b* by factor *t*."""
    return a + (b - a) * t


def inverse_lerp(a: float, b: float, value: float) -> float:
    """Return the interpolation parameter *t* such that ``lerp(a, b, t) == value``.

    Returns 0.0 when ``a == b`` to avoid division by zero.
    """
    if b == a:
        return 0.0
    return (value - a) / (b - a)


def fast_erfc(x: float) -> float:
    """Fast approximation of the complementary error function erfc(x).

    Uses the Horner-form polynomial approximation (Abramowitz & Stegun 7.1.26)
    with five coefficients.  Maximum absolute error ~ 1.5e-7.
    """
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    # erfc is symmetric: erfc(-x) = 2 - erfc(x)
    sign = 1.0
    if x < 0.0:
        sign = -1.0
        x = -x

    t = 1.0 / (1.0 + 0.3275911 * x)
    # Horner evaluation of the polynomial a1*t + a2*t^2 + ... + a5*t^5
    poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
    y = poly * np.exp(-(x * x))

    if sign < 0.0:
        return 2.0 - y
    return float(y)


def smooth_step(edge0: float, edge1: float, x: float) -> float:
    """Hermite interpolation (smooth-step) between *edge0* and *edge1*.

    Returns 0.0 for ``x <= edge0``, 1.0 for ``x >= edge1``, and a smooth
    cubic transition in between.
    """
    t = clamp((x - edge0) / (edge1 - edge0) if edge1 != edge0 else 0.0, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def normalize_angle(angle: float) -> float:
    """Normalize *angle* (radians) to the interval [-pi, pi]."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))
