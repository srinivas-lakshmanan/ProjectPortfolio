"""Time-parameterized circular reference with tangent heading."""

import math

import numpy as np
from numpy.typing import NDArray


def generate_reference(t: float, N: int = 20, dt: float = 0.1) -> NDArray[np.float64]:
    """Return [x, y, theta] columns with shape (3, N+1)."""
    if not math.isfinite(t):
        raise ValueError("Reference start time must be finite.")
    if isinstance(N, bool) or not isinstance(N, int) or N < 1:
        raise ValueError("Reference horizon must be a positive integer.")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("Reference sampling interval must be finite and positive.")

    ref = np.zeros((3, N + 1))
    for k in range(N + 1):
        tk = t + k * dt
        x = 2.0 * np.cos(0.2 * tk)
        y = 2.0 * np.sin(0.2 * tk)
        dx = -0.4 * np.sin(0.2 * tk)
        dy = 0.4 * np.cos(0.2 * tk)
        theta = np.arctan2(dy, dx)
        ref[:, k] = [x, y, theta]
    return ref
