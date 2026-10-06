"""Bicubic-convolution interpolation kernel (sncosmo / snfit convention)."""

import numpy as np

__all__ = ["kernel_weights"]


def _keys(x, a=-0.5, xp=np):
    """Keys cubic-convolution kernel.

    Parameters
    ----------
    x : array_like
        Distance to the node, in units of the node spacing.

    a : float, optional
        Kernel parameter. The default is -0.5 (as sncosmo).

    xp : module, optional
        Array module (``numpy`` or ``jax.numpy``). The default is numpy.

    Returns
    -------
    array_like
        Kernel values.
    """
    x = xp.abs(x)
    return xp.where(x <= 1, (a + 2) * x**3 - (a + 3) * x**2 + 1,
                    xp.where(x < 2, a * (x**3 - 5 * x**2 + 8 * x - 4), 0.))


def kernel_weights(u, n, xp=np):
    """Get the nodes and weights of sncosmo's 1-D interpolation kernel.

    sncosmo's ``BicubicInterpolator`` (``Grid2DFunction`` in snfit) is a
    separable cubic convolution (Keys, a=-0.5), linear in the first and last
    interval and 0 outside the grid.

    Parameters
    ----------
    u : array_like
        Fractional node coordinates, i.e. ``(x - x_0) / step``.

    n : int
        Number of nodes.

    xp : module, optional
        Array module (``numpy`` or ``jax.numpy``). The default is numpy.

    Returns
    -------
    idx : array_like
        Indices of the 4 nodes used for each point, shape ``u.shape + (4,)``.

    weights : array_like
        Corresponding weights. Cubic convolution inside the grid, linear in
        the first and last interval, 0 outside ``[0, n-1]``.
    """
    inside = (u >= 0) & (u <= n - 1)
    i = xp.clip(xp.floor(u), 0, n - 2).astype(int)
    f = u - i
    edge = (i == 0) | (i == n - 2)
    # Keys kernel at the distances 1+f, f, 1-f, 2-f (0 <= f <= 1 inside the grid)
    a, g = -0.5, 1 - f
    wc = xp.stack([a * f * g * g, ((a + 2) * f - (a + 3)) * f * f + 1,
                   ((a + 2) * g - (a + 3)) * g * g + 1, a * g * f * f], -1)
    wl = xp.stack([xp.zeros_like(f), 1 - f, f, xp.zeros_like(f)], -1)
    w = xp.where(edge[..., None], wl, wc) * inside[..., None]
    idx = xp.clip(i[..., None] + xp.arange(-1, 3), 0, n - 1)
    return idx, w
