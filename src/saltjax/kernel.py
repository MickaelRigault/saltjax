"""Bicubic-convolution interpolation kernel (sncosmo / snfit convention).

sncosmo's ``BicubicInterpolator`` (``Grid2DFunction`` in snfit) interpolates
the SALT surfaces with a separable cubic convolution (Keys kernel, a=-0.5),
linear in the first and last interval of the grid and 0 outside.

- :func:`keys_kernel` is the kernel itself, as a function of the distance to a
  node (useful to check or plot it).
- :func:`kernel_weights` gives, for any coordinate, the 4 nodes it uses and
  their weights (used everywhere in saltjax).

Both work with ``numpy`` or ``jax.numpy`` (see `xp`).
"""

import numpy as np

__all__ = ["kernel_weights", "keys_kernel"]

KEYS_A = -0.5    # Keys kernel parameter (as sncosmo)


def keys_kernel(x, a=KEYS_A, xp=np):
    """Get the Keys cubic-convolution kernel.

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
        Kernel values, same shape as `x` (0 for ``|x| >= 2``).

    See Also
    --------
    kernel_weights : Nodes and weights of the kernel on a grid.
    """
    x = xp.abs(x)
    inner = (a + 2) * x**3 - (a + 3) * x**2 + 1          # |x| <= 1
    outer = a * (x**3 - 5 * x**2 + 8 * x - 4)             # 1 < |x| < 2
    return xp.where(x <= 1, inner, xp.where(x < 2, outer, 0.))


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

    See Also
    --------
    keys_kernel : The cubic-convolution kernel.
    """
    inside = (u >= 0) & (u <= n - 1)
    # interval [node, node+1] of each point, and position within it (0 <= frac <= 1 inside)
    node = xp.clip(xp.floor(u), 0, n - 2).astype(int)
    frac = u - node
    on_edge = (node == 0) | (node == n - 2)              # first or last interval: linear

    # Keys kernel at the distances 1+frac, frac, 1-frac, 2-frac of the 4 nodes
    a, rest = KEYS_A, 1 - frac
    weights_cubic = xp.stack([a * frac * rest * rest,
                              ((a + 2) * frac - (a + 3)) * frac * frac + 1,
                              ((a + 2) * rest - (a + 3)) * rest * rest + 1,
                              a * rest * frac * frac], -1)
    weights_linear = xp.stack([xp.zeros_like(frac), 1 - frac, frac, xp.zeros_like(frac)], -1)

    weights = xp.where(on_edge[..., None], weights_linear, weights_cubic) * inside[..., None]
    idx = xp.clip(node[..., None] + xp.arange(-1, 3), 0, n - 1)   # nodes node-1 ... node+2
    return idx, weights
