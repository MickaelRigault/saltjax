"""SALT2 / SALT3 band fluxes and model covariance in JAX (one target)."""

import jax.numpy as jnp

from .kernel import kernel_weights

__all__ = ["get_model_functions"]


def phase_grids(tables):
    """Get a hashable description of the phase grids and of the model
    variance (static for the JIT).

    Parameters
    ----------
    tables : dict
        Output of :func:`saltjax.tables.build_tables`.

    Returns
    -------
    tuple
        (first node, step, number of nodes) of the M0/M1 phase grid, then of
        the LCRV/errscale phase grid, then whether the model variance is
        relative to M0 (True: SALT2) or in flux units (False: SALT3).
    """
    phM, phV = tables["phM"], tables["phV"]
    return (float(phM[0]), float(phM[1] - phM[0]), len(phM),
            float(phV[0]), float(phV[1] - phV[0]), len(phV),
            bool(tables.get("relative_var", True)))


def _interp(T, Te, band, p, p0, dp, n):
    """Interpolate T[band] at phases p with sncosmo's kernel.

    In the first and last phase intervals, the node values seen from inside
    these intervals (Te, see :data:`saltjax.tables.EDGE_NODES`) are used.
    """
    idx, w = kernel_weights((p - p0) / dp, n, xp=jnp)
    Tb, Teb = T[band], Te[band]
    shape = (idx.shape[0],) + (1,) * (Tb.ndim - 2) + (4,)
    g = jnp.take_along_axis(Tb, idx.reshape(shape), axis=-1)       # (npoints, ..., 4)
    first, last = (idx[:, 1] == 0).reshape(shape[:-1]), (idx[:, 1] == n - 2).reshape(shape[:-1])
    g = g.at[..., 1].set(jnp.where(first, Teb[..., 0], jnp.where(last, Teb[..., 2], g[..., 1])))
    g = g.at[..., 2].set(jnp.where(first, Teb[..., 1], jnp.where(last, Teb[..., 3], g[..., 2])))
    return jnp.sum(g * w.reshape(shape), -1)


def bandflux(theta, d, grid):
    """Model band fluxes of a target, in the zero points of the data.

    Parameters
    ----------
    theta : array_like
        (t0, x0, x1, c).

    d : dict
        Packed arrays of the target (see :func:`saltjax.data.pack_lightcurves`).

    grid : tuple
        Output of :func:`phase_grids`.

    Returns
    -------
    jax.Array
        Fluxes, one per lightcurve point.
    """
    t0, x0, x1, c = theta
    phM0, dM, nM = grid[:3]
    cpow = c ** jnp.arange(d["K"].shape[2])
    factor = (10 ** (-0.4 * c * d["CLb"]))[:, None, None]
    H = jnp.einsum("bkcp,c->bkp", d["K"], cpow) * factor
    He = jnp.einsum("bkcp,c->bkp", d["Ke"], cpow) * factor
    Hk = _interp(H, He, d["band"], (d["t"] - t0) / (1. + d["z"]), phM0, dM, nM)
    return x0 * (Hk[:, 0] + x1 * Hk[:, 1]) * d["zpn"]


def rvar_and_cdisp(theta, d, grid):
    """Relative model variance and color dispersion of each point (as sncosmo).

    Parameters
    ----------
    theta : array_like
        (t0, x0, x1, c).

    d : dict
        Packed arrays of the target.

    grid : tuple
        Output of :func:`phase_grids`.

    Returns
    -------
    rvar : jax.Array
        Diagonal relative variance (``SALT2Source._bandflux_rvar_single``,
        or ``SALT3Source._bandflux_rvar_single``).

    cdisp : jax.Array
        Color dispersion of the band of each point (``SALT2Source._colordisp``).
    """
    t0, x0, x1, c = theta
    phM0, dM, nM, phV0, dV, nV, relative_var = grid
    p = (d["t"] - t0) / (1. + d["z"])
    f = _interp(d["F"], d["Fe"], d["band"], p, phM0, dM, nM)
    v = _interp(d["V"], d["Ve"], d["band"], p, phV0, dV, nV)
    f0, ftot = f[:, 0], f[:, 0] + x1 * f[:, 1]
    var = v[:, 0] + 2. * x1 * v[:, 2] + x1 * x1 * v[:, 1]
    var = jnp.where(var < 0., 1e-4, var)
    # SALT2: var (f0 / ftot)^2 errscale^2; SALT3: var / ftot^2 (errscale = 1)
    num = f0 if relative_var else 1.
    rvar = jnp.where(ftot <= 0., 1e4,
                     var * (num / jnp.where(ftot == 0, 1., ftot)) ** 2 * v[:, 3] ** 2)
    return rvar, d["CD"][d["band"]]


def get_model_functions(tables):
    """Get the JAX functions of the model for a single target.

    Parameters
    ----------
    tables : dict
        Output of :func:`saltjax.tables.build_tables`.

    Returns
    -------
    flux : callable
        ``flux(theta, d)``: model fluxes (in the data zero points) for
        ``theta = (t0, x0, x1, c)`` and the packed target arrays `d`.

    rcov : callable
        ``rcov(theta, d)``: relative model covariance (as
        ``sncosmo.SALT2Source.bandflux_rcov``), as a dense matrix.

    mcov : callable
        ``mcov(theta, d)``: model covariance (as the second output of
        ``sncosmo.Model.bandfluxcov``), as a dense matrix.

    Notes
    -----
    The fit never builds these dense matrices: the model covariance is a
    diagonal plus one constant block per band (a low-rank term), so the chi2
    is computed exactly with the Woodbury identity in O(number of points).
    """
    grid = phase_grids(tables)

    def flux(theta, d):
        return bandflux(theta, d, grid)

    def rcov(theta, d):
        rvar, cd = rvar_and_cdisp(theta, d, grid)
        same = d["band"][:, None] == d["band"][None, :]
        return jnp.diag(rvar) + same * cd[:, None] ** 2

    def mcov(theta, d):
        f = flux(theta, d)
        return f[:, None] * rcov(theta, d) * f[None, :]

    return flux, rcov, mcov
