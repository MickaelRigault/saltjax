"""SALT2 / SALT3 band fluxes and model covariance in JAX (one target).

The functions of this module work on the packed arrays of a single target
(one row of :func:`saltjax.data.pack_lightcurves`), and are vectorised over
targets and compiled by :mod:`saltjax.fitter`.

- :func:`bandflux`: model band fluxes, from the band integrals at the phase
  nodes, interpolated at the phase of each point (:func:`interpolate_phase`).
- :func:`rvar_and_cdisp`: relative model variance and color dispersion.
- :func:`get_model_functions`: these, plus the dense model covariance, as
  ``f(theta, packed)`` functions (to check against sncosmo).
"""

import jax.numpy as jnp

from .kernel import kernel_weights

__all__ = ["get_model_functions", "phase_grids", "interpolate_phase", "bandflux", "rvar_and_cdisp"]


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
    phase_model, phase_var = tables["phM"], tables["phV"]
    return (float(phase_model[0]), float(phase_model[1] - phase_model[0]), len(phase_model),
            float(phase_var[0]), float(phase_var[1] - phase_var[0]), len(phase_var),
            bool(tables.get("relative_var", True)))


def interpolate_phase(table, table_edge, band, phase, phase0, dphase, nphase):
    """Interpolate ``table[band]`` at the given phases with sncosmo's kernel.

    In the first and last phase intervals, sncosmo's kernel is bilinear: the
    node values seen from inside these intervals (`table_edge`, see
    :data:`saltjax.tables.EDGE_NODES`) are used there.

    Parameters
    ----------
    table : jax.Array
        Values at the phase nodes, shape (nbands, ..., nphase).

    table_edge : jax.Array
        Values at the 4 edge nodes (linear in wavelength), shape (nbands, ..., 4).

    band : jax.Array
        (npoints,) band index of each point.

    phase : jax.Array
        (npoints,) rest-frame phase of each point [day].

    phase0, dphase : float
        First phase node and node spacing [day].

    nphase : int
        Number of phase nodes.

    Returns
    -------
    jax.Array
        Interpolated values, shape (npoints, ...).
    """
    idx, weights = kernel_weights((phase - phase0) / dphase, nphase, xp=jnp)   # (npoints, 4)
    table_band, table_edge_band = table[band], table_edge[band]               # (npoints, ..., nphase|4)
    shape = (idx.shape[0],) + (1,) * (table_band.ndim - 2) + (4,)             # broadcast idx over "..."
    nodes = jnp.take_along_axis(table_band, idx.reshape(shape), axis=-1)       # (npoints, ..., 4)

    # first / last interval: the 2 central nodes take their edge values
    first = (idx[:, 1] == 0).reshape(shape[:-1])
    last = (idx[:, 1] == nphase - 2).reshape(shape[:-1])
    nodes = nodes.at[..., 1].set(jnp.where(first, table_edge_band[..., 0],
                                           jnp.where(last, table_edge_band[..., 2], nodes[..., 1])))
    nodes = nodes.at[..., 2].set(jnp.where(first, table_edge_band[..., 1],
                                           jnp.where(last, table_edge_band[..., 3], nodes[..., 2])))
    return jnp.sum(nodes * weights.reshape(shape), -1)


def bandflux(theta, packed, grid):
    """Model band fluxes of a target, in the zero points of the data.

    The color factor is ``10**(-0.4 c CLb) sum_n c^n K[..., n, :]`` (see
    :func:`saltjax.data.band_integrals`).

    Parameters
    ----------
    theta : array_like
        (t0, x0, x1, c).

    packed : dict
        Packed arrays of the target (see :func:`saltjax.data.pack_lightcurves`).

    grid : tuple
        Output of :func:`phase_grids`.

    Returns
    -------
    jax.Array
        Fluxes, one per lightcurve point.
    """
    t0, x0, x1, c = theta
    phase0, dphase, nphase = grid[:3]

    # color factor applied to the band integrals at the phase nodes
    color_powers = c ** jnp.arange(packed["K"].shape[2])                   # (nc,)
    factor = (10 ** (-0.4 * c * packed["CLb"]))[:, None, None]             # (nbands, 1, 1)
    flux_nodes = jnp.einsum("bkcp,c->bkp", packed["K"], color_powers) * factor        # (nbands, 2, nphase)
    flux_edge = jnp.einsum("bkcp,c->bkp", packed["Ke"], color_powers) * factor        # (nbands, 2, 4)

    # M0 and M1 terms at the rest-frame phase of each point
    phase = (packed["t"] - t0) / (1. + packed["z"])
    flux_terms = interpolate_phase(flux_nodes, flux_edge, packed["band"], phase, phase0, dphase, nphase)
    return x0 * (flux_terms[:, 0] + x1 * flux_terms[:, 1]) * packed["zpn"]


def rvar_and_cdisp(theta, packed, grid):
    """Relative model variance and color dispersion of each point (as sncosmo).

    Parameters
    ----------
    theta : array_like
        (t0, x0, x1, c).

    packed : dict
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
    phase0, dphase, nphase, phase0_var, dphase_var, nphase_var, relative_var = grid
    phase = (packed["t"] - t0) / (1. + packed["z"])

    # M0, M1 in the rest-frame bands; LCRV00, LCRV11, LCRV01, errscale
    flux = interpolate_phase(packed["F"], packed["Fe"], packed["band"], phase, phase0, dphase, nphase)
    var_terms = interpolate_phase(packed["V"], packed["Ve"], packed["band"], phase,
                                  phase0_var, dphase_var, nphase_var)
    flux0, flux_total = flux[:, 0], flux[:, 0] + x1 * flux[:, 1]
    variance = var_terms[:, 0] + 2. * x1 * var_terms[:, 2] + x1 * x1 * var_terms[:, 1]
    variance = jnp.where(variance < 0., 1e-4, variance)        # as sncosmo

    # SALT2: var (f0 / ftot)^2 errscale^2; SALT3: var / ftot^2 (errscale = 1)
    numerator = flux0 if relative_var else 1.
    rvar = jnp.where(flux_total <= 0., 1e4,
                     variance * (numerator / jnp.where(flux_total == 0, 1., flux_total)) ** 2
                     * var_terms[:, 3] ** 2)
    return rvar, packed["CD"][packed["band"]]


def get_model_functions(tables):
    """Get the JAX functions of the model for a single target.

    Parameters
    ----------
    tables : dict
        Output of :func:`saltjax.tables.build_tables`.

    Returns
    -------
    flux : callable
        ``flux(theta, packed)``: model fluxes (in the data zero points) for
        ``theta = (t0, x0, x1, c)`` and the packed target arrays.

    rcov : callable
        ``rcov(theta, packed)``: relative model covariance (as
        ``sncosmo.SALT2Source.bandflux_rcov``), as a dense matrix.

    mcov : callable
        ``mcov(theta, packed)``: model covariance (as the second output of
        ``sncosmo.Model.bandfluxcov``), as a dense matrix.

    Notes
    -----
    The fit never builds these dense matrices: the model covariance is a
    diagonal plus one constant block per band (a low-rank term), so the chi2
    is computed exactly with the Woodbury identity in O(number of points).
    """
    grid = phase_grids(tables)

    def flux(theta, packed):
        return bandflux(theta, packed, grid)

    def rcov(theta, packed):
        rvar, cdisp = rvar_and_cdisp(theta, packed, grid)
        same_band = packed["band"][:, None] == packed["band"][None, :]     # (npoints, npoints)
        return jnp.diag(rvar) + same_band * cdisp[:, None] ** 2

    def mcov(theta, packed):
        flux_ = flux(theta, packed)
        return flux_[:, None] * rcov(theta, packed) * flux_[None, :]

    return flux, rcov, mcov
