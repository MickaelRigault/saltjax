"""Pack lightcurves and per-target band integrals into padded arrays for the fit.

The fit works on fixed-shape arrays (one row per target, one column per
lightcurve point, padded), so that it can be vectorised and compiled with JAX.
This module builds these arrays:

- :func:`band_integrals` integrates the SALT surfaces in each band, for each
  target (its redshift and its effects), at the surface phase nodes.
- :func:`pack_lightcurves` selects the lightcurve points used by the fit,
  places them in padded arrays and adds the band integrals and the redshift
  tables of each target.

Each step is also available on its own, to check or plot intermediate
products: :func:`get_colorlaw_terms`, :func:`get_band_integrand`,
:func:`get_wavelength_kernels`, :func:`project_surfaces` (band integrals);
:func:`check_magsys`, :func:`get_target_position`,
:func:`select_covered_points`, :func:`select_phase_range`,
:func:`get_lightcurve_slots`, :func:`get_lightcurve_arrays`,
:func:`get_redshift_tables` (lightcurves).
"""

import warnings

import numpy as np
from scipy import sparse

from .kernel import kernel_weights
from .tables import EDGE_NODES, LN10_04
from .effects import get_effects, parameter_table, transmission, wave_limits

__all__ = ["pack_lightcurves", "band_integrals",
           # band integrals, step by step
           "get_colorlaw_terms", "get_band_integrand", "get_wavelength_kernels",
           "project_surfaces",
           # lightcurves, step by step
           "get_time_key", "check_magsys", "get_target_position",
           "select_covered_points", "select_phase_range",
           "get_lightcurve_slots", "get_lightcurve_arrays", "get_redshift_tables"]

# keys of the packed arrays, see pack_lightcurves()
LC_KEYS = ("t", "flux", "err", "zpn", "band", "mask")     # (ntargets, npoints) arrays
FIT_KEYS = ("K", "Ke", "CLb", "F", "Fe", "V", "Ve", "CD", "z") + LC_KEYS  # used by the fit
CHUNK = 256    # targets per block of the band integrals (limits memory)

# tables interpolated on the redshift grid, see tables.build_tables()
_REDSHIFT_TABLES = ("F", "Fe", "V", "Ve", "CD")


# ================== #
#                    #
#   Band integrals   #
#                    #
# ================== #
def band_integrals(tables, z, effects=()):
    """Get the band integrals of the SALT surfaces of each target, with its effects.

    For a target at redshift z (a = 1/(1+z)), in band b and for the color term
    n, at each phase node p of the M0/M1 surfaces M_k, this computes (exactly
    as ``sncosmo.Model.bandflux``, on the same wavelength grid)::

        K[b, k, n, p] = sum_lambda a M_k(p, a lambda) T_b(lambda) lambda dlambda / (hc zp_b)
                        x E(lambda) x (-0.4 ln10 dCL(a lambda))^n / n!

    with E the transmission of the effects and dCL the deviation of the color
    law from its band mean CLb (so that the color factor is
    ``10**(-0.4 c CLb) sum_n c^n K[..., n, :]``).

    Targets are processed by blocks of `CHUNK` to limit memory.

    Parameters
    ----------
    tables : dict
        Output of :func:`saltjax.tables.build_tables`.

    z : numpy.ndarray
        Redshifts of the targets.

    effects : list of dict, optional
        Effects and their parameters for these targets (output of
        :func:`saltjax.effects.get_effects`). The default is no effect.

    Returns
    -------
    K : numpy.ndarray
        Band integrals, shape (ntargets, nbands, 2, nc, nphase).

    Ke : numpy.ndarray
        Same at the edge-interval nodes, with a linear kernel in wavelength
        (see :data:`saltjax.tables.EDGE_NODES`), shape (..., 4).

    CLb : numpy.ndarray
        Band mean of the color law, shape (ntargets, nbands).

    See Also
    --------
    pack_lightcurves : Pack the lightcurves and these band integrals for the fit.
    """
    z = np.asarray(z, float)
    blocks = [_get_block_band_integrals_(tables, z[start:start + CHUNK],
                                         _select_effects_(effects, slice(start, start + CHUNK)))
              for start in range(0, len(z), CHUNK)]

    # no target: empty arrays with the right shapes
    if len(blocks) == 0:
        nk, nph, _ = tables["M"].shape
        nb, nc = len(tables["band_wave"]), tables["nc"]
        return (np.zeros((0, nb, nk, nc, nph)), np.zeros((0, nb, nk, nc, len(EDGE_NODES))),
                np.zeros((0, nb)))

    K, Ke, CLb = (np.concatenate(block_) for block_ in zip(*blocks))
    return K, Ke, CLb


def _get_block_band_integrals_(tables, z, effects):
    """Get the band integrals of a block of targets (see :func:`band_integrals`).

    Parameters
    ----------
    tables : dict
        Output of :func:`saltjax.tables.build_tables`.

    z : numpy.ndarray
        Redshifts of the targets of the block.

    effects : list of dict
        Effects with their parameters for the targets of the block.

    Returns
    -------
    K, Ke, CLb : numpy.ndarray
        As :func:`band_integrals`, for the targets of the block.
    """
    surfaces = tables["M"]                                  # (nk, nphase, nwave)
    surfaces_edge = surfaces[:, list(EDGE_NODES)]           # (nk, 4, nwave)
    wave = tables["band_wave"]                              # (nbands, nlambda) observer frame [A]

    # rest-frame wavelengths of the band integration grids
    a = 1. / (1. + z)                                       # (ntargets,)
    rest_wave = a[:, None, None] * wave[None]               # (ntargets, nbands, nlambda)

    # color law: band mean and deviation from it
    colorlaw_mean, colorlaw_dev = get_colorlaw_terms(tables, rest_wave)

    # transmission of the effects, at the observer-frame wavelengths
    trans = transmission(effects, wave.ravel(), a).reshape(rest_wave.shape)
    integrand = get_band_integrand(tables, a, trans, colorlaw_dev)  # (ntargets, nbands, nlambda, nc)

    # interpolation kernel of the surfaces at the rest wavelengths:
    # cubic for the regular phase nodes, linear for the edge phase nodes (as sncosmo)
    idx, weights, weights_linear = get_wavelength_kernels(rest_wave, tables["M_wave"])
    lo, hi = idx.min((0, 2, 3)), idx.max((0, 2, 3)) + 1     # surface nodes spanned by each band

    K = project_surfaces(surfaces, idx, weights, integrand, lo, hi)
    Ke = project_surfaces(surfaces_edge, idx, weights_linear, integrand, lo, hi)
    return K, Ke, colorlaw_mean


def _select_effects_(effects, sel):
    """Get the effects with their parameters restricted to the targets `sel`.

    Parameters
    ----------
    effects : list of dict
        Effects (output of :func:`saltjax.effects.get_effects`).

    sel : slice
        Targets to keep.

    Returns
    -------
    list of dict
        Effects with their parameter arrays sliced (new dicts, the inputs are
        not modified).
    """
    return [effect_ | dict(params={name_: values_[sel] for name_, values_ in effect_["params"].items()})
            for effect_ in effects]


def get_colorlaw_terms(tables, rest_wave):
    """Get the band mean of the color law and the deviation from it.

    Parameters
    ----------
    tables : dict
        Output of :func:`saltjax.tables.build_tables`.

    rest_wave : numpy.ndarray
        Rest-frame wavelengths [A], shape (ntargets, nbands, nlambda).

    Returns
    -------
    colorlaw_mean : numpy.ndarray
        Band mean of the color law (CLb), weighted by the band integration
        weights, shape (ntargets, nbands).

    colorlaw_dev : numpy.ndarray
        -0.4 ln10 (CL - CLb), shape (ntargets, nbands, nlambda).
    """
    band_weights = tables["band_wgt"]                       # (nbands, nlambda)
    colorlaw = tables["source"]._colorlaw(rest_wave.ravel()).reshape(rest_wave.shape)
    colorlaw_mean = (colorlaw * band_weights).sum(-1) / band_weights.sum(-1)
    colorlaw_dev = LN10_04 * (colorlaw - colorlaw_mean[..., None])
    return colorlaw_mean, colorlaw_dev


def get_band_integrand(tables, a, trans, colorlaw_dev):
    """Get the band integrand of each color term (but the surfaces).

    ``integrand[..., n] = a base trans colorlaw_dev^n / n!``, the Taylor terms
    of ``exp(c colorlaw_dev)`` in the color c.

    Parameters
    ----------
    tables : dict
        Output of :func:`saltjax.tables.build_tables` (uses ``"band_base"``,
        the band transmission x lambda dlambda / (hc zp), and ``"nc"``, the
        number of color terms).

    a : numpy.ndarray
        Scale factors 1/(1+z), shape (ntargets,).

    trans : numpy.ndarray
        Transmission of the effects, shape (ntargets, nbands, nlambda).

    colorlaw_dev : numpy.ndarray
        Deviation of the color law from its band mean (see
        :func:`get_colorlaw_terms`), shape (ntargets, nbands, nlambda).

    Returns
    -------
    numpy.ndarray
        Integrand, shape (ntargets, nbands, nlambda, nc).
    """
    ncolor = tables["nc"]
    integrand = np.empty(trans.shape + (ncolor,))
    integrand[..., 0] = a[:, None, None] * tables["band_base"][None] * trans
    # recursion: term n = term n-1 x colorlaw_dev / n
    for n in range(1, ncolor):
        integrand[..., n] = integrand[..., n - 1] * colorlaw_dev / n

    return integrand


def get_wavelength_kernels(rest_wave, surface_wave):
    """Get the interpolation kernels of the surfaces at the rest wavelengths.

    As in sncosmo, the kernel is applied in node-index space (non-uniform
    wavelength grids are supported).

    Parameters
    ----------
    rest_wave : numpy.ndarray
        Rest-frame wavelengths [A], shape (ntargets, nbands, nlambda).

    surface_wave : numpy.ndarray
        Wavelength nodes of the surfaces [A], shape (nwave,).

    Returns
    -------
    idx : numpy.ndarray
        Indices of the 4 surface nodes used at each wavelength,
        shape (ntargets, nbands, nlambda, 4).

    weights : numpy.ndarray
        Cubic-convolution weights (see :func:`saltjax.kernel.kernel_weights`),
        same shape as `idx`.

    weights_linear : numpy.ndarray
        Linear-interpolation weights, used for the edge phase nodes
        (sncosmo's kernel is bilinear in the first and last phase intervals),
        same shape as `idx`.
    """
    nwave = len(surface_wave)
    # fractional node index; outside the grid -> weights are 0
    u = np.interp(rest_wave, surface_wave, np.arange(nwave), left=-1., right=nwave)
    idx, weights = kernel_weights(u, nwave)

    # linear kernel on the 2 central nodes, 0 where the cubic kernel is 0 (outside the grid)
    frac = u - idx[..., 1]
    inside = weights.sum(-1, keepdims=True) > 0
    weights_linear = np.stack([np.zeros_like(frac), 1 - frac, frac, np.zeros_like(frac)], -1) * inside
    return idx, weights, weights_linear


def project_surfaces(surfaces, idx, weights, integrand, lo, hi):
    """Integrate the surfaces in each band: one matrix product per band.

    Parameters
    ----------
    surfaces : numpy.ndarray
        Surfaces at their phase nodes, shape (nk, nphase, nwave).

    idx, weights : numpy.ndarray
        Wavelength interpolation kernel (see :func:`get_wavelength_kernels`),
        shape (ntargets, nbands, nlambda, 4).

    integrand : numpy.ndarray
        Band integrand of each color term (see :func:`get_band_integrand`),
        shape (ntargets, nbands, nlambda, nc).

    lo, hi : numpy.ndarray
        First and last+1 surface wavelength nodes spanned by each band,
        shape (nbands,).

    Returns
    -------
    numpy.ndarray
        Band integrals, shape (ntargets, nbands, nk, nc, nphase).
    """
    ntargets, nbands, _, ncolor = integrand.shape
    nk, nphase, _ = surfaces.shape

    node_weights = _get_node_weights_(idx, weights, integrand, lo, hi)
    out = np.zeros((ntargets, nbands, nk, ncolor, nphase))
    for band_ in range(nbands):
        # (nk*nphase, nnodes) @ (nnodes, ntargets*nc)
        integral = surfaces[:, :, lo[band_]:hi[band_]].reshape(-1, hi[band_] - lo[band_]) @ node_weights[band_]
        out[:, band_] = integral.reshape(nk, nphase, ntargets, ncolor).transpose(2, 0, 3, 1)

    return out


def _get_node_weights_(idx, weights, integrand, lo, hi):
    """Get the weights G of the surface wavelength nodes in each band.

    They are such that the band integrals of band b are
    ``M[..., lo[b]:hi[b]] @ G[b]``::

        G[b][node, target, term] = sum_lambda W[lambda, node] P[target, b, lambda, term]

    with W the interpolation kernel and P the integrand. W is built as a
    sparse matrix since each wavelength only uses 4 nodes.

    Parameters
    ----------
    idx, weights : numpy.ndarray
        Wavelength interpolation kernel (see :func:`get_wavelength_kernels`),
        shape (ntargets, nbands, nlambda, 4).

    integrand : numpy.ndarray
        Band integrand P, shape (ntargets, nbands, nlambda, nc).

    lo, hi : numpy.ndarray
        First and last+1 surface wavelength nodes spanned by each band,
        shape (nbands,).

    Returns
    -------
    list of scipy.sparse.csr_matrix
        One matrix per band, shape (hi[b] - lo[b], ntargets * nc).
    """
    ntargets, nbands, nlambda, ncolor = integrand.shape
    nnodes = hi - lo                                         # (nbands,) nodes per band
    offset = np.r_[0, np.cumsum(nnodes)[:-1]]                # first row of each band

    # sparse W: rows (band, node, target), columns (target, band, lambda)
    target_ = np.arange(ntargets)[:, None, None, None]
    band_ = np.arange(nbands)[None, :, None, None]
    rows = (offset[band_] + idx - lo[band_]) * ntargets + target_
    cols = np.broadcast_to(np.arange(ntargets * nbands * nlambda).reshape(ntargets, nbands, nlambda, 1),
                           idx.shape)
    W = sparse.csr_matrix((weights.ravel(), (rows.ravel(), cols.ravel())),
                          shape=(nnodes.sum() * ntargets, ntargets * nbands * nlambda))

    G = W @ integrand.reshape(ntargets * nbands * nlambda, ncolor)  # (sum(nnodes)*ntargets, nc)
    # split per band: (nnodes[b], ntargets * nc)
    return [G[offset_ * ntargets:(offset_ + nnodes_) * ntargets].reshape(nnodes_, ntargets * ncolor)
            for offset_, nnodes_ in zip(offset, nnodes)]


# ================== #
#                    #
#   Lightcurves      #
#                    #
# ================== #
def pack_lightcurves(data, targets, tables, indexes=None, phase_range=(-10, 40),
                     effects=None, effect_names=None, effect_frames=None,
                     mwebv_key="mwebv", mw_r_v=3.1, time_key=None, pad_to=16):
    """Pack the lightcurves and the per-target tables into padded arrays.

    This:

    1. gets the effects of each target and their band integrals
       (:func:`band_integrals`),
    2. drops the points whose band is not covered by the model (with a
       warning, as ``sncosmo.fit_lc``) and those outside `phase_range`,
    3. places the points of each target in a row of padded arrays,
    4. interpolates the redshift tables (model covariance) at each target
       redshift.

    Parameters
    ----------
    data : pandas.DataFrame
        Lightcurves, with a (target, observation) multi-index and columns
        time (see `time_key`), 'band', 'flux', 'fluxerr', 'zp' and optionally
        'zpsys'.

    targets : pandas.DataFrame
        Targets, indexed as the first level of `data`, with a 'z' column, a
        't0' column if `phase_range` is given, and the parameters of the
        effects (see :func:`saltjax.effects.get_effects`).

    tables : dict
        Output of :func:`saltjax.tables.build_tables`.

    indexes : list or None, optional
        Targets to pack. If None, all those of `targets`. The default is None.

    phase_range : list or None, optional
        Rest-frame phase range (relative to the targets' t0) of the data used.
        If None, all points are used. The default is (-10, 40).

    effects, effect_names, effect_frames : list or None, optional
        Propagation effects (e.g. dust), their names and frames ('obs' or
        'rest'), as in ``sncosmo.Model``; see
        :func:`saltjax.effects.get_effects`. The default is None (Milky Way
        dust from `mwebv_key` and `mw_r_v`).

    mwebv_key : str or None, optional
        Only if `effects` is None: column of `targets` with the Milky Way
        E(B-V) (CCM89 dust in the observer frame). If None or absent, no
        effect. The default is 'mwebv'.

    mw_r_v : float, optional
        Only if `effects` is None: R_V of the Milky Way dust. The default is 3.1.

    time_key : str or None, optional
        Time column of the lightcurves. If None, the first of 'time', 'mjd' or
        'jd' found. The default is None.

    pad_to : int, optional
        The number of points per lightcurve is padded to a multiple of
        `pad_to`. The default is 16.

    Returns
    -------
    dict
        Arrays used by the fit (one row per target, see `FIT_KEYS`): the
        lightcurves (`LC_KEYS`, shape (ntargets, npoints), padded points have
        ``mask=0``), the band integrals ('K', 'Ke', 'CLb'), the redshift
        tables ('F', 'Fe', 'V', 'Ve', 'CD') and 'z'; plus 'effects': the
        parameters of the effects (DataFrame with columns ``{name}{param}``)
        and 'index': the packed targets.

    Raises
    ------
    NotImplementedError
        If a magnitude system other than 'ab' is used.

    Warns
    -----
    UserWarning
        If points are dropped because their band is out of the model
        wavelength range.
    """
    indexes = list(targets.index if indexes is None else indexes)
    truth = targets.loc[indexes]
    z = truth["z"].to_numpy(float)

    # Model: effects and band integrals of each target
    effects = get_effects(truth, effects, effect_names, effect_frames,
                          mwebv_key=mwebv_key, mw_r_v=mw_r_v)
    K, Ke, CLb = band_integrals(tables, z, effects)

    # Data: points used by the fit
    data = data.loc[indexes]
    time_key = get_time_key(data, time_key)
    check_magsys(data)
    position = get_target_position(data, indexes)          # row of each point
    data, position = select_covered_points(data, position, tables, z, effects)
    if phase_range is not None:
        t0 = truth["t0"].to_numpy(float)
        data, position = select_phase_range(data, position, t0, z, phase_range, time_key)

    # Packing
    lightcurves = get_lightcurve_arrays(data, position, len(indexes), tables["bands"],
                                           time_key=time_key, pad_to=pad_to)
    redshift_tables = get_redshift_tables(tables, z)
    return lightcurves | redshift_tables | dict(K=K, Ke=Ke, CLb=CLb, z=z,
                                                effects=parameter_table(effects, index=truth.index),
                                                index=np.asarray(indexes))


def get_time_key(data, time_key=None):
    """Get the time column of the lightcurves.

    Parameters
    ----------
    data : pandas.DataFrame
        Lightcurve data.

    time_key : str or None, optional
        Time column. If None, the first of 'time', 'mjd' or 'jd' found.
        The default is None.

    Returns
    -------
    str
        Name of the time column.

    Raises
    ------
    ValueError
        If `time_key` is None and no time column is found.
    """
    if time_key is not None:
        return time_key

    time_key = next((k for k in ("time", "mjd", "jd") if k in data.columns), None)
    if time_key is None:
        raise ValueError(f"cannot parse the time column, provide time_key. {list(data.columns)=}")

    return time_key


def check_magsys(data):
    """Check that all points are in the 'ab' magnitude system.

    Parameters
    ----------
    data : pandas.DataFrame
        Lightcurves. Points are assumed 'ab' if there is no 'zpsys' column.

    Raises
    ------
    NotImplementedError
        If a magnitude system other than 'ab' is used.
    """
    if "zpsys" in data.columns and not data["zpsys"].str.lower().eq("ab").all():
        magsys = sorted(data["zpsys"].str.lower().unique())
        raise NotImplementedError(f"only the 'ab' magnitude system is supported, {magsys=}")


def get_target_position(data, indexes):
    """Get the row (position in `indexes`) of the target of each point.

    Parameters
    ----------
    data : pandas.DataFrame
        Lightcurves, the target is the first level of the index.

    indexes : list
        Packed targets.

    Returns
    -------
    numpy.ndarray
        (npoints,) row of each point.
    """
    target_to_row = {index_: row_ for row_, index_ in enumerate(indexes)}
    return np.fromiter((target_to_row[index_] for index_ in data.index.get_level_values(0)),
                       int, len(data))


def select_covered_points(data, position, tables, z, effects):
    """Drop the points whose band is out of the model wavelength range.

    As ``sncosmo.fit_lc``: the band must be within the (redshifted) model
    wavelength range, and within the range allowed by the effects.

    Parameters
    ----------
    data : pandas.DataFrame
        Lightcurves.

    position : numpy.ndarray
        (npoints,) row of the target of each point.

    tables : dict
        Output of :func:`saltjax.tables.build_tables`.

    z : numpy.ndarray
        (ntargets,) redshifts.

    effects : list of dict
        Effects of the targets (see :func:`saltjax.effects.get_effects`).

    Returns
    -------
    data : pandas.DataFrame
        Lightcurves, covered points only.

    position : numpy.ndarray
        Row of the target of each kept point.

    Warns
    -----
    UserWarning
        If points are dropped.
    """
    band_ids = data["band"].map({band_: i for i, band_ in enumerate(tables["bands"])}).to_numpy()
    band_range = tables["band_range"][band_ids]            # (npoints, 2) observer frame [A]
    wmin, wmax = tables["wave_range"]                      # rest frame [A]
    effect_min, effect_max = wave_limits(effects, z)       # (ntargets,) observer frame [A]

    zpoint = z[position]
    covered = ((band_range[:, 0] >= np.maximum(wmin * (1 + zpoint), effect_min[position])) &
               (band_range[:, 1] <= np.minimum(wmax * (1 + zpoint), effect_max[position])))
    if np.all(covered):
        return data, position

    dropped_bands = ", ".join(sorted(data["band"][~covered].unique()))
    warnings.warn(f"{np.sum(~covered)} points dropped: band out of the model wavelength range "
                  f"({dropped_bands})")
    return data[covered], position[covered]


def select_phase_range(data, position, t0, z, phase_range, time_key=None):
    """Keep the points within the rest-frame phase range.

    Parameters
    ----------
    data : pandas.DataFrame
        Lightcurves.

    position : numpy.ndarray
        (npoints,) row of the target of each point.

    t0 : numpy.ndarray
        (ntargets,) times of maximum.

    z : numpy.ndarray
        (ntargets,) redshifts.

    phase_range : list
        [min, max] rest-frame phase [day].

    time_key : str or None, optional
        Time column of `data`. If None, the first of 'time', 'mjd' or 'jd'
        found (see :func:`get_time_key`). The default is None.

    Returns
    -------
    data : pandas.DataFrame
        Lightcurves, points within `phase_range` only.

    position : numpy.ndarray
        Row of the target of each kept point.
    """
    time_key = get_time_key(data, time_key)
    phase = (data[time_key].to_numpy() - t0[position]) / (1 + z[position])  # rest frame [day]
    keep = (phase >= phase_range[0]) & (phase <= phase_range[1])
    return data[keep], position[keep]


def get_lightcurve_slots(position, pad_to=16):
    """Get the column of each point in the padded arrays.

    Points keep their order within each target.

    Parameters
    ----------
    position : numpy.ndarray
        (npoints,) row of the target of each point.

    pad_to : int, optional
        The number of columns is padded to a multiple of `pad_to`.
        The default is 16.

    Returns
    -------
    slot : numpy.ndarray
        (npoints,) column of each point.

    npoints_max : int
        Number of columns of the padded arrays.
    """
    order = np.argsort(position, kind="stable")             # points grouped by target
    sorted_position = position[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_position)) + 1]  # first point of each target
    counts = np.diff(np.r_[starts, len(sorted_position)])    # points per target

    slot = np.empty(len(sorted_position), int)
    slot[order] = np.arange(len(sorted_position)) - np.repeat(starts, counts)

    if len(slot) == 0:
        return slot, pad_to

    npoints_max = int(np.ceil((slot.max() + 1) / pad_to) * pad_to)
    return slot, npoints_max


def get_lightcurve_arrays(data, position, ntargets, bands, time_key=None, pad_to=16):
    """Place the lightcurve points in padded (ntargets, npoints) arrays.

    Parameters
    ----------
    data : pandas.DataFrame
        Lightcurves (selected points).

    position : numpy.ndarray
        (npoints,) row of the target of each point.

    ntargets : int
        Number of rows.

    bands : list of str
        Bands of the tables; 'band' is stored as the index in this list.

    time_key : str or None, optional
        Time column of `data`. If None, the first of 'time', 'mjd' or 'jd'
        found (see :func:`get_time_key`). The default is None.

    pad_to : int, optional
        The number of columns is padded to a multiple of `pad_to`.
        The default is 16.

    Returns
    -------
    dict
        One array per key of `LC_KEYS`. Padded points have ``mask=0`` and
        ``err=1`` (to avoid divisions by 0).
    """
    time_key = get_time_key(data, time_key)
    slot, npoints_max = get_lightcurve_slots(position, pad_to=pad_to)
    band_ids = {band_: i for i, band_ in enumerate(bands)}

    out = {k: np.zeros((ntargets, npoints_max)) for k in LC_KEYS}
    out["t"][position, slot] = data[time_key]
    out["flux"][position, slot] = data["flux"]
    out["err"][position, slot] = data["fluxerr"]
    out["zpn"][position, slot] = 10 ** (0.4 * data["zp"].to_numpy(float))  # zero point in flux
    out["band"][position, slot] = data["band"].map(band_ids)
    out["mask"][position, slot] = 1.

    out["err"][out["mask"] == 0] = 1.          # padded points: avoid divisions by 0
    out["band"] = out["band"].astype(int)
    return out


def get_redshift_tables(tables, z):
    """Interpolate the redshift tables at the target redshifts.

    The tables of the model covariance (``F``, ``Fe``, ``V``, ``Ve``, ``CD``)
    are tabulated on ``tables["zgrid"]`` and interpolated with the same
    kernel as the surfaces (see :func:`saltjax.kernel.kernel_weights`).

    Parameters
    ----------
    tables : dict
        Output of :func:`saltjax.tables.build_tables`.

    z : numpy.ndarray
        (ntargets,) redshifts.

    Returns
    -------
    dict
        Interpolated tables, one row per target.
    """
    zgrid = tables["zgrid"]
    idx, weights = kernel_weights((z - zgrid[0]) / (zgrid[1] - zgrid[0]), len(zgrid))  # (ntargets, 4)
    return {key_: np.einsum("sj,sj...->s...", weights, tables[key_][idx])
            for key_ in _REDSHIFT_TABLES}
