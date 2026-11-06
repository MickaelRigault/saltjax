"""Batched SALT2 / SALT3 fit of many lightcurves.

:func:`fit_salt` is the entry point of saltjax. Its steps are also available
on their own, to check or reproduce a fit: :func:`select_fittable`,
:func:`get_initial_theta`, :func:`get_bucket_length`, :func:`get_batches`,
:func:`fit_batches`, :func:`get_results_table` and :func:`check_results`.
"""

import warnings

import numpy as np
import pandas
import jax
import jax.numpy as jnp

from .tables import get_salt_source, get_tables
from .data import LC_KEYS, FIT_KEYS, pack_lightcurves
from .model import phase_grids
from .fitter import fit_batch, initial_batch, compile_all, get_compiled, enable_x64

__all__ = ["fit_salt",
           # steps of fit_salt
           "select_fittable", "get_initial_theta", "get_bucket_length", "get_batches",
           "fit_batches", "get_results_table", "check_results"]

PARAMS = ["t0", "x0", "x1", "c"]
LENGTH_BASE = 2      # growth factor of the lightcurve-length buckets
LENGTH_MIN = 16      # smallest bucket (number of points)
# initial guess scan (guess='data'): t0 offsets [day] around the highest S/N point, and colors
T0_OFFSETS = tuple(np.arange(-25., 25.5, 1.))
C_VALUES = (-0.2, 0., 0.2, 0.4)


# ================== #
#                    #
#   Fit              #
#                    #
# ================== #
def fit_salt(data, targets, indexes=None, modelcov=True, phase_range=[-10, 40],
             guess="data", source="salt2", version=None, modeldir=None,
             effects=None, effect_names=None, effect_frames=None, mwebv_key="mwebv", mw_r_v=3.1,
             c_max=1.5, dz=2.5e-4, batch_points=2048, minsnr=5., nrefit=10,
             time_key=None, progress_bar=False, verbose=False):
    """Fit a SALT model on many lightcurves at once, reproducing ``sncosmo.fit_lc``.

    This:

    1. gets the (cached) model tables and packs the lightcurves
       (:func:`saltjax.tables.get_tables`, :func:`saltjax.data.pack_lightcurves`),
    2. selects the targets that can be fitted (:func:`select_fittable`),
    3. gets the starting point of each target (:func:`get_initial_theta`),
    4. groups the targets in batches of fixed shapes (:func:`get_batches`),
    5. compiles and runs the fit of each batch (:func:`fit_batches`),
    6. gathers the results in a table (:func:`get_results_table`) and warns
       about invalid or non-converged fits (:func:`check_results`).

    Same model, same chi2 and same model-covariance procedure as
    ``sncosmo.fit_lc``; parameters (t0, x0, x1, c) are not bounded, and the
    redshift and the parameters of the effects (e.g. dust) are fixed.

    Targets are grouped by number of lightcurve points (padded to 16, 32, 64,
    ... points) and fitted by batches whose shape only depends on that number.
    The model tables are cached per (bands, redshift range rounded to 0.01),
    and the compiled fit is reused by any later call using the same tables.
    The first call of a session includes a few seconds of tabulation and
    (parallel) compilation; such later calls do not. The band fluxes of each
    target, with its effects (e.g. Milky Way and host dust), are integrated
    exactly at each call. Prefer one call over many small ones.

    Parameters
    ----------
    data : pandas.DataFrame
        Lightcurves, with a (target, observation) multi-index and columns
        time (see `time_key`), 'band', 'flux', 'fluxerr', 'zp' and optionally
        'zpsys' (only 'ab' is supported).

    targets : pandas.DataFrame
        Targets, indexed as the first level of `data`, with a 'z' column; a
        't0' column if `phase_range` is given; 't0', 'x1' and 'c' columns if
        ``guess='truth'``; and the parameters of the effects: columns
        ``{name}{param}`` (e.g. 'mwebv', 'hostebv', 'hostr_v'), see `effects`.

    indexes : list or None, optional
        Targets to fit. If None, all the targets with data. The default is None.

    modelcov : bool, optional
        Include the SALT model covariance in the chi2, as in
        ``sncosmo.fit_lc(..., modelcov=True)``. The default is True.

    phase_range : list or None, optional
        Rest-frame phase range (relative to the targets' t0) of the data used.
        The default is [-10, 40].

    guess : {'data', 'truth'}, optional
        Initial guess. 'data': x1 = 0 and the (t0, c) with the lowest chi2 on a
        grid (t0 within +-25 days of the highest S/N point in 1-day steps, c in
        {-0.2, 0, 0.2, 0.4}); 'truth': the targets' t0, x1 and c. In both cases
        x0 is solved linearly. The default is 'data'.

    source : str or sncosmo.SALT2Source or sncosmo.SALT3Source, optional
        SALT model: the name of any SALT2 or SALT3 source registered in
        sncosmo (e.g. 'salt2', 'salt2-extended', 'salt2-h17', 'salt3',
        'salt3-nir'), or an instance (see `modeldir`). The default is 'salt2'.

    version : str or None, optional
        Version of the source, if given by name. The default is None (latest).

    modeldir : str or None, optional
        Directory of the source files (with sncosmo's default file names).
        If None, it is found from the sncosmo registry, so it is required for
        a source instance that is not registered in sncosmo (e.g.
        ``sncosmo.SALT3Source(modeldir=...)``). The default is None.

    effects : list of sncosmo.PropagationEffect or None, optional
        Propagation effects, as in ``sncosmo.Model``: any sncosmo dust
        (``CCM89Dust``, ``OD94Dust``, ``F99Dust``) or any other effect that
        is a deterministic transmission in wavelength. The parameter ``p`` of
        the effect named ``name`` is read per target from the column
        ``f"{name}{p}"`` of `targets` (e.g. 'hostebv', 'hostr_v'), or else
        taken from the effect instance. If None, Milky Way CCM89 dust from
        `mwebv_key` and `mw_r_v`. The default is None.

    effect_names : list of str or None, optional
        Names of the effects (required with `effects`), e.g. ['mw', 'host'].
        The default is None.

    effect_frames : list of str or None, optional
        Frames of the effects (required with `effects`): 'obs' (e.g. Milky
        Way dust) or 'rest' (e.g. host dust). The default is None.

    mwebv_key : str or None, optional
        Only if `effects` is None: column of `targets` with the Milky Way
        E(B-V) (``sncosmo.CCM89Dust`` in the observer frame, named 'mw'). If
        None or absent, no effect. The default is 'mwebv'.

    mw_r_v : float, optional
        Only if `effects` is None: R_V of the Milky Way dust. The default is 3.1.

    c_max : float, optional
        Largest absolute color for which the model is exact; a warning is
        issued if a fit goes beyond. The default is 1.5.

    dz : float, optional
        Redshift step of the model tables. The default is 2.5e-4.

    batch_points : int, optional
        Number of lightcurve points fitted together: targets with at most L
        points are fitted by batches of ``max(1, batch_points // L)``.
        The default is 2048.

    minsnr : float, optional
        Targets without any point with flux/fluxerr >= `minsnr` are not
        fitted. The default is 5.

    nrefit : int, optional
        Maximum number of refits with model covariance. The default is 10.

    time_key : str or None, optional
        Time column of the lightcurves (None: 'time', 'mjd' or 'jd').
        The default is None.

    progress_bar : bool, optional
        Display a progress bar (requires tqdm). Targets are fitted by batches,
        so it advances one batch at a time. The default is False.

    verbose : bool, optional
        Print progress information. The default is False.

    Returns
    -------
    pandas.DataFrame
        One row per fitted target: ``z``, ``t0``, ``x0``, ``x1``, ``c``, their
        ``_err``, ``cov_{p}{q}``, the fixed parameters of the effects
        (``{name}{param}``, e.g. ``mwebv`` and ``mwr_v``), ``chi2``, ``ndof``,
        ``converged``, ``nrefit`` and ``valid`` (False if the covariance is
        not positive: errors are then NaN).

    See Also
    --------
    select_fittable, get_initial_theta, get_batches, fit_batches, get_results_table :
        The steps of the fit, which can be called on their own.

    Notes
    -----
    As ``sncosmo.fit_lc``, points whose band is out of the model wavelength
    range are dropped (with a warning). Targets without any point with S/N >=
    `minsnr`, or with no more points than parameters, are not fitted.

    Raises
    ------
    NotImplementedError
        If the source is not a SALT2 or SALT3 source, a magnitude system other
        than 'ab' is used, or an effect is not supported ('free' frame, random
        scattering effects).

    ValueError
        If `guess` is not 'data' or 'truth', or the effects are inconsistent.

    Warns
    -----
    UserWarning
        If points are dropped (band out of the model range), targets are not
        fitted, fits have an invalid covariance or ``|c| > c_max``, or did not
        converge.
    """
    if guess not in ("data", "truth"):
        raise ValueError(f"guess must be 'data' or 'truth', not {guess!r}")

    if indexes is None:
        indexes = data.index.get_level_values(0).unique()
    indexes = list(indexes)

    # Model tables (cached)
    src = get_salt_source(source, version)
    truth = targets.loc[indexes]
    bands = sorted(data.loc[indexes]["band"].unique())
    tables = get_tables(bands, src, truth["z"].min(), truth["z"].max(), c_max=c_max, dz=dz,
                        modeldir=modeldir)
    grid, nbands = phase_grids(tables), len(tables["bands"])
    if verbose:
        print(f"tables: {len(tables['zgrid'])} redshifts, {tables['nc']} color terms")

    with enable_x64(True):
        # Data
        packed = pack_lightcurves(data, targets, tables, indexes=indexes, phase_range=phase_range,
                                  effects=effects, effect_names=effect_names,
                                  effect_frames=effect_frames, mwebv_key=mwebv_key,
                                  mw_r_v=mw_r_v, time_key=time_key)
        fittable = select_fittable(packed, minsnr=minsnr)

        # Fit, by batches
        theta_in, scan = get_initial_theta(packed, truth, guess=guess)
        batches = get_batches(packed, theta_in, fittable, batch_points=batch_points)
        out = fit_batches(batches, grid, nbands, scan=scan, modelcov=modelcov, nrefit=nrefit,
                          progress_bar=progress_bar, verbose=verbose)

    if not out:
        return pandas.DataFrame(columns=PARAMS)

    results = get_results_table(out, packed, indexes, index_name=targets.index.name)
    check_results(results, c_max=c_max, modelcov=modelcov, nrefit=nrefit)
    return results


# ================== #
#                    #
#   Steps            #
#                    #
# ================== #
def select_fittable(packed, minsnr=5.):
    """Select the targets that can be fitted.

    A target is fitted if it has at least one point with S/N >= `minsnr`, and
    more points than parameters (4).

    Parameters
    ----------
    packed : dict
        Output of :func:`saltjax.data.pack_lightcurves`.

    minsnr : float, optional
        Minimum S/N of the best point. The default is 5.

    Returns
    -------
    numpy.ndarray
        (ntargets,) True for the targets to fit.

    Warns
    -----
    UserWarning
        If targets are not fitted (and why).
    """
    npoints = packed["mask"].sum(1).astype(int)
    detected = np.any((packed["flux"] / packed["err"] >= minsnr) & (packed["mask"] > 0), axis=1)
    constrained = npoints > len(PARAMS)
    if not np.all(detected):
        warnings.warn(f"{np.sum(~detected)} targets without data or detection are not fitted")
    if np.any(detected & ~constrained):
        warnings.warn(f"{np.sum(detected & ~constrained)} targets with <= {len(PARAMS)} points "
                      "(no degree of freedom) are not fitted")

    return detected & constrained


def get_initial_theta(packed, truth, guess="data"):
    """Get the starting (t0, x0, x1, c) of each target, and the scan of the initial guess.

    The returned values are refined by :func:`saltjax.fitter.initial_batch`,
    which scans t0 (and c) and solves x0 linearly.

    Parameters
    ----------
    packed : dict
        Output of :func:`saltjax.data.pack_lightcurves`.

    truth : pandas.DataFrame
        Targets (same order as `packed`), with 't0', 'x1' and 'c' columns if
        ``guess='truth'``.

    guess : {'data', 'truth'}, optional
        'data': t0 of the highest S/N point, x1 = c = 0, and a scan of t0
        (within +-25 days, 1-day steps) and c (`C_VALUES`). 'truth': the
        targets' t0, x1 and c (no scan). The default is 'data'.

    Returns
    -------
    theta_in : numpy.ndarray
        (ntargets, 4) starting (t0, x0=1, x1, c).

    scan : dict
        Scan of the initial guess: ``t0_offsets`` and ``c_values`` (keyword
        arguments of :func:`saltjax.fitter.initial_batch`).

    Raises
    ------
    ValueError
        If `guess` is not 'data' or 'truth'.
    """
    if guess == "truth":
        t0_x1_c = truth[["t0", "x1", "c"]].to_numpy(float)
        scan = dict(t0_offsets=(0.,))
    elif guess == "data":
        # t0: time of the highest S/N point
        snr = np.where(packed["mask"] > 0, packed["flux"] / packed["err"], -np.inf)
        t0_best = packed["t"][np.arange(len(snr)), np.argmax(snr, axis=1)]
        t0_x1_c = np.column_stack([t0_best, np.zeros(len(snr)), np.zeros(len(snr))])
        scan = dict(t0_offsets=T0_OFFSETS, c_values=C_VALUES)
    else:
        raise ValueError(f"guess must be 'data' or 'truth', not {guess!r}")

    theta_in = np.column_stack([t0_x1_c[:, 0], np.ones(len(t0_x1_c)), t0_x1_c[:, 1], t0_x1_c[:, 2]])
    return theta_in, scan


def get_bucket_length(npoints, base=LENGTH_BASE, minimum=LENGTH_MIN):
    """Get the padded lightcurve length of a target: smallest ``minimum * base**k >= npoints``.

    Parameters
    ----------
    npoints : int
        Number of points of the lightcurve.

    base : int, optional
        Growth factor of the buckets. The default is `LENGTH_BASE` (2).

    minimum : int, optional
        Smallest bucket. The default is `LENGTH_MIN` (16).

    Returns
    -------
    int
        Padded length.
    """
    length = minimum
    while length < npoints:
        length *= base
    return length


def get_batches(packed, theta_in, fittable, batch_points=2048):
    """Group the targets in batches of fixed shapes.

    Targets are bucketed by number of points (:func:`get_bucket_length`);
    a bucket of length L is cut into batches of ``max(1, batch_points // L)``
    targets, the last one being padded (repeating its first target). The
    batch shapes thus only depend on the bucket, so that compiled functions
    are reused by any later call.

    Parameters
    ----------
    packed : dict
        Output of :func:`saltjax.data.pack_lightcurves`.

    theta_in : numpy.ndarray
        (ntargets, 4) starting parameters (see :func:`get_initial_theta`).

    fittable : numpy.ndarray
        (ntargets,) True for the targets to fit (see :func:`select_fittable`).

    batch_points : int, optional
        Number of lightcurve points fitted together. The default is 2048.

    Returns
    -------
    list of tuple
        One ``(length, sel, packed_batch, theta_batch)`` per batch: the bucket
        length, the rows of the (non-padding) targets, their packed arrays
        (jax, cut to `length` points) and starting parameters (jax).
    """
    npoints = packed["mask"].sum(1).astype(int)
    length = np.array([get_bucket_length(n_) for n_ in npoints])

    batches = []
    for length_ in np.unique(length[fittable]):
        members = np.flatnonzero(fittable & (length == length_))
        nbatch = max(1, int(batch_points) // int(length_))         # targets per batch
        for start in range(0, len(members), nbatch):
            sel = members[start:start + nbatch]
            padded = np.r_[sel, np.repeat(sel[:1], nbatch - len(sel))]   # fixed batch size
            packed_batch = {key_: jnp.asarray(packed[key_][padded][:, :length_] if key_ in LC_KEYS
                                              else packed[key_][padded])
                            for key_ in FIT_KEYS}
            batches.append((length_, sel, packed_batch, jnp.asarray(theta_in[padded])))

    return batches


def fit_batches(batches, grid, nbands, scan, modelcov=True, nrefit=10, progress_bar=False, verbose=False):
    """Compile (in parallel, once per shape) and run the fit of each batch.

    Parameters
    ----------
    batches : list of tuple
        Output of :func:`get_batches`.

    grid : tuple
        Output of :func:`saltjax.model.phase_grids`.

    nbands : int
        Number of bands of the tables.

    scan : dict
        Scan of the initial guess (see :func:`get_initial_theta`).

    modelcov : bool, optional
        Include the model covariance. The default is True.

    nrefit : int, optional
        Maximum number of refits with model covariance. The default is 10.

    progress_bar : bool, optional
        Display a progress bar (requires tqdm). The default is False.

    verbose : bool, optional
        Print progress information. The default is False.

    Returns
    -------
    dict
        Concatenated outputs of :func:`saltjax.fitter.fit_batch` (theta, cov,
        chi2, ndof, converged, nrefit) and 'pos', the row of each fitted
        target in the packed arrays. Empty if there is no batch.
    """
    init_static = dict(grid=grid, **scan)
    fit_static = dict(grid=grid, nbands=nbands, modelcov=modelcov, nrefit=nrefit)

    pbar = None
    if progress_bar:
        from tqdm.auto import tqdm
        pbar = tqdm(total=int(sum(len(batch_[1]) for batch_ in batches)), unit="SN")
        pbar.set_description("compiling")

    # compile every needed shape first, in parallel
    compile_all([(initial_batch, (packed_, theta_), init_static) for _, _, packed_, theta_ in batches] +
                [(fit_batch, (theta_, packed_), fit_static) for _, _, packed_, theta_ in batches])
    if pbar is not None:
        pbar.set_description("fitting")

    out = {}
    for length_, sel_, packed_, theta_ in batches:
        theta0 = get_compiled(initial_batch, (packed_, theta_), init_static)(packed_, theta_)
        result = get_compiled(fit_batch, (theta0, packed_), fit_static)(theta0, packed_)
        # drop the padding targets
        result = {key_: np.asarray(value_)[:len(sel_)] for key_, value_ in jax.block_until_ready(result).items()}
        for key_, value_ in result.items():
            out.setdefault(key_, []).append(value_)
        out.setdefault("pos", []).append(sel_)

        if pbar is not None:
            pbar.update(len(sel_))
        if verbose:
            print(f"fitted {len(sel_)} targets with <= {length_} points")

    if pbar is not None:
        pbar.close()

    return {key_: np.concatenate(values_) for key_, values_ in out.items()}


def get_results_table(out, packed, indexes, index_name=None):
    """Gather the fit outputs in a table, one row per fitted target.

    Parameters
    ----------
    out : dict
        Output of :func:`fit_batches`.

    packed : dict
        Output of :func:`saltjax.data.pack_lightcurves`.

    indexes : list
        Packed targets (rows of `packed`).

    index_name : str or None, optional
        Name of the index of the table. The default is None.

    Returns
    -------
    pandas.DataFrame
        See :func:`fit_salt`. Rows follow the order of `indexes`.
    """
    out = dict(out)
    pos = out.pop("pos")
    results = pandas.DataFrame(out["theta"], columns=PARAMS, index=np.asarray(indexes)[pos])

    # errors and covariance; valid: finite covariance with positive variances
    variances = np.diagonal(out["cov"], axis1=1, axis2=2)
    valid = np.all(np.isfinite(out["cov"]), axis=(1, 2)) & np.all(variances > 0, axis=1)
    for i, param_i in enumerate(PARAMS):
        results[f"{param_i}_err"] = np.sqrt(np.where(variances[:, i] > 0, variances[:, i], np.nan))
        for j, param_j in enumerate(PARAMS):
            results[f"cov_{param_i}{param_j}"] = out["cov"][:, i, j]

    # fixed parameters
    results["z"] = packed["z"][pos]
    for col_ in packed["effects"]:
        results[col_] = packed["effects"][col_].to_numpy()[pos]

    for key_ in ("chi2", "ndof", "converged", "nrefit"):
        results[key_] = out[key_]
    results["valid"] = valid

    results = results.iloc[np.argsort(pos)]      # back to the order of indexes
    results.index.name = index_name
    return results


def check_results(results, c_max=1.5, modelcov=True, nrefit=10):
    """Warn about invalid, out-of-range or non-converged fits.

    Parameters
    ----------
    results : pandas.DataFrame
        Output of :func:`get_results_table`.

    c_max : float, optional
        Largest absolute color for which the model is exact. The default is 1.5.

    modelcov : bool, optional
        Whether the model covariance was included (convergence is only
        checked then). The default is True.

    nrefit : int, optional
        Maximum number of refits (for the message). The default is 10.

    Warns
    -----
    UserWarning
        For each kind of problem found.
    """
    if not results["valid"].all():
        warnings.warn(f"{(~results['valid']).sum()} fits have an invalid covariance "
                      "(errors set to NaN, see the 'valid' column)")
    if (results["c"].abs() > c_max).any():
        warnings.warn(f"{(results['c'].abs() > c_max).sum()} fits have |c| > c_max={c_max}: "
                      "the model is not exact there, increase c_max.")
    if modelcov and not results["converged"].all():
        warnings.warn(f"{(~results['converged']).sum()} targets did not converge in {nrefit} refits")
