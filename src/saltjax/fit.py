"""Batched SALT2 fit of many lightcurves."""

import warnings

import numpy as np
import pandas
import jax
import jax.numpy as jnp

from .tables import get_salt2_source, get_tables
from .data import LC_KEYS, FIT_KEYS, pack_lightcurves
from .model import phase_grids
from .fitter import fit_batch, initial_batch, compile_all, get_compiled

__all__ = ["fit_salt"]

PARAMS = ["t0", "x0", "x1", "c"]
LENGTH_BASE = 2   # growth factor of the lightcurve-length buckets


def _next_pow(n, base=2, minimum=1):
    """Smallest minimum * base**k >= n."""
    size = minimum
    while size < n:
        size *= base
    return size


def fit_salt(data, targets, indexes=None, modelcov=True, phase_range=[-10, 40],
             guess="data", source="salt2", version=None, mwebv_key="mwebv", mw_r_v=3.1,
             c_max=1.5, dz=2.5e-4, batch_points=2048, minsnr=5., nrefit=10,
             time_key=None, progress_bar=False, verbose=False):
    """Fit SALT2 on many lightcurves at once, reproducing ``sncosmo.fit_lc``.

    Same model, same chi2 and same model-covariance procedure as
    ``sncosmo.fit_lc``; parameters (t0, x0, x1, c) are not bounded and the
    redshift is fixed.

    Targets are grouped by number of lightcurve points (padded to 16, 32, 64,
    ... points) and fitted by batches whose shape only depends on that number.
    The model tables are cached per (bands, redshift range rounded to 0.01,
    E(B-V) range rounded to 0.1), and the compiled fit is reused by any later
    call using the same tables. The first call of a session includes a few
    seconds of tabulation and (parallel) compilation; such later calls do not.
    Prefer one call over many small ones.

    Parameters
    ----------
    data : pandas.DataFrame
        Lightcurves, with a (target, observation) multi-index and columns
        time (see `time_key`), 'band', 'flux', 'fluxerr', 'zp' and optionally
        'zpsys' (only 'ab' is supported).

    targets : pandas.DataFrame
        Targets, indexed as the first level of `data`, with a 'z' column; a
        't0' column if `phase_range` is given; 't0', 'x1' and 'c' columns if
        ``guess='truth'``; and a `mwebv_key` column for the Milky Way dust.

    indexes : list or None, optional
        Targets to fit. If None, all the targets with data. The default is None.

    modelcov : bool, optional
        Include the SALT2 model covariance in the chi2, as in
        ``sncosmo.fit_lc(..., modelcov=True)``. The default is True.

    phase_range : list or None, optional
        Rest-frame phase range (relative to the targets' t0) of the data used.
        The default is [-10, 40].

    guess : {'data', 'truth'}, optional
        Initial guess. 'data': x1 = 0 and the (t0, c) with the lowest chi2 on a
        grid (t0 within +-25 days of the highest S/N point in 1-day steps, c in
        {-0.2, 0, 0.2, 0.4}); 'truth': the targets' t0, x1 and c. In both cases
        x0 is solved linearly. The default is 'data'.

    source : str or sncosmo.SALT2Source, optional
        SALT2 source. The default is 'salt2'.

    version : str or None, optional
        Version of the source, if given by name. The default is None (latest).

    mwebv_key : str or None, optional
        Column of `targets` with the Milky Way E(B-V) (CCM89 dust in the
        observer frame, as ``sncosmo.CCM89Dust``). If None or absent, no
        Milky Way dust. The default is 'mwebv'.

    mw_r_v : float, optional
        R_V of the Milky Way dust. The default is 3.1.

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
        ``_err``, ``cov_{p}{q}``, the fixed ``mwebv`` and ``mwr_v`` (if Milky Way
        dust), ``chi2``, ``ndof``, ``converged``, ``nrefit`` and ``valid``
        (False if the covariance is not positive: errors are then NaN).

    Notes
    -----
    As ``sncosmo.fit_lc``, points whose band is out of the model wavelength
    range are dropped (with a warning). Targets without any point with S/N >=
    `minsnr`, or with no more points than parameters, are not fitted.

    Raises
    ------
    NotImplementedError
        If the source is not a SALT2 source, or a magnitude system other than
        'ab' is used.

    ValueError
        If `guess` is not 'data' or 'truth'.
    """
    if guess not in ("data", "truth"):
        raise ValueError(f"guess must be 'data' or 'truth', not {guess!r}")
    if indexes is None:
        indexes = data.index.get_level_values(0).unique()
    indexes = list(indexes)
    src = get_salt2_source(source, version)
    truth = targets.loc[indexes]
    has_mw = mwebv_key is not None and mwebv_key in truth
    ebv = truth[mwebv_key].to_numpy(float) if has_mw else np.zeros(len(truth))
    bands = sorted(data.loc[indexes]["band"].unique())
    tables = get_tables(bands, src, truth["z"].min(), truth["z"].max(),
                        ebv_max=ebv.max() if has_mw else 0., r_v=mw_r_v, c_max=c_max, dz=dz)
    grid, nbands = phase_grids(tables), len(tables["bands"])
    if verbose:
        print(f"tables: {len(tables['zgrid'])} redshifts, {tables['nc']} color "
              f"and {tables['nm']} dust terms")

    with jax.enable_x64(True):
        arr = pack_lightcurves(data, targets, tables, indexes=indexes, phase_range=phase_range,
                               mwebv_key=mwebv_key if has_mw else None, time_key=time_key)
        npts = arr["mask"].sum(1).astype(int)
        detected = np.any((arr["flux"] / arr["err"] >= minsnr) & (arr["mask"] > 0), axis=1)
        constrained = npts > len(PARAMS)
        good = detected & constrained
        if not np.all(detected):
            warnings.warn(f"{np.sum(~detected)} targets without data or detection are not fitted")
        if np.any(detected & ~constrained):
            warnings.warn(f"{np.sum(detected & ~constrained)} targets with <= {len(PARAMS)} points "
                          "(no degree of freedom) are not fitted")

        # starting point
        if guess == "truth":
            th_in = truth[["t0", "x1", "c"]].to_numpy(float)
            scan = dict(t0_offsets=(0.,))
        else:
            snr = np.where(arr["mask"] > 0, arr["flux"] / arr["err"], -np.inf)
            th_in = np.column_stack([arr["t"][np.arange(len(snr)), np.argmax(snr, axis=1)],
                                     np.zeros(len(snr)), np.zeros(len(snr))])
            scan = dict(t0_offsets=tuple(np.arange(-25., 25.5, 1.)), c_values=(-0.2, 0., 0.2, 0.4))
        th_in = np.column_stack([th_in[:, 0], np.ones(len(th_in)), th_in[:, 1], th_in[:, 2]])

        # bucket by number of points; batches of a bucket are padded to a size that
        # only depends on the bucket, so that compilations are reused by any call.
        length = np.array([_next_pow(n, LENGTH_BASE, 16) for n in npts])
        batches = []
        for L in np.unique(length[good]):
            members = np.flatnonzero(good & (length == L))
            nbatch = max(1, int(batch_points) // int(L))
            for start in range(0, len(members), nbatch):
                sel = members[start:start + nbatch]
                padded = np.r_[sel, np.repeat(sel[:1], nbatch - len(sel))]
                d = {k: jnp.asarray(arr[k][padded][:, :L] if k in LC_KEYS else arr[k][padded])
                     for k in FIT_KEYS}
                batches.append((L, sel, d, jnp.asarray(th_in[padded])))

        # compile every needed shape first, in parallel
        init_static = dict(grid=grid, **scan)
        fit_static = dict(grid=grid, nbands=nbands, modelcov=modelcov, nrefit=nrefit)
        pbar = None
        if progress_bar:
            from tqdm.auto import tqdm
            pbar = tqdm(total=int(sum(len(b[1]) for b in batches)), unit="SN")
            pbar.set_description("compiling")
        compile_all([(initial_batch, (d, th), init_static) for _, _, d, th in batches] +
                    [(fit_batch, (th, d), fit_static) for _, _, d, th in batches])
        if pbar is not None:
            pbar.set_description("fitting")

        out = {}
        for L, sel, d, th in batches:
            theta0 = get_compiled(initial_batch, (d, th), init_static)(d, th)
            res = get_compiled(fit_batch, (theta0, d), fit_static)(theta0, d)
            res = {k: np.asarray(v)[:len(sel)] for k, v in jax.block_until_ready(res).items()}
            for k, v in res.items():
                out.setdefault(k, []).append(v)
            out.setdefault("pos", []).append(sel)
            if pbar is not None:
                pbar.update(len(sel))
            if verbose:
                print(f"fitted {len(sel)} targets with <= {L} points")
        if pbar is not None:
            pbar.close()

    if not out:
        return pandas.DataFrame(columns=PARAMS)
    out = {k: np.concatenate(v) for k, v in out.items()}
    pos = out.pop("pos")
    results = pandas.DataFrame(out["theta"], columns=PARAMS, index=np.asarray(indexes)[pos])
    variances = np.diagonal(out["cov"], axis1=1, axis2=2)
    valid = np.all(np.isfinite(out["cov"]), axis=(1, 2)) & np.all(variances > 0, axis=1)
    for i, p in enumerate(PARAMS):
        results[f"{p}_err"] = np.sqrt(np.where(variances[:, i] > 0, variances[:, i], np.nan))
        for j, q in enumerate(PARAMS):
            results[f"cov_{p}{q}"] = out["cov"][:, i, j]
    results["z"] = arr["z"][pos]
    if has_mw:
        results["mwebv"] = arr["ebv"][pos]
        results["mwr_v"] = mw_r_v
    for k in ("chi2", "ndof", "converged", "nrefit"):
        results[k] = out[k]
    results["valid"] = valid
    results = results.iloc[np.argsort(pos)]
    results.index.name = targets.index.name
    if not results["valid"].all():
        warnings.warn(f"{(~results['valid']).sum()} fits have an invalid covariance "
                      "(errors set to NaN, see the 'valid' column)")
    if (results["c"].abs() > c_max).any():
        warnings.warn(f"{(results['c'].abs() > c_max).sum()} fits have |c| > c_max={c_max}: "
                      "the model is not exact there, increase c_max.")
    if modelcov and not results["converged"].all():
        warnings.warn(f"{(~results['converged']).sum()} targets did not converge in {nrefit} refits")
    return results
