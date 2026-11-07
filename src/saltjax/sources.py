"""Fit lightcurves with any supported sncosmo source: :func:`fit_lc`.

:func:`fit_lc` is the entry point of saltjax, as ``sncosmo.fit_lc``: it parses
the source and calls the fitter of its family::

    import saltjax
    results = saltjax.fit_lc(data, targets, source="salt2")    # SALT2 / SALT3
    results = saltjax.fit_lc(data, targets, source="hsiao")    # time series

- SALT2 / SALT3 sources (e.g. 'salt2', 'salt3'): :func:`saltjax.fit.fit_salt`,
  fitting t0, x0, x1 and c.
- time-series sources (e.g. 'hsiao', 'nugent-sn2p'):
  :func:`saltjax.timeseries.fit_timeseries`, fitting t0 and amplitude.

Both take the same lightcurve tables and the same effects (dust).
"""

import inspect

import sncosmo

from .fit import fit_salt
from .timeseries import TimeSeriesTemplate, fit_timeseries

__all__ = ["fit_lc", "get_source_kind"]

# fitter of each source family
FITTERS = {"salt": fit_salt, "timeseries": fit_timeseries}


def get_source_kind(source, version=None):
    """Get the family of a source ('salt' or 'timeseries'), and the source.

    Parameters
    ----------
    source : str or sncosmo.Source or saltjax.timeseries.TimeSeriesTemplate
        Source name registered in sncosmo (e.g. 'salt2', 'salt3', 'hsiao',
        'nugent-sn2p'), or an instance.

    version : str or None, optional
        Version of the source, if given by name. The default is None (latest).

    Returns
    -------
    kind : str
        'salt' (``sncosmo.SALT2Source``, ``sncosmo.SALT3Source``) or
        'timeseries' (``sncosmo.TimeSeriesSource``, or a `TimeSeriesTemplate`).

    source : sncosmo.Source or saltjax.timeseries.TimeSeriesTemplate
        The source instance (loaded by sncosmo if given by name).

    Raises
    ------
    NotImplementedError
        If the source is of another family (e.g. ``sncosmo.SNEMOSource``,
        ``sncosmo.MLCS2k2Source``, ``sncosmo.StretchSource``).
    """
    if isinstance(source, TimeSeriesTemplate):
        return "timeseries", source

    if isinstance(source, str):
        source = sncosmo.get_source(source, version=version)

    # exact types: subclasses may change the model
    if type(source) in (sncosmo.SALT2Source, sncosmo.SALT3Source):
        return "salt", source

    if type(source) is sncosmo.TimeSeriesSource:
        return "timeseries", source

    raise NotImplementedError(f"{type(source).__name__} sources are not supported ({source.name!r}). "
                              "Supported: sncosmo.SALT2Source, sncosmo.SALT3Source (e.g. 'salt2', 'salt3') "
                              "and sncosmo.TimeSeriesSource (e.g. 'hsiao', 'nugent-sn2p').")


def fit_lc(data, targets, source="salt2", version=None, **kwargs):
    """Fit lightcurves with any supported sncosmo source, reproducing ``sncosmo.fit_lc``.

    The source family sets the fitter (see :func:`get_source_kind`):

    - SALT2 / SALT3 (e.g. 'salt2', 'salt3', 'salt3-nir'): :func:`saltjax.fit.fit_salt`,
      fitting t0, x0, x1 and c (with the SALT model covariance).
    - time series (e.g. 'hsiao', 'nugent-sn1a', 'nugent-sn2p', SNANA templates):
      :func:`saltjax.timeseries.fit_timeseries`, fitting t0 and amplitude.

    Parameters
    ----------
    data : pandas.DataFrame
        Lightcurves, with a (target, observation) multi-index and columns
        time ('time', 'mjd' or 'jd'), 'band', 'flux', 'fluxerr', 'zp' and
        optionally 'zpsys' (only 'ab' is supported).

    targets : pandas.DataFrame
        Targets, indexed as the first level of `data`, with a 'z' column and
        the parameters of the effects (e.g. 'mwebv', 'hostebv').

    source : str or sncosmo.Source or saltjax.timeseries.TimeSeriesTemplate, optional
        Source name registered in sncosmo, or an instance. The default is 'salt2'.

    version : str or None, optional
        Version of the source, if given by name. The default is None (latest).

    **kwargs
        Options of the fitter of the source family. Common to both: `indexes`,
        `phase_range`, `guess`, `effects`, `effect_names`, `effect_frames`,
        `mwebv_key`, `mw_r_v`, `batch_points`, `minsnr`, `time_key`,
        `progress_bar`. SALT only: e.g. `modelcov`, `c_max`, `nrefit`.
        Time series only: `t0_offsets`, `niter`. See :func:`saltjax.fit.fit_salt`
        and :func:`saltjax.timeseries.fit_timeseries` (also for the defaults,
        e.g. `phase_range`).

    Returns
    -------
    pandas.DataFrame
        One row per fitted target: the fitted parameters (t0, x0, x1, c for
        SALT; t0, amplitude for time series), their errors and covariance,
        z, the fixed effect parameters, chi2, ndof and valid.

    Raises
    ------
    NotImplementedError
        If the source family is not supported (see :func:`get_source_kind`).

    TypeError
        If an option of `kwargs` is not an option of the fitter of the source
        family (e.g. `modelcov` with a time-series source).

    Examples
    --------
    >>> results = saltjax.fit_lc(data, targets, source="salt3", modelcov=True)  # doctest: +SKIP
    >>> results = saltjax.fit_lc(data, targets, source="nugent-sn2p",
    ...                          effects=[sncosmo.CCM89Dust()], effect_names=["host"],
    ...                          effect_frames=["rest"])  # doctest: +SKIP
    """
    kind, source = get_source_kind(source, version=version)
    fitter = FITTERS[kind]

    # options of the other family: clear error, rather than a silent or cryptic one
    options = inspect.signature(fitter).parameters
    unknown = [key_ for key_ in kwargs if key_ not in options]
    if unknown:
        raise TypeError(f"{unknown} not options for {source.name!r} ({kind} source, fitted by "
                        f"{fitter.__module__}.{fitter.__name__}).")

    return fitter(data, targets, source=source, **kwargs)
