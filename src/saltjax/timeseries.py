"""Fit of time-series templates (``sncosmo.TimeSeriesSource``) with JAX.

Most sncosmo sources are time series: a spectral template F(phase, wave)
scaled by an amplitude (e.g. 'hsiao', 'nugent-sn2p', the SNANA core-collapse
templates). :func:`fit_timeseries` fits (t0, amplitude) on many lightcurves at
once, reproducing ``sncosmo.fit_lc``::

    import saltjax
    results = saltjax.fit_timeseries(data, targets, source="hsiao")

How it works (as for SALT, see :mod:`saltjax.fit`): sncosmo interpolates the
template with a tensor-product spline (``scipy.interpolate.RectBivariateSpline``),
so the flux is linear in the template and separable in phase and wavelength:

    F(p, wave) = sum_i B_i(p) S_i(wave)

with B_i the phase B-spline basis and S_i a spline in wavelength. The band
integral of each S_i (with the redshift and the effects of the target) is
computed once per target (:func:`get_band_integrals`); at fit time, the band
flux at any phase is ``amplitude x sum_i B_i(p) G_i`` (:func:`timeseries_bandflux`),
exact and fast.

As sncosmo (FITPACK), the template is constant outside its phase range (its
edge values), or 0 before its first phase if the source has ``zero_before``.
"""

import warnings

import numpy as np
import pandas
import sncosmo
import jax
import jax.numpy as jnp
from functools import partial

from .data import (LC_KEYS, check_magsys, get_time_key, get_target_position,
                   select_covered_points, select_phase_range, get_lightcurve_arrays)
from .effects import get_effects, parameter_table, transmission
from .tables import get_band_grids, pad_band_grids
from .fit import get_bucket_length
from .fitter import enable_x64, get_compiled, compile_all

__all__ = ["fit_timeseries", "TimeSeriesTemplate",
           # building blocks
           "get_bspline_weights", "get_band_integrals", "get_timeseries_tables", "pack_timeseries",
           "timeseries_bandflux", "fit_timeseries_one", "fit_timeseries_batch", "initial_timeseries_batch",
           "select_fittable", "get_initial_t0", "get_t0_offsets", "get_timeseries_batches", "fit_timeseries_batches",
           "get_results_table"]

PARAMS = ["t0", "amplitude"]
TS_FIT_KEYS = ("G", "z") + LC_KEYS        # packed arrays used by the fit
T0_STEP = 1.                              # step of the initial t0 scan [rest-frame day]
CHUNK = 64                                # targets per block of the band integrals (limits memory)


# ================== #
#                    #
#   Template         #
#                    #
# ================== #
class TimeSeriesTemplate:
    """A time-series spectral template: the spline of an ``sncosmo.TimeSeriesSource``.

    The template flux is ``F(p, wave) = sum_ij coefs[i, j] B_i(p) C_j(wave)``,
    with B_i (C_j) the phase (wavelength) B-spline basis.

    Instances are usually loaded with :meth:`TimeSeriesTemplate.from_name`
    (e.g. ``"hsiao"``) or :meth:`TimeSeriesTemplate.from_source`.

    Parameters
    ----------
    phase_knots, wave_knots : numpy.ndarray
        Knots of the phase [day] and wavelength [A] splines.

    coefs : numpy.ndarray
        Spline coefficients, shape (nphase_basis, nwave_basis).

    phase_degree, wave_degree : int
        Degrees of the phase and wavelength splines.

    zero_before : bool, optional
        If True, the flux is 0 before the first phase of the template (else,
        it is the value at the first phase). The default is False.

    source : sncosmo.TimeSeriesSource or None, optional
        The sncosmo source, if any (used to plot and to name the template).
        The default is None.
    """

    def __init__(self, phase_knots, wave_knots, coefs, phase_degree, wave_degree,
                 zero_before=False, source=None):
        """Initialize the TimeSeriesTemplate (see the class docstring for parameters)."""
        self._phase_knots = np.asarray(phase_knots, float)
        self._wave_knots = np.asarray(wave_knots, float)
        self._coefs = np.asarray(coefs, float)
        self._phase_degree = int(phase_degree)
        self._wave_degree = int(wave_degree)
        self._zero_before = bool(zero_before)
        self._source = source

    @classmethod
    def from_source(cls, source):
        """Load the template of an ``sncosmo.TimeSeriesSource``.

        Parameters
        ----------
        source : sncosmo.TimeSeriesSource
            The source (its amplitude is ignored: it is fitted).

        Returns
        -------
        TimeSeriesTemplate
            The template.

        Raises
        ------
        NotImplementedError
            If the source is not an ``sncosmo.TimeSeriesSource`` (subclasses,
            which may change the flux, are not supported either).
        """
        if type(source) is not sncosmo.TimeSeriesSource:
            hint = (" Use saltjax.fit_salt for SALT sources."
                    if isinstance(source, (sncosmo.SALT2Source, sncosmo.SALT3Source)) else "")
            raise NotImplementedError("only sncosmo.TimeSeriesSource is supported, "
                                      f"got {type(source).__name__}.{hint}")

        spline = source._model_flux                     # scipy RectBivariateSpline
        phase_knots, wave_knots, coefs = spline.tck
        phase_degree, wave_degree = spline.degrees
        nphase_basis = len(phase_knots) - phase_degree - 1
        coefs = np.asarray(coefs).reshape(nphase_basis, -1)   # (nphase_basis, nwave_basis)
        return cls(phase_knots, wave_knots, coefs, phase_degree, wave_degree,
                   zero_before=source._zero_before, source=source)

    @classmethod
    def from_name(cls, name, version=None):
        """Load the template of a time-series source registered in sncosmo.

        Parameters
        ----------
        name : str
            Source name (e.g. 'hsiao', 'nugent-sn1a', 'nugent-sn2p',
            'snana-2004fe'). Files are downloaded by sncosmo if needed.

        version : str or None, optional
            Version of the source. The default is None (latest).

        Returns
        -------
        TimeSeriesTemplate
            The template.

        Raises
        ------
        NotImplementedError
            If the source is not an ``sncosmo.TimeSeriesSource``.
        """
        return cls.from_source(sncosmo.get_source(name, version=version))

    def __str__(self):
        """Return a human-readable summary of the template."""
        return (f"TimeSeriesTemplate {self.name}: phase {self.phase_range[0]:g} to "
                f"{self.phase_range[1]:g} days (spline degree {self.phase_degree}"
                f"{', zero before' if self.zero_before else ''}), "
                f"wavelength {self.wave_range[0]:g} to {self.wave_range[1]:g} A")

    def __repr__(self):
        """Return a human-readable summary of the template."""
        return self.__str__()

    # --------- #
    #  GETTER   #
    # --------- #
    def get_flux(self, phase, wave):
        """Get the template flux (amplitude = 1), as ``sncosmo.TimeSeriesSource``.

        Parameters
        ----------
        phase : float or array_like
            Rest-frame phases [day].

        wave : float or array_like
            Rest-frame wavelengths [A].

        Returns
        -------
        numpy.ndarray
            Flux [erg/s/cm^2/A], shape (nphase, nwave).
        """
        phase, wave = np.atleast_1d(np.asarray(phase, float)), np.atleast_1d(np.asarray(wave, float))
        phase_idx, phase_weights = get_bspline_weights(phase, self.phase_knots, self.phase_degree)
        wave_idx, wave_weights = get_bspline_weights(wave, self.wave_knots, self.wave_degree)
        # sum_ij coefs[i, j] B_i(p) C_j(wave), with the non-zero basis functions only
        coefs = self.coefs[phase_idx[:, None, :, None], wave_idx[None, :, None, :]]   # (np, nw, kp+1, kw+1)
        flux = np.einsum("pwij,pi,wj->pw", coefs, phase_weights, wave_weights)
        if self.zero_before:
            flux[phase < self.phase_range[0]] = 0.

        return flux

    def get_peak_phase(self):
        """Get the phase of the maximum of the template (summed over wavelengths).

        Returns
        -------
        float
            Phase [day] of the maximum of the template flux.
        """
        phase = np.linspace(*self.phase_range, 500)
        wave = np.linspace(*self.wave_range, 200)
        return float(phase[np.argmax(self.get_flux(phase, wave).sum(1))])

    def get_sncosmo_model(self, effects=None, effect_names=None, effect_frames=None, mwebv=None, **params):
        """Get the ``sncosmo.Model`` of this template, with effects (to compare or plot).

        Parameters
        ----------
        effects, effect_names, effect_frames : list or None, optional
            Propagation effects, as in ``sncosmo.Model``. If None, Milky Way
            CCM89 dust (named 'mw', observer frame) if `mwebv` is given, as
            :func:`fit_timeseries`. The default is None.

        mwebv : float or None, optional
            Milky Way E(B-V). Only used with the default effect. The default is None.

        **params
            Model parameters, e.g. ``z``, ``t0``, ``amplitude``, ``hostebv``.

        Returns
        -------
        sncosmo.Model
            The model.

        Raises
        ------
        ValueError
            If the template was not built from an sncosmo source, or a
            parameter is not in the model.
        """
        if self.source is None:
            raise ValueError("this template has no sncosmo source (built from its spline only)")

        if effects is None:
            if mwebv is not None:          # default of fit_timeseries: Milky Way dust
                effects, effect_names, effect_frames = [sncosmo.CCM89Dust()], ["mw"], ["obs"]
                params = params | {"mwebv": mwebv}
            else:
                effects, effect_names, effect_frames = [], [], []
        elif mwebv is not None:
            params = params | {"mwebv": mwebv}

        model = sncosmo.Model(self.source, effects=effects, effect_names=effect_names,
                              effect_frames=effect_frames)
        unknown = [key_ for key_ in params if key_ not in model.param_names]
        if unknown:
            raise ValueError(f"parameters not in the model: {unknown}, model parameters: {model.param_names}")

        model.set(**params)
        return model

    # --------- #
    #  Plotting #
    # --------- #
    def show(self, ax=None, phases=None, **kwargs):  # pragma: no cover
        """Show the template spectrum at a few phases.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, optional
            Axes to plot on. If None, a new figure is created. The default is None.

        phases : list of float or None, optional
            Phases [day] to show. If None, 5 phases spanning the template.
            The default is None.

        **kwargs
            Passed to `matplotlib.axes.Axes.plot`.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot.
        """
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=[7, 3.5])
        else:
            fig = ax.figure

        if phases is None:
            phases = np.linspace(*self.phase_range, 7)[1:-1]

        wave = np.linspace(*self.wave_range, 1000)
        flux = self.get_flux(phases, wave)                 # (nphase, nwave)
        for phase_, flux_ in zip(phases, flux):
            ax.plot(wave, flux_, label=f"{phase_:+.0f} d", **kwargs)

        ax.set_xlabel("Rest-frame wavelength [Å]", fontsize="large")
        ax.set_ylabel("Flux (amplitude = 1)", fontsize="large")
        ax.set_title(self.name, fontsize="medium")
        ax.legend(fontsize="small", frameon=False)
        return fig

    def show_fit(self, data, results, index, effects=None, effect_names=None, effect_frames=None,
                 ax=None, time_key=None, zp=25.):  # pragma: no cover
        """Show the lightcurve of a target and its best-fit model.

        The model is the ``sncosmo.Model`` of this template (see
        :meth:`get_sncosmo_model`) with the effects used in the fit, and the
        parameters of the target in `results` (z, t0, amplitude and the
        effect parameters, e.g. 'mwebv', 'hostebv').

        Parameters
        ----------
        data : pandas.DataFrame
            Lightcurves (as given to :func:`fit_timeseries`).

        results : pandas.DataFrame
            Output of :func:`fit_timeseries`.

        index : object
            Target to show (index of `results`).

        effects, effect_names, effect_frames : list or None, optional
            The effects given to :func:`fit_timeseries`. If None, Milky Way
            CCM89 dust if `results` has a 'mwebv' column (the default of
            :func:`fit_timeseries`). The default is None.

        ax : matplotlib.axes.Axes or None, optional
            Axes to plot on. If None, a new figure is created. The default is None.

        time_key : str or None, optional
            Time column (None: 'time', 'mjd' or 'jd'). The default is None.

        zp : float, optional
            Zero point of the plotted fluxes. The default is 25.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot.

        Warns
        -----
        UserWarning
            If `results` has effect parameters that are not in the model (the
            effects of the fit were not given): the model is then not the fitted one.
        """
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=[7, 4])
        else:
            fig = ax.figure

        # Model: best-fit parameters and effect parameters of the target
        row = results.loc[index]
        params = {key_: row[key_] for key_ in row.index if not key_.endswith("_err")
                  and not key_.startswith("cov_") and key_ not in ("chi2", "ndof", "valid")}
        model = self.get_sncosmo_model(effects=effects, effect_names=effect_names, effect_frames=effect_frames,
                                       mwebv=params.get("mwebv") if effects is None else None)
        missing = [key_ for key_ in params if key_ not in model.param_names]
        if missing:
            warnings.warn(f"results have parameters not in the model ({missing}): give the effects "
                          "used in the fit (effects, effect_names, effect_frames) to show the fitted model.")
        model.set(**{key_: value_ for key_, value_ in params.items() if key_ in model.param_names})

        # Data and model, per band
        lightcurve = data.xs(index)
        time_key = get_time_key(lightcurve, time_key)
        times = np.linspace(lightcurve[time_key].min() - 10, lightcurve[time_key].max() + 10, 300)
        for i, band_ in enumerate(sorted(lightcurve["band"].unique())):
            color = f"C{i}"
            points = lightcurve[lightcurve["band"] == band_]
            coef = 10 ** (-0.4 * (points["zp"] - zp))          # to the plotted zero point
            ax.errorbar(points[time_key], points["flux"] * coef, yerr=points["fluxerr"] * coef,
                        ls="None", marker="o", ms=4, color=color, label=band_)
            ax.plot(times, model.bandflux(band_, times, zp=zp, zpsys="ab"), color=color, lw=1.5)

        ax.set_xlabel("Time [day]", fontsize="large")
        ax.set_ylabel(f"Flux (zp={zp:g}, AB)", fontsize="large")
        ax.set_title(f"{self.name}, target {index}: chi2/ndof = {row['chi2']:.1f}/{row['ndof']:.0f}",
                     fontsize="medium")
        ax.legend(fontsize="small", frameon=False)
        return fig

    # ============= #
    #   Properties  #
    # ============= #
    @property
    def name(self):
        """Name of the sncosmo source (None if unknown)."""
        return None if self.source is None else self.source.name

    @property
    def source(self):
        """The ``sncosmo.TimeSeriesSource`` (None if built from a spline)."""
        return self._source

    @property
    def phase_knots(self):
        """Knots of the phase spline [day]."""
        return self._phase_knots

    @property
    def wave_knots(self):
        """Knots of the wavelength spline [A]."""
        return self._wave_knots

    @property
    def coefs(self):
        """Spline coefficients, shape (nphase_basis, nwave_basis)."""
        return self._coefs

    @property
    def phase_degree(self):
        """Degree of the phase spline (sncosmo's ``time_spline_degree``)."""
        return self._phase_degree

    @property
    def wave_degree(self):
        """Degree of the wavelength spline."""
        return self._wave_degree

    @property
    def zero_before(self):
        """Whether the flux is 0 before the first phase of the template."""
        return self._zero_before

    @property
    def phase_range(self):
        """Phase range of the template [day]."""
        return (self.phase_knots[self.phase_degree], self.phase_knots[-self.phase_degree - 1])

    @property
    def wave_range(self):
        """Wavelength range of the template [A]."""
        return (self.wave_knots[self.wave_degree], self.wave_knots[-self.wave_degree - 1])

    @property
    def phase_spline(self):
        """Hashable description of the phase spline (static for the JIT).

        (phase knots, degree, zero_before): see :func:`timeseries_bandflux`.
        """
        return (tuple(float(knot_) for knot_ in self.phase_knots), self.phase_degree, self.zero_before)


# ================== #
#                    #
#   B-splines        #
#                    #
# ================== #
def get_bspline_weights(x, knots, degree, xp=np):
    """Get the non-zero B-spline basis functions at given points.

    A spline ``s(x) = sum_i c_i B_i(x)`` is then ``sum(c[idx] * weights, -1)``.
    As FITPACK (``scipy.interpolate.RectBivariateSpline``, used by sncosmo),
    points outside the spline domain are clamped to its edges.

    Parameters
    ----------
    x : array_like
        Points.

    knots : array_like
        Knots of the spline.

    degree : int
        Degree of the spline.

    xp : module, optional
        Array module (``numpy`` or ``jax.numpy``). The default is numpy.

    Returns
    -------
    idx : array_like
        Indices of the ``degree + 1`` non-zero basis functions at each point,
        shape ``x.shape + (degree + 1,)``.

    weights : array_like
        Their values, same shape (they sum to 1).
    """
    knots = xp.asarray(knots)
    nbasis = len(knots) - degree - 1
    x = xp.clip(x, knots[degree], knots[nbasis])             # clamped to the domain (as FITPACK)
    # interval [knots[l], knots[l+1]) of each point: basis functions l-degree ... l are non-zero
    interval = xp.clip(xp.searchsorted(knots, x, side="right") - 1, degree, nbasis - 1)

    # de Boor's recursion, one degree at a time (as scipy's evaluate_all_bspl)
    values = [xp.ones_like(x)]
    for j in range(1, degree + 1):
        previous, values = values, [xp.zeros_like(x) for _ in range(j + 1)]
        for n in range(1, j + 1):
            right, left = knots[interval + n], knots[interval + n - j]
            span = right - left
            w = xp.where(span > 0, previous[n - 1] / xp.where(span > 0, span, 1.), 0.)
            values[n - 1] = values[n - 1] + w * (right - x)
            values[n] = w * (x - left)

    idx = interval[..., None] + xp.arange(-degree, 1)
    return idx, xp.stack(values, -1)


# ================== #
#                    #
#   Band integrals   #
#                    #
# ================== #
def get_timeseries_tables(template, bands, spacing=5.):
    """Get the band integration grids of the template and its wavelength range.

    Parameters
    ----------
    template : TimeSeriesTemplate
        The template.

    bands : list of str
        Names of the (sncosmo-registered) bandpasses.

    spacing : float, optional
        Wavelength step of the band integration [A] (sncosmo's
        ``MODEL_BANDFLUX_SPACING``). The default is 5.

    Returns
    -------
    dict
        'bands', 'band_wave', 'band_base' (integration weights), 'band_range'
        and 'wave_range' (rest frame), as :func:`saltjax.tables.build_tables`
        (so that :func:`saltjax.data.select_covered_points` can use it).
    """
    band_wave, band_base, _, band_range = pad_band_grids(get_band_grids(bands, spacing=spacing))
    return dict(bands=list(bands), band_wave=band_wave, band_base=band_base, band_range=band_range,
                wave_range=template.wave_range)


def get_band_integrals(template, tables, z, effects=()):
    """Get the band integrals of each phase basis function, for each target.

    For a target at redshift z (a = 1/(1+z)), in band b, this computes (as
    ``sncosmo.Model.bandflux``, on the same wavelength grid)::

        G[b, i] = sum_lambda a S_i(a lambda) T_b(lambda) lambda dlambda / (hc zp_b) x E(lambda)

    with S_i the wavelength spline of phase basis function i and E the
    transmission of the effects. The band flux at phase p is then
    ``amplitude x sum_i B_i(p) G[b, i]``.

    Targets are processed by blocks of `CHUNK` to limit memory.

    Parameters
    ----------
    template : TimeSeriesTemplate
        The template.

    tables : dict
        Output of :func:`get_timeseries_tables`.

    z : numpy.ndarray
        Redshifts of the targets.

    effects : list of dict, optional
        Effects and their parameters for these targets (output of
        :func:`saltjax.effects.get_effects`). The default is no effect.

    Returns
    -------
    numpy.ndarray
        Band integrals, shape (ntargets, nbands, nphase_basis).
    """
    z = np.asarray(z, float)
    nbands, nphase_basis = len(tables["bands"]), len(template.coefs)
    blocks = [np.zeros((0, nbands, nphase_basis))]
    for start in range(0, len(z), CHUNK):
        sel = slice(start, start + CHUNK)
        effects_ = [effect_ | dict(params={name_: values_[sel] for name_, values_ in effect_["params"].items()})
                    for effect_ in effects]
        blocks.append(_get_block_band_integrals_(template, tables, z[sel], effects_))

    return np.concatenate(blocks)


def _get_block_band_integrals_(template, tables, z, effects):
    """Band integrals of a block of targets (see :func:`get_band_integrals`)."""
    wave = tables["band_wave"]                              # (nbands, nlambda) observer frame [A]
    a = 1. / (1. + z)                                       # (ntargets,)
    rest_wave = a[:, None, None] * wave[None]               # (ntargets, nbands, nlambda)

    # integration weight of each wavelength: a x base x transmission of the effects
    trans = transmission(effects, wave.ravel(), a).reshape(rest_wave.shape)
    weight = a[:, None, None] * tables["band_base"][None] * trans

    # H[t, b, j] = sum_lambda weight C_j(rest wave): only degree+1 basis functions per wavelength
    idx, values = get_bspline_weights(rest_wave, template.wave_knots, template.wave_degree)
    ntargets, nbands, _ = rest_wave.shape
    nwave_basis = template.coefs.shape[1]
    flat_index = ((np.arange(ntargets)[:, None, None, None] * nbands +
                   np.arange(nbands)[None, :, None, None]) * nwave_basis + idx)
    H = np.bincount(flat_index.ravel(), weights=(weight[..., None] * values).ravel(),
                    minlength=ntargets * nbands * nwave_basis).reshape(ntargets, nbands, nwave_basis)

    # G[t, b, i] = sum_j coefs[i, j] H[t, b, j]
    return H @ template.coefs.T


# ================== #
#                    #
#   Lightcurves      #
#                    #
# ================== #
def pack_timeseries(data, targets, template, indexes=None, phase_range=None,
                    effects=None, effect_names=None, effect_frames=None,
                    mwebv_key="mwebv", mw_r_v=3.1, time_key=None, pad_to=16):
    """Pack the lightcurves and the per-target band integrals into padded arrays.

    As :func:`saltjax.data.pack_lightcurves`, for a time-series template.

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

    template : TimeSeriesTemplate
        The template.

    indexes : list or None, optional
        Targets to pack. If None, all those of `targets`. The default is None.

    phase_range : list or None, optional
        Rest-frame phase range (relative to the targets' t0) of the data used.
        If None, all points. The default is None.

    effects, effect_names, effect_frames : list or None, optional
        Propagation effects, as in ``sncosmo.Model`` (see
        :func:`saltjax.effects.get_effects`). The default is None (Milky Way
        dust from `mwebv_key` and `mw_r_v`).

    mwebv_key : str or None, optional
        Only if `effects` is None: column of `targets` with the Milky Way
        E(B-V). The default is 'mwebv'.

    mw_r_v : float, optional
        Only if `effects` is None: R_V of the Milky Way dust. The default is 3.1.

    time_key : str or None, optional
        Time column (None: 'time', 'mjd' or 'jd'). The default is None.

    pad_to : int, optional
        The number of points per lightcurve is padded to a multiple of
        `pad_to`. The default is 16.

    Returns
    -------
    dict
        Arrays used by the fit (one row per target, see `TS_FIT_KEYS`): the
        lightcurves (`LC_KEYS`), the band integrals 'G' (ntargets, nbands,
        nphase_basis) and 'z'; plus 'effects' (DataFrame of the effect
        parameters) and 'index' (the packed targets).

    Raises
    ------
    NotImplementedError
        If a magnitude system other than 'ab' is used.

    Warns
    -----
    UserWarning
        If points are dropped because their band is out of the template
        wavelength range.
    """
    indexes = list(targets.index if indexes is None else indexes)
    truth = targets.loc[indexes]
    z = truth["z"].to_numpy(float)
    tables = get_timeseries_tables(template, sorted(data.loc[indexes]["band"].unique()))

    # Model: effects and band integrals of each target
    effects = get_effects(truth, effects, effect_names, effect_frames,
                          mwebv_key=mwebv_key, mw_r_v=mw_r_v)
    G = get_band_integrals(template, tables, z, effects)

    # Data: points used by the fit
    data = data.loc[indexes]
    time_key = get_time_key(data, time_key)
    check_magsys(data)
    position = get_target_position(data, indexes)
    data, position = select_covered_points(data, position, tables, z, effects)
    if phase_range is not None:
        data, position = select_phase_range(data, position, truth["t0"].to_numpy(float), z,
                                            phase_range, time_key)

    lightcurves = get_lightcurve_arrays(data, position, len(indexes), tables["bands"],
                                        time_key=time_key, pad_to=pad_to)
    return lightcurves | dict(G=G, z=z, effects=parameter_table(effects, index=truth.index),
                              index=np.asarray(indexes))


# ================== #
#                    #
#   Model (JAX)      #
#                    #
# ================== #
def timeseries_bandflux(theta, packed, phase_spline):
    """Model band fluxes of a target, in the zero points of the data.

    Parameters
    ----------
    theta : array_like
        (t0, amplitude).

    packed : dict
        Packed arrays of the target (one row of :func:`pack_timeseries`).

    phase_spline : tuple
        :attr:`TimeSeriesTemplate.phase_spline`: (knots, degree, zero_before).

    Returns
    -------
    jax.Array
        Fluxes, one per lightcurve point.
    """
    t0, amplitude = theta
    knots, degree, zero_before = phase_spline
    phase = (packed["t"] - t0) / (1. + packed["z"])            # rest frame [day]

    idx, weights = get_bspline_weights(phase, jnp.asarray(knots), degree, xp=jnp)   # (npoints, degree+1)
    flux = jnp.sum(packed["G"][packed["band"][:, None], idx] * weights, -1)
    if zero_before:                                             # as sncosmo: 0 before the template
        flux = flux * (phase >= knots[degree])

    return amplitude * flux * packed["zpn"]


# ================== #
#                    #
#   Fit (JAX)        #
#                    #
# ================== #
def fit_timeseries_one(theta0, packed, phase_spline, niter=20):
    """Fit (t0, amplitude) of one target by Levenberg-Marquardt.

    It is traced and compiled by :func:`fit_timeseries_batch`; calling it
    directly (not compiled) is useful to check a single fit.

    Parameters
    ----------
    theta0 : jax.Array
        Initial (t0, amplitude).

    packed : dict
        Packed arrays of the target.

    phase_spline : tuple
        :attr:`TimeSeriesTemplate.phase_spline`.

    niter : int, optional
        Number of iterations (fixed, for the JIT). The default is 20.

    Returns
    -------
    dict
        theta, cov (2 x the inverse Hessian of the chi2, as Minuit HESSE),
        chi2 and ndof.
    """
    scale = jnp.array([1., theta0[1]])                      # amplitude rescaled to ~1
    inverse_variance = packed["mask"] / packed["err"] ** 2

    def model(q):
        return timeseries_bandflux(q * scale, packed, phase_spline) * packed["mask"]

    def chi2(q):
        residual = packed["flux"] * packed["mask"] - model(q)
        return jnp.sum(inverse_variance * residual ** 2)

    def step(carry, _):
        q, damping, chi2_ = carry
        residual = packed["flux"] * packed["mask"] - model(q)
        J = jax.jacfwd(model)(q)                            # (npoints, 2)
        A = J.T @ (inverse_variance[:, None] * J)           # (2, 2) normal matrix
        q_new = q + jnp.linalg.solve(A + damping * jnp.diag(jnp.diag(A) + 1e-12),
                                     J.T @ (inverse_variance * residual))
        chi2_new = chi2(q_new)
        # accept the step if the chi2 decreases (and trust the quadratic model more)
        better = chi2_new < chi2_
        return (jnp.where(better, q_new, q), jnp.where(better, damping * 0.3, damping * 10.),
                jnp.where(better, chi2_new, chi2_)), None

    q0 = theta0 / scale
    (q, _, chi2_), _ = jax.lax.scan(step, (q0, 1e-3, chi2(q0)), None, length=niter)
    cov_q = 2. * jnp.linalg.inv(jax.hessian(chi2)(q))
    return dict(theta=q * scale, cov=cov_q * jnp.outer(scale, scale), chi2=chi2_,
                ndof=jnp.sum(packed["mask"]) - len(PARAMS))


@partial(jax.jit, static_argnames=("phase_spline", "niter"))
def fit_timeseries_batch(theta0, packed, phase_spline, niter=20):
    """Fit a batch of targets (vectorised :func:`fit_timeseries_one`).

    Parameters
    ----------
    theta0 : jax.Array
        Initial (t0, amplitude), shape (ntargets, 2).

    packed : dict
        Packed arrays of the targets (one row per target).

    phase_spline : tuple
        :attr:`TimeSeriesTemplate.phase_spline`.

    niter : int, optional
        Number of Levenberg-Marquardt iterations. The default is 20.

    Returns
    -------
    dict
        theta, cov, chi2 and ndof, one entry per target.
    """
    return jax.vmap(lambda theta_, packed_: fit_timeseries_one(theta_, packed_, phase_spline, niter)
                    )(theta0, packed)


@partial(jax.jit, static_argnames=("phase_spline", "t0_offsets"))
def initial_timeseries_batch(packed, t0_in, phase_spline, t0_offsets=(0.,)):
    """Get the initial (t0, amplitude) of a batch of targets.

    t0 is scanned over ``t0_in + t0_offsets x (1+z)``, the amplitude is
    solved linearly, and the lowest chi2 is kept.

    Parameters
    ----------
    packed : dict
        Packed arrays of the targets (one row per target).

    t0_in : jax.Array
        (ntargets,) starting t0.

    phase_spline : tuple
        :attr:`TimeSeriesTemplate.phase_spline`.

    t0_offsets : tuple, optional
        Offsets added to the starting t0, in rest-frame days (see
        :func:`get_t0_offsets`). The default is (0.,): no scan.

    Returns
    -------
    jax.Array
        Initial (t0, amplitude), shape (ntargets, 2).
    """
    offsets = jnp.asarray(t0_offsets)

    def initial_one(packed_, t0_):
        weight = packed_["mask"] / packed_["err"] ** 2

        def trial(offset_):
            t0_trial = t0_ + offset_ * (1. + packed_["z"])           # rest frame -> observer frame
            model_flux = timeseries_bandflux(jnp.array([t0_trial, 1.]), packed_, phase_spline)
            # amplitude: weighted linear least squares
            norm = jnp.sum(weight * model_flux * model_flux)
            amplitude = jnp.sum(weight * model_flux * packed_["flux"]) / jnp.where(norm > 0, norm, 1.)
            amplitude = jnp.where(amplitude > 0, amplitude, 1e-30)   # non-zero (fit scale)
            return jnp.array([t0_trial, amplitude]), jnp.sum(weight * (packed_["flux"] - amplitude * model_flux) ** 2)

        thetas, chi2 = jax.vmap(trial)(offsets)
        return thetas[jnp.argmin(chi2)]

    return jax.vmap(initial_one)(packed, t0_in)


# ================== #
#                    #
#   Fit              #
#                    #
# ================== #
def fit_timeseries(data, targets, source="hsiao", version=None, indexes=None, phase_range=None,
                   guess="data", t0_offsets=None,
                   effects=None, effect_names=None, effect_frames=None, mwebv_key="mwebv", mw_r_v=3.1,
                   batch_points=2048, minsnr=5., niter=20, time_key=None, progress_bar=False):
    """Fit a time-series template on many lightcurves at once, reproducing ``sncosmo.fit_lc``.

    Fits (t0, amplitude) of an ``sncosmo.TimeSeriesSource`` (e.g. 'hsiao',
    'nugent-sn2p', 'snana-2004fe'), the redshift and the parameters of the
    effects (e.g. dust) being fixed. Same model and chi2 as ``sncosmo.fit_lc``
    (time-series sources have no model covariance).

    This:

    1. loads the template (:class:`TimeSeriesTemplate`) and packs the
       lightcurves with their band integrals (:func:`pack_timeseries`),
    2. selects the targets that can be fitted (:func:`select_fittable`),
    3. scans t0 for the initial guess (:func:`get_initial_t0`,
       :func:`get_t0_offsets`, :func:`initial_timeseries_batch`),
    4. fits by batches of fixed shapes, compiled once per shape
       (:func:`fit_timeseries_batch`),
    5. gathers the results in a table (:func:`get_results_table`).

    Parameters
    ----------
    data : pandas.DataFrame
        Lightcurves, with a (target, observation) multi-index and columns
        time (see `time_key`), 'band', 'flux', 'fluxerr', 'zp' and optionally
        'zpsys' (only 'ab' is supported).

    targets : pandas.DataFrame
        Targets, indexed as the first level of `data`, with a 'z' column; a
        't0' column if `phase_range` is given or ``guess='truth'``; and the
        parameters of the effects (e.g. 'mwebv').

    source : str or sncosmo.TimeSeriesSource or TimeSeriesTemplate, optional
        The template: the name of a time-series source registered in sncosmo,
        or an instance. The default is 'hsiao'.

    version : str or None, optional
        Version of the source, if given by name. The default is None (latest).

    indexes : list or None, optional
        Targets to fit. If None, all the targets with data. The default is None.

    phase_range : list or None, optional
        Rest-frame phase range (relative to the targets' t0) of the data used.
        If None, all points (as ``sncosmo.fit_lc``). The default is None.

    guess : {'data', 'truth'}, optional
        Initial t0. 'data': t0 is scanned so that the highest S/N point is at
        every phase of the template (1-day steps), which is robust for any
        lightcurve shape (e.g. the plateau of SNe II-P); 'truth': the targets'
        t0. In both cases, the amplitude is solved linearly. The default is 'data'.

    t0_offsets : tuple or None, optional
        Custom initial t0 scan: offsets in rest-frame days added to the
        starting t0 (see :func:`get_t0_offsets`). If None, the scan of
        `guess`. The default is None.

    effects, effect_names, effect_frames : list or None, optional
        Propagation effects, as in ``sncosmo.Model`` (see
        :func:`saltjax.fit_salt`). The default is None (Milky Way dust from
        `mwebv_key` and `mw_r_v`).

    mwebv_key : str or None, optional
        Only if `effects` is None: column of `targets` with the Milky Way
        E(B-V). If None or absent, no effect. The default is 'mwebv'.

    mw_r_v : float, optional
        Only if `effects` is None: R_V of the Milky Way dust. The default is 3.1.

    batch_points : int, optional
        Number of lightcurve points fitted together. The default is 2048.

    minsnr : float, optional
        Targets without any point with flux/fluxerr >= `minsnr` are not
        fitted. The default is 5.

    niter : int, optional
        Number of Levenberg-Marquardt iterations. The default is 20.

    time_key : str or None, optional
        Time column (None: 'time', 'mjd' or 'jd'). The default is None.

    progress_bar : bool, optional
        Display a progress bar (requires tqdm). The default is False.

    Returns
    -------
    pandas.DataFrame
        One row per fitted target: ``t0``, ``amplitude``, their ``_err``,
        ``cov_{p}{q}``, ``z``, the fixed parameters of the effects (e.g.
        ``mwebv``), ``chi2``, ``ndof`` and ``valid`` (False if the covariance
        is not positive: errors are then NaN).

    Raises
    ------
    NotImplementedError
        If the source is not an ``sncosmo.TimeSeriesSource`` (use
        :func:`saltjax.fit_salt` for SALT), or a magnitude system other than
        'ab' is used.

    ValueError
        If `guess` is not 'data' or 'truth'.

    Warns
    -----
    UserWarning
        If points are dropped (band out of the template range), targets are
        not fitted, or fits have an invalid covariance.

    See Also
    --------
    TimeSeriesTemplate.show_fit : Show a fitted lightcurve.
    """
    if guess not in ("data", "truth"):
        raise ValueError(f"guess must be 'data' or 'truth', not {guess!r}")

    template = source if isinstance(source, TimeSeriesTemplate) else (
        TimeSeriesTemplate.from_name(source, version=version) if isinstance(source, str)
        else TimeSeriesTemplate.from_source(source))
    if indexes is None:
        indexes = data.index.get_level_values(0).unique()
    indexes = list(indexes)

    with enable_x64(True):
        # Data
        packed = pack_timeseries(data, targets, template, indexes=indexes, phase_range=phase_range,
                                 effects=effects, effect_names=effect_names, effect_frames=effect_frames,
                                 mwebv_key=mwebv_key, mw_r_v=mw_r_v, time_key=time_key)
        fittable = select_fittable(packed, minsnr=minsnr)

        # Fit, by batches
        t0_in = get_initial_t0(packed, targets.loc[indexes], guess=guess)
        if t0_offsets is None:
            t0_offsets = get_t0_offsets(template, guess=guess)
        batches = get_timeseries_batches(packed, t0_in, fittable, batch_points=batch_points)
        out = fit_timeseries_batches(batches, template.phase_spline, t0_offsets=tuple(t0_offsets),
                                     niter=niter, progress_bar=progress_bar)

    if not out:
        return pandas.DataFrame(columns=PARAMS)

    results = get_results_table(out, packed, indexes, index_name=targets.index.name)
    if not results["valid"].all():
        warnings.warn(f"{(~results['valid']).sum()} fits have an invalid covariance "
                      "(errors set to NaN, see the 'valid' column)")
    return results


# ================== #
#                    #
#   Steps            #
#                    #
# ================== #
def select_fittable(packed, minsnr=5.):
    """Select the targets that can be fitted.

    A target is fitted if it has at least one point with S/N >= `minsnr`, and
    more points than parameters (2).

    Parameters
    ----------
    packed : dict
        Output of :func:`pack_timeseries`.

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


def get_initial_t0(packed, truth, guess="data"):
    """Get the starting t0 of each target (then scanned, see :func:`get_t0_offsets`).

    Parameters
    ----------
    packed : dict
        Output of :func:`pack_timeseries`.

    truth : pandas.DataFrame
        Targets (same order as `packed`), with a 't0' column if ``guess='truth'``.

    guess : {'data', 'truth'}, optional
        'data': time of the highest S/N point; 'truth': the targets' t0.
        The default is 'data'.

    Returns
    -------
    numpy.ndarray
        (ntargets,) starting t0.

    Raises
    ------
    ValueError
        If `guess` is not 'data' or 'truth'.
    """
    if guess == "truth":
        return truth["t0"].to_numpy(float)

    if guess == "data":
        snr = np.where(packed["mask"] > 0, packed["flux"] / packed["err"], -np.inf)
        return packed["t"][np.arange(len(snr)), np.argmax(snr, axis=1)]

    raise ValueError(f"guess must be 'data' or 'truth', not {guess!r}")


def get_t0_offsets(template, guess="data", step=T0_STEP):
    """Get the offsets of the initial t0 scan, in rest-frame days.

    The trial t0 of a target are ``t0_in + offsets x (1+z)``, with ``t0_in``
    from :func:`get_initial_t0`.

    Parameters
    ----------
    template : TimeSeriesTemplate
        The template.

    guess : {'data', 'truth'}, optional
        'data': ``t0_in`` is the time of the highest S/N point, and the
        offsets put it at every phase of the template (``-phase``); 'truth':
        no scan (``(0.,)``). The default is 'data'.

    step : float, optional
        Step of the scan [rest-frame day]. The default is 1.

    Returns
    -------
    tuple of float
        Offsets [rest-frame day].

    Raises
    ------
    ValueError
        If `guess` is not 'data' or 'truth'.
    """
    if guess == "truth":
        return (0.,)

    if guess == "data":
        phases = np.arange(template.phase_range[0], template.phase_range[1] + step / 2, step)
        return tuple(float(-phase_) for phase_ in phases)

    raise ValueError(f"guess must be 'data' or 'truth', not {guess!r}")


def get_timeseries_batches(packed, t0_in, fittable, batch_points=2048):
    """Group the targets in batches of fixed shapes.

    As :func:`saltjax.fit.get_batches`: targets are bucketed by number of
    points, so that compiled functions are reused by any later call.

    Parameters
    ----------
    packed : dict
        Output of :func:`pack_timeseries`.

    t0_in : numpy.ndarray
        (ntargets,) starting t0 (see :func:`get_initial_t0`).

    fittable : numpy.ndarray
        (ntargets,) True for the targets to fit (see :func:`select_fittable`).

    batch_points : int, optional
        Number of lightcurve points fitted together. The default is 2048.

    Returns
    -------
    list of tuple
        One ``(length, sel, packed_batch, t0_batch)`` per batch.
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
                            for key_ in TS_FIT_KEYS}
            batches.append((length_, sel, packed_batch, jnp.asarray(t0_in[padded])))

    return batches


def fit_timeseries_batches(batches, phase_spline, t0_offsets=(0.,), niter=20, progress_bar=False):
    """Compile (in parallel, once per shape) and run the fit of each batch.

    Parameters
    ----------
    batches : list of tuple
        Output of :func:`get_timeseries_batches`.

    phase_spline : tuple
        :attr:`TimeSeriesTemplate.phase_spline`.

    t0_offsets : tuple, optional
        Offsets of the initial t0 scan [rest-frame day] (see
        :func:`get_t0_offsets`). The default is (0.,): no scan.

    niter : int, optional
        Number of Levenberg-Marquardt iterations. The default is 20.

    progress_bar : bool, optional
        Display a progress bar (requires tqdm). The default is False.

    Returns
    -------
    dict
        Concatenated outputs of :func:`fit_timeseries_batch` (theta, cov,
        chi2, ndof) and 'pos', the row of each fitted target. Empty if there
        is no batch.
    """
    init_static = dict(phase_spline=phase_spline, t0_offsets=t0_offsets)
    fit_static = dict(phase_spline=phase_spline, niter=niter)

    pbar = None
    if progress_bar:
        from tqdm.auto import tqdm
        pbar = tqdm(total=int(sum(len(batch_[1]) for batch_ in batches)), unit="SN")
        pbar.set_description("compiling")

    # compile every needed shape first, in parallel (the theta shape is known: (nbatch, 2))
    compile_all([(initial_timeseries_batch, (packed_, t0_), init_static) for _, _, packed_, t0_ in batches] +
                [(fit_timeseries_batch, (jnp.zeros((len(t0_), len(PARAMS))), packed_), fit_static)
                 for _, _, packed_, t0_ in batches])
    if pbar is not None:
        pbar.set_description("fitting")

    out = {}
    for _, sel_, packed_, t0_ in batches:
        theta0 = get_compiled(initial_timeseries_batch, (packed_, t0_), init_static)(packed_, t0_)
        result = get_compiled(fit_timeseries_batch, (theta0, packed_), fit_static)(theta0, packed_)
        # drop the padding targets
        result = {key_: np.asarray(value_)[:len(sel_)] for key_, value_ in jax.block_until_ready(result).items()}
        for key_, value_ in result.items():
            out.setdefault(key_, []).append(value_)
        out.setdefault("pos", []).append(sel_)
        if pbar is not None:
            pbar.update(len(sel_))

    if pbar is not None:
        pbar.close()

    return {key_: np.concatenate(values_) for key_, values_ in out.items()}


def get_results_table(out, packed, indexes, index_name=None):
    """Gather the fit outputs in a table, one row per fitted target.

    Parameters
    ----------
    out : dict
        Output of :func:`fit_timeseries_batches`.

    packed : dict
        Output of :func:`pack_timeseries`.

    indexes : list
        Packed targets (rows of `packed`).

    index_name : str or None, optional
        Name of the index of the table. The default is None.

    Returns
    -------
    pandas.DataFrame
        See :func:`fit_timeseries`. Rows follow the order of `indexes`.
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

    results["chi2"] = out["chi2"]
    results["ndof"] = out["ndof"]
    results["valid"] = valid

    results = results.iloc[np.argsort(pos)]      # back to the order of indexes
    results.index.name = index_name
    return results
