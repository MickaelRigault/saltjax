"""Propagation effects (dust) applied to the SALT model, as in ``sncosmo.Model``.

Effects are given as in sncosmo (``effects``, ``effect_names``,
``effect_frames``). Their parameters are fixed per target, read from the
``targets`` columns ``{name}{param}`` (e.g. ``mwebv``, ``hostebv``,
``hostr_v``), or else taken from the effect instance.

An effect is a transmission T(wavelength) that multiplies the model flux in
the observer frame ('obs') or in the rest frame ('rest'). saltjax folds it,
exactly, into the band integrals of each target (see
:func:`saltjax.data.band_integrals`).
"""

import copy

import numpy as np
import pandas
import sncosmo
import extinction

__all__ = ["get_effects", "transmission"]

FRAMES = ("obs", "rest")
# random scattering effects: not a deterministic transmission
_UNSUPPORTED = tuple(getattr(sncosmo.models, n) for n in ("G10", "C11") if hasattr(sncosmo.models, n))
# laws that are exactly ebv * (r_v * a(wave) + b(wave))
_LINEAR_RV_LAWS = {sncosmo.CCM89Dust: extinction.ccm89, sncosmo.OD94Dust: extinction.odonnell94}


def get_effects(targets, effects=None, effect_names=None, effect_frames=None,
                mwebv_key="mwebv", mw_r_v=3.1):
    """Get the effects and their parameters for each target.

    Parameters
    ----------
    targets : pandas.DataFrame
        Targets. The parameter ``p`` of the effect named ``name`` is read from
        the column ``f"{name}{p}"`` if present (e.g. 'mwebv', 'hostr_v').

    effects : list of sncosmo.PropagationEffect or None, optional
        Effects, as in ``sncosmo.Model``. If None, Milky Way CCM89 dust in the
        observer frame (named 'mw') if the `mwebv_key` column exists, else no
        effect. The default is None.

    effect_names : list of str or None, optional
        Names of the effects (prefixes of their parameters). Required with
        `effects`. The default is None.

    effect_frames : list of str or None, optional
        Frames of the effects, 'obs' or 'rest'. Required with `effects`.
        The default is None.

    mwebv_key : str or None, optional
        Only if `effects` is None: column with the Milky Way E(B-V).
        The default is 'mwebv'.

    mw_r_v : float, optional
        Only if `effects` is None: R_V of the Milky Way dust. The default is 3.1.

    Returns
    -------
    list of dict
        One dict per effect: 'effect', 'name', 'frame' and 'params' (a dict
        of arrays, one value per target, for every parameter of the effect).

    Raises
    ------
    ValueError
        If `effects`, `effect_names` and `effect_frames` do not match, or a
        frame is not 'obs', 'rest' or 'free'.

    NotImplementedError
        For the 'free' frame, and for random scattering effects (G10, C11).
    """
    n = len(targets)
    if effects is None:
        if mwebv_key is None or mwebv_key not in targets:
            return []
        return [dict(effect=sncosmo.CCM89Dust(), name="mw", frame="obs",
                     params={"ebv": targets[mwebv_key].to_numpy(float),
                             "r_v": np.full(n, float(mw_r_v))})]

    effects = list(effects)
    if effect_names is None or effect_frames is None:
        raise ValueError("effect_names and effect_frames are required with effects")
    effect_names, effect_frames = list(effect_names), list(effect_frames)
    if not len(effects) == len(effect_names) == len(effect_frames):
        raise ValueError("effects, effect_names and effect_frames must have the same length")
    if len(set(effect_names)) != len(effect_names):
        raise ValueError("effect_names must be unique")
    out = []
    for effect, name, frame in zip(effects, effect_names, effect_frames):
        if not isinstance(effect, sncosmo.PropagationEffect):
            raise TypeError(f"effect {name!r} is not an sncosmo.PropagationEffect")
        if isinstance(effect, _UNSUPPORTED):
            raise NotImplementedError(f"random scattering effects are not supported ({type(effect).__name__})")
        if frame == "free":
            raise NotImplementedError("the 'free' effect frame is not supported")
        if frame not in FRAMES:
            raise ValueError(f"effect frame must be one of 'obs', 'rest' (or 'free'), not {frame!r}")
        params = {p: (targets[name + p].to_numpy(float) if name + p in targets else np.full(n, float(v)))
                  for p, v in zip(effect.param_names, effect.parameters)}
        out.append(dict(effect=effect, name=name, frame=frame, params=params))
    return out


def parameter_table(effects, index=None):
    """Get the parameters of the effects as a DataFrame (columns ``{name}{param}``).

    Parameters
    ----------
    effects : list of dict
        Output of :func:`get_effects`.

    index : array_like or None, optional
        Index of the output. The default is None.

    Returns
    -------
    pandas.DataFrame
        One column per parameter of each effect.
    """
    return pandas.DataFrame({e["name"] + p: v for e in effects for p, v in e["params"].items()},
                            index=index)


def _extinction(effect, params, wave, sel):
    """Extinction in magnitudes A(wave) of an effect, for the targets `sel`.

    `wave` has shape (nwave,) (same for all targets) or (len(sel), nwave).
    """
    law = _LINEAR_RV_LAWS.get(type(effect))
    if law is not None:     # exact: A = ebv * (r_v * a + b), with law(w, a_v, r_v) = a_v * (a + b / r_v)
        p, q = law(wave.ravel(), 1., 1.).reshape(wave.shape), law(wave.ravel(), 1., 2.).reshape(wave.shape)
        a, b = 2. * q - p, 2. * (p - q)
        return params["ebv"][sel, None] * (params["r_v"][sel, None] * a + b)
    if type(effect) is sncosmo.F99Dust:   # linear in A_V, R_V fixed by the instance
        g = effect._f(wave.ravel(), 1.).reshape(wave.shape)
        return params["ebv"][sel, None] * effect._r_v * g
    return None


def transmission(effects, wave, a):
    """Transmission of all the effects, for each target, at observer-frame wavelengths.

    Parameters
    ----------
    effects : list of dict
        Output of :func:`get_effects`, restricted to the targets.

    wave : numpy.ndarray
        Observer-frame wavelengths, shape (nwave,).

    a : numpy.ndarray
        Scale factor 1/(1+z) of each target, shape (ntargets,).

    Returns
    -------
    numpy.ndarray
        Transmission, shape (ntargets, nwave).
    """
    a = np.asarray(a, float)
    sel = np.arange(len(a))
    trans = np.ones((len(a), len(wave)))
    for e in effects:
        w = wave if e["frame"] == "obs" else a[:, None] * wave[None, :]
        A = _extinction(e["effect"], e["params"], w, sel)
        if A is not None:
            trans *= 10 ** (-0.4 * A)
            continue
        # any other effect: its own propagate, target per target
        effect = copy.deepcopy(e["effect"])
        for i in sel:
            effect.set(**{p: v[i] for p, v in e["params"].items()})
            wi = w if w.ndim == 1 else w[i]
            trans[i] *= effect.propagate(wi, np.ones(len(wi)))
    return trans


def wave_limits(effects, z):
    """Observer-frame wavelength range allowed by the effects, as ``sncosmo.Model``.

    Parameters
    ----------
    effects : list of dict
        Output of :func:`get_effects`.

    z : numpy.ndarray
        Redshifts.

    Returns
    -------
    wmin, wmax : numpy.ndarray
        Range for each target (no limit: 0 and inf).
    """
    wmin, wmax = np.zeros(len(z)), np.full(len(z), np.inf)
    for e in effects:
        shift = 1. + z if e["frame"] == "rest" else 1.
        if e["effect"].minwave() is not None:
            wmin = np.maximum(wmin, e["effect"].minwave() * shift)
        if e["effect"].maxwave() is not None:
            wmax = np.minimum(wmax, e["effect"].maxwave() * shift)
    return wmin, wmax
