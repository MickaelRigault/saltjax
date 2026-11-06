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

__all__ = ["get_effects", "parameter_table", "get_extinction", "transmission", "wave_limits"]

FRAMES = ("obs", "rest")      # supported effect frames
# random scattering effects: not a deterministic transmission
_UNSUPPORTED = tuple(getattr(sncosmo.models, n) for n in ("G10", "C11") if hasattr(sncosmo.models, n))
# laws that are exactly ebv * (r_v * a(wave) + b(wave))
_LINEAR_RV_LAWS = {sncosmo.CCM89Dust: extinction.ccm89, sncosmo.OD94Dust: extinction.odonnell94}


# ================== #
#                    #
#   Effects          #
#                    #
# ================== #
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

    TypeError
        If an effect is not an ``sncosmo.PropagationEffect``.

    NotImplementedError
        For the 'free' frame, and for random scattering effects (G10, C11).
    """
    ntargets = len(targets)

    # default: Milky Way dust, if the targets have its E(B-V)
    if effects is None:
        if mwebv_key is None or mwebv_key not in targets:
            return []

        return [dict(effect=sncosmo.CCM89Dust(), name="mw", frame="obs",
                     params={"ebv": targets[mwebv_key].to_numpy(float),
                             "r_v": np.full(ntargets, float(mw_r_v))})]

    # user-given effects: check them
    effects = list(effects)
    if effect_names is None or effect_frames is None:
        raise ValueError("effect_names and effect_frames are required with effects")

    effect_names, effect_frames = list(effect_names), list(effect_frames)
    if not len(effects) == len(effect_names) == len(effect_frames):
        raise ValueError("effects, effect_names and effect_frames must have the same length, "
                         f"{len(effects)=}, {len(effect_names)=}, {len(effect_frames)=}")
    if len(set(effect_names)) != len(effect_names):
        raise ValueError(f"effect_names must be unique, {effect_names=}")

    out = []
    for effect_, name_, frame_ in zip(effects, effect_names, effect_frames):
        _check_effect_(effect_, name_, frame_)
        # parameter p: column {name}{p} of the targets, else the effect value
        params = {param_: (targets[name_ + param_].to_numpy(float) if name_ + param_ in targets
                           else np.full(ntargets, float(value_)))
                  for param_, value_ in zip(effect_.param_names, effect_.parameters)}
        out.append(dict(effect=effect_, name=name_, frame=frame_, params=params))

    return out


def _check_effect_(effect, name, frame):
    """Check that an effect and its frame are supported (see :func:`get_effects`)."""
    if not isinstance(effect, sncosmo.PropagationEffect):
        raise TypeError(f"effect {name!r} is not an sncosmo.PropagationEffect")
    if isinstance(effect, _UNSUPPORTED):
        raise NotImplementedError(f"random scattering effects are not supported ({type(effect).__name__})")
    if frame == "free":
        raise NotImplementedError("the 'free' effect frame is not supported")
    if frame not in FRAMES:
        raise ValueError(f"effect frame must be one of 'obs', 'rest' (or 'free'), not {frame!r}")


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
    return pandas.DataFrame({effect_["name"] + param_: values_
                             for effect_ in effects for param_, values_ in effect_["params"].items()},
                            index=index)


# ================== #
#                    #
#   Transmission     #
#                    #
# ================== #
def get_extinction(effect, params, wave):
    """Get the extinction in magnitudes A(wave) of a dust effect, for each target.

    Only the dust laws for which this is exact and fast are handled:
    ``sncosmo.CCM89Dust`` and ``sncosmo.OD94Dust`` (linear in E(B-V) and in
    E(B-V) R_V), and ``sncosmo.F99Dust`` (linear in E(B-V), R_V fixed by the
    instance). Any other effect returns None (see :func:`transmission`).

    Parameters
    ----------
    effect : sncosmo.PropagationEffect
        The effect.

    params : dict
        Parameters of the effect, one array per parameter with one value per
        target (``"ebv"``, and ``"r_v"`` for CCM89 and OD94).

    wave : numpy.ndarray
        Wavelengths [A] at which the effect applies (e.g. rest frame for host
        dust): shape (nwave,) (same for all targets) or (ntargets, nwave).

    Returns
    -------
    numpy.ndarray or None
        Extinction [mag], shape (ntargets, nwave); None if the effect is not a
        supported dust law.
    """
    law = _LINEAR_RV_LAWS.get(type(effect))
    if law is not None:
        # exact: A = ebv * (r_v * a + b), with law(wave, a_v, r_v) = a_v * (a + b / r_v)
        law_rv1 = law(wave.ravel(), 1., 1.).reshape(wave.shape)
        law_rv2 = law(wave.ravel(), 1., 2.).reshape(wave.shape)
        a, b = 2. * law_rv2 - law_rv1, 2. * (law_rv1 - law_rv2)
        return params["ebv"][:, None] * (params["r_v"][:, None] * a + b)

    if type(effect) is sncosmo.F99Dust:    # linear in A_V, R_V fixed by the instance
        extinction_per_av = effect._f(wave.ravel(), 1.).reshape(wave.shape)
        return params["ebv"][:, None] * effect._r_v * extinction_per_av

    return None


def transmission(effects, wave, a):
    """Transmission of all the effects, for each target, at observer-frame wavelengths.

    Dust laws use :func:`get_extinction` (vectorised over targets); any other
    effect uses its own ``propagate``, target per target.

    Parameters
    ----------
    effects : list of dict
        Output of :func:`get_effects`, restricted to the targets.

    wave : numpy.ndarray
        Observer-frame wavelengths [A], shape (nwave,).

    a : numpy.ndarray
        Scale factor 1/(1+z) of each target, shape (ntargets,).

    Returns
    -------
    numpy.ndarray
        Transmission, shape (ntargets, nwave).
    """
    a = np.asarray(a, float)
    trans = np.ones((len(a), len(wave)))
    for effect_ in effects:
        # wavelengths in the frame of the effect
        effect_wave = wave if effect_["frame"] == "obs" else a[:, None] * wave[None, :]  # [A]

        extinction_ = get_extinction(effect_["effect"], effect_["params"], effect_wave)
        if extinction_ is not None:
            trans *= 10 ** (-0.4 * extinction_)
            continue

        # any other effect: its own propagate, target per target (on a copy, not to change the input)
        effect = copy.deepcopy(effect_["effect"])
        for i in range(len(a)):
            effect.set(**{param_: values_[i] for param_, values_ in effect_["params"].items()})
            wave_i = effect_wave if effect_wave.ndim == 1 else effect_wave[i]
            trans[i] *= effect.propagate(wave_i, np.ones(len(wave_i)))

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
        Range for each target [A] (no limit: 0 and inf).
    """
    wmin, wmax = np.zeros(len(z)), np.full(len(z), np.inf)
    for effect_ in effects:
        shift = 1. + z if effect_["frame"] == "rest" else 1.    # rest frame -> observer frame
        if effect_["effect"].minwave() is not None:
            wmin = np.maximum(wmin, effect_["effect"].minwave() * shift)
        if effect_["effect"].maxwave() is not None:
            wmax = np.minimum(wmax, effect_["effect"].maxwave() * shift)

    return wmin, wmax
