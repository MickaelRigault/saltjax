import numpy as np
import pandas
import jax
import jax.numpy as jnp
import pytest
import sncosmo

import saltjax
from saltjax import build_tables, pack_lightcurves, get_model_functions
from saltjax.data import LC_KEYS, FIT_KEYS
from saltjax.effects import get_effects, transmission

PARAMS = ["t0", "x0", "x1", "c"]


class GrayDust(sncosmo.PropagationEffect):
    """A user-defined effect (not one of sncosmo's dust laws)."""
    _param_names = ["tau"]
    param_names_latex = [r"\tau"]
    _minwave = 1000.
    _maxwave = 30000.

    def __init__(self):
        self._parameters = np.array([0.1])

    def propagate(self, wave, flux):
        return flux * np.exp(-self._parameters[0] * 5000. / wave)


# (effects, names, frames): Milky Way and host dust with every sncosmo law
CASES = {
    "ccm89 mw + ccm89 host": ([sncosmo.CCM89Dust(), sncosmo.CCM89Dust()], ["mw", "host"], ["obs", "rest"]),
    "od94 host": ([sncosmo.OD94Dust()], ["host"], ["rest"]),
    "f99 mw + f99 host": ([sncosmo.F99Dust(), sncosmo.F99Dust(r_v=2.0)], ["mw", "host"], ["obs", "rest"]),
    "custom rest": ([GrayDust()], ["gray"], ["rest"]),
}


@pytest.fixture(scope="module")
def dusty(simulation):
    """The simulation, with per-target Milky Way R_V and host dust parameters."""
    data, targets = simulation
    rng = np.random.default_rng(7)
    targets = targets.assign(mwr_v=rng.uniform(2.5, 3.5, len(targets)),
                             hostebv=rng.uniform(0., 0.4, len(targets)),
                             hostr_v=rng.uniform(1.5, 4.5, len(targets)),
                             graytau=rng.uniform(0., 0.3, len(targets)))
    return data, targets


def _sncosmo_model(row, effects, names, frames, source="salt2"):
    model = sncosmo.Model(source, effects=effects, effect_names=names, effect_frames=frames)
    model.set(**{p: row[p] for p in model.param_names if p in row.index})
    return model


# ------------------------------------------------------------------ #
#  Effects and their parameters                                      #
# ------------------------------------------------------------------ #
def test_default_effect_is_mw_ccm89(dusty):
    _, targets = dusty
    eff = get_effects(targets)
    assert len(eff) == 1 and type(eff[0]["effect"]) is sncosmo.CCM89Dust
    assert (eff[0]["name"], eff[0]["frame"]) == ("mw", "obs")
    np.testing.assert_array_equal(eff[0]["params"]["ebv"], targets["mwebv"])
    np.testing.assert_array_equal(eff[0]["params"]["r_v"], 3.1)     # mw_r_v, not the mwr_v column
    assert get_effects(targets, mw_r_v=2.5)[0]["params"]["r_v"][0] == 2.5
    assert get_effects(targets.drop(columns="mwebv")) == []
    assert get_effects(targets, mwebv_key=None) == []


def test_effect_parameters_from_columns_or_instance(dusty):
    _, targets = dusty
    host = sncosmo.CCM89Dust()
    host.set(r_v=2.2)
    eff = get_effects(targets.drop(columns="hostr_v"), [host], ["host"], ["rest"])
    np.testing.assert_array_equal(eff[0]["params"]["ebv"], targets["hostebv"])
    np.testing.assert_array_equal(eff[0]["params"]["r_v"], 2.2)
    assert get_effects(targets, [], [], []) == []


def test_bad_effects_raise(dusty):
    _, targets = dusty
    dust = sncosmo.CCM89Dust()
    with pytest.raises(ValueError):
        get_effects(targets, [dust])                                   # no names/frames
    with pytest.raises(ValueError):
        get_effects(targets, [dust], ["host", "mw"], ["rest"])          # lengths differ
    with pytest.raises(ValueError):
        get_effects(targets, [dust, dust], ["host", "host"], ["rest", "rest"])
    with pytest.raises(ValueError):
        get_effects(targets, [dust], ["host"], ["observer"])
    with pytest.raises(NotImplementedError):
        get_effects(targets, [dust], ["host"], ["free"])
    with pytest.raises(NotImplementedError):
        get_effects(targets, [sncosmo.models.G10(sncosmo.get_source("salt2"))], ["g10"], ["rest"])
    with pytest.raises(TypeError):
        get_effects(targets, ["ccm89"], ["host"], ["rest"])


@pytest.mark.parametrize("case", CASES)
def test_transmission_matches_sncosmo(dusty, case):
    _, targets = dusty
    effects, names, frames = CASES[case]
    eff = get_effects(targets, effects, names, frames)
    wave = np.linspace(3000., 10000., 300)
    a = 1. / (1. + targets["z"].to_numpy())
    trans = transmission(eff, wave, a)
    for i, (_, row) in enumerate(targets.iterrows()):
        model = _sncosmo_model(row, effects, names, frames)
        ref = np.ones_like(wave)
        for effect, frame in zip(model._effects, model._effect_frames):
            ref = effect.propagate(wave * (a[i] if frame == "rest" else 1.), ref)
        np.testing.assert_allclose(trans[i], ref, rtol=1e-12)


# ------------------------------------------------------------------ #
#  Model and fit                                                     #
# ------------------------------------------------------------------ #
@pytest.mark.parametrize("source", ["salt2", "salt3"])
@pytest.mark.parametrize("case", CASES)
def test_model_with_effects_matches_sncosmo(dusty, case, source):
    data, targets = dusty
    effects, names, frames = CASES[case]
    tab = build_tables(sorted(data["band"].unique()), targets["z"].min(), targets["z"].max(),
                       source=source)
    with jax.enable_x64(True):
        arr = pack_lightcurves(data, targets, tab, phase_range=None, effects=effects,
                               effect_names=names, effect_frames=frames)
        flux, _, mcov = get_model_functions(tab)
        for i, (_, row) in enumerate(targets.iloc[:6].iterrows()):
            n = int(arr["mask"][i].sum())
            d = {k: jnp.asarray(arr[k][i][:n] if k in LC_KEYS else arr[k][i]) for k in FIT_KEYS}
            row = row.copy()
            row[PARAMS] = [row["t0"] + 1.3, 2e-3, 0.7, 0.08]
            bands = np.array(tab["bands"])[np.asarray(d["band"])]
            ref_flux, ref_cov = _sncosmo_model(row, effects, names, frames, source).bandfluxcov(
                bands, np.asarray(d["t"]), zp=25., zpsys="ab")
            theta = jnp.asarray(row[PARAMS].to_numpy(float))
            np.testing.assert_allclose(np.asarray(flux(theta, d)), ref_flux,
                                       rtol=1e-10, atol=1e-12 * np.abs(ref_flux).max())
            np.testing.assert_allclose(np.asarray(mcov(theta, d)), ref_cov,
                                       rtol=1e-3, atol=1e-6 * np.abs(ref_cov).max())


@pytest.mark.parametrize("modelcov", [False, True])
def test_fit_with_host_dust_matches_sncosmo(dusty, modelcov):
    pytest.importorskip("iminuit")
    data, targets = dusty
    effects, names, frames = CASES["ccm89 mw + ccm89 host"]
    res = saltjax.fit_salt(data, targets, modelcov=modelcov, effects=effects,
                           effect_names=names, effect_frames=frames)
    for name, row in targets.iloc[:8].iterrows():
        lc = data.xs(name)
        lc = lc[((lc["time"] - row["t0"]) / (1 + row["z"])).between(-10, 40)]
        model = _sncosmo_model(row, effects, names, frames)
        bounds = {"t0": (row["t0"] - 15, row["t0"] + 15), "x1": (row["x1"] - 5, row["x1"] + 5),
                  "c": (row["c"] - 1, row["c"] + 1)}
        ref, _ = sncosmo.fit_lc({k: lc[k].to_numpy() for k in lc.columns}, model, PARAMS,
                                bounds=bounds, modelcov=modelcov)
        for p, v in zip(ref.param_names, ref.parameters):
            if p in PARAMS:
                assert abs(res.loc[name, p] - v) / ref.errors[p] < 0.05, p
                assert abs(res.loc[name, f"{p}_err"] / ref.errors[p] - 1) < 0.02, p
    for col in ("mwebv", "mwr_v", "hostebv", "hostr_v"):
        np.testing.assert_allclose(res[col], targets.loc[res.index, col])


def test_mw_shortcuts_equal_effects(dusty):
    data, targets = dusty
    short = saltjax.fit_salt(data, targets, modelcov=False, mw_r_v=2.5)
    full = saltjax.fit_salt(data, targets.assign(mwr_v=2.5), modelcov=False,
                            effects=[sncosmo.CCM89Dust()], effect_names=["mw"], effect_frames=["obs"])
    np.testing.assert_allclose(short[PARAMS].astype(float), full.loc[short.index, PARAMS].astype(float),
                               rtol=1e-12)
    assert (short["mwr_v"] == 2.5).all()


def test_no_effect(dusty):
    data, targets = dusty
    res = saltjax.fit_salt(data, targets, modelcov=False, effects=[], effect_names=[], effect_frames=[])
    assert not {"mwebv", "mwr_v"} & set(res.columns)
    assert len(res) == len(targets)


def test_effect_wave_range_drops_points(dusty):
    data, targets = dusty

    class RedOnly(GrayDust):
        _minwave = 5000.        # cuts bessellb and bessellv in the observer frame

    with pytest.warns(UserWarning, match="out of the model wavelength range"):
        res = saltjax.fit_salt(data, targets, modelcov=False, effects=[RedOnly()],
                               effect_names=["gray"], effect_frames=["obs"])
    ref = saltjax.fit_salt(data[data["band"] == "bessellr"], targets, modelcov=False,
                           effects=[RedOnly()], effect_names=["gray"], effect_frames=["obs"])
    np.testing.assert_allclose(res[PARAMS].astype(float), ref.loc[res.index, PARAMS].astype(float),
                               rtol=1e-12)
