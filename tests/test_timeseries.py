import numpy as np
import pandas
import jax.numpy as jnp
import pytest
import sncosmo
from scipy.interpolate import BSpline

import saltjax
from saltjax import timeseries
from saltjax.data import LC_KEYS
from saltjax.fitter import enable_x64

BANDS = ["bessellb", "bessellv", "bessellr"]


def make_source(degree=3, zero_before=False):
    """A local time-series source (no download): gaussian lightcurve x smooth spectrum."""
    phase = np.linspace(-15., 60., 76)
    wave = np.linspace(2000., 11000., 451)
    light = np.exp(-0.5 * (phase / 12.) ** 2)[:, None] + 0.05
    sed = (wave / 5000.) ** -2 * (1 + 0.3 * np.sin(wave / 300.))
    return sncosmo.TimeSeriesSource(phase, wave, light * sed[None] * 1e-15, zero_before=zero_before,
                                    time_spline_degree=degree, name=f"test-degree{degree}")


def simulate(source, n=12, seed=0, sorted_times=True):
    """(data, targets) observed with Milky Way dust, with noise."""
    rng = np.random.default_rng(seed)
    targets = pandas.DataFrame({"z": rng.uniform(0.02, 0.08, n), "t0": rng.uniform(59_000, 59_100, n),
                                "amplitude": rng.uniform(0.5, 2., n), "mwebv": rng.uniform(0., 0.2, n)},
                               index=pandas.Index(np.arange(n), name="index"))
    lcs = []
    for i, row in targets.iterrows():
        model = sncosmo.Model(source, effects=[sncosmo.CCM89Dust()], effect_names=["mw"], effect_frames=["obs"])
        model.set(z=row["z"], t0=row["t0"], amplitude=row["amplitude"], mwebv=row["mwebv"])
        time = row["t0"] + rng.uniform(-15, 60, 40)
        time = np.sort(time) if sorted_times else time
        band = rng.choice(BANDS, len(time))
        flux = np.array([model.bandflux(b_, t_, zp=25., zpsys="ab") for b_, t_ in zip(band, time)])
        fluxerr = 0.03 * flux.max() + 0.02 * np.abs(flux)
        lcs.append(pandas.DataFrame({"time": time, "band": band, "zp": 25., "zpsys": "ab",
                                     "flux": flux + rng.normal(0, fluxerr), "fluxerr": fluxerr}))
    data = pandas.concat(lcs, keys=targets.index, names=["index", "index_obs"])
    return data, targets


@pytest.fixture(scope="module")
def source():
    return make_source()


@pytest.fixture(scope="module")
def simulation(source):
    return simulate(source)


# tests for the func get_bspline_weights
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_bspline_weights_match_scipy(degree):
    knots = np.r_[[0.] * degree, np.linspace(0., 10., 12), [10.] * degree]
    x = np.linspace(0., 10., 101)
    idx, weights = timeseries.get_bspline_weights(x, knots, degree)
    dense = np.zeros((len(x), len(knots) - degree - 1))
    np.put_along_axis(dense, idx, weights, axis=1)
    np.testing.assert_allclose(dense, BSpline.design_matrix(x, knots, degree).toarray(), atol=1e-14)
    np.testing.assert_allclose(weights.sum(-1), 1.)                     # partition of unity

    # same with jax
    with enable_x64(True):
        idx_jax, weights_jax = timeseries.get_bspline_weights(jnp.asarray(x), jnp.asarray(knots), degree, xp=jnp)
    np.testing.assert_array_equal(np.asarray(idx_jax), idx)
    np.testing.assert_allclose(np.asarray(weights_jax), weights, atol=1e-14)


def test_bspline_weights_clamped_outside():
    knots = np.r_[[0.] * 3, np.linspace(0., 10., 12), [10.] * 3]
    idx, weights = timeseries.get_bspline_weights(np.array([-5., 0., 10., 15.]), knots, 3)
    np.testing.assert_array_equal(idx[0], idx[1])                      # clamped to the edges
    np.testing.assert_allclose(weights[0], weights[1])
    np.testing.assert_allclose(weights[3], weights[2])


# tests for the class TimeSeriesTemplate
def test_template_from_source(source):
    template = timeseries.TimeSeriesTemplate.from_source(source)
    assert template.name == "test-degree3"
    assert template.phase_degree == 3 and template.wave_degree == 3
    assert template.phase_range == (source.minphase(), source.maxphase())
    assert template.wave_range == (source.minwave(), source.maxwave())
    assert "test-degree3" in str(template)
    assert template.get_peak_phase() == pytest.approx(0., abs=0.5)    # gaussian centered on 0


@pytest.mark.parametrize("degree, zero_before", [(3, False), (1, True)])
def test_template_flux_matches_sncosmo(degree, zero_before):
    source = make_source(degree, zero_before)
    template = timeseries.TimeSeriesTemplate.from_source(source)
    phase = np.linspace(-25., 70., 40)                                 # incl. outside the template
    wave = np.linspace(2500., 10000., 30)
    np.testing.assert_allclose(template.get_flux(phase, wave), source._flux(phase, wave),
                               rtol=1e-12, atol=1e-30)


def test_template_unsupported_source_raises():
    with pytest.raises(NotImplementedError, match="fit_salt"):
        timeseries.TimeSeriesTemplate.from_source(sncosmo.get_source("salt2"))
    stretch = sncosmo.StretchSource(np.linspace(0, 10, 11), np.linspace(3000, 9000, 61), np.ones((11, 61)))
    with pytest.raises(NotImplementedError):
        timeseries.TimeSeriesTemplate.from_source(stretch)


# tests for the func get_sncosmo_model
def test_get_sncosmo_model(source):
    template = timeseries.TimeSeriesTemplate.from_source(source)
    model = template.get_sncosmo_model(z=0.05, t0=10., amplitude=2.)
    assert model.param_names == ["z", "t0", "amplitude"]
    assert model.get("amplitude") == 2.

    # default effect, as fit_timeseries: Milky Way dust
    model = template.get_sncosmo_model(z=0.05, mwebv=0.1)
    assert model.get("mwebv") == 0.1

    # any effects
    model = template.get_sncosmo_model(effects=[sncosmo.CCM89Dust()], effect_names=["host"],
                                       effect_frames=["rest"], hostebv=0.2, hostr_v=2.5)
    assert model.get("hostr_v") == 2.5

    with pytest.raises(ValueError, match="not in the model"):
        template.get_sncosmo_model(hostebv=0.2)


# tests for the func timeseries_bandflux
@pytest.mark.parametrize("degree, zero_before", [(3, False), (1, True)])
def test_bandflux_matches_sncosmo(degree, zero_before):
    """ exact, with Milky Way and host dust, incl. phases outside the template """
    source = make_source(degree, zero_before)
    data, targets = simulate(source, n=4, seed=1, sorted_times=False)
    data["time"] = data["time"] + np.tile([-60., 0., 0., 80.], len(data) // 4)   # some outside the template
    targets = targets.assign(hostebv=[0., 0.1, 0.2, 0.3])
    effects, names, frames = [sncosmo.CCM89Dust(), sncosmo.CCM89Dust()], ["mw", "host"], ["obs", "rest"]
    template = timeseries.TimeSeriesTemplate.from_source(source)
    with enable_x64(True):
        packed = timeseries.pack_timeseries(data, targets, template, effects=effects,
                                            effect_names=names, effect_frames=frames)
        for i, (index_, row) in enumerate(targets.iterrows()):
            npoints = int(packed["mask"][i].sum())
            one = {k: jnp.asarray(packed[k][i][:npoints] if k in LC_KEYS else packed[k][i])
                   for k in timeseries.TS_FIT_KEYS}
            mine = timeseries.timeseries_bandflux(jnp.array([row["t0"], 1.3]), one, template.phase_spline)

            model = sncosmo.Model(source, effects=effects, effect_names=names, effect_frames=frames)
            model.set(z=row["z"], t0=row["t0"], amplitude=1.3, mwebv=row["mwebv"], hostebv=row["hostebv"])
            bands = np.asarray(sorted(data["band"].unique()))[packed["band"][i][:npoints]]
            ref = [model.bandflux(b_, t_, zp=25., zpsys="ab") for b_, t_ in zip(bands, packed["t"][i][:npoints])]
            np.testing.assert_allclose(np.asarray(mine), ref, rtol=1e-12, atol=1e-12 * np.max(np.abs(ref)))


# tests for the initial t0
def test_get_t0_offsets(source):
    template = timeseries.TimeSeriesTemplate.from_source(source)
    offsets = timeseries.get_t0_offsets(template)
    assert offsets[0] == -template.phase_range[0] and offsets[-1] == -template.phase_range[1]
    assert np.allclose(np.diff(offsets), -1.)
    assert timeseries.get_t0_offsets(template, guess="truth") == (0.,)
    with pytest.raises(ValueError):
        timeseries.get_t0_offsets(template, guess="random")


def test_fit_plateau_observed_late():
    """ a plateau lightcurve observed only after its peak: the t0 scan must find the truth """
    phase = np.linspace(0., 150., 151)
    wave = np.linspace(2000., 11000., 451)
    rise, drop = 1. / (1. + np.exp(-(phase - 8.) / 2.)), 1. / (1. + np.exp((phase - 100.) / 4.))
    light = rise * (1. - phase / 400.) * (0.2 + 0.8 * drop) + 0.01                # smooth rise, plateau, drop
    sed = (wave / 5000.) ** -2
    plateau = sncosmo.TimeSeriesSource(phase, wave, light[:, None] * sed[None] * 1e-15, name="plateau")

    rng = np.random.default_rng(2)
    targets = pandas.DataFrame({"z": [0.02, 0.03, 0.04], "t0": [59_000., 59_010., 59_020.], "amplitude": 1.},
                               index=pandas.Index(np.arange(3), name="index"))
    lcs = []
    for index_, row in targets.iterrows():
        model = sncosmo.Model(plateau)
        model.set(z=row["z"], t0=row["t0"], amplitude=1.)
        time = row["t0"] + np.sort(rng.uniform(40, 140, 50))              # plateau and drop only
        band = rng.choice(BANDS, len(time))
        flux = np.array([model.bandflux(b_, t_, zp=25., zpsys="ab") for b_, t_ in zip(band, time)])
        fluxerr = np.full(len(time), 0.02 * flux.max())
        lcs.append(pandas.DataFrame({"time": time, "band": band, "zp": 25., "zpsys": "ab",
                                     "flux": flux + rng.normal(0, fluxerr), "fluxerr": fluxerr}))
    data = pandas.concat(lcs, keys=targets.index, names=["index", "index_obs"])

    from_data = saltjax.fit_timeseries(data, targets, source=plateau)
    from_truth = saltjax.fit_timeseries(data, targets, source=plateau, guess="truth")
    np.testing.assert_allclose(from_data["chi2"], from_truth["chi2"], rtol=1e-6)
    assert ((from_data["t0"] - targets["t0"]).abs() / from_data["t0_err"]).max() < 5


# tests for the func fit_timeseries
def test_fit_matches_sncosmo(source, simulation):
    pytest.importorskip("iminuit")
    data, targets = simulation
    results = saltjax.fit_timeseries(data, targets, source=source)
    for index_, row in targets.iterrows():
        model = sncosmo.Model(source, effects=[sncosmo.CCM89Dust()], effect_names=["mw"], effect_frames=["obs"])
        model.set(z=row["z"], t0=row["t0"], amplitude=row["amplitude"], mwebv=row["mwebv"])
        lc = data.xs(index_)
        res, _ = sncosmo.fit_lc({k: lc[k].to_numpy() for k in ["time", "band", "flux", "fluxerr", "zp", "zpsys"]},
                                model, ["t0", "amplitude"], bounds={"t0": (row["t0"] - 20, row["t0"] + 20)})
        for k, param_ in enumerate(["t0", "amplitude"]):
            error = res.errors[param_]
            assert abs(results.loc[index_, param_] - res.parameters[k + 1]) / error < 0.05, param_
            assert results.loc[index_, f"{param_}_err"] / error == pytest.approx(1., abs=0.01), param_


def test_fit_output(source, simulation):
    data, targets = simulation
    results = saltjax.fit_timeseries(data, targets, source=source)
    assert list(results.index) == list(targets.index)
    for column_ in ["t0", "amplitude", "t0_err", "amplitude_err", "cov_t0amplitude", "z", "mwebv",
                    "chi2", "ndof", "valid"]:
        assert column_ in results, column_
    assert results["valid"].all()
    assert (results["ndof"] == data.groupby(level=0).size() - 2).all()
    # recovers the truth
    pull = (results["t0"] - targets["t0"]) / results["t0_err"]
    assert pull.abs().max() < 5


def test_fit_inputs_are_equivalent(source, simulation):
    """ source as a template, an instance, and guess='truth' give the same fit """
    data, targets = simulation
    template = timeseries.TimeSeriesTemplate.from_source(source)
    ref = saltjax.fit_timeseries(data, targets, source=source)
    for kwargs in [dict(source=template), dict(source=source, guess="truth")]:
        res = saltjax.fit_timeseries(data, targets, **kwargs)
        np.testing.assert_allclose(res[["t0", "amplitude"]], ref[["t0", "amplitude"]], rtol=1e-6)


def test_fit_warns_and_raises(source, simulation):
    data, targets = simulation
    faint = data.copy()
    faint.loc[0, "flux"] = faint.loc[0, "flux"].to_numpy() * 1e-6       # no detection
    with pytest.warns(UserWarning, match="not fitted"):
        results = saltjax.fit_timeseries(faint, targets, source=source)
    assert 0 not in results.index

    with pytest.raises(ValueError):
        saltjax.fit_timeseries(data, targets, source=source, guess="random")
    with pytest.raises(NotImplementedError):
        saltjax.fit_timeseries(data, targets, source="salt2")


# tests for the plots
def test_show(source, simulation):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    data, targets = simulation
    template = timeseries.TimeSeriesTemplate.from_source(source)
    results = saltjax.fit_timeseries(data, targets, source=template)
    assert isinstance(template.show(), matplotlib.figure.Figure)
    assert isinstance(template.show_fit(data, results, index=0), matplotlib.figure.Figure)

    # with effects: the same as the fit, else a warning
    dust = dict(effects=[sncosmo.CCM89Dust(), sncosmo.CCM89Dust()], effect_names=["mw", "host"],
                effect_frames=["obs", "rest"])
    results = saltjax.fit_timeseries(data, targets.assign(hostebv=0.1), source=template, **dust)
    assert isinstance(template.show_fit(data, results, index=0, **dust), matplotlib.figure.Figure)
    with pytest.warns(UserWarning, match="not in the model"):
        template.show_fit(data, results, index=0)
