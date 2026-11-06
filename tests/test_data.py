import numpy as np
import pytest
import sncosmo

from saltjax.tables import build_tables
from saltjax.data import band_integrals
from saltjax.effects import get_effects
from saltjax import data as sjdata

BANDS = ["bessellb", "bessellv", "bessellr"]
# a near-infrared band out of the SALT2 wavelength range, defined locally (no download)
NIR_BAND = "saltjax_test_nir"
sncosmo.register(sncosmo.Bandpass([11_000., 11_500., 13_000., 13_500.], [0., 1., 1., 0.], name=NIR_BAND),
                 force=True)


@pytest.fixture(scope="module")
def tables():
    return build_tables(BANDS, 0.02, 0.08, dz=1e-3)


# tests for the func check_magsys
def test_check_magsys(simulation):
    data, _ = simulation
    sjdata.check_magsys(data)                       # all 'ab': no error
    sjdata.check_magsys(data.drop(columns="zpsys"))  # no zpsys: assumed 'ab'
    with pytest.raises(NotImplementedError):
        sjdata.check_magsys(data.assign(zpsys="vega"))


# tests for the func get_time_key
def test_get_time_key(simulation):
    data, _ = simulation
    assert sjdata.get_time_key(data) == "time"
    assert sjdata.get_time_key(data.rename(columns={"time": "mjd"})) == "mjd"
    with pytest.raises(ValueError):
        sjdata.get_time_key(data.drop(columns="time"))


# tests for the func get_target_position
def test_get_target_position(simulation):
    data, targets = simulation
    indexes = list(targets.index[::-1])             # reversed order
    position = sjdata.get_target_position(data, indexes)
    assert position.shape == (len(data),)
    assert np.all(np.asarray(indexes)[position] == data.index.get_level_values(0))


# tests for the func get_lightcurve_slots
def test_get_lightcurve_slots():
    position = np.array([1, 0, 1, 1, 0])
    slot, npoints_max = sjdata.get_lightcurve_slots(position, pad_to=4)
    assert list(slot) == [0, 0, 1, 2, 1], "points must keep their order within each target"
    assert npoints_max == 4

    slot, npoints_max = sjdata.get_lightcurve_slots(np.array([], int), pad_to=4)
    assert len(slot) == 0 and npoints_max == 4


# tests for the func select_phase_range
def test_select_phase_range(simulation):
    data, targets = simulation
    position = sjdata.get_target_position(data, list(targets.index))
    t0, z = targets["t0"].to_numpy(), targets["z"].to_numpy()
    selected, position_sel = sjdata.select_phase_range(data, position, t0, z, (-10, 40))
    phase = (selected["time"].to_numpy() - t0[position_sel]) / (1 + z[position_sel])
    assert len(selected) == len(position_sel) < len(data)
    assert np.all((phase >= -10) & (phase <= 40))


# tests for the func select_covered_points
def test_select_covered_points_warns(simulation):
    data, targets = simulation
    tab = build_tables(BANDS + [NIR_BAND], 0.02, 0.08, dz=1e-3)
    data = data.copy()
    data.loc[data.index[:5], "band"] = NIR_BAND      # out of the SALT2 wavelength range
    position = sjdata.get_target_position(data, list(targets.index))
    effects = get_effects(targets)
    with pytest.warns(UserWarning, match="5 points dropped"):
        selected, position_sel = sjdata.select_covered_points(data, position, tab,
                                                              targets["z"].to_numpy(), effects)
    assert len(selected) == len(position_sel) == len(data) - 5


# tests for the func get_lightcurve_arrays
def test_get_lightcurve_arrays(simulation, tables):
    data, targets = simulation
    position = sjdata.get_target_position(data, list(targets.index))
    arrays = sjdata.get_lightcurve_arrays(data, position, len(targets), tables["bands"], pad_to=16)
    assert set(arrays) == set(sjdata.LC_KEYS)
    assert arrays["flux"].shape[0] == len(targets) and arrays["flux"].shape[1] % 16 == 0
    assert arrays["mask"].sum() == len(data)
    assert np.all(arrays["err"] > 0), "padded points must have err > 0"


# tests for the func get_redshift_tables
def test_get_redshift_tables(tables):
    z = np.array([0.03, 0.05])
    ztables = sjdata.get_redshift_tables(tables, z)
    assert set(ztables) == {"F", "Fe", "V", "Ve", "CD"}
    for key_, table_ in ztables.items():
        assert table_.shape == (len(z),) + tables[key_].shape[1:]


# tests for the band integrals building blocks
def test_colorlaw_terms(tables):
    z = np.array([0.03, 0.05])
    rest_wave = (1 / (1 + z))[:, None, None] * tables["band_wave"][None]
    colorlaw_mean, colorlaw_dev = sjdata.get_colorlaw_terms(tables, rest_wave)
    assert colorlaw_mean.shape == (len(z), len(BANDS))
    # by construction, the deviation has zero weighted band mean
    band_weights = tables["band_wgt"]
    weighted_mean = (colorlaw_dev * band_weights).sum(-1) / band_weights.sum(-1)
    assert np.allclose(weighted_mean, 0, atol=1e-12)


def test_band_integrals_steps(tables):
    """ band_integrals() == the building blocks put together (no effect) """
    z = np.array([0.03, 0.05])
    a = 1 / (1 + z)
    rest_wave = a[:, None, None] * tables["band_wave"][None]
    _, colorlaw_dev = sjdata.get_colorlaw_terms(tables, rest_wave)
    integrand = sjdata.get_band_integrand(tables, a, np.ones_like(rest_wave), colorlaw_dev)
    assert integrand.shape == rest_wave.shape + (tables["nc"],)

    idx, weights, weights_linear = sjdata.get_wavelength_kernels(rest_wave, tables["M_wave"])
    assert np.allclose(weights_linear.sum(-1)[weights.sum(-1) > 0], 1)
    lo, hi = idx.min((0, 2, 3)), idx.max((0, 2, 3)) + 1
    K = sjdata.project_surfaces(tables["M"], idx, weights, integrand, lo, hi)

    K_ref, _, _ = band_integrals(tables, z)
    np.testing.assert_allclose(K, K_ref)


def test_band_integrals_empty(tables):
    K, Ke, CLb = band_integrals(tables, [])
    assert len(K) == len(Ke) == len(CLb) == 0
