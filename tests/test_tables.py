import numpy as np
import pytest
import sncosmo

from saltjax import tables

BANDS = ["bessellb", "bessellv", "bessellr"]


def test_build_tables_shapes():
    tab = tables.build_tables(BANDS, 0.02, 0.05, dz=1e-3)
    nz, nb, nph = len(tab["zgrid"]), len(BANDS), len(tab["phM"])
    assert tab["F"].shape == (nz, nb, 2, nph)
    assert tab["V"].shape == (nz, nb, 4, len(tab["phV"]))
    assert tab["CD"].shape == (nz, nb)
    assert tab["M"].shape == (2, nph, len(tab["M_wave"]))
    assert tab["band_wave"].shape == tab["band_base"].shape == tab["band_wgt"].shape
    assert tab["band_wave"].shape[0] == nb
    assert tab["zgrid"][0] <= 0.02 and tab["zgrid"][-1] >= 0.05
    assert tab["nc"] > 1


def test_get_tables_is_cached():
    src = tables.get_salt_source("salt2")
    tab1 = tables.get_tables(BANDS, src, 0.021, 0.048)
    tab2 = tables.get_tables(BANDS, src, 0.025, 0.045)   # same snapped ranges
    assert tab1 is tab2


def test_salt3_tables():
    tab = tables.build_tables(BANDS, 0.02, 0.03, source="salt3", dz=1e-3)
    assert isinstance(tab["source"], sncosmo.SALT3Source)
    assert not tab["relative_var"]
    np.testing.assert_array_equal(tab["V"][:, :, 3], 1.)      # no error scaling
    assert tables.build_tables(BANDS, 0.02, 0.03, dz=1e-3)["relative_var"]


@pytest.mark.parametrize("name", ["salt2", "salt3", "salt3-nir", "salt2-extended", "salt2-h17"])
def test_supported_sources(name):
    assert tables.get_salt_source(name).name == name


def test_unsupported_source_raises():
    with pytest.raises(NotImplementedError):
        tables.get_salt_source("hsiao")
    with pytest.raises(NotImplementedError):
        tables.get_salt_source(sncosmo.get_source("nugent-sn1a"))


# tests for the class Surface
def test_surface_get_values_at_nodes():
    src = tables.get_salt_source("salt2")
    surfaces = tables.load_surfaces(tables.get_modeldir(src), "salt2", src._SCALE_FACTOR)
    surface = surfaces["M0"]
    assert isinstance(surface, tables.Surface)
    assert surface.values.shape == (len(surface.phase), len(surface.wave))
    # at the wavelength nodes, the interpolation is exact
    np.testing.assert_allclose(surface.get_values(surface.wave[10:20]), surface.values[:, 10:20])
    # outside the grid: 0
    np.testing.assert_array_equal(surface.get_values([surface.wave[0] - 100.]), 0.)


def test_surface_non_uniform_phase_raises():
    with pytest.raises(NotImplementedError):
        tables.Surface(np.array([0., 1., 3.]), np.arange(4.), np.zeros((3, 4)))


def test_check_surface_grids_raises():
    src = tables.get_salt_source("salt2")
    surfaces = dict(tables.load_surfaces(tables.get_modeldir(src), "salt2", src._SCALE_FACTOR))
    tables.check_surface_grids(surfaces)           # no error
    m1 = surfaces["M1"]
    surfaces["M1"] = tables.Surface(m1.phase, m1.wave + 1., m1.values)
    with pytest.raises(NotImplementedError):
        tables.check_surface_grids(surfaces)


# tests for the func get_ncolor_terms
def test_get_ncolor_terms():
    nterms = tables.get_ncolor_terms(2.)
    remainder = 2. ** nterms / np.prod(np.arange(1, nterms + 1))
    assert remainder < 1e-13
    assert tables.get_ncolor_terms(4.) > nterms


# tests for the band grids
def test_band_grids():
    band_grids = tables.get_band_grids(BANDS)
    assert len(band_grids) == len(BANDS)
    band_wave, band_base, band_wgt, band_range = tables.pad_band_grids(band_grids)
    nlambda = max(len(grid_["wave"]) for grid_ in band_grids)
    assert band_wave.shape == band_base.shape == band_wgt.shape == (len(BANDS), nlambda)
    assert band_range.shape == (len(BANDS), 2)
    for i, grid_ in enumerate(band_grids):
        np.testing.assert_array_equal(band_base[i, len(grid_["wave"]):], 0.)   # padding: no weight


# tests for the cache
def test_clear_cache():
    src = tables.get_salt_source("salt2")
    tables.get_tables(BANDS, src, 0.021, 0.048)
    assert tables.cache_info().currsize >= 1
    tables.clear_cache()
    assert tables.cache_info().currsize == 0
