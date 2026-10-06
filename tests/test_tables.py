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
