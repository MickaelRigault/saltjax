import numpy as np
import pytest
import sncosmo

from saltjax import tables

BANDS = ["bessellb", "bessellv", "bessellr"]


def test_build_tables_shapes():
    tab = tables.build_tables(BANDS, 0.02, 0.05, ebv_max=0.2, dz=1e-3)
    nz, nb, nph = len(tab["zgrid"]), len(BANDS), len(tab["phM"])
    assert tab["K"].shape == (nz, nb, 2, tab["nc"], tab["nm"], nph)
    assert tab["F"].shape == (nz, nb, 2, nph)
    assert tab["V"].shape == (nz, nb, 4, len(tab["phV"]))
    assert tab["CD"].shape == tab["CLb"].shape == (nz, nb)
    assert tab["zgrid"][0] <= 0.02 and tab["zgrid"][-1] >= 0.05
    assert tab["nm"] > 1


def test_no_dust_single_term():
    tab = tables.build_tables(BANDS, 0.02, 0.03, ebv_max=0., dz=1e-3)
    assert tab["nm"] == 1


def test_get_tables_is_cached():
    src = tables.get_salt2_source("salt2")
    tab1 = tables.get_tables(BANDS, src, 0.021, 0.048, ebv_max=0.15)
    tab2 = tables.get_tables(BANDS, src, 0.025, 0.045, ebv_max=0.12)   # same snapped ranges
    assert tab1 is tab2


def test_unsupported_source_raises():
    with pytest.raises(NotImplementedError):
        tables.get_salt2_source("salt3")
    with pytest.raises(NotImplementedError):
        tables.get_salt2_source(sncosmo.get_source("hsiao"))
