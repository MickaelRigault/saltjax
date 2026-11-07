import numpy as np
import pytest
import sncosmo

import saltjax
from saltjax import sources, timeseries
from test_timeseries import make_source, simulate


# tests for the func get_source_kind
def test_get_source_kind():
    kind, source = sources.get_source_kind("salt2")
    assert kind == "salt" and isinstance(source, sncosmo.SALT2Source)
    assert sources.get_source_kind("salt3")[0] == "salt"

    local = make_source()
    assert sources.get_source_kind(local) == ("timeseries", local)
    template = timeseries.TimeSeriesTemplate.from_source(local)
    assert sources.get_source_kind(template) == ("timeseries", template)


def test_get_source_kind_unsupported_raises():
    stretch = sncosmo.StretchSource(np.linspace(0, 10, 11), np.linspace(3000, 9000, 61), np.ones((11, 61)))
    with pytest.raises(NotImplementedError, match="StretchSource"):
        sources.get_source_kind(stretch)


# tests for the func fit_lc
def test_fit_lc_salt_equals_fit_salt(simulation):
    data, targets = simulation
    results = saltjax.fit_lc(data, targets, source="salt2", modelcov=False)
    ref = saltjax.fit.fit_salt(data, targets, source="salt2", modelcov=False)
    np.testing.assert_allclose(results[["t0", "x0", "x1", "c"]].astype(float),
                               ref[["t0", "x0", "x1", "c"]].astype(float), rtol=1e-12)


def test_fit_lc_timeseries_equals_fit_timeseries():
    source = make_source()
    data, targets = simulate(source, n=5)
    for source_ in [source, timeseries.TimeSeriesTemplate.from_source(source)]:
        results = saltjax.fit_lc(data, targets, source=source_)
        ref = timeseries.fit_timeseries(data, targets, source=source)
        np.testing.assert_allclose(results[["t0", "amplitude"]], ref[["t0", "amplitude"]], rtol=1e-12)


def test_fit_lc_wrong_option_raises(simulation):
    source = make_source()
    data, targets = simulate(source, n=3)
    with pytest.raises(TypeError, match="modelcov"):       # SALT option, time-series source
        saltjax.fit_lc(data, targets, source=source, modelcov=True)

    data, targets = simulation
    with pytest.raises(TypeError, match="t0_offsets"):     # time-series option, SALT source
        saltjax.fit_lc(data, targets, source="salt2", t0_offsets=(0.,))
