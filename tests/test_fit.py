import numpy as np
import pandas
import pytest
import sncosmo

import saltjax

PARAMS = ["t0", "x0", "x1", "c"]


def _fit_lc(data, targets, sncosmo_model, modelcov, phase_range=(-10, 40), source="salt2"):
    """sncosmo.fit_lc per target, started at the truth with wide bounds."""
    out = {}
    for name, row in targets.iterrows():
        lc = data.xs(name)
        phase = (lc["time"] - row["t0"]) / (1 + row["z"])
        lc = lc[phase.between(*phase_range)]
        lc = {k: lc[k].to_numpy() for k in ["time", "band", "flux", "fluxerr", "zp", "zpsys"]}
        model = sncosmo_model(row, source)
        bounds = {"t0": (row["t0"] - 15, row["t0"] + 15), "x1": (row["x1"] - 5, row["x1"] + 5),
                  "c": (row["c"] - 1, row["c"] + 1)}
        res, _ = sncosmo.fit_lc(lc, model, PARAMS, bounds=bounds, modelcov=modelcov)
        out[name] = {p: v for p, v in zip(res.param_names, res.parameters)} | \
                    {f"{p}_err": res.errors[p] for p in PARAMS}
    return pandas.DataFrame(out).T


@pytest.mark.parametrize("source", ["salt2", "salt3"])
@pytest.mark.parametrize("modelcov", [False, True])
def test_fit_matches_sncosmo(simulation, sncosmo_model, modelcov, source):
    pytest.importorskip("iminuit")
    data, targets = simulation
    jres = saltjax.fit_salt(data, targets, modelcov=modelcov, source=source)
    sres = _fit_lc(data, targets, sncosmo_model, modelcov, source=source)
    assert set(jres.index) == set(sres.index)
    for p in PARAMS:
        pull = (jres.loc[sres.index, p] - sres[p]) / sres[f"{p}_err"]
        assert pull.abs().max() < 0.05, p
        ratio = jres.loc[sres.index, f"{p}_err"] / sres[f"{p}_err"]
        assert (ratio - 1).abs().max() < 0.02, p
    if modelcov:
        assert jres["converged"].all()


def test_output_columns(simulation):
    data, targets = simulation
    res = saltjax.fit_salt(data, targets, modelcov=False)
    expected = {"z", "chi2", "ndof", "converged", "nrefit", "mwebv", "mwr_v"} | set(PARAMS) | \
        {f"{p}_err" for p in PARAMS} | {f"cov_{p}{q}" for p in PARAMS for q in PARAMS}
    assert expected <= set(res.columns)
    assert res.index.name == targets.index.name
    np.testing.assert_allclose(res["z"], targets.loc[res.index, "z"])


def test_guess_truth_equals_guess_data(simulation):
    data, targets = simulation
    r1 = saltjax.fit_salt(data, targets, modelcov=False, guess="data")
    r2 = saltjax.fit_salt(data, targets, modelcov=False, guess="truth")
    pull = ((r1[PARAMS] - r2.loc[r1.index, PARAMS]) / r1[[f"{p}_err" for p in PARAMS]].to_numpy()).abs()
    assert pull.max().max() < 0.01


def test_no_dust_column(simulation):
    data, targets = simulation
    res = saltjax.fit_salt(data, targets.drop(columns="mwebv"), modelcov=False)
    assert "mwebv" not in res.columns


def test_target_without_detection_is_dropped(simulation):
    data, targets = simulation
    data = data.copy()
    data.loc[0, "flux"] = 0.
    with pytest.warns(UserWarning, match="not fitted"):
        res = saltjax.fit_salt(data, targets, modelcov=False)
    assert 0 not in res.index and len(res) == len(targets) - 1


def test_bad_guess_raises(simulation):
    data, targets = simulation
    with pytest.raises(ValueError):
        saltjax.fit_salt(data, targets, guess="random")


def test_progress_bar(simulation):
    pytest.importorskip("tqdm")
    data, targets = simulation
    res = saltjax.fit_salt(data, targets, modelcov=False, progress_bar=True)
    assert len(res) == len(targets)


def test_out_of_range_band_is_dropped(simulation):
    data, targets = simulation
    data = data.copy()
    rows = data.index[data.index.get_level_values(0) == 0][:5]
    data.loc[rows, "band"] = "desy"           # red edge beyond SALT2 at these redshifts
    with pytest.warns(UserWarning, match="out of the model wavelength range"):
        res = saltjax.fit_salt(data, targets, modelcov=False, phase_range=None)
    ref = saltjax.fit_salt(data.drop(index=rows), targets, modelcov=False, phase_range=None)
    assert 0 in res.index
    np.testing.assert_allclose(res.loc[0, PARAMS].astype(float), ref.loc[0, PARAMS].astype(float))


def test_underconstrained_target_is_skipped(simulation):
    data, targets = simulation
    brightest = data.loc[0, "flux"].nlargest(3).index      # 3 detected points only
    keep = ~((data.index.get_level_values(0) == 0) & ~data.index.get_level_values(1).isin(brightest))
    with pytest.warns(UserWarning, match="no degree of freedom"):
        res = saltjax.fit_salt(data[keep], targets, modelcov=False, phase_range=None)
    assert 0 not in res.index and len(res) == len(targets) - 1


def test_valid_column(simulation):
    data, targets = simulation
    res = saltjax.fit_salt(data, targets, modelcov=False)
    assert res["valid"].dtype == bool and res["valid"].all()
    assert np.isfinite(res[[f"{p}_err" for p in PARAMS]].to_numpy()).all()


def test_unregistered_source_needs_modeldir(simulation):
    data, targets = simulation
    modeldir = sncosmo.builtins.DATADIR.abspath("models/salt3/salt3-f22", isdir=True)
    src = sncosmo.SALT3Source(modeldir=modeldir)          # not registered: no name
    with pytest.raises(ValueError, match="modeldir"):
        saltjax.fit_salt(data, targets, modelcov=False, source=src)
    res = saltjax.fit_salt(data, targets, modelcov=False, source=src, modeldir=modeldir)
    ref = saltjax.fit_salt(data, targets, modelcov=False, source="salt3", version="2.0")
    np.testing.assert_allclose(res[PARAMS].astype(float), ref.loc[res.index, PARAMS].astype(float),
                               rtol=1e-10)
