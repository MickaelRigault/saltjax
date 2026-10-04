import numpy as np
import jax
import jax.numpy as jnp
import pytest

from saltjax import build_tables, pack_lightcurves, get_model_functions
from saltjax.data import LC_KEYS, FIT_KEYS


@pytest.fixture(scope="module")
def packed(simulation):
    data, targets = simulation
    tab = build_tables(sorted(data["band"].unique()), targets["z"].min(), targets["z"].max(),
                       ebv_max=targets["mwebv"].max())
    with jax.enable_x64(True):
        arr = pack_lightcurves(data, targets, tab, phase_range=None)
    return tab, arr


def _target_arrays(arr, i):
    n = int(arr["mask"][i].sum())
    return {k: jnp.asarray(arr[k][i][:n] if k in LC_KEYS else arr[k][i]) for k in FIT_KEYS}


def test_pack_lightcurves(simulation, packed):
    data, targets = simulation
    tab, arr = packed
    assert arr["t"].shape[0] == len(targets)
    assert arr["t"].shape[1] % 16 == 0
    np.testing.assert_array_equal(arr["mask"].sum(1), data.groupby(level=0).size().to_numpy())
    np.testing.assert_allclose(arr["ebv"], targets["mwebv"])


def test_model_matches_sncosmo_bandfluxcov(simulation, packed, sncosmo_model):
    data, targets = simulation
    tab, arr = packed
    with jax.enable_x64(True):
        flux, rcov, mcov = get_model_functions(tab)
        for i, (_, row) in enumerate(targets.iloc[:8].iterrows()):
            d = _target_arrays(arr, i)
            row = row.copy()
            row[["t0", "x0", "x1", "c"]] = [row["t0"] + 1.3, 2e-3, 0.7, 0.08]
            theta = jnp.asarray(row[["t0", "x0", "x1", "c"]].to_numpy(float))
            bands = np.array(tab["bands"])[np.asarray(d["band"])]
            ref_flux, ref_cov = sncosmo_model(row).bandfluxcov(bands, np.asarray(d["t"]),
                                                               zp=25., zpsys="ab")
            np.testing.assert_allclose(np.asarray(flux(theta, d)), ref_flux,
                                       rtol=1e-5, atol=1e-6 * np.abs(ref_flux).max())
            np.testing.assert_allclose(np.asarray(mcov(theta, d)), ref_cov,
                                       rtol=1e-3, atol=1e-6 * np.abs(ref_cov).max())


def test_non_ab_raises(simulation, packed):
    data, targets = simulation
    tab, _ = packed
    data = data.assign(zpsys="vega")
    with pytest.raises(NotImplementedError):
        pack_lightcurves(data, targets, tab)
