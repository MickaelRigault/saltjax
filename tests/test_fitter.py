import numpy as np
import jax.numpy as jnp
import pytest

from saltjax import fitter
from saltjax.tables import build_tables
from saltjax.data import pack_lightcurves, LC_KEYS, FIT_KEYS
from saltjax.model import phase_grids, bandflux

PARAMS = ["t0", "x0", "x1", "c"]


@pytest.fixture(scope="module")
def one_target(simulation):
    """(packed arrays, truth theta, grid, nbands) of the first target."""
    data, targets = simulation
    tab = build_tables(sorted(data["band"].unique()), targets["z"].min(), targets["z"].max())
    with fitter.enable_x64(True):
        arr = pack_lightcurves(data, targets, tab)
    npoints = int(arr["mask"][0].sum())
    packed = {k: jnp.asarray(arr[k][0][:npoints] if k in LC_KEYS else arr[k][0]) for k in FIT_KEYS}
    theta = targets.iloc[0][PARAMS].to_numpy(float)
    return packed, theta, phase_grids(tab), len(tab["bands"])


# tests for the func enable_x64
def test_enable_x64():
    with fitter.enable_x64(True):
        assert jnp.ones(2).dtype == jnp.float64


# tests for the func get_cinv_operator
def test_cinv_operator_matches_dense(one_target):
    packed, theta, grid, nbands = one_target
    with fitter.enable_x64(True):
        theta = jnp.asarray(theta)
        # without model covariance: 1 / err^2
        v = jnp.ones(len(packed["flux"]))
        np.testing.assert_allclose(fitter.get_cinv_operator(packed)(v), 1 / packed["err"] ** 2)

        # with model covariance: the Woodbury operator is the inverse of the dense matrix
        from saltjax.model import rvar_and_cdisp
        flux = bandflux(theta, packed, grid)
        rvar, cdisp = rvar_and_cdisp(theta, packed, grid)
        same_band = packed["band"][:, None] == packed["band"][None, :]
        cov = jnp.diag(packed["err"] ** 2 + flux ** 2 * rvar) + \
            same_band * (cdisp * flux)[:, None] * (cdisp * flux)[None, :]
        cinv = fitter.get_cinv_operator(packed, theta, grid, nbands)
        np.testing.assert_allclose(cinv(jnp.eye(len(flux))) @ cov, np.eye(len(flux)), atol=1e-8)


# tests for the func fit_one
def test_fit_one_matches_fit_batch(one_target):
    packed, theta, grid, nbands = one_target
    with fitter.enable_x64(True):
        theta0 = jnp.asarray(theta * np.array([1., 1.1, 1., 1.]) + np.array([0.5, 0., 0.2, 0.02]))
        single = fitter.fit_one(theta0, packed, grid, nbands)
        batch = fitter.fit_batch(theta0[None], {k: v[None] for k, v in packed.items()}, grid, nbands)
    # same fit up to rounding (compiled vs not compiled), in units of the fit errors
    sigma = np.sqrt(np.diag(batch["cov"][0]))
    assert np.all(np.abs(single["theta"] - batch["theta"][0]) / sigma < 1e-6)
    np.testing.assert_allclose(single["theta"], theta, rtol=0.1)   # close to the truth


# tests for the cache
def test_clear_cache(one_target):
    packed, theta, grid, nbands = one_target
    with fitter.enable_x64(True):
        args = (jnp.asarray(theta)[None], {k: v[None] for k, v in packed.items()})
        fitter.get_compiled(fitter.fit_batch, args, dict(grid=grid, nbands=nbands))
    assert fitter.get_ncompiled() >= 1
    fitter.clear_cache()
    assert fitter.get_ncompiled() == 0
