import numpy as np
import sncosmo

from saltjax.kernel import kernel_weights


def test_kernel_matches_sncosmo_interpolator():
    src = sncosmo.get_source("salt2")
    wave = src._wave[[300, 700, 1000]]          # wavelength nodes: phase-only interpolation
    phase = np.random.default_rng(0).uniform(src._phase[0] - 2, src._phase[-1] + 2, 500)
    ref = src._model["M0"](phase, wave)
    nodes = src._model["M0"](src._phase, wave)
    idx, w = kernel_weights(phase - src._phase[0], len(src._phase))
    mine = np.einsum("pj,pjw->pw", w, nodes[idx])
    np.testing.assert_allclose(mine, ref, rtol=0, atol=1e-12 * np.abs(ref).max())


def test_kernel_weights_properties():
    u = np.array([-0.5, 0., 0.3, 1.7, 5., 8.6, 9., 9.5])
    idx, w = kernel_weights(u, 10)
    assert idx.shape == w.shape == (len(u), 4)
    inside = (u >= 0) & (u <= 9)
    np.testing.assert_allclose(w.sum(1), inside.astype(float))   # partition of unity inside
    np.testing.assert_allclose(w[~inside], 0.)                     # zero outside
    # interpolating: exact at the nodes
    vals = np.arange(10.) ** 2
    idx, w = kernel_weights(np.arange(10.), 10)
    np.testing.assert_allclose((w * vals[idx]).sum(1), vals)
