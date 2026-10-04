"""Shared fixtures: SALT2 lightcurves simulated with sncosmo."""
import numpy as np
import pandas
import pytest
import sncosmo

BANDS = ["bessellb", "bessellv", "bessellr"]


def _sncosmo_model(row):
    model = sncosmo.Model("salt2", effects=[sncosmo.CCM89Dust()],
                          effect_names=["mw"], effect_frames=["obs"])
    model.set(z=row["z"], t0=row["t0"], x0=row["x0"], x1=row["x1"], c=row["c"],
              mwebv=row["mwebv"])
    return model


@pytest.fixture(scope="session")
def simulation():
    """(data, targets) of 20 SNe Ia observed every ~2 days in 3 bands, with noise."""
    rng = np.random.default_rng(42)
    n = 20
    z = rng.uniform(0.02, 0.08, n)
    targets = pandas.DataFrame({"z": z,
                                "t0": rng.uniform(59_000, 59_100, n),
                                "x0": 2e-3 * (0.03 / z) ** 2,
                                "x1": rng.normal(0, 1, n),
                                "c": rng.normal(0, 0.1, n),
                                "mwebv": rng.uniform(0., 0.3, n)},
                               index=pandas.Index(np.arange(n), name="index"))
    lcs = []
    for i, row in targets.iterrows():
        time = np.sort(row["t0"] + rng.uniform(-20, 50, 60))
        band = rng.choice(BANDS, len(time))
        flux = _sncosmo_model(row).bandflux(band, time, zp=25., zpsys="ab")
        fluxerr = 0.03 * flux.max() + 0.02 * np.abs(flux)
        lcs.append(pandas.DataFrame({"time": time, "band": band, "zp": 25., "zpsys": "ab",
                                     "flux": flux + rng.normal(0, fluxerr), "fluxerr": fluxerr}))
    data = pandas.concat(lcs, keys=targets.index, names=["index", "index_obs"])
    return data, targets


@pytest.fixture(scope="session")
def sncosmo_model():
    """Get the sncosmo model (with Milky Way dust) of a target row."""
    return _sncosmo_model
