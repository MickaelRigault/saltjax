"""Fast batched SALT2 / SALT3 lightcurve fitting with JAX, reproducing sncosmo.

The entry points are :func:`saltjax.fit_salt` (SALT2 / SALT3) and
:func:`saltjax.fit_timeseries` (time-series templates, e.g. 'hsiao')::

    import saltjax
    results = saltjax.fit_salt(data, targets)
    results = saltjax.fit_timeseries(data, targets, source="hsiao")

The building blocks are in the submodules: `saltjax.tables` (model tables),
`saltjax.effects` (dust), `saltjax.data` (packed lightcurves and band
integrals), `saltjax.model` (model fluxes and covariance in JAX),
`saltjax.kernel` (interpolation kernel), `saltjax.fitter` (compiled fit) and
`saltjax.timeseries` (time-series templates).
"""

__version__ = "0.3.0"

# Top-level API: what users are expected to call
from .fit import fit_salt # noqa: F401, E402
from .timeseries import fit_timeseries # noqa: F401, E402

# Submodules, e.g. saltjax.tables.build_tables
from . import kernel, tables, effects, data, model, fitter, timeseries # noqa: F401, E402
