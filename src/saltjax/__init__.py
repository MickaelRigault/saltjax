"""Fast batched SALT2 / SALT3 lightcurve fitting with JAX, reproducing sncosmo.

The entry point is :func:`saltjax.fit_salt`::

    import saltjax
    results = saltjax.fit_salt(data, targets)

The building blocks are in the submodules: `saltjax.tables` (model tables),
`saltjax.effects` (dust), `saltjax.data` (packed lightcurves and band
integrals), `saltjax.model` (model fluxes and covariance in JAX),
`saltjax.kernel` (interpolation kernel) and `saltjax.fitter` (compiled fit).
"""

__version__ = "0.3.0"

# Top-level API: what users are expected to call
from .fit import fit_salt # noqa: F401, E402

# Submodules, e.g. saltjax.tables.build_tables
from . import kernel, tables, effects, data, model, fitter # noqa: F401, E402
