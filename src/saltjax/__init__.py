"""Fast batched lightcurve fitting with JAX, reproducing sncosmo's ``fit_lc``.

The entry point is :func:`saltjax.fit_lc`: give it the lightcurves, the targets
and any supported sncosmo source, by name or instance::

    import saltjax
    results = saltjax.fit_lc(data, targets, source="salt2")   # SALT2 / SALT3: t0, x0, x1, c
    results = saltjax.fit_lc(data, targets, source="hsiao")   # time series: t0, amplitude

with any sncosmo dust (``effects``, ``effect_names``, ``effect_frames``), as in
``sncosmo.Model``.

The building blocks are in the submodules: `saltjax.sources` (source families),
`saltjax.tables` (SALT tables), `saltjax.effects` (dust), `saltjax.data` (packed
lightcurves and band integrals), `saltjax.model` (SALT model in JAX),
`saltjax.kernel` (interpolation kernel), `saltjax.fit` and `saltjax.fitter`
(SALT fit) and `saltjax.timeseries` (time-series templates and their fit).
"""

__version__ = "0.4.0"

# Top-level API: what users are expected to call
from .sources import fit_lc # noqa: F401, E402

# Submodules, e.g. saltjax.tables.build_tables
from . import kernel, tables, effects, data, model, fitter, timeseries, sources # noqa: F401, E402
