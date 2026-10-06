For sncosmo users
=================

If you fit SALT2 with ``sncosmo.fit_lc``, saltjax gives you the same parameters and
errors for many lightcurves at once. Here is what changes.

From one table per lightcurve to two tables
-------------------------------------------

sncosmo fits one ``astropy.table.Table`` at a time, with the redshift set on the
model. saltjax fits *all* lightcurves in one call, from two :class:`pandas.DataFrame`:

``data``
   All the lightcurves, indexed by (target, observation), with the same columns as
   sncosmo: ``time`` (or ``mjd``), ``band``, ``flux``, ``fluxerr``, ``zp`` and
   ``zpsys`` (only ``'ab'``).

``targets``
   One row per target: ``z`` (required), the parameters of the effects (e.g.
   ``mwebv`` for the Milky Way extinction, ``hostebv`` and ``hostr_v`` for host
   dust; optional), and ``t0`` if you select data with ``phase_range``.

.. code-block:: python

   import pandas
   import saltjax

   # lcs: list of sncosmo-style astropy Tables, with meta["z"] (and meta["mwebv"])
   data = pandas.concat([lc.to_pandas() for lc in lcs], keys=range(len(lcs)))
   targets = pandas.DataFrame([{"z": lc.meta["z"], "mwebv": lc.meta.get("mwebv", 0)}
                               for lc in lcs])

   results = saltjax.fit_salt(data, targets, phase_range=None)

The notebook below shows the full conversion, and compares saltjax with
``sncosmo.fit_lc`` on the same lightcurves.

Translation table
-----------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - ``sncosmo.fit_lc``
     - ``saltjax.fit_salt``
   * - ``data`` (one Table)
     - ``data`` (all lightcurves) and ``targets`` (z, mwebv, ...)
   * - ``model=sncosmo.Model("salt2")``
     - ``source="salt2"`` (any registered ``sncosmo.SALT2Source`` or
       ``sncosmo.SALT3Source``, e.g. ``"salt3"``, ``"salt2-extended"``)
   * - ``effects``, ``effect_names``, ``effect_frames`` of ``sncosmo.Model``
     - the same ``effects``, ``effect_names``, ``effect_frames`` arguments
       (``"obs"`` or ``"rest"``); see :doc:`/examples/dust`
   * - ``model.set(mwebv=..., hostebv=..., hostr_v=...)``
     - columns ``mwebv``, ``hostebv``, ``hostr_v`` of ``targets`` (one value per
       target; a missing column means the effect's own value)
   * - ``CCM89Dust`` effect named ``mw`` (``frame="obs"``)
     - the default when ``effects`` is not given: ``targets["mwebv"]``
       (``mwebv_key``) and ``mw_r_v``
   * - ``vparam_names=['t0', 'x0', 'x1', 'c']``
     - always these four parameters; ``z`` is fixed
   * - ``modelcov=True``
     - ``modelcov=True`` (same procedure)
   * - ``bounds``
     - not supported: parameters are unbounded
   * - ``guess_t0``, ``guess_amplitude``, ``minsnr``
     - automatic initial guess (``guess="data"``); ``minsnr`` drops targets
       without any point above that S/N
   * - ``phase_range`` (relative to the fitted :math:`t_0`, with refits)
     - ``phase_range`` relative to ``targets["t0"]`` (a known or preliminary
       :math:`t_0`); use ``None`` to keep all the data
   * - ``result.parameters``, ``result.errors``, ``result.covariance``
     - columns ``t0, x0, x1, c``, ``*_err`` and ``cov_{p}{q}`` of the output table

.. toctree::
   :maxdepth: 1

   from_sncosmo
