.. _api:

API reference
=============

.. currentmodule:: saltjax

Top-level API
-------------

The main entry point, directly accessible as ``saltjax.fit_salt``.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   fit_salt

The steps of :func:`fit_salt` (``saltjax.fit``), to check or reproduce a fit.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   fit.select_fittable
   fit.get_initial_theta
   fit.get_bucket_length
   fit.get_batches
   fit.fit_batches
   fit.get_results_table
   fit.check_results

The building blocks below live in their submodules (e.g.
``saltjax.tables.build_tables``). You do not need them to fit lightcurves, but
they let you check or reproduce each step of the fit (see
:doc:`/how_it_works`).

Model tables (``saltjax.tables``)
---------------------------------

SALT2 / SALT3 surfaces, band integration grids, and the model covariance tabulated on a
redshift grid.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   tables.build_tables
   tables.get_tables
   tables.get_salt_source

The steps of :func:`~tables.build_tables`, the SALT surfaces, and the cache.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   tables.Surface
   tables.load_surfaces
   tables.check_surface_grids
   tables.get_modeldir
   tables.get_band_grids
   tables.get_max_colorlaw_deviation
   tables.get_ncolor_terms
   tables.get_covariance_tables
   tables.pad_band_grids
   tables.clear_cache
   tables.cache_info

Effects (``saltjax.effects``)
-----------------------------

Propagation effects (dust), as in ``sncosmo.Model`` (see :doc:`/examples/dust`).

.. autosummary::
   :toctree: generated/
   :nosignatures:

   effects.get_effects
   effects.parameter_table
   effects.get_extinction
   effects.transmission
   effects.wave_limits

Data (``saltjax.data``)
-----------------------

Pack lightcurves into arrays, with the band integrals of each target.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   data.pack_lightcurves
   data.band_integrals

The steps of :func:`~data.pack_lightcurves` and :func:`~data.band_integrals`, to
check or plot intermediate products.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   data.get_colorlaw_terms
   data.get_band_integrand
   data.get_wavelength_kernels
   data.project_surfaces
   data.get_time_key
   data.check_magsys
   data.get_target_position
   data.select_covered_points
   data.select_phase_range
   data.get_lightcurve_slots
   data.get_lightcurve_arrays
   data.get_redshift_tables

Model (``saltjax.model`` and ``saltjax.kernel``)
------------------------------------------------

Evaluate the model (fluxes and model covariance) of a target in JAX.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   model.get_model_functions
   model.phase_grids
   model.interpolate_phase
   model.bandflux
   model.rvar_and_cdisp
   kernel.kernel_weights
   kernel.keys_kernel

Low-level fitter (``saltjax.fitter``)
-------------------------------------

The compiled, batched functions used by :func:`fit_salt`.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   fitter.fit_batch
   fitter.initial_batch
   fitter.compile_all
   fitter.get_compiled
   fitter.enable_x64
   fitter.clear_cache
   fitter.get_ncompiled

Fit of a single target (traced and compiled by :func:`~fitter.fit_batch`).

.. autosummary::
   :toctree: generated/
   :nosignatures:

   fitter.fit_one
   fitter.levenberg_marquardt
   fitter.get_cinv_operator
