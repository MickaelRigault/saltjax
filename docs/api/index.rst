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

Effects (``saltjax.effects``)
-----------------------------

Propagation effects (dust), as in ``sncosmo.Model`` (see :doc:`/examples/dust`).

.. autosummary::
   :toctree: generated/
   :nosignatures:

   effects.get_effects
   effects.transmission

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
   model.bandflux
   model.rvar_and_cdisp
   kernel.kernel_weights

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
