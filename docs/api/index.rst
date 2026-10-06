.. _api:

API reference
=============

.. currentmodule:: saltjax

Fitting
-------

The main entry point.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   fit_salt

Model tables
------------

SALT2 / SALT3 surfaces, band integration grids, and the model covariance tabulated on a
redshift grid (see :doc:`/how_it_works`).

.. autosummary::
   :toctree: generated/
   :nosignatures:

   build_tables
   get_tables
   get_salt_source

Effects
-------

Propagation effects (dust), as in ``sncosmo.Model`` (see :doc:`/examples/dust`).

.. autosummary::
   :toctree: generated/
   :nosignatures:

   get_effects
   effects.transmission

Data and model
--------------

Pack lightcurves into arrays (with the band integrals of each target), and evaluate
the model (fluxes and model covariance) of a target in JAX.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pack_lightcurves
   band_integrals
   get_model_functions
   kernel_weights
   model.phase_grids
   model.bandflux
   model.rvar_and_cdisp

Low-level fitter
----------------

The compiled, batched functions used by :func:`fit_salt`.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   fitter.fit_batch
   fitter.initial_batch
   fitter.compile_all
   fitter.get_compiled
