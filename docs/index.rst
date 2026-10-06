saltjax: fast SALT2 / SALT3 fits with JAX
=========================================

**saltjax** fits the SALT2 or SALT3 supernova model to thousands of lightcurves at once with
`JAX <https://docs.jax.dev>`_. It reproduces `sncosmo <https://sncosmo.readthedocs.io>`_'s
``fit_lc``: same model, same :math:`\chi^2`, same model covariance and the same dust
(any sncosmo dust law, Milky Way and host), and the same parameters and errors, much faster.

.. grid:: 1 2 3 3
   :gutter: 3
   :margin: 4 4 0 0

   .. grid-item-card:: :octicon:`check-circle;1.5em;sd-mr-1` Same answer as sncosmo
      :class-card: sd-border-0
      :shadow: none

      Same interpolation kernel, model covariance and dust as sncosmo:
      parameters agree to :math:`\sim 10^{-2}\sigma` at worst.
      See :doc:`examples/validation`.

   .. grid-item-card:: :octicon:`zap;1.5em;sd-mr-1` Much faster
      :class-card: sd-border-0
      :shadow: none

      Precomputed band integrals, exact gradients, all lightcurves fitted at
      once and compiled once: 16 to 25 times faster than sncosmo per SN on a
      ZTF sample, Milky Way dust included (5 to 6 times on a first call,
      compilation included). Dust does not slow it down.
      See :doc:`examples/speed`.

   .. grid-item-card:: :octicon:`plug;1.5em;sd-mr-1` Easy to adopt
      :class-card: sd-border-0
      :shadow: none

      Two pandas tables in, one table out. Coming from sncosmo? See
      :doc:`sncosmo_users/index`. Using skysurvey? One line.

Installation
------------

.. code-block:: bash

   pip install git+https://github.com/MickaelRigault/saltjax.git

See :doc:`installation` for details.

In one call
-----------

.. code-block:: python

   import saltjax

   # data:    lightcurves, indexed by (target, observation), with columns
   #          time (or mjd), band, flux, fluxerr, zp [, zpsys]
   # targets: one row per target with z [, t0, x1, c, mwebv]
   results = saltjax.fit_salt(data, targets, modelcov=True, progress_bar=True)
   results[["t0", "x0", "x1", "c", "x1_err", "c_err", "chi2", "ndof"]]

Any SALT2 or SALT3 source of sncosmo, and any sncosmo dust, e.g. Milky Way and host dust
with per-target :math:`E(B-V)` and :math:`R_V` (columns ``mwebv``, ``hostebv``, ``hostr_v``
of ``targets``):

.. code-block:: python

   import sncosmo

   results = saltjax.fit_salt(data, targets, source="salt3",
                              effects=[sncosmo.CCM89Dust(), sncosmo.CCM89Dust()],
                              effect_names=["mw", "host"], effect_frames=["obs", "rest"])

See :doc:`examples/dust`.

With `skysurvey <https://skysurvey.readthedocs.io>`_, fit a simulated dataset directly:

.. code-block:: python

   from skysurvey.lcfit import fit_salt_jax
   results = fit_salt_jax(dset)

Where to go next
----------------

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: :octicon:`rocket;1.5em;sd-mr-1` Quickstart
      :link: quickstart
      :link-type: doc

      Simulate, fit and check SNe Ia in five minutes.

   .. grid-item-card:: :octicon:`arrow-switch;1.5em;sd-mr-1` For sncosmo users
      :link: sncosmo_users/index
      :link-type: doc

      From ``sncosmo.fit_lc`` to ``saltjax.fit_salt``.

   .. grid-item-card:: :octicon:`gear;1.5em;sd-mr-1` How it works
      :link: how_it_works
      :link-type: doc

      Why the results are exactly sncosmo's, and why it is fast.

   .. grid-item-card:: :octicon:`check-circle;1.5em;sd-mr-1` Validation
      :link: examples/validation
      :link-type: doc

      Model and fits compared with sncosmo, point by point.

   .. grid-item-card:: :octicon:`stopwatch;1.5em;sd-mr-1` Speed
      :link: examples/speed
      :link-type: doc

      Timings against sncosmo on a realistic ZTF simulation.

   .. grid-item-card:: :octicon:`code;1.5em;sd-mr-1` API reference
      :link: api/index
      :link-type: doc

      Every public function.

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting started

   installation
   quickstart
   how_it_works

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Guides

   sncosmo_users/index
   examples/index

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Reference

   api/index
