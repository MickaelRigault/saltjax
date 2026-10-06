Examples
========

Complete, executed notebooks. Download one (button at the top of each page) and adapt it.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`check-circle;1.5em;sd-mr-1` Validation against sncosmo
      :link: validation
      :link-type: doc

      Interpolation, model fluxes, model covariance, fitted parameters, errors
      and :math:`\chi^2`, compared with sncosmo.

   .. grid-item-card:: :octicon:`globe;1.5em;sd-mr-1` Dust and other effects
      :link: dust
      :link-type: doc

      Milky Way and host dust with any sncosmo dust law, per-target :math:`E(B-V)`
      and :math:`R_V`, user-defined effects, compared with sncosmo.

   .. grid-item-card:: :octicon:`stopwatch;1.5em;sd-mr-1` Speed: saltjax vs sncosmo
      :link: speed
      :link-type: doc

      Timing breakdown and scaling with the number of SNe on a realistic ZTF
      simulation.

.. toctree::
   :hidden:
   :maxdepth: 1

   validation
   dust
   speed
