Installation
============

saltjax requires Python 3.10 or later. Install it from GitHub:

.. code-block:: bash

   pip install git+https://github.com/MickaelRigault/saltjax.git

or, for development:

.. code-block:: bash

   git clone https://github.com/MickaelRigault/saltjax.git
   cd saltjax
   pip install -e ".[tests]"

Dependencies
------------

The dependencies (numpy, pandas, scipy, jax, sncosmo, extinction) are installed
automatically. ``pip install jax`` gives the CPU version of JAX; to run on a GPU,
install the matching JAX build (see the `JAX installation guide
<https://docs.jax.dev/en/latest/installation.html>`_).

.. list-table:: Optional packages
   :header-rows: 1
   :widths: 25 75

   * - Package
     - Needed for
   * - ``tqdm``
     - ``progress_bar=True``.
   * - ``iminuit``
     - Running ``sncosmo.fit_lc`` (used in the tests and in the comparisons).
   * - ``skysurvey``
     - Simulating datasets, and fitting them with ``skysurvey.lcfit.fit_salt_jax``.

.. note::

   The SALT2 model files and the bandpasses are downloaded by sncosmo the first
   time you use them, and cached afterwards.

saltjax computes in double precision (``float64``) internally, without changing
your global JAX configuration.

Run the tests
-------------

.. code-block:: bash

   pytest

Build this documentation
------------------------

.. code-block:: bash

   pip install -e ".[docs]"
   cd docs
   make html   # pages are written to docs/_build/html

Notebooks are stored with their outputs and are not executed at build time.
