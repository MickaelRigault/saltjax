How it works
============

saltjax gives the same results as ``sncosmo.fit_lc`` because it computes the same
model and the same :math:`\chi^2`. It is faster because it computes them in a
different order. This page explains both.

The SALT2 band flux
-------------------

For a target at redshift :math:`z`, with :math:`a = 1/(1+z)`, the flux in band
:math:`b` at observer-frame time :math:`t` (rest-frame phase
:math:`p = a\,(t - t_0)`) is, as in sncosmo,

.. math::

   F_b(t) = \frac{10^{0.4\,zp}}{F^{AB}_b}\sum_{\lambda} a\,x_0
   \left[M_0(p, a\lambda) + x_1\,M_1(p, a\lambda)\right]
   10^{-0.4\,c\,CL(a\lambda)}\,10^{-0.4\,E(B-V)\,k(\lambda)}\,
   T_b(\lambda)\,\frac{\lambda\,\Delta\lambda}{hc}

where :math:`M_0, M_1` are the SALT2 surfaces, :math:`CL` the color law,
:math:`k(\lambda)` the CCM89 Milky Way extinction (observer frame) and
:math:`T_b` the bandpass sampled every :math:`\Delta\lambda = 5` Å.

sncosmo evaluates this sum over a few hundred wavelengths every time the
minimizer asks for a :math:`\chi^2`, typically hundreds of times per lightcurve.

What saltjax precomputes
------------------------

**Interpolation, done once and exactly.**
sncosmo interpolates :math:`M_0, M_1` (and the model-covariance surfaces) with a
bicubic-convolution kernel (Keys, :math:`a=-0.5`), which becomes bilinear when the
phase is in the first or last interval of the grid. The band sum is linear in the
surfaces, so saltjax integrates the surfaces through the bands *at the native phase
nodes* (with the same kernel in wavelength), then applies *the same kernel in phase*
when fitting. This gives the same result as sncosmo, not an approximation of it.

**Color and dust as exact series.**
The color and dust factors depend on wavelength, so they cannot be pulled out of
the sum directly. saltjax writes each one as an exact band-mean factor times a power
series of the deviation from that mean:

.. math::

   10^{-0.4\,c\,CL(a\lambda)} = 10^{-0.4\,c\,\overline{CL}_b}
   \sum_{n} \frac{\left(-0.4\ln 10\; c\,[CL(a\lambda) - \overline{CL}_b]\right)^n}{n!}

The number of terms is chosen so that the remainder is below :math:`10^{-13}` for
:math:`|c| \le` ``c_max`` (1.5 by default) and for the largest Milky Way E(B-V) of
the sample. E(B-V) is fixed, so the dust series is summed once per target.

**Redshift tables.**
These band integrals are tabulated once on a fine redshift grid
(:math:`\Delta z = 2.5\times10^{-4}` by default) and interpolated in redshift with
the same cubic kernel. **This is the only approximation:** model fluxes differ from
sncosmo's by less than :math:`10^{-6}` in relative terms.

Evaluating the model during the fit therefore costs a few operations per point
instead of a sum over hundreds of wavelengths.

The fit
-------

**Same** :math:`\chi^2`.
:math:`\chi^2 = r^T C^{-1} r` with :math:`r` the flux residuals. Without model
covariance, :math:`C` is the diagonal of the flux variances.

**Same model covariance.**
With ``modelcov=True``, saltjax follows ``sncosmo.fit_lc``:

#. It fits once without model covariance.
#. It refits with :math:`C = \mathrm{diag}(\sigma^2) + f f^T \, R(\theta)`, where :math:`R` is
   sncosmo's relative model covariance (``SALT2Source.bandflux_rcov``) computed at the
   start of each refit and kept fixed during it.
#. It stops once every parameter moves by less than :math:`0.1\sigma`.

:math:`R` is a diagonal plus one constant block per band. So :math:`C^{-1}` is applied
exactly with the Woodbury identity, at a cost proportional to the number of points.

**Minimizer.**
A Levenberg–Marquardt algorithm with exact (automatic) Jacobians. It starts from a
coarse grid in (:math:`t_0`, :math:`c`) around the highest-S/N point, with
:math:`x_0` solved linearly.

**Errors.**
The errors are :math:`2H^{-1}`, with :math:`H` the exact Hessian of the
:math:`\chi^2`, as with Minuit's HESSE in sncosmo.

Speed
-----

**All lightcurves at once.**
All lightcurves are fitted together: the one-lightcurve fit is vectorized
(``jax.vmap``) and compiled (``jax.jit``).

**Length buckets.**
Lightcurves are grouped by number of points (16, 32, 64, … points), and each group is
fitted in batches whose shape depends only on that number. Compiled functions are
therefore reused by any later call.

**Compiled once per session.**
All the needed shapes are compiled before fitting, in parallel threads, and the redshift
tables are cached. The first call of a session pays for tabulation and compilation (a
few seconds); later calls do not.

See :doc:`examples/speed` for measured timings.

Scope and limitations
---------------------

**Supported:**

* SALT2 sources (``sncosmo.SALT2Source``, e.g. ``"salt2"``).
* Milky Way CCM89 dust in the observer frame.
* AB zero points.
* :math:`t_0, x_0, x_1, c` free, redshift fixed.

**Not supported (yet):**

* SALT3.
* Other effects (e.g. host dust).
* Parameter bounds.
* Fitting the redshift.
