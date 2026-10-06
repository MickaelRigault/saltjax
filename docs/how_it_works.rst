How it works
============

saltjax gives the same results as ``sncosmo.fit_lc`` because it computes the same
model and the same :math:`\chi^2`. It is faster because it computes them in a
different order. This page explains both.

The SALT band flux
------------------

For a target at redshift :math:`z`, with :math:`a = 1/(1+z)`, the flux in band
:math:`b` at observer-frame time :math:`t` (rest-frame phase
:math:`p = a\,(t - t_0)`) is, as in sncosmo,

.. math::

   F_b(t) = \frac{10^{0.4\,zp}}{F^{AB}_b}\sum_{\lambda} a\,x_0
   \left[M_0(p, a\lambda) + x_1\,M_1(p, a\lambda)\right]
   10^{-0.4\,c\,CL(a\lambda)}\,E(\lambda)\,
   T_b(\lambda)\,\frac{\lambda\,\Delta\lambda}{hc}

where :math:`M_0, M_1` are the SALT2 or SALT3 surfaces, :math:`CL` the color law,
:math:`T_b` the bandpass sampled every :math:`\Delta\lambda = 5` Å, and
:math:`E(\lambda)` the transmission of the propagation effects (dust): the product,
over the effects, of :math:`10^{-0.4\,A(\lambda)}` in the observer frame (e.g. Milky
Way dust) or :math:`10^{-0.4\,A(a\lambda)}` in the rest frame (e.g. host dust).

sncosmo evaluates this sum over a few hundred wavelengths every time the
minimizer asks for a :math:`\chi^2`, typically hundreds of times per lightcurve.

What saltjax computes before fitting
------------------------------------

**Interpolation, exactly as sncosmo.**
sncosmo interpolates :math:`M_0, M_1` (and the model-covariance surfaces) with a
bicubic-convolution kernel (Keys, :math:`a=-0.5`), which becomes bilinear when the
phase is in the first or last interval of the grid. The band sum is linear in the
surfaces, so saltjax integrates the surfaces through the bands *at the native phase
nodes* (with the same kernel in wavelength), then applies *the same kernel in phase*
when fitting. This gives the same result as sncosmo, not an approximation of it.

**Effects, exactly, per target.**
During the fit, the redshift and the parameters of the effects (e.g. :math:`E(B-V)`
and :math:`R_V` of the Milky Way and of the host) are fixed. So :math:`E(\lambda)` is a
fixed weight of the band sum, like the bandpass. saltjax computes, for each target,
the band integrals of the surfaces at its own redshift, with its own transmission
(:func:`saltjax.band_integrals`), on sncosmo's wavelength grid. This is exact for any
effect that is a transmission in wavelength, whatever its law or frame, and whatever
the spread of its parameters between targets.

**Color as an exact series.**
The color is fitted, so it cannot be fixed in the band sum. saltjax writes the color
factor as an exact band-mean factor times a power series of the deviation from that
mean:

.. math::

   10^{-0.4\,c\,CL(a\lambda)} = 10^{-0.4\,c\,\overline{CL}_b}
   \sum_{n} \frac{\left(-0.4\ln 10\; c\,[CL(a\lambda) - \overline{CL}_b]\right)^n}{n!}

and integrates each term of the series. The number of terms is chosen so that the
remainder is below :math:`10^{-13}` for :math:`|c| \le` ``c_max`` (1.5 by default).
Model fluxes therefore equal sncosmo's to machine precision (about
:math:`10^{-15}` in relative terms).

**Model-covariance tables.**
As in sncosmo, the model covariance does not depend on the effects. Its band integrals
are tabulated once on a fine redshift grid (:math:`\Delta z = 2.5\times10^{-4}` by
default, cached) and interpolated in redshift with the same cubic kernel. **This is the
only approximation:** model covariances differ from sncosmo's by about
:math:`10^{-6}` (median) in relative terms.

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
   sncosmo's relative model covariance (``SALT2Source.bandflux_rcov``; for SALT3 the
   variance surfaces are in flux units, as in ``SALT3Source``) computed at the
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
few seconds); later calls do not. The band integrals of each target (with its effects)
are computed at each call, in well under a millisecond per target.

**Dust costs nothing during the fit.**
The effects are folded into the band integrals before the fit, so a model flux costs the
same with or without dust (Milky Way, host, any law), while sncosmo applies them to the
spectrum at every model evaluation.

See :doc:`examples/speed` for measured timings on a ZTF sample with Milky Way dust (16 to
25 times faster than sncosmo per SN on later calls, 5 to 6 times on a first call), and
:doc:`examples/dust` (section *Speed with dust*) for the cost of host dust.

Scope and limitations
---------------------

**Supported:**

* SALT2 and SALT3 sources (``sncosmo.SALT2Source`` and ``sncosmo.SALT3Source``),
  given by name with ``source=`` (and ``version=``): ``"salt2"``, ``"salt2-extended"``,
  ``"salt2-h17"``, ``"salt3"``, ``"salt3-nir"``, ...
* Propagation effects, as in ``sncosmo.Model`` (``effects``, ``effect_names``,
  ``effect_frames``), in the ``"obs"`` and ``"rest"`` frames: all sncosmo dust laws
  (``CCM89Dust``, ``OD94Dust``, ``F99Dust``) and any user-defined
  ``sncosmo.PropagationEffect`` that is a transmission in wavelength. Their parameters
  are fixed per target (columns ``{name}{param}`` of ``targets``, e.g. ``mwebv``,
  ``hostebv``, ``hostr_v``). See :doc:`examples/dust`.
* AB zero points.
* :math:`t_0, x_0, x_1, c` free; redshift and effect parameters fixed.

**Not supported (yet):**

* Fitting the parameters of the effects (e.g. host :math:`E(B-V)`).
* The ``"free"`` effect frame, random scattering effects (``G10``, ``C11``) and
  phase-dependent effects.
* Parameter bounds.
* Fitting the redshift.
