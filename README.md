# saltjax

Fast, batched SALT2 lightcurve fitting with [JAX](https://docs.jax.dev), reproducing
[sncosmo](https://sncosmo.readthedocs.io)'s `fit_lc`.

`saltjax` fits SALT2 to thousands of lightcurves at once. It uses the same model as sncosmo
(same interpolation kernel, model covariance and Milky Way dust), and gives the same
parameters and errors (to ~1e-2 sigma at worst) one to two orders of magnitude faster.

## Install

```bash
git clone https://github.com/MickaelRigault/saltjax.git
cd saltjax
pip install .
```

## Quick start

```python
import saltjax

# data: lightcurves, indexed by (target, observation), with columns
#       time (or mjd), band, flux, fluxerr, zp [, zpsys]
# targets: one row per target with z [, t0, x1, c, mwebv]
results = saltjax.fit_salt(data, targets, modelcov=True, progress_bar=True)
results[["t0", "x0", "x1", "c", "t0_err", "x0_err", "x1_err", "c_err"]]
```

`saltjax` is used by [skysurvey](https://github.com/MickaelRigault/skysurvey) through
`skysurvey.lcfit.fit_salt_jax(dataset)`.

## How it reproduces sncosmo

* **Interpolation.** sncosmo interpolates the SALT2 surfaces with a separable bicubic-convolution kernel. Band
  integration is linear, so `saltjax` band-integrates the surfaces at the native phase nodes
  and applies the *same* kernel in phase. This is exact.
* **Colour and dust.** The colour law and the Milky Way extinction (CCM89, observer frame) are written as an exact
  band-mean factor times power series truncated below machine precision.
* **Redshift tables.** The band integrals are tabulated on a fine redshift grid. This is the only approximation:
  relative flux differences are below 1e-6.
* **Model covariance.** `modelcov=True` follows `sncosmo.fit_lc`. The model covariance (a diagonal plus one block
  per band) is inverted exactly with the Woodbury identity.
* **Errors.** Errors are `2 H^-1`, with `H` the exact Hessian of the chi2 (as Minuit HESSE).

## Limitations

* SALT2 sources only (`sncosmo.SALT2Source`).
* The only effect supported is Milky Way CCM89 dust in the observer frame.
* AB magnitude system only.
* No parameter bounds.
* Fixed redshift.
