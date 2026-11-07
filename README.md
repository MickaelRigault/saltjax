[![PyPI](https://img.shields.io/pypi/v/saltjax.svg?style=flat-square)](https://pypi.org/project/saltjax/)
[![Documentation Status](https://readthedocs.org/projects/saltjax/badge/?version=latest)](https://saltjax.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/MickaelRigault/saltjax/branch/main/graph/badge.svg)](https://codecov.io/gh/MickaelRigault/saltjax)

# saltjax

Fast, batched SALT2 / SALT3 lightcurve fitting with [JAX](https://docs.jax.dev), reproducing
[sncosmo](https://sncosmo.readthedocs.io)'s `fit_lc`.

`saltjax` fits SALT2 or SALT3 to thousands of lightcurves at once. It uses the same model as sncosmo
(same interpolation kernel, model covariance and dust), and gives the same parameters and errors
(to ~1e-2 sigma at worst), much faster.

## Speed

On ~1000 simulated ZTF SNe Ia with Milky Way extinction (SFD map), on a laptop CPU (Apple M1 Max),
with the same model in both (SALT2 + Milky Way CCM89 dust):

| per SN | sncosmo `fit_lc` | saltjax, first call | saltjax, later call |
|---|---|---|---|
| `modelcov=False` | 58 ms | 9 ms (6x) | 2.4 ms (25x) |
| `modelcov=True` | 107 ms | 20 ms (5x) | 6.7 ms (16x) |

The first call of a session includes tabulation and JAX compilation. Dust costs nothing during the fit: host
dust (any law, per-SN E(B-V) and R_V) gives the same saltjax time, while sncosmo gets slower. See the
[speed](docs/examples/speed.ipynb) and [dust](docs/examples/dust.ipynb) notebooks.

## Install

```bash
pip install saltjax
```

## What can be fitted

Any combination of a **source** and **effects** supported by sncosmo below:

| Source (sncosmo) | Function | Fitted parameters |
|---|---|---|
| any SALT2 or SALT3 source: `"salt2"`, `"salt2-extended"`, `"salt2-h17"`, `"salt3"`, `"salt3-nir"`… | `saltjax.fit_salt` | `t0`, `x0`, `x1`, `c` (+ model covariance) |
| any time-series source (`sncosmo.TimeSeriesSource`): `"hsiao"`, `"nugent-sn1a"`, `"nugent-sn2p"`, the SNANA core-collapse templates… | `saltjax.fit_timeseries` | `t0`, `amplitude` |

**Effects**, with both functions, given as in `sncosmo.Model` (`effects`, `effect_names`, `effect_frames`):

* any sncosmo dust law (`CCM89Dust`, `OD94Dust`, `F99Dust`), or any user-defined `sncosmo.PropagationEffect`
  that is a deterministic transmission in wavelength;
* in the observer frame (`"obs"`, e.g. Milky Way dust) or the rest frame (`"rest"`, e.g. host dust);
* with parameters set **per target**, from the columns `{name}{param}` of `targets` (e.g. `mwebv`, `hostebv`,
  `hostr_v`), else from the effect instance. They are fixed during the fit.

By default (no `effects`), Milky Way CCM89 dust is applied from the `mwebv` column, if present.

## Quick start

```python
import saltjax

# data: lightcurves, indexed by (target, observation), with columns
#       time (or mjd), band, flux, fluxerr, zp [, zpsys]
# targets: one row per target with z [, t0, x1, c, mwebv]
results = saltjax.fit_salt(data, targets, modelcov=True, progress_bar=True)
results[["t0", "x0", "x1", "c", "t0_err", "x0_err", "x1_err", "c_err"]]

# any SALT2 or SALT3 source registered in sncosmo
results = saltjax.fit_salt(data, targets, source="salt3")
results = saltjax.fit_salt(data, targets, source="salt2", version="T23")

# any time-series source: fits t0 and amplitude
results = saltjax.fit_timeseries(data, targets, source="hsiao")
results = saltjax.fit_timeseries(data, targets, source="nugent-sn2p")
```

With effects, the same arguments for both functions, e.g. Milky Way and host dust whose parameters are read per
target from the columns `mwebv`, `hostebv` and `hostr_v` of `targets`:

```python
import sncosmo

dust = dict(effects=[sncosmo.CCM89Dust(), sncosmo.F99Dust(r_v=2.0)],
            effect_names=["mw", "host"], effect_frames=["obs", "rest"])

results = saltjax.fit_salt(data, targets, source="salt3", **dust)
results = saltjax.fit_timeseries(data, targets, source="nugent-sn2p", **dust)
```

Check a time-series fit by eye (give the same effects as the fit, if any):

```python
template = saltjax.timeseries.TimeSeriesTemplate.from_name("nugent-sn2p")
template.show_fit(data, results, index=results.index[0], **dust)
```

`saltjax` is used by [skysurvey](https://github.com/MickaelRigault/skysurvey) through
`skysurvey.lcfit.fit_salt_jax(dataset)`.

## How it reproduces sncosmo

* **Interpolation.** sncosmo interpolates the SALT2 surfaces with a separable bicubic-convolution kernel. Band
  integration is linear, so `saltjax` band-integrates the surfaces at the native phase nodes
  and applies the *same* kernel in phase. This is exact.
* **Dust and other effects.** Their parameters are fixed per target, so their transmission is folded, exactly, into
  the band integrals of each target, at its own redshift, on sncosmo's wavelength grid. Any sncosmo dust law
  (CCM89, OD94, F99), in the observer or rest frame, with per-target E(B-V) and R_V.
* **Colour.** The colour law is written as an exact band-mean factor times a power series truncated below machine
  precision. Model fluxes equal sncosmo's to ~1e-15.
* **Redshift tables.** The model covariance is tabulated on a fine redshift grid. This is the only approximation:
  relative differences of ~1e-6 (median).
* **Model covariance.** `modelcov=True` follows `sncosmo.fit_lc`. The model covariance (a diagonal plus one block
  per band) is inverted exactly with the Woodbury identity.
* **Errors.** Errors are `2 H^-1`, with `H` the exact Hessian of the chi2 (as Minuit HESSE).
* **Time-series sources.** sncosmo interpolates them with a tensor-product spline, also linear in the template
  and separable in phase and wavelength: the band integral of each phase spline function is computed once per
  target, and the band flux at any phase is the amplitude times a sum over a few spline functions, exactly as sncosmo.

## Limitations

* SALT2 and SALT3 sources (`sncosmo.SALT2Source`, `sncosmo.SALT3Source`, e.g.
  `"salt2"`, `"salt2-extended"`, `"salt3"`, `"salt3-nir"`) with `fit_salt`, and time-series
  sources (`sncosmo.TimeSeriesSource`, e.g. `"hsiao"`, `"nugent-sn2p"`) with `fit_timeseries`.
  Other sources (`StretchSource`, `SNEMOSource`, `MLCS2k2Source`, `SUGARSource`) are not supported.
* Effects (dust) in the observer or rest frame only, with fixed parameters (not fitted); no random scattering
  effects (G10, C11).
* Time-series sources: no model covariance (as in sncosmo, where only SALT has one).
* AB magnitude system only.
* No parameter bounds.
* Fixed redshift.

## Acknowledgements

`saltjax` was developed by Mickael Rigault with the help of [Claude Code](https://claude.com/claude-code)
(Anthropic), an AI coding assistant, which contributed to the implementation, tests and documentation
(including support for all SALT2/SALT3 sources and for sncosmo dust effects).
