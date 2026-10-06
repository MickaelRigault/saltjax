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

# any sncosmo dust (effects), e.g. Milky Way and host dust; their parameters are
# read per target from the columns mwebv, mwr_v, hostebv, hostr_v of targets
import sncosmo
results = saltjax.fit_salt(data, targets,
                           effects=[sncosmo.CCM89Dust(), sncosmo.F99Dust(r_v=2.0)],
                           effect_names=["mw", "host"], effect_frames=["obs", "rest"])
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

## Limitations

* SALT2 and SALT3 sources only (`sncosmo.SALT2Source`, `sncosmo.SALT3Source`, e.g.
  `"salt2"`, `"salt2-extended"`, `"salt3"`, `"salt3-nir"`).
* Effects (dust) in the observer or rest frame only, with fixed parameters (not fitted); no random scattering
  effects (G10, C11).
* AB magnitude system only.
* No parameter bounds.
* Fixed redshift.

## Acknowledgements

`saltjax` was developed by Mickael Rigault with the help of [Claude Code](https://claude.com/claude-code)
(Anthropic), an AI coding assistant, which contributed to the implementation, tests and documentation
(including support for all SALT2/SALT3 sources and for sncosmo dust effects).
