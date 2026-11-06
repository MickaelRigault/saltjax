"""Batched Levenberg-Marquardt SALT fit (compiled with JAX).

- One target: :func:`get_cinv_operator` (inverse data + model covariance),
  :func:`levenberg_marquardt` (minimisation) and :func:`fit_one` (first fit,
  then refits with the model covariance, as ``sncosmo.fit_lc``).
- Batches: :func:`initial_batch` (initial guess) and :func:`fit_batch`
  (vectorised :func:`fit_one`), compiled by :func:`get_compiled` /
  :func:`compile_all` and cached (see :func:`clear_cache`).
"""

import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import jax
import jax.numpy as jnp

from .model import bandflux, rvar_and_cdisp

__all__ = ["fit_batch", "initial_batch", "compile_all", "get_compiled", "enable_x64",
           # one target
           "get_cinv_operator", "levenberg_marquardt", "fit_one",
           # cache
           "clear_cache", "get_ncompiled"]


def enable_x64(enabled=True):
    """Context manager to enable (or disable) 64-bit precision in JAX.

    Works with any jax version: recent jax provides ``jax.enable_x64``,
    older jax (e.g. the last versions supporting python 3.10) only
    ``jax.experimental.enable_x64``.

    Parameters
    ----------
    enabled : bool, optional
        Enable (True) or disable (False) 64-bit precision. The default is True.

    Returns
    -------
    contextmanager
        To be used as ``with enable_x64(): ...``. The flag is thread-local.

    Examples
    --------
    >>> with enable_x64():
    ...     x = jax.numpy.ones(3)  # float64
    """
    if hasattr(jax, "enable_x64"):          # recent jax
        return jax.enable_x64(enabled)

    from jax.experimental import enable_x64 as experimental_enable_x64  # older jax
    return experimental_enable_x64(enabled)


# ================== #
#                    #
#   One target       #
#                    #
# ================== #
def get_cinv_operator(packed, theta=None, grid=None, nbands=None):
    """Get the operator v -> C^-1 v of the data covariance (+ model covariance).

    Without `theta`: C = diag(fluxerr^2). With `theta`: C = diag(fluxerr^2) +
    f f^T rcov(theta) = D + U U^T, with D diagonal and U (npoints, nbands)
    holding cdisp_b * f on the points of band b, inverted exactly with the
    Woodbury identity (the dense covariance is never built).

    Parameters
    ----------
    packed : dict
        Packed arrays of the target (see :func:`saltjax.data.pack_lightcurves`).

    theta : jax.Array or None, optional
        (t0, x0, x1, c) at which the model covariance is computed. If None,
        no model covariance. The default is None.

    grid : tuple or None, optional
        Output of :func:`saltjax.model.phase_grids`. Required with `theta`.
        The default is None.

    nbands : int or None, optional
        Number of bands of the tables. Required with `theta`. The default is None.

    Returns
    -------
    callable
        ``apply(v)``: C^-1 v, for v of shape (npoints,) or (npoints, k).
        Masked (padded) points have C^-1 = 0.
    """
    mask = packed["mask"]
    if theta is None:
        dinv = mask / packed["err"] ** 2

        def apply_diagonal(v):
            return dinv.reshape(dinv.shape + (1,) * (v.ndim - 1)) * v
        return apply_diagonal

    # D: data variance + diagonal model variance; U: color dispersion (one column per band)
    flux = bandflux(theta, packed, grid)
    rvar, cdisp = rvar_and_cdisp(theta, packed, grid)
    dinv = mask / (packed["err"] ** 2 + flux ** 2 * rvar)                                 # (npoints,)
    U = (cdisp * flux * mask)[:, None] * (packed["band"][:, None] == jnp.arange(nbands)[None, :])  # (npoints, nbands)
    DU = dinv[:, None] * U
    S = jnp.linalg.inv(jnp.eye(nbands) + U.T @ DU)                                       # (nbands, nbands)

    def apply(v):
        # Woodbury: (D + U U^T)^-1 = D^-1 - D^-1 U (1 + U^T D^-1 U)^-1 U^T D^-1
        dv = dinv.reshape(dinv.shape + (1,) * (v.ndim - 1)) * v
        return dv - DU @ (S @ (DU.T @ v))
    return apply


def levenberg_marquardt(q0, scale, packed, grid, cinv, niter):
    """Minimise r^T C^-1 r, with r = flux - model, by Levenberg-Marquardt.

    The parameters are rescaled (``theta = q * scale``) so that they are all
    of order 1 (x0 is ~1e-5).

    Parameters
    ----------
    q0 : jax.Array
        Initial rescaled parameters, shape (4,).

    scale : jax.Array
        Scale of each parameter, shape (4,).

    packed : dict
        Packed arrays of the target.

    grid : tuple
        Output of :func:`saltjax.model.phase_grids`.

    cinv : callable
        Inverse covariance operator (see :func:`get_cinv_operator`).

    niter : int
        Number of iterations (fixed, for the JIT).

    Returns
    -------
    q : jax.Array
        Best-fit rescaled parameters.

    chi2 : jax.Array
        Chi2 at `q`.

    cov_q : jax.Array
        Covariance of `q`: 2 x the inverse Hessian of the chi2 (as Minuit HESSE).
    """
    def model(q):
        return bandflux(q * scale, packed, grid) * packed["mask"]

    def chi2(q):
        residual = (packed["flux"] * packed["mask"] - model(q))
        return residual @ cinv(residual)

    def step(carry, _):
        q, damping, chi2_ = carry
        residual = packed["flux"] * packed["mask"] - model(q)
        J = jax.jacfwd(model)(q)                      # (npoints, 4)
        CJ = cinv(J)
        A = J.T @ CJ                                  # (4, 4) normal matrix
        q_new = q + jnp.linalg.solve(A + damping * jnp.diag(jnp.diag(A) + 1e-12), CJ.T @ residual)
        chi2_new = chi2(q_new)
        # accept the step if the chi2 decreases (and trust the quadratic model more)
        better = chi2_new < chi2_
        return (jnp.where(better, q_new, q), jnp.where(better, damping * 0.3, damping * 10.),
                jnp.where(better, chi2_new, chi2_)), None

    (q, _, chi2_), _ = jax.lax.scan(step, (q0, 1e-3, chi2(q0)), None, length=niter)
    cov_q = 2. * jnp.linalg.inv(jax.hessian(chi2)(q))     # as Minuit HESSE
    return q, chi2_, cov_q


def fit_one(theta0, packed, grid, nbands, modelcov=True, niter=20, niter_refit=10, nrefit=10, tol=0.1):
    """Fit one target, following ``sncosmo.fit_lc`` for the model covariance.

    A first fit without model covariance, then (if `modelcov`) refits with
    the model covariance computed at the current best fit, until all
    parameters move by less than `tol` sigma (or `nrefit` refits).

    It is traced and compiled by :func:`fit_batch` (one call per target);
    calling it directly (not compiled) is useful to check a single fit.

    Parameters
    ----------
    theta0 : jax.Array
        Initial (t0, x0, x1, c).

    packed : dict
        Packed arrays of the target.

    grid : tuple
        Output of :func:`saltjax.model.phase_grids`.

    nbands : int
        Number of bands of the tables.

    modelcov, niter, niter_refit, nrefit, tol : optional
        See :func:`fit_batch`.

    Returns
    -------
    dict
        theta, cov, chi2, ndof, converged and nrefit.
    """
    scale = jnp.array([1., theta0[1], 1., 1.])       # x0 rescaled to ~1
    q, chi2, cov_q = levenberg_marquardt(theta0 / scale, scale, packed, grid,
                                         get_cinv_operator(packed), niter)
    nrefit_done, converged = jnp.array(0), jnp.array(True)

    if modelcov:
        def refit(carry, _):
            q, chi2, cov_q, converged, nrefit_done = carry
            # model covariance at the current best fit (not differentiated)
            cinv = get_cinv_operator(packed, jax.lax.stop_gradient(q * scale), grid, nbands)
            q_new, chi2_new, cov_new = levenberg_marquardt(q, scale, packed, grid, cinv, niter_refit)
            moved_little = jnp.all(jnp.abs(q_new - q) / jnp.sqrt(jnp.diag(cov_new)) < tol)
            # once converged, keep the values (scan has a fixed length)
            return (jnp.where(converged, q, q_new), jnp.where(converged, chi2, chi2_new),
                    jnp.where(converged, cov_q, cov_new), converged | moved_little,
                    nrefit_done + (~converged)), None

        (q, chi2, cov_q, converged, nrefit_done), _ = jax.lax.scan(
            refit, (q, chi2, cov_q, jnp.array(False), nrefit_done), None, length=nrefit)

    return dict(theta=q * scale, cov=cov_q * jnp.outer(scale, scale), chi2=chi2,
                ndof=jnp.sum(packed["mask"]) - 4, converged=converged, nrefit=nrefit_done)


# ================== #
#                    #
#   Batches          #
#                    #
# ================== #
@partial(jax.jit, static_argnames=("grid", "nbands", "modelcov", "niter", "niter_refit",
                                   "nrefit", "tol"))
def fit_batch(theta0, packed, grid, nbands, modelcov=True, niter=20, niter_refit=10,
              nrefit=10, tol=0.1):
    """Fit a batch of targets (vectorised :func:`fit_one`).

    Parameters
    ----------
    theta0 : jax.Array
        Initial (t0, x0, x1, c), shape (ntargets, 4).

    packed : dict
        Packed arrays of the targets (one row per target).

    grid : tuple
        Output of :func:`saltjax.model.phase_grids`.

    nbands : int
        Number of bands of the tables.

    modelcov : bool, optional
        Include the model covariance (as ``sncosmo.fit_lc``). The default is True.

    niter : int, optional
        Number of Levenberg-Marquardt iterations of the first fit.
        The default is 20.

    niter_refit : int, optional
        Number of iterations of each refit with model covariance.
        The default is 10.

    nrefit : int, optional
        Maximum number of refits with model covariance. The default is 10.

    tol : float, optional
        Refits stop when all parameters move by less than `tol` sigma.
        The default is 0.1 (as sncosmo).

    Returns
    -------
    dict
        theta, cov, chi2, ndof, converged and nrefit, one entry per target.
    """
    return jax.vmap(lambda theta_, packed_: fit_one(theta_, packed_, grid, nbands, modelcov, niter,
                                                    niter_refit, nrefit, tol))(theta0, packed)


@partial(jax.jit, static_argnames=("grid", "t0_offsets", "c_values"))
def initial_batch(packed, theta_in, grid, t0_offsets=(0.,), c_values=None):
    """Get the initial guess of a batch of targets.

    t0 is scanned over `t0_offsets` (and c over `c_values`, if given), x0 is
    solved linearly, and the lowest chi2 is kept.

    Parameters
    ----------
    packed : dict
        Packed arrays of the targets (one row per target).

    theta_in : jax.Array
        Starting (t0, x0, x1, c), shape (ntargets, 4); x0 is ignored.

    grid : tuple
        Output of :func:`saltjax.model.phase_grids`.

    t0_offsets : tuple, optional
        Offsets added to the starting t0. The default is (0.,).

    c_values : tuple or None, optional
        Colors to try. If None, the starting c. The default is None.

    Returns
    -------
    jax.Array
        Initial (t0, x0, x1, c), shape (ntargets, 4).
    """
    # grid of (t0 offset, c) trials; c = nan means "keep the starting c"
    dt0, color = jnp.meshgrid(jnp.asarray(t0_offsets),
                              jnp.asarray(c_values if c_values is not None else (jnp.nan,)))
    dt0, color = dt0.ravel(), color.ravel()

    def initial_one(packed_, theta_):
        weight = packed_["mask"] / packed_["err"] ** 2

        def trial(dt0_, color_):
            theta_trial = theta_.at[0].add(dt0_).at[1].set(1.).at[3].set(
                jnp.where(jnp.isnan(color_), theta_[3], color_))
            model_flux = bandflux(theta_trial, packed_, grid)          # for x0 = 1
            # x0: weighted linear least squares
            norm = jnp.sum(weight * model_flux * model_flux)
            x0 = jnp.clip(jnp.sum(weight * model_flux * packed_["flux"]) / jnp.where(norm > 0, norm, 1.),
                          1e-15, None)
            return theta_trial.at[1].set(x0), jnp.sum(weight * (packed_["flux"] - x0 * model_flux) ** 2)

        thetas, chi2 = jax.vmap(trial)(dt0, color)
        return thetas[jnp.argmin(chi2)]

    return jax.vmap(initial_one)(packed, theta_in)


# ================== #
#                    #
#   Compilation      #
#                    #
# ================== #
_COMPILED = {}   # compiled functions, per (function, argument shapes, static arguments)


def _get_compile_key_(fn, args, static):
    """Key of `_COMPILED`: function name, argument structure/shapes/dtypes and static arguments."""
    leaves, tree = jax.tree_util.tree_flatten(args)
    return (fn.__name__, tree, tuple((leaf_.shape, str(leaf_.dtype)) for leaf_ in leaves),
            tuple(sorted(static.items())))


def get_compiled(fn, args, static):
    """Get `fn` compiled for these argument shapes (cached).

    Parameters
    ----------
    fn : callable
        A jitted function (e.g. :func:`fit_batch`).

    args : tuple
        Positional (array) arguments.

    static : dict
        Static keyword arguments.

    Returns
    -------
    callable
        The compiled function, called with `args` only.
    """
    key = _get_compile_key_(fn, args, static)
    if key not in _COMPILED:
        with enable_x64(True):   # the x64 flag is thread-local
            _COMPILED[key] = fn.lower(*args, **static).compile()
    return _COMPILED[key]


def compile_all(tasks, max_workers=None):
    """Compile (fn, args, static) tasks in parallel threads (XLA releases the GIL).

    Parameters
    ----------
    tasks : list of tuple
        (fn, args, static) to compile (see :func:`get_compiled`).

    max_workers : int or None, optional
        Number of threads. If None, one per task, up to the number of CPUs.
        The default is None.
    """
    todo = {_get_compile_key_(*task_): task_ for task_ in tasks if _get_compile_key_(*task_) not in _COMPILED}
    if len(todo) > 1:
        with ThreadPoolExecutor(max_workers=max_workers or min(len(todo), os.cpu_count() or 1)) as executor:
            list(executor.map(lambda task_: get_compiled(*task_), todo.values()))
    else:
        for task_ in todo.values():
            get_compiled(*task_)


def clear_cache():
    """Empty the cache of compiled functions (they are recompiled when needed)."""
    _COMPILED.clear()


def get_ncompiled():
    """Get the number of compiled functions in the cache.

    Returns
    -------
    int
        Number of compiled (function, shapes, static arguments).
    """
    return len(_COMPILED)
