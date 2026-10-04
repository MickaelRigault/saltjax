"""Batched Levenberg-Marquardt SALT2 fit (compiled with JAX)."""

import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import jax
import jax.numpy as jnp

from .model import bandflux, rvar_and_cdisp

__all__ = ["fit_batch", "initial_batch", "compile_all", "get_compiled"]


# ================================================================== #
#  Chi2 and minimisation (one target)                                #
# ================================================================== #
def _cinv_operator(d, theta=None, grid=None, nbands=None):
    """Return v -> C^-1 v for the data covariance (+ model covariance).

    Without `theta`: C = diag(fluxerr^2). With `theta`: C = diag(fluxerr^2) +
    f f^T rcov(theta) = D + U U^T, with D diagonal and U (npoints, nbands)
    holding cdisp_b * f on the points of band b (Woodbury identity, exact).
    """
    m = d["mask"]
    if theta is None:
        dinv = m / d["err"] ** 2
        return lambda v: dinv.reshape(dinv.shape + (1,) * (v.ndim - 1)) * v
    f = bandflux(theta, d, grid)
    rvar, cd = rvar_and_cdisp(theta, d, grid)
    dinv = m / (d["err"] ** 2 + f ** 2 * rvar)
    U = (cd * f * m)[:, None] * (d["band"][:, None] == jnp.arange(nbands)[None, :])
    DU = dinv[:, None] * U
    S = jnp.linalg.inv(jnp.eye(nbands) + U.T @ DU)

    def apply(v):
        dv = dinv.reshape(dinv.shape + (1,) * (v.ndim - 1)) * v
        return dv - DU @ (S @ (DU.T @ v))
    return apply


def _lm(q0, scale, d, grid, cinv, n):
    """Levenberg-Marquardt minimisation of r^T C^-1 r, with r = flux - model."""
    def model(q):
        return bandflux(q * scale, d, grid) * d["mask"]

    def chi2(q):
        r = (d["flux"] * d["mask"] - model(q))
        return r @ cinv(r)

    def step(carry, _):
        q, lam, c2 = carry
        r = d["flux"] * d["mask"] - model(q)
        J = jax.jacfwd(model)(q)                      # (npoints, 4)
        CJ = cinv(J)
        A = J.T @ CJ
        qn = q + jnp.linalg.solve(A + lam * jnp.diag(jnp.diag(A) + 1e-12), CJ.T @ r)
        c2n = chi2(qn)
        ok = c2n < c2
        return (jnp.where(ok, qn, q), jnp.where(ok, lam * 0.3, lam * 10.),
                jnp.where(ok, c2n, c2)), None

    (q, _, c2), _ = jax.lax.scan(step, (q0, 1e-3, chi2(q0)), None, length=n)
    cov_q = 2. * jnp.linalg.inv(jax.hessian(chi2)(q))     # as Minuit HESSE
    return q, c2, cov_q


def _fit_one(theta0, d, grid, nbands, modelcov, niter, niter_refit, nrefit, tol):
    """Fit one target, following sncosmo.fit_lc for the model covariance."""
    scale = jnp.array([1., theta0[1], 1., 1.])
    q, c2, cov_q = _lm(theta0 / scale, scale, d, grid, _cinv_operator(d), niter)
    nref, done = jnp.array(0), jnp.array(True)
    if modelcov:
        def refit(carry, _):
            q, c2, cov_q, done, n = carry
            cinv = _cinv_operator(d, jax.lax.stop_gradient(q * scale), grid, nbands)
            qn, c2n, covn = _lm(q, scale, d, grid, cinv, niter_refit)
            conv = jnp.all(jnp.abs(qn - q) / jnp.sqrt(jnp.diag(covn)) < tol)
            return (jnp.where(done, q, qn), jnp.where(done, c2, c2n),
                    jnp.where(done, cov_q, covn), done | conv, n + (~done)), None
        (q, c2, cov_q, done, nref), _ = jax.lax.scan(
            refit, (q, c2, cov_q, jnp.array(False), nref), None, length=nrefit)
    return dict(theta=q * scale, cov=cov_q * jnp.outer(scale, scale), chi2=c2,
                ndof=jnp.sum(d["mask"]) - 4, converged=done, nrefit=nref)


# ================================================================== #
#  Batched, compiled functions                                       #
# ================================================================== #
@partial(jax.jit, static_argnames=("grid", "nbands", "modelcov", "niter", "niter_refit",
                                   "nrefit", "tol"))
def fit_batch(theta0, d, grid, nbands, modelcov=True, niter=20, niter_refit=10,
              nrefit=10, tol=0.1):
    """Fit a batch of targets.

    Parameters
    ----------
    theta0 : jax.Array
        Initial (t0, x0, x1, c), shape (ntargets, 4).

    d : dict
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
    return jax.vmap(lambda th, dd: _fit_one(th, dd, grid, nbands, modelcov, niter,
                                            niter_refit, nrefit, tol))(theta0, d)


@partial(jax.jit, static_argnames=("grid", "t0_offsets", "c_values"))
def initial_batch(d, theta_in, grid, t0_offsets=(0.,), c_values=None):
    """Get the initial guess of a batch of targets.

    t0 is scanned over `t0_offsets` (and c over `c_values`, if given), x0 is
    solved linearly, and the lowest chi2 is kept.

    Parameters
    ----------
    d : dict
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
    dt, cc = jnp.meshgrid(jnp.asarray(t0_offsets),
                          jnp.asarray(c_values if c_values is not None else (jnp.nan,)))
    dt, cc = dt.ravel(), cc.ravel()

    def one(dd, th):
        w = dd["mask"] / dd["err"] ** 2

        def trial(dt, c):
            tt = th.at[0].add(dt).at[1].set(1.).at[3].set(jnp.where(jnp.isnan(c), th[3], c))
            mf = bandflux(tt, dd, grid)
            norm = jnp.sum(w * mf * mf)
            x0 = jnp.clip(jnp.sum(w * mf * dd["flux"]) / jnp.where(norm > 0, norm, 1.), 1e-15, None)
            return tt.at[1].set(x0), jnp.sum(w * (dd["flux"] - x0 * mf) ** 2)
        ths, chi2 = jax.vmap(trial)(dt, cc)
        return ths[jnp.argmin(chi2)]
    return jax.vmap(one)(d, theta_in)


# ================================================================== #
#  Ahead-of-time compilation                                         #
# ================================================================== #
_COMPILED = {}


def _compile_key(fn, args, static):
    leaves, tree = jax.tree_util.tree_flatten(args)
    return (fn.__name__, tree, tuple((x.shape, str(x.dtype)) for x in leaves),
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
    key = _compile_key(fn, args, static)
    if key not in _COMPILED:
        with jax.enable_x64(True):   # the x64 flag is thread-local
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
    todo = {_compile_key(*t): t for t in tasks if _compile_key(*t) not in _COMPILED}
    if len(todo) > 1:
        with ThreadPoolExecutor(max_workers=max_workers or min(len(todo), os.cpu_count() or 1)) as ex:
            list(ex.map(lambda t: get_compiled(*t), todo.values()))
    else:
        for t in todo.values():
            get_compiled(*t)
