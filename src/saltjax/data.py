"""Pack lightcurves and per-target band integrals into padded arrays for the fit."""

import warnings

import numpy as np
from scipy import sparse

from .kernel import kernel_weights
from .tables import EDGE_NODES, LN10_04
from .effects import get_effects, parameter_table, transmission, wave_limits

__all__ = ["pack_lightcurves", "band_integrals"]

LC_KEYS = ("t", "flux", "err", "zpn", "band", "mask")
FIT_KEYS = ("K", "Ke", "CLb", "F", "Fe", "V", "Ve", "CD", "z") + LC_KEYS
CHUNK = 256    # targets per block of the band integrals


def get_time_key(data, time_key=None):
    """Get the time column of the lightcurves.

    Parameters
    ----------
    data : pandas.DataFrame
        Lightcurve data.

    time_key : str or None, optional
        Time column. If None, the first of 'time', 'mjd' or 'jd' found.
        The default is None.

    Returns
    -------
    str
        Name of the time column.

    Raises
    ------
    ValueError
        If `time_key` is None and no time column is found.
    """
    if time_key is not None:
        return time_key
    time_key = next((k for k in ("time", "mjd", "jd") if k in data.columns), None)
    if time_key is None:
        raise ValueError("cannot parse the time column, provide time_key.")
    return time_key


def _node_weights(idx, w, P, lo, hi):
    """Weights G of the surface wavelength nodes, such that the band integrals
    of band b are ``M[..., lo[b]:hi[b]] @ G[b]``::

        G[b][node, target, term] = sum_lambda W[lambda, node] P[target, b, lambda, term]

    with W the interpolation kernel (nodes `idx` and weights `w`, shape
    (ntargets, nbands, nlambda, 4)); band b only spans the nodes lo[b] to hi[b].
    """
    nt, nb, nl, nterm = P.shape
    size = hi - lo
    offset = np.r_[0, np.cumsum(size)[:-1]]
    # sparse (band, node, target) x (target, band, lambda) matrix
    t, b = np.arange(nt)[:, None, None, None], np.arange(nb)[None, :, None, None]
    rows = (offset[b] + idx - lo[b]) * nt + t
    cols = np.broadcast_to(np.arange(nt * nb * nl).reshape(nt, nb, nl, 1), idx.shape)
    W = sparse.csr_matrix((w.ravel(), (rows.ravel(), cols.ravel())), shape=(size.sum() * nt, nt * nb * nl))
    G = W @ P.reshape(nt * nb * nl, nterm)
    return [G[o * nt:(o + s) * nt].reshape(s, nt * nterm) for o, s in zip(offset, size)]


def band_integrals(tables, z, effects=()):
    """Band integrals of the SALT surfaces of each target, with its effects.

    For a target at redshift z (a = 1/(1+z)), in band b and for the color term
    n, at each phase node p of the M0/M1 surfaces M_k, this computes (exactly
    as ``sncosmo.Model.bandflux``, on the same wavelength grid)::

        K[b, k, n, p] = sum_lambda a M_k(p, a lambda) T_b(lambda) lambda dlambda / (hc zp_b)
                        x E(lambda) x (-0.4 ln10 dCL(a lambda))^n / n!

    with E the transmission of the effects and dCL the deviation of the color
    law from its band mean CLb (so that the color factor is
    ``10**(-0.4 c CLb) sum_n c^n K[..., n, :]``).

    Parameters
    ----------
    tables : dict
        Output of :func:`saltjax.tables.build_tables`.

    z : numpy.ndarray
        Redshifts of the targets.

    effects : list of dict, optional
        Effects and their parameters for these targets (output of
        :func:`saltjax.effects.get_effects`). The default is no effect.

    Returns
    -------
    K : numpy.ndarray
        Band integrals, shape (ntargets, nbands, 2, nc, nphase).

    Ke : numpy.ndarray
        Same at the edge-interval nodes, with a linear kernel in wavelength
        (see :data:`saltjax.tables.EDGE_NODES`), shape (..., 4).

    CLb : numpy.ndarray
        Band mean of the color law, shape (ntargets, nbands).
    """
    z = np.asarray(z, float)
    src, nc, M, M_wave = tables["source"], tables["nc"], tables["M"], tables["M_wave"]
    wave, base, wgt = tables["band_wave"], tables["band_base"], tables["band_wgt"]
    nb, nl = wave.shape
    nk, nph, nw = M.shape
    Me = M[:, list(EDGE_NODES)]
    K = np.zeros((len(z), nb, nk, nc, nph))
    Ke = np.zeros((len(z), nb, nk, nc, len(EDGE_NODES)))
    CLb = np.zeros((len(z), nb))
    for start in range(0, len(z), CHUNK):
        sel = slice(start, min(start + CHUNK, len(z)))
        nt = sel.stop - sel.start
        a = 1. / (1. + z[sel])
        rwave = a[:, None, None] * wave[None]
        cl = src._colorlaw(rwave.ravel()).reshape(rwave.shape)
        CLb[sel] = (cl * wgt).sum(-1) / wgt.sum(-1)
        dcl = LN10_04 * (cl - CLb[sel][..., None])
        trans = transmission([e | dict(params={p: v[sel] for p, v in e["params"].items()})
                              for e in effects], wave.ravel(), a).reshape(rwave.shape)
        # P[..., n] = a base trans (-0.4 ln10 dCL)^n / n!
        P = np.empty(rwave.shape + (nc,))
        P[..., 0] = a[:, None, None] * base[None] * trans
        for k in range(1, nc):
            P[..., k] = P[..., k - 1] * dcl / k

        # interpolation kernel at the rest wavelengths: cubic, and linear (edge phase nodes)
        u = np.interp(rwave, M_wave, np.arange(nw), left=-1., right=nw)
        idx, w = kernel_weights(u, nw)
        f = u - idx[..., 1]
        wl = np.stack([np.zeros_like(f), 1 - f, f, np.zeros_like(f)], -1) * (w.sum(-1, keepdims=True) > 0)
        lo, hi = idx.min((0, 2, 3)), idx.max((0, 2, 3)) + 1       # nodes spanned by each band
        for MM, weights, out in ((M, w, K), (Me, wl, Ke)):
            G = _node_weights(idx, weights, P, lo, hi)
            for b in range(nb):         # one matrix product per band
                Kb = MM[:, :, lo[b]:hi[b]].reshape(-1, hi[b] - lo[b]) @ G[b]
                out[sel, b] = Kb.reshape(nk, MM.shape[1], nt, nc).transpose(2, 0, 3, 1)
    return K, Ke, CLb


def pack_lightcurves(data, targets, tables, indexes=None, phase_range=(-10, 40),
                     effects=None, effect_names=None, effect_frames=None,
                     mwebv_key="mwebv", mw_r_v=3.1, time_key=None, pad_to=16):
    """Pack the lightcurves and the per-target tables into padded arrays.

    Parameters
    ----------
    data : pandas.DataFrame
        Lightcurves, with a (target, observation) multi-index and columns
        time (see `time_key`), 'band', 'flux', 'fluxerr', 'zp' and optionally
        'zpsys'.

    targets : pandas.DataFrame
        Targets, indexed as the first level of `data`, with a 'z' column, a
        't0' column if `phase_range` is given, and the parameters of the
        effects (see :func:`saltjax.effects.get_effects`).

    tables : dict
        Output of :func:`saltjax.tables.build_tables`.

    indexes : list or None, optional
        Targets to pack. If None, all those of `targets`. The default is None.

    phase_range : list or None, optional
        Rest-frame phase range (relative to the targets' t0) of the data used.
        The default is (-10, 40).

    effects, effect_names, effect_frames : list or None, optional
        Propagation effects (e.g. dust), their names and frames ('obs' or
        'rest'), as in ``sncosmo.Model``; see
        :func:`saltjax.effects.get_effects`. The default is None (Milky Way
        dust from `mwebv_key` and `mw_r_v`).

    mwebv_key : str or None, optional
        Only if `effects` is None: column of `targets` with the Milky Way
        E(B-V) (CCM89 dust in the observer frame). If None or absent, no
        effect. The default is 'mwebv'.

    mw_r_v : float, optional
        Only if `effects` is None: R_V of the Milky Way dust. The default is 3.1.

    time_key : str or None, optional
        Time column of the lightcurves. If None, the first of 'time', 'mjd' or
        'jd' found. The default is None.

    pad_to : int, optional
        The number of points per lightcurve is padded to a multiple of
        `pad_to`. The default is 16.

    Returns
    -------
    dict
        Arrays used by the fit (one row per target), and 'effects': the
        parameters of the effects (DataFrame with columns ``{name}{param}``).

    Raises
    ------
    NotImplementedError
        If a magnitude system other than 'ab' is used.
    """
    indexes = list(targets.index if indexes is None else indexes)
    truth = targets.loc[indexes]
    z = truth["z"].to_numpy(float)
    effects = get_effects(truth, effects, effect_names, effect_frames,
                          mwebv_key=mwebv_key, mw_r_v=mw_r_v)
    zg = tables["zgrid"]
    idx, w = kernel_weights((z - zg[0]) / (zg[1] - zg[0]), len(zg))

    def zi(table):
        return np.einsum("sj,sj...->s...", w, table[idx])
    K, Ke, CLb = band_integrals(tables, z, effects)

    data = data.loc[indexes]
    time_key = get_time_key(data, time_key)
    if "zpsys" in data.columns and not data["zpsys"].str.lower().eq("ab").all():
        raise NotImplementedError("only the 'ab' magnitude system is supported")
    lut = {k: j for j, k in enumerate(indexes)}
    pos = np.fromiter((lut[v] for v in data.index.get_level_values(0)), int, len(data))
    # as sncosmo.fit_lc: drop the points whose band is out of the model wavelength range
    bandid = {b: j for j, b in enumerate(tables["bands"])}
    bid = data["band"].map(bandid).to_numpy()
    wmin, wmax = tables["wave_range"]
    emin, emax = wave_limits(effects, z)
    brange = tables["band_range"][bid]
    covered = ((brange[:, 0] >= np.maximum(wmin * (1 + z[pos]), emin[pos])) &
               (brange[:, 1] <= np.minimum(wmax * (1 + z[pos]), emax[pos])))
    if not np.all(covered):
        warnings.warn(f"{np.sum(~covered)} points dropped: band out of the model wavelength range "
                      f"({', '.join(sorted(data['band'][~covered].unique()))})")
        data, pos = data[covered], pos[covered]
    if phase_range is not None:
        ph = (data[time_key].to_numpy() - truth["t0"].to_numpy(float)[pos]) / (1 + z[pos])
        keep = (ph >= phase_range[0]) & (ph <= phase_range[1])
        data, pos = data[keep], pos[keep]
    order = np.argsort(pos, kind="stable")
    ps = pos[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ps)) + 1]
    slot = np.empty(len(ps), int)
    slot[order] = np.arange(len(ps)) - np.repeat(starts, np.diff(np.r_[starts, len(ps)]))
    nsn = len(indexes)
    nobs = int(np.ceil((slot.max() + 1) / pad_to) * pad_to) if len(slot) else pad_to
    out = {k: np.zeros((nsn, nobs)) for k in LC_KEYS}
    out["t"][pos, slot] = data[time_key]
    out["flux"][pos, slot] = data["flux"]
    out["err"][pos, slot] = data["fluxerr"]
    out["zpn"][pos, slot] = 10 ** (0.4 * data["zp"].to_numpy(float))
    out["band"][pos, slot] = data["band"].map(bandid)
    out["mask"][pos, slot] = 1.
    out["err"][out["mask"] == 0] = 1.
    out["band"] = out["band"].astype(int)
    return out | dict(K=K, Ke=Ke, CLb=CLb,
                      F=zi(tables["F"]), Fe=zi(tables["Fe"]),
                      V=zi(tables["V"]), Ve=zi(tables["Ve"]), CD=zi(tables["CD"]), z=z,
                      effects=parameter_table(effects, index=truth.index), index=np.asarray(indexes))
