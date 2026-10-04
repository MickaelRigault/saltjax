"""Pack lightcurves and per-target tables into padded arrays for the fit."""

import warnings

import numpy as np
import jax
import jax.numpy as jnp

from .kernel import kernel_weights

__all__ = ["pack_lightcurves"]

LC_KEYS = ("t", "flux", "err", "zpn", "band", "mask")
FIT_KEYS = ("K", "Ke", "CLb", "Tmw", "F", "Fe", "V", "Ve", "CD", "z") + LC_KEYS


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


@jax.jit
def _interp_contract(K, idx, w, epow):
    """Interpolate K in redshift and contract the dust series, per target."""
    def one(i, ww, e):
        return jnp.einsum("j,m,jbkcmp->bkcp", ww, e, K[i])
    return jax.vmap(one)(idx, w, epow)


def pack_lightcurves(data, targets, tables, indexes=None, phase_range=(-10, 40),
                     mwebv_key="mwebv", time_key=None, pad_to=16):
    """Pack the lightcurves and the per-target tables into padded arrays.

    Parameters
    ----------
    data : pandas.DataFrame
        Lightcurves, with a (target, observation) multi-index and columns
        time (see `time_key`), 'band', 'flux', 'fluxerr', 'zp' and optionally
        'zpsys'.

    targets : pandas.DataFrame
        Targets, indexed as the first level of `data`, with a 'z' column, a
        't0' column if `phase_range` is given and a `mwebv_key` column for the
        Milky Way extinction.

    tables : dict
        Output of :func:`saltjax.tables.build_tables`.

    indexes : list or None, optional
        Targets to pack. If None, all those of `targets`. The default is None.

    phase_range : list or None, optional
        Rest-frame phase range (relative to the targets' t0) of the data used.
        The default is (-10, 40).

    mwebv_key : str or None, optional
        Column of `targets` with the Milky Way E(B-V). If None or absent, no
        Milky Way extinction. The default is 'mwebv'.

    time_key : str or None, optional
        Time column of the lightcurves. If None, the first of 'time', 'mjd' or
        'jd' found. The default is None.

    pad_to : int, optional
        The number of points per lightcurve is padded to a multiple of
        `pad_to`. The default is 16.

    Returns
    -------
    dict
        Arrays used by the fit (one row per target).

    Raises
    ------
    NotImplementedError
        If a magnitude system other than 'ab' is used.
    """
    indexes = list(targets.index if indexes is None else indexes)
    truth = targets.loc[indexes]
    z = truth["z"].to_numpy(float)
    has_mw = mwebv_key is not None and mwebv_key in truth
    ebv = truth[mwebv_key].to_numpy(float) if has_mw else np.zeros(len(z))
    zg = tables["zgrid"]
    idx, w = kernel_weights((z - zg[0]) / (zg[1] - zg[0]), len(zg))

    def zi(table):
        return np.einsum("sj,sj...->s...", w, table[idx])
    epow = ebv[:, None] ** np.arange(tables["nm"])
    with jax.enable_x64(True):
        K, Ke = (np.asarray(_interp_contract(jnp.asarray(tables[key]), jnp.asarray(idx),
                                             jnp.asarray(w), jnp.asarray(epow)))
                 for key in ("K", "Ke"))

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
    brange = tables["band_range"][bid]
    covered = (brange[:, 0] >= wmin * (1 + z[pos])) & (brange[:, 1] <= wmax * (1 + z[pos]))
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
    # exact band-mean dust factor
    Tmw = 10 ** (-0.4 * ebv[:, None] * tables["Kb"][None, :])
    return out | dict(K=K, Ke=Ke, CLb=zi(tables["CLb"]), Tmw=Tmw,
                      F=zi(tables["F"]), Fe=zi(tables["Fe"]),
                      V=zi(tables["V"]), Ve=zi(tables["Ve"]), CD=zi(tables["CD"]), z=z,
                      ebv=ebv, index=np.asarray(indexes))
