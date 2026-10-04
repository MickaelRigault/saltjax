"""Band-integrated SALT2 surfaces tabulated on a redshift grid.

The SALT2 band flux is linear in the (phase, wavelength) surfaces, so the
surfaces are band-integrated once at their native phase nodes; the color law
and the Milky Way extinction are written as an exact band-mean factor times a
power series of the deviation from that mean.

sncosmo interpolates the surfaces with a bicubic convolution, which becomes
bilinear (in both dimensions) when the phase is in the first or last interval
of the grid. The tables are therefore built from the raw grid files with the
same kernel in wavelength: cubic at every phase node ("regular" tables), and
linear for the two nodes of each edge interval ("edge" tables).
"""

import math
import os
from functools import lru_cache

import numpy as np
from scipy import sparse
import sncosmo
import extinction
from sncosmo.constants import HC_ERG_AA
from sncosmo.utils import integration_grid
from sncosmo.models import read_griddata_ascii

from .kernel import kernel_weights

__all__ = ["get_salt2_source", "build_tables", "get_tables"]

LN10_04 = -0.4 * np.log(10)
EDGE_NODES = (0, 1, -2, -1)       # nodes of the first and last phase intervals

# file names of the SALT2Source surfaces (sncosmo defaults)
_SURFACE_FILES = {"M0": "salt2_template_0.dat",
                  "M1": "salt2_template_1.dat",
                  "LCRV00": "salt2_lc_relative_variance_0.dat",
                  "LCRV11": "salt2_lc_relative_variance_1.dat",
                  "LCRV01": "salt2_lc_relative_covariance_01.dat",
                  "errscale": "salt2_lc_dispersion_scaling.dat"}


def _nterms(xmax, tol=1e-13):
    """Number of terms of the exponential series so that the remainder < tol."""
    return next(n for n in range(1, 200) if xmax ** n / math.factorial(n) < tol)


class _Surface:
    """A SALT2 surface on its (phase, wavelength) grid, interpolated in
    wavelength with sncosmo's kernel at the phase nodes."""

    def __init__(self, filename, scale=1.):
        self.phase, self.wave, values = read_griddata_ascii(filename)
        self.values = values * scale
        for nodes in (self.phase, self.wave):
            if not np.allclose(np.diff(nodes), nodes[1] - nodes[0]):
                raise NotImplementedError("non-uniform SALT2 grids are not supported")

    def at_nodes(self, wave, rows=slice(None), linear=False):
        """Values at the phase nodes `rows` and wavelengths `wave`.

        Parameters
        ----------
        wave : numpy.ndarray
            Wavelengths.

        rows : slice or list, optional
            Phase nodes. The default is all.

        linear : bool, optional
            Linear (instead of cubic convolution) interpolation in wavelength.
            The default is False.

        Returns
        -------
        numpy.ndarray
            Values, shape (nrows, len(wave)).
        """
        u = (np.asarray(wave, float) - self.wave[0]) / (self.wave[1] - self.wave[0])
        idx, w = kernel_weights(u, len(self.wave))
        if linear:
            f = u - idx[:, 1]
            w = np.stack([np.zeros_like(f), 1 - f, f, np.zeros_like(f)], -1) * (w.sum(-1, keepdims=True) > 0)
        # sparse (len(wave), nwave_nodes) interpolation matrix: one product for all phase nodes
        W = sparse.csr_matrix((w.ravel(), (np.repeat(np.arange(len(u)), 4), idx.ravel())),
                              shape=(len(u), len(self.wave)))
        return (W @ self.values[rows].T).T


@lru_cache(maxsize=8)
def _load_surfaces(modeldir, scale_factor):
    """Read (once per session) the SALT2 surfaces of a model directory."""
    return {key: _Surface(os.path.join(modeldir, fname),
                          scale=scale_factor if key in ("M0", "M1") else 1.)
            for key, fname in _SURFACE_FILES.items()}


def get_salt2_source(source, version=None):
    """Get a SALT2 source and check it is supported.

    Parameters
    ----------
    source : str or sncosmo.SALT2Source
        Source name (e.g. 'salt2') or instance.

    version : str or None, optional
        Version of the source, if given by name. The default is None (latest).

    Returns
    -------
    sncosmo.SALT2Source
        The source.

    Raises
    ------
    NotImplementedError
        If the source is not an ``sncosmo.SALT2Source`` (e.g. SALT3).
    """
    src = sncosmo.get_source(source, version=version) if isinstance(source, str) else source
    if type(src) is not sncosmo.SALT2Source:
        raise NotImplementedError(f"only sncosmo.SALT2Source is supported, got {type(src).__name__}")
    return src


def _get_modeldir(src):
    """Directory of the files of a registered SALT2 source."""
    from sncosmo.builtins import DATADIR
    loader = sncosmo.models._SOURCES._loaders.get((src.name, src.version))
    if loader is None:
        raise ValueError(f"cannot locate the files of source {src.name!r} ({src.version}); "
                         "provide modeldir.")
    return DATADIR.abspath(loader[1][0], isdir=True)


def build_tables(bands, zmin, zmax, source="salt2", version=None, modeldir=None,
                 ebv_max=0., r_v=3.1, c_max=1.5, dz=2.5e-4, spacing=5.):
    """Tabulate the band-integrated SALT2 surfaces on a redshift grid.

    Parameters
    ----------
    bands : list of str
        Names of the (sncosmo-registered) bandpasses.

    zmin, zmax : float
        Redshift range to tabulate.

    source : str or sncosmo.SALT2Source, optional
        SALT2 source (name or instance). The default is 'salt2'.

    version : str or None, optional
        Version of the source, if given by name. The default is None (latest).

    modeldir : str or None, optional
        Directory of the source files. If None, it is found from the sncosmo
        registry. The default is None.

    ebv_max : float, optional
        Maximum Milky Way E(B-V) to support (sets the number of dust terms).
        The default is 0 (no dust).

    r_v : float, optional
        R_V of the CCM89 Milky Way dust. The default is 3.1.

    c_max : float, optional
        Maximum absolute SALT2 color to support exactly (sets the number of
        color terms). The default is 1.5.

    dz : float, optional
        Redshift step of the tables. The default is 2.5e-4.

    spacing : float, optional
        Wavelength step of the band integration, in Angstrom (sncosmo's
        ``MODEL_BANDFLUX_SPACING``). The default is 5.

    Returns
    -------
    dict
        The tables and their grids. Arrays ending with 'e' (``Ke``, ``Fe``,
        ``Ve``) hold the edge-interval nodes (see :data:`EDGE_NODES`).

    Raises
    ------
    NotImplementedError
        If the source grids are not supported (non-uniform grids, or LCRV and
        errscale grids not sharing their phase nodes).
    """
    src = get_salt2_source(source, version)
    modeldir = _get_modeldir(src) if modeldir is None else modeldir
    surf = _load_surfaces(modeldir, src._SCALE_FACTOR)
    phM, phV = surf["M0"].phase, surf["LCRV00"].phase
    if not all(np.allclose(surf[k].phase, phV) for k in ("LCRV11", "LCRV01", "errscale")):
        raise NotImplementedError("LCRV and errscale grids must share their phase nodes")
    edge = list(EDGE_NODES)

    zgrid = np.arange(max(zmin - 3 * dz, 0.), zmax + 4 * dz, dz)
    ab = sncosmo.get_magsystem("ab")
    grids = {}
    for b in bands:
        bp = sncosmo.get_bandpass(b)
        wave, dwave = integration_grid(bp.minwave(), bp.maxwave(), spacing)
        trans = bp(wave)
        kmw = extinction.ccm89(wave, r_v, r_v)          # A(wave) for E(B-V)=1
        wgt = wave * trans
        grids[b] = dict(bp=bp, wave=wave, dwave=dwave, trans=trans, kmw=kmw,
                        kb=np.sum(wgt * kmw) / np.sum(wgt))
        grids[b]["cl"] = np.array([src._colorlaw(wave / (1 + z)) for z in zgrid])
        grids[b]["clb"] = (grids[b]["cl"] * wgt).sum(1) / wgt.sum()

    xc = abs(LN10_04) * c_max * max(np.abs(g["cl"] - g["clb"][:, None]).max() for g in grids.values())
    xm = abs(LN10_04) * ebv_max * max(np.abs(g["kmw"] - g["kb"]).max() for g in grids.values())
    nc, nm = _nterms(xc), (_nterms(xm) if ebv_max > 0 else 1)
    cfac = np.array([LN10_04 ** n / math.factorial(n) for n in range(nc)])
    mfac = np.array([LN10_04 ** m / math.factorial(m) for m in range(nm)])

    nz, nb = len(zgrid), len(bands)
    K = np.zeros((nz, nb, 2, nc, nm, len(phM)))
    Ke = np.zeros((nz, nb, 2, nc, nm, 4))
    F = np.zeros((nz, nb, 2, len(phM)))
    Fe = np.zeros((nz, nb, 2, 4))
    V = np.zeros((nz, nb, 4, len(phV)))
    Ve = np.zeros((nz, nb, 4, 4))
    CD = np.zeros((nz, nb))
    CLb = np.zeros((nz, nb))
    Kb = np.array([grids[b]["kb"] for b in bands])
    avals = 1. / (1. + zgrid)
    for ib, b in enumerate(bands):
        g = grids[b]
        base = g["wave"] * g["trans"] * g["dwave"] / HC_ERG_AA / ab.zpbandflux(g["bp"])
        dk = g["kmw"] - g["kb"]
        mpow = mfac[:, None] * dk[None, :] ** np.arange(nm)[:, None]
        CLb[:, ib] = g["clb"]

        # observer-frame band integrals: surfaces at all (z, wave) at once
        rw = (g["wave"][None, :] * avals[:, None]).ravel()
        shape = (nz, len(g["wave"]))
        surfaces = {key: (surf[key].at_nodes(rw).reshape(-1, *shape),
                          surf[key].at_nodes(rw, edge, linear=True).reshape(-1, *shape))
                    for key in ("M0", "M1")}
        for iz in range(nz):
            dcl = g["cl"][iz] - g["clb"][iz]
            cpow = cfac[:, None] * dcl[None, :] ** np.arange(nc)[:, None]
            P = (cpow[:, None, :] * mpow[None, :, :]).reshape(nc * nm, -1).T * (avals[iz] * base)[:, None]
            for k, key in enumerate(("M0", "M1")):
                K[iz, ib, k] = (surfaces[key][0][:, iz] @ P).T.reshape(nc, nm, -1)
                Ke[iz, ib, k] = (surfaces[key][1][:, iz] @ P).T.reshape(nc, nm, -1)

        # model covariance, computed as sncosmo does: in the rest-frame (shifted) band.
        rbands = [g["bp"].shifted(a) for a in avals]
        rgrids = [integration_grid(rb.minwave(), rb.maxwave(), spacing) for rb in rbands]
        rwave = np.concatenate([rwv for rwv, _ in rgrids])
        weight = np.concatenate([rb(rwv) * rwv * rdw / HC_ERG_AA for rb, (rwv, rdw) in zip(rbands, rgrids)])
        starts = np.cumsum([0] + [len(rwv) for rwv, _ in rgrids[:-1]])
        for k, key in enumerate(("M0", "M1")):
            F[:, ib, k] = np.add.reduceat(surf[key].at_nodes(rwave) * weight, starts, axis=1).T
            Fe[:, ib, k] = np.add.reduceat(surf[key].at_nodes(rwave, edge, linear=True) * weight,
                                           starts, axis=1).T
        weff = np.array([rb.wave_eff for rb in rbands])
        for j, key in enumerate(("LCRV00", "LCRV11", "LCRV01", "errscale")):
            V[:, ib, j] = surf[key].at_nodes(weff).T
            Ve[:, ib, j] = surf[key].at_nodes(weff, edge, linear=True).T
        CD[:, ib] = src._colordisp(weff)

    band_range = np.array([[grids[b]["bp"].minwave(), grids[b]["bp"].maxwave()] for b in bands])
    return dict(K=K, Ke=Ke, F=F, Fe=Fe, V=V, Ve=Ve, CD=CD, CLb=CLb, Kb=Kb,
                zgrid=zgrid, phM=phM, phV=phV, bands=list(bands),
                band_range=band_range, wave_range=(src.minwave(), src.maxwave()),
                nc=nc, nm=nm, r_v=r_v, source=src)


@lru_cache(maxsize=16)
def _cached_tables(bands, source_key, zmin, zmax, ebv_max, r_v, c_max, dz):
    name, version = source_key
    return build_tables(list(bands), zmin, zmax, source=name, version=version,
                        ebv_max=ebv_max, r_v=r_v, c_max=c_max, dz=dz)


def get_tables(bands, source, zmin, zmax, ebv_max=0., r_v=3.1, c_max=1.5, dz=2.5e-4):
    """Get tables covering the given ranges (cached).

    The redshift range is snapped to a 0.01 grid and E(B-V) to a 0.1 grid, so
    that similar calls share the same (cached) tables.

    Parameters
    ----------
    bands : list of str
        Names of the bandpasses.

    source : sncosmo.SALT2Source
        SALT2 source.

    zmin, zmax : float
        Redshift range to cover.

    ebv_max : float, optional
        Maximum Milky Way E(B-V) to support. The default is 0 (no dust).

    r_v : float, optional
        R_V of the CCM89 Milky Way dust. The default is 3.1.

    c_max : float, optional
        Maximum absolute SALT2 color to support exactly. The default is 1.5.

    dz : float, optional
        Redshift step of the tables. The default is 2.5e-4.

    Returns
    -------
    dict
        Output of :func:`build_tables`.

    See Also
    --------
    build_tables : Build the tables (no cache).
    """
    zmin, zmax = np.floor(zmin * 100) / 100, np.ceil(zmax * 100) / 100 + 0.01
    ebv_max = np.ceil(ebv_max * 10) / 10 if ebv_max > 0 else 0.
    if source.name is None:  # unregistered source: no cache
        return build_tables(list(bands), zmin, zmax, source=source, ebv_max=ebv_max,
                            r_v=r_v, c_max=c_max, dz=dz)
    return _cached_tables(tuple(bands), (source.name, source.version), float(zmin), float(zmax),
                          float(ebv_max), float(r_v), float(c_max), float(dz))
