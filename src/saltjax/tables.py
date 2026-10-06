"""SALT2 / SALT3 model tables: surfaces, band grids and model covariance.

The SALT band flux is linear in the (phase, wavelength) surfaces, so the
surfaces are band-integrated at their native phase nodes (per target, with its
effects, see :func:`saltjax.data.band_integrals`); the color law is written as
an exact band-mean factor times a power series of the deviation from that
mean. The model covariance does not depend on the effects and is tabulated on
a redshift grid.

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
from sncosmo.constants import HC_ERG_AA
from sncosmo.utils import integration_grid
from sncosmo.models import read_griddata_ascii

from .kernel import kernel_weights

__all__ = ["get_salt_source", "build_tables", "get_tables"]

LN10_04 = -0.4 * np.log(10)
EDGE_NODES = (0, 1, -2, -1)       # nodes of the first and last phase intervals

# file names of the surfaces (sncosmo defaults), per source type. SALT2 gives
# the model variance relative to M0 (with an error scaling); SALT3 gives it in
# flux units (scaled as M0**2), without error scaling.
_SURFACE_FILES = {
    "salt2": {"M0": "salt2_template_0.dat",
              "M1": "salt2_template_1.dat",
              "LCRV00": "salt2_lc_relative_variance_0.dat",
              "LCRV11": "salt2_lc_relative_variance_1.dat",
              "LCRV01": "salt2_lc_relative_covariance_01.dat",
              "errscale": "salt2_lc_dispersion_scaling.dat"},
    "salt3": {"M0": "salt3_template_0.dat",
              "M1": "salt3_template_1.dat",
              "LCRV00": "salt3_lc_variance_0.dat",
              "LCRV11": "salt3_lc_variance_1.dat",
              "LCRV01": "salt3_lc_covariance_01.dat"},
}
_SOURCE_KINDS = {sncosmo.SALT2Source: "salt2", sncosmo.SALT3Source: "salt3"}


def _nterms(xmax, tol=1e-13):
    """Number of terms of the exponential series so that the remainder < tol."""
    return next(n for n in range(1, 200) if xmax ** n / math.factorial(n) < tol)


class _Surface:
    """A SALT surface on its (phase, wavelength) grid, interpolated in
    wavelength with sncosmo's kernel at the phase nodes.

    As in sncosmo, the kernel is applied in node-index space, so non-uniform
    wavelength grids are supported; the phase grid must be uniform.
    """

    def __init__(self, filename, scale=1.):
        self.phase, self.wave, values = read_griddata_ascii(filename)
        self.values = values * scale
        if not np.allclose(np.diff(self.phase), self.phase[1] - self.phase[0]):
            raise NotImplementedError("non-uniform SALT phase grids are not supported")

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
        # fractional node index (-1 or n outside the grid, where the kernel is 0)
        u = np.interp(np.asarray(wave, float), self.wave, np.arange(len(self.wave)),
                      left=-1., right=len(self.wave))
        idx, w = kernel_weights(u, len(self.wave))
        if linear:
            f = u - idx[:, 1]
            w = np.stack([np.zeros_like(f), 1 - f, f, np.zeros_like(f)], -1) * (w.sum(-1, keepdims=True) > 0)
        # sparse (len(wave), nwave_nodes) interpolation matrix: one product for all phase nodes
        W = sparse.csr_matrix((w.ravel(), (np.repeat(np.arange(len(u)), 4), idx.ravel())),
                              shape=(len(u), len(self.wave)))
        return (W @ self.values[rows].T).T


@lru_cache(maxsize=8)
def _load_surfaces(modeldir, kind, scale_factor):
    """Read (once per session) the SALT surfaces of a model directory."""
    def scale(key):
        if key in ("M0", "M1"):
            return scale_factor
        return scale_factor ** 2 if kind == "salt3" and key.startswith("LCRV") else 1.
    return {key: _Surface(os.path.join(modeldir, fname), scale=scale(key))
            for key, fname in _SURFACE_FILES[kind].items()}


def get_salt_source(source, version=None):
    """Get a SALT2 or SALT3 source and check it is supported.

    Parameters
    ----------
    source : str or sncosmo.SALT2Source or sncosmo.SALT3Source
        Source name (any SALT2 or SALT3 source registered in sncosmo, e.g.
        'salt2', 'salt2-extended', 'salt3', 'salt3-nir') or instance.

    version : str or None, optional
        Version of the source, if given by name. The default is None (latest).

    Returns
    -------
    sncosmo.SALT2Source or sncosmo.SALT3Source
        The source.

    Raises
    ------
    NotImplementedError
        If the source is neither an ``sncosmo.SALT2Source`` nor an
        ``sncosmo.SALT3Source`` (their subclasses are not supported).
    """
    src = sncosmo.get_source(source, version=version) if isinstance(source, str) else source
    if type(src) not in _SOURCE_KINDS:
        raise NotImplementedError("only sncosmo.SALT2Source and sncosmo.SALT3Source are "
                                  f"supported, got {type(src).__name__}")
    return src



def _get_modeldir(src):
    """Directory of the files of a registered SALT source."""
    from sncosmo.builtins import DATADIR
    loader = sncosmo.models._SOURCES._loaders.get((src.name, src.version))
    if loader is None:
        raise ValueError(f"cannot locate the files of source {src.name!r} ({src.version}); "
                         "provide modeldir.")
    return DATADIR.abspath(loader[1][0], isdir=True)


def build_tables(bands, zmin, zmax, source="salt2", version=None, modeldir=None,
                 c_max=1.5, dz=2.5e-4, spacing=5.):
    """Prepare the SALT2 / SALT3 model for the given bands and redshift range.

    The band fluxes are integrated per target (see
    :func:`saltjax.data.band_integrals`) from the band integration grids and
    the M0/M1 surfaces returned here. The model covariance, which does not
    depend on the effects (as in sncosmo), is tabulated on a redshift grid.

    Parameters
    ----------
    bands : list of str
        Names of the (sncosmo-registered) bandpasses.

    zmin, zmax : float
        Redshift range to tabulate.

    source : str or sncosmo.SALT2Source or sncosmo.SALT3Source, optional
        SALT2 or SALT3 source (name or instance, see :func:`get_salt_source`).
        The default is 'salt2'.

    version : str or None, optional
        Version of the source, if given by name. The default is None (latest).

    modeldir : str or None, optional
        Directory of the source files (with sncosmo's default file names).
        If None, it is found from the sncosmo registry, so it is required
        for a source that is not registered in sncosmo. The default is None.

    c_max : float, optional
        Maximum absolute SALT color to support exactly (sets the number of
        color terms). The default is 1.5.

    dz : float, optional
        Redshift step of the model-covariance tables. The default is 2.5e-4.

    spacing : float, optional
        Wavelength step of the band integration, in Angstrom (sncosmo's
        ``MODEL_BANDFLUX_SPACING``). The default is 5.

    Returns
    -------
    dict
        The model-covariance tables (``F``, ``V``, ``CD``) and their grids;
        arrays ending with 'e' (``Fe``, ``Ve``) hold the edge-interval nodes
        (see :data:`EDGE_NODES`). The band integration grids (``band_wave``,
        ``band_base``, ``band_wgt``, zero-padded to the same length), the
        M0/M1 surfaces at their nodes (``M``, ``M_wave``) and the number of
        color terms ``nc``. ``relative_var`` is True for SALT2 (model
        variance relative to M0, with an error scaling) and False for SALT3
        (model variance in flux units: ``F`` then holds band averages and the
        error scaling is 1).

    Raises
    ------
    NotImplementedError
        If the source grids are not supported (non-uniform phase grids, M0
        and M1 not sharing their grid, or LCRV and errscale grids not sharing
        their phase nodes).
    """
    src = get_salt_source(source, version)
    kind = _SOURCE_KINDS[type(src)]
    relative_var = kind == "salt2"
    modeldir = _get_modeldir(src) if modeldir is None else modeldir
    surf = _load_surfaces(modeldir, kind, src._SCALE_FACTOR)
    vkeys = [k for k in ("LCRV00", "LCRV11", "LCRV01", "errscale") if k in surf]
    phM, phV = surf["M0"].phase, surf["LCRV00"].phase
    if not all(np.allclose(surf[k].phase, phV) for k in vkeys):
        raise NotImplementedError("LCRV and errscale grids must share their phase nodes")
    if not (np.array_equal(surf["M0"].phase, surf["M1"].phase) and
            np.array_equal(surf["M0"].wave, surf["M1"].wave)):
        raise NotImplementedError("M0 and M1 must share their grid")
    edge = list(EDGE_NODES)

    zgrid = np.arange(max(zmin - 3 * dz, 0.), zmax + 4 * dz, dz)
    avals = 1. / (1. + zgrid)
    ab = sncosmo.get_magsystem("ab")
    nz, nb = len(zgrid), len(bands)
    grids, xc = [], 0.
    for b in bands:
        bp = sncosmo.get_bandpass(b)
        wave, dwave = integration_grid(bp.minwave(), bp.maxwave(), spacing)
        trans = bp(wave)
        wgt = wave * trans
        grids.append(dict(bp=bp, wave=wave, wgt=wgt,
                          base=wgt * dwave / HC_ERG_AA / ab.zpbandflux(bp)))
        # largest color-law deviation from its band mean, over the redshift range
        cl = src._colorlaw((wave[None, :] * avals[:, None]).ravel()).reshape(nz, -1)
        xc = max(xc, np.abs(cl - (cl * wgt).sum(1, keepdims=True) / wgt.sum()).max())
    nc = _nterms(abs(LN10_04) * c_max * xc)

    F = np.zeros((nz, nb, 2, len(phM)))
    Fe = np.zeros((nz, nb, 2, 4))
    V = np.zeros((nz, nb, 4, len(phV)))
    Ve = np.zeros((nz, nb, 4, 4))
    CD = np.zeros((nz, nb))
    for ib, g in enumerate(grids):
        # model covariance, computed as sncosmo does: in the rest-frame (shifted) band
        # (SALT2: band integral; SALT3: band average).
        rbands = [g["bp"].shifted(a) for a in avals]
        rgrids = [integration_grid(rb.minwave(), rb.maxwave(), spacing) for rb in rbands]
        rwave = np.concatenate([rwv for rwv, _ in rgrids])
        rwgts = [rb(rwv) * rwv for rb, (rwv, _) in zip(rbands, rgrids)]
        weight = np.concatenate([w * rdw / HC_ERG_AA if relative_var else w / w.sum()
                                 for w, (_, rdw) in zip(rwgts, rgrids)])
        starts = np.cumsum([0] + [len(rwv) for rwv, _ in rgrids[:-1]])
        for k, key in enumerate(("M0", "M1")):
            F[:, ib, k] = np.add.reduceat(surf[key].at_nodes(rwave) * weight, starts, axis=1).T
            Fe[:, ib, k] = np.add.reduceat(surf[key].at_nodes(rwave, edge, linear=True) * weight,
                                           starts, axis=1).T
        weff = np.array([rb.wave_eff for rb in rbands])
        for j, key in enumerate(vkeys):
            V[:, ib, j] = surf[key].at_nodes(weff).T
            Ve[:, ib, j] = surf[key].at_nodes(weff, edge, linear=True).T
        if "errscale" not in surf:
            V[:, ib, 3] = Ve[:, ib, 3] = 1.
        CD[:, ib] = src._colordisp(weff)

    # band integration grids, padded to the same length (zero weight)
    nw = max(len(g["wave"]) for g in grids)
    band_wave = np.array([np.pad(g["wave"], (0, nw - len(g["wave"])), mode="edge") for g in grids])
    band_base, band_wgt = (np.array([np.pad(g[k], (0, nw - len(g[k]))) for g in grids])
                           for k in ("base", "wgt"))
    band_range = np.array([[g["bp"].minwave(), g["bp"].maxwave()] for g in grids])
    return dict(F=F, Fe=Fe, V=V, Ve=Ve, CD=CD,
                M=np.stack([surf["M0"].values, surf["M1"].values]), M_wave=surf["M0"].wave,
                band_wave=band_wave, band_base=band_base, band_wgt=band_wgt,
                zgrid=zgrid, phM=phM, phV=phV, bands=list(bands),
                band_range=band_range, wave_range=(src.minwave(), src.maxwave()),
                nc=nc, relative_var=relative_var, source=src)


@lru_cache(maxsize=16)
def _cached_tables(bands, source, version, modeldir, zmin, zmax, c_max, dz):
    return build_tables(list(bands), zmin, zmax, source=source, version=version,
                        modeldir=modeldir, c_max=c_max, dz=dz)


def get_tables(bands, source, zmin, zmax, c_max=1.5, dz=2.5e-4, modeldir=None):
    """Get tables covering the given ranges (cached).

    The redshift range is snapped to a 0.01 grid, so that similar calls
    share the same (cached) tables.

    Parameters
    ----------
    bands : list of str
        Names of the bandpasses.

    source : sncosmo.SALT2Source or sncosmo.SALT3Source
        SALT source (see :func:`get_salt_source`).

    zmin, zmax : float
        Redshift range to cover.

    c_max : float, optional
        Maximum absolute SALT color to support exactly. The default is 1.5.

    dz : float, optional
        Redshift step of the model-covariance tables. The default is 2.5e-4.

    modeldir : str or None, optional
        Directory of the source files (see :func:`build_tables`); required
        for a source that is not registered in sncosmo. The default is None.

    Returns
    -------
    dict
        Output of :func:`build_tables`.

    See Also
    --------
    build_tables : Build the tables (no cache).
    """
    zmin, zmax = np.floor(zmin * 100) / 100, np.ceil(zmax * 100) / 100 + 0.01
    # registered sources are cached by (name, version), others by instance
    key = (source.name, source.version) if source.name is not None else (source, None)
    modeldir = None if modeldir is None else os.path.abspath(modeldir)
    return _cached_tables(tuple(bands), *key, modeldir, float(zmin), float(zmax),
                          float(c_max), float(dz))
