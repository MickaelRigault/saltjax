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

__all__ = ["get_salt_source", "build_tables", "get_tables",
           # building blocks
           "Surface", "load_surfaces", "check_surface_grids", "get_modeldir", "get_ncolor_terms",
           "get_band_grids", "get_max_colorlaw_deviation", "get_covariance_tables", "pad_band_grids",
           "clear_cache", "cache_info"]

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


# ================== #
#                    #
#   SALT surfaces    #
#                    #
# ================== #
class Surface:
    """A SALT surface on its (phase, wavelength) grid.

    It is interpolated in wavelength with sncosmo's kernel at its phase nodes
    (see :meth:`get_values`). As in sncosmo, the kernel is applied in
    node-index space, so non-uniform wavelength grids are supported; the
    phase grid must be uniform.

    Instances are usually loaded with :meth:`Surface.from_file` (or all at once
    with :func:`load_surfaces`).

    Parameters
    ----------
    phase : numpy.ndarray
        Phase nodes [day], uniformly spaced.

    wave : numpy.ndarray
        Wavelength nodes [A].

    values : numpy.ndarray
        Values at the nodes, shape (nphase, nwave).

    Raises
    ------
    NotImplementedError
        If the phase grid is not uniform.
    """

    def __init__(self, phase, wave, values):
        """Initialize the Surface (see the class docstring for parameters)."""
        if not np.allclose(np.diff(phase), phase[1] - phase[0]):
            raise NotImplementedError("non-uniform SALT phase grids are not supported")

        self._phase = phase
        self._wave = wave
        self._values = values

    @classmethod
    def from_file(cls, filename, scale=1.):
        """Load a surface from a SALT grid file (sncosmo's ascii format).

        Parameters
        ----------
        filename : str
            Path of the file.

        scale : float, optional
            Multiplicative factor of the values. The default is 1.

        Returns
        -------
        Surface
            The loaded surface.
        """
        phase, wave, values = read_griddata_ascii(filename)
        return cls(phase, wave, values * scale)

    # --------- #
    #  GETTER   #
    # --------- #
    def get_values(self, wave, rows=slice(None), linear=False):
        """Get the values at the phase nodes `rows` and wavelengths `wave`.

        Parameters
        ----------
        wave : numpy.ndarray
            Wavelengths [A].

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
        nwave = len(self.wave)
        # fractional node index (-1 or n outside the grid, where the kernel is 0)
        u = np.interp(np.asarray(wave, float), self.wave, np.arange(nwave), left=-1., right=nwave)
        idx, weights = kernel_weights(u, nwave)
        if linear:
            frac = u - idx[:, 1]
            inside = weights.sum(-1, keepdims=True) > 0
            weights = np.stack([np.zeros_like(frac), 1 - frac, frac, np.zeros_like(frac)], -1) * inside

        # sparse (len(wave), nwave) interpolation matrix: one product for all phase nodes
        W = sparse.csr_matrix((weights.ravel(), (np.repeat(np.arange(len(u)), 4), idx.ravel())),
                              shape=(len(u), nwave))
        return (W @ self.values[rows].T).T

    # --------- #
    #  Plotting #
    # --------- #
    def show(self, ax=None, phases=None, **kwargs):  # pragma: no cover
        """Show the surface at a few phase nodes, as a function of wavelength.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, optional
            Axes to plot on. If None, a new figure is created. The default is None.

        phases : list of float or None, optional
            Phases [day] to show (the closest nodes are used). If None, -10,
            0, +10, +20 and +40 days. The default is None.

        **kwargs
            Passed to `matplotlib.axes.Axes.plot`.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot.
        """
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=[7, 3.5])
        else:
            fig = ax.figure

        if phases is None:
            phases = [-10, 0, 10, 20, 40]

        for phase_ in phases:
            row = np.argmin(np.abs(self.phase - phase_))      # closest phase node
            ax.plot(self.wave, self.values[row], label=f"{self.phase[row]:+.0f} d", **kwargs)

        ax.set_xlabel("Rest-frame wavelength [Å]", fontsize="large")
        ax.legend(fontsize="small", frameon=False)
        return fig

    # ============= #
    #   Properties  #
    # ============= #
    @property
    def phase(self):
        """Phase nodes [day]."""
        return self._phase

    @property
    def wave(self):
        """Wavelength nodes [A]."""
        return self._wave

    @property
    def values(self):
        """Values at the nodes, shape (nphase, nwave)."""
        return self._values


@lru_cache(maxsize=8)
def load_surfaces(modeldir, kind, scale_factor):
    """Load (once per session, cached) the SALT surfaces of a model directory.

    Parameters
    ----------
    modeldir : str
        Directory of the source files (see :func:`get_modeldir`).

    kind : {'salt2', 'salt3'}
        Kind of source: sets the file names.

    scale_factor : float
        Flux scale of the source (``source._SCALE_FACTOR``): M0 and M1 are
        scaled by it, and the SALT3 variances (in flux units) by its square.

    Returns
    -------
    dict
        `Surface` per key: 'M0', 'M1', 'LCRV00', 'LCRV11', 'LCRV01', and
        'errscale' (SALT2 only).
    """
    def get_scale(key):
        if key in ("M0", "M1"):
            return scale_factor
        return scale_factor ** 2 if kind == "salt3" and key.startswith("LCRV") else 1.

    return {key: Surface.from_file(os.path.join(modeldir, filename), scale=get_scale(key))
            for key, filename in _SURFACE_FILES[kind].items()}


def check_surface_grids(surfaces):
    """Check that the surface grids are supported.

    Parameters
    ----------
    surfaces : dict
        Output of :func:`load_surfaces`.

    Raises
    ------
    NotImplementedError
        If M0 and M1 do not share their grid, or the LCRV and errscale grids
        do not share their phase nodes.
    """
    phase_var = surfaces["LCRV00"].phase
    if not all(np.allclose(surfaces[key].phase, phase_var) for key in _variance_keys(surfaces)):
        raise NotImplementedError("LCRV and errscale grids must share their phase nodes")
    if not (np.array_equal(surfaces["M0"].phase, surfaces["M1"].phase) and
            np.array_equal(surfaces["M0"].wave, surfaces["M1"].wave)):
        raise NotImplementedError("M0 and M1 must share their grid")


def _variance_keys(surfaces):
    """Keys of the model-variance surfaces present (errscale is SALT2 only)."""
    return [key for key in ("LCRV00", "LCRV11", "LCRV01", "errscale") if key in surfaces]


# ================== #
#                    #
#   Sources          #
#                    #
# ================== #
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


def get_modeldir(source):
    """Get the directory of the files of a registered SALT source.

    Parameters
    ----------
    source : sncosmo.SALT2Source or sncosmo.SALT3Source
        A source registered in sncosmo (see :func:`get_salt_source`).

    Returns
    -------
    str
        Directory of the source files (downloaded by sncosmo if needed).

    Raises
    ------
    ValueError
        If the source is not registered in sncosmo.
    """
    from sncosmo.builtins import DATADIR
    loader = sncosmo.models._SOURCES._loaders.get((source.name, source.version))
    if loader is None:
        raise ValueError(f"cannot locate the files of source {source.name!r} ({source.version}); "
                         "provide modeldir.")
    return DATADIR.abspath(loader[1][0], isdir=True)


def get_ncolor_terms(xmax, tol=1e-13):
    """Get the number of terms of the exponential series so that the remainder < tol.

    The color factor ``exp(x)`` with ``x = -0.4 ln10 c dCL`` is written as its
    series; ``xmax`` is the largest ``|x|`` (see :func:`build_tables`).

    Parameters
    ----------
    xmax : float
        Largest absolute value of the exponent.

    tol : float, optional
        Tolerance on the first neglected term. The default is 1e-13.

    Returns
    -------
    int
        Number of terms.
    """
    return next(n for n in range(1, 200) if xmax ** n / math.factorial(n) < tol)


# ================== #
#                    #
#   Tables           #
#                    #
# ================== #
def build_tables(bands, zmin, zmax, source="salt2", version=None, modeldir=None,
                 c_max=1.5, dz=2.5e-4, spacing=5.):
    """Prepare the SALT2 / SALT3 model for the given bands and redshift range.

    The band fluxes are integrated per target (see
    :func:`saltjax.data.band_integrals`) from the band integration grids and
    the M0/M1 surfaces returned here. The model covariance, which does not
    depend on the effects (as in sncosmo), is tabulated on a redshift grid.

    This:

    1. loads the source surfaces (:func:`load_surfaces`) and checks their grids,
    2. builds the band integration grids (:func:`get_band_grids`),
    3. sets the number of color terms needed up to `c_max`
       (:func:`get_ncolor_terms`),
    4. tabulates the model covariance on the redshift grid
       (:func:`get_covariance_tables`),
    5. pads the band grids to a common length (:func:`pad_band_grids`).

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
    # Source
    src = get_salt_source(source, version)
    kind = _SOURCE_KINDS[type(src)]
    relative_var = kind == "salt2"     # SALT2: variance relative to M0; SALT3: in flux units
    modeldir = get_modeldir(src) if modeldir is None else modeldir
    surfaces = load_surfaces(modeldir, kind, src._SCALE_FACTOR)
    check_surface_grids(surfaces)

    # Bands and color terms
    zgrid = np.arange(max(zmin - 3 * dz, 0.), zmax + 4 * dz, dz)  # margin for the kernel
    band_grids = get_band_grids(bands, spacing=spacing)
    xmax = abs(LN10_04) * c_max * get_max_colorlaw_deviation(src, band_grids, zgrid)
    ncolor = get_ncolor_terms(xmax)

    # Model covariance on the redshift grid
    F, Fe, V, Ve, CD = get_covariance_tables(surfaces, band_grids, zgrid, src,
                                             relative_var=relative_var, spacing=spacing)

    band_wave, band_base, band_wgt, band_range = pad_band_grids(band_grids)
    return dict(F=F, Fe=Fe, V=V, Ve=Ve, CD=CD,
                M=np.stack([surfaces["M0"].values, surfaces["M1"].values]), M_wave=surfaces["M0"].wave,
                band_wave=band_wave, band_base=band_base, band_wgt=band_wgt,
                zgrid=zgrid, phM=surfaces["M0"].phase, phV=surfaces["LCRV00"].phase, bands=list(bands),
                band_range=band_range, wave_range=(src.minwave(), src.maxwave()),
                nc=ncolor, relative_var=relative_var, source=src)


def get_band_grids(bands, spacing=5.):
    """Get the band integration grids (as ``sncosmo.Model.bandflux``).

    Parameters
    ----------
    bands : list of str
        Names of the (sncosmo-registered) bandpasses.

    spacing : float, optional
        Wavelength step of the integration [A]. The default is 5.

    Returns
    -------
    list of dict
        One dict per band: 'bp' (the ``sncosmo.Bandpass``), 'wave' (observer
        frame [A]), 'wgt' (wave x transmission) and 'base' (the integration
        weight, wgt dwave / (hc zp_ab)).
    """
    ab = sncosmo.get_magsystem("ab")
    band_grids = []
    for band_ in bands:
        bandpass = sncosmo.get_bandpass(band_)
        wave, dwave = integration_grid(bandpass.minwave(), bandpass.maxwave(), spacing)
        wgt = wave * bandpass(wave)
        band_grids.append(dict(bp=bandpass, wave=wave, wgt=wgt,
                               base=wgt * dwave / HC_ERG_AA / ab.zpbandflux(bandpass)))
    return band_grids


def get_max_colorlaw_deviation(source, band_grids, zgrid):
    """Get the largest deviation of the color law from its band mean.

    Parameters
    ----------
    source : sncosmo.SALT2Source or sncosmo.SALT3Source
        SALT source.

    band_grids : list of dict
        Output of :func:`get_band_grids`.

    zgrid : numpy.ndarray
        Redshifts.

    Returns
    -------
    float
        Largest ``|CL - CLb|`` over the bands and redshifts.
    """
    a = 1. / (1. + zgrid)
    deviation = 0.
    for grid_ in band_grids:
        rest_wave = grid_["wave"][None, :] * a[:, None]                    # (nz, nlambda)
        colorlaw = source._colorlaw(rest_wave.ravel()).reshape(len(zgrid), -1)
        colorlaw_mean = (colorlaw * grid_["wgt"]).sum(1, keepdims=True) / grid_["wgt"].sum()
        deviation = max(deviation, np.abs(colorlaw - colorlaw_mean).max())
    return deviation


def get_covariance_tables(surfaces, band_grids, zgrid, source, relative_var=True, spacing=5.):
    """Tabulate the model covariance of each band on a redshift grid.

    Computed as sncosmo does, in the rest-frame (shifted) band: band integral
    (SALT2) or band average (SALT3) of M0 and M1, and the variance surfaces
    at the band effective wavelength.

    Parameters
    ----------
    surfaces : dict
        Output of :func:`load_surfaces`.

    band_grids : list of dict
        Output of :func:`get_band_grids`.

    zgrid : numpy.ndarray
        Redshifts.

    source : sncosmo.SALT2Source or sncosmo.SALT3Source
        SALT source (for its color dispersion).

    relative_var : bool, optional
        True for SALT2 (variance relative to M0: band integrals), False for
        SALT3 (band averages). The default is True.

    spacing : float, optional
        Wavelength step of the integration [A]. The default is 5.

    Returns
    -------
    F : numpy.ndarray
        M0 and M1 in each rest-frame band, shape (nz, nbands, 2, nphase).

    Fe : numpy.ndarray
        Same at the edge phase nodes (linear in wavelength), shape (nz, nbands, 2, 4).

    V : numpy.ndarray
        LCRV00, LCRV11, LCRV01 and errscale, shape (nz, nbands, 4, nphase_var).

    Ve : numpy.ndarray
        Same at the edge phase nodes, shape (nz, nbands, 4, 4).

    CD : numpy.ndarray
        Color dispersion, shape (nz, nbands).
    """
    edge = list(EDGE_NODES)
    variance_keys = _variance_keys(surfaces)
    a = 1. / (1. + zgrid)
    nz, nbands = len(zgrid), len(band_grids)
    nphase, nphase_var = len(surfaces["M0"].phase), len(surfaces["LCRV00"].phase)

    F = np.zeros((nz, nbands, 2, nphase))
    Fe = np.zeros((nz, nbands, 2, 4))
    V = np.zeros((nz, nbands, 4, nphase_var))
    Ve = np.zeros((nz, nbands, 4, 4))
    CD = np.zeros((nz, nbands))
    for iband, grid_ in enumerate(band_grids):
        # rest-frame band at each redshift, and its integration grid
        rest_bands = [grid_["bp"].shifted(a_) for a_ in a]
        rest_grids = [integration_grid(band_.minwave(), band_.maxwave(), spacing) for band_ in rest_bands]
        rest_wave = np.concatenate([wave_ for wave_, _ in rest_grids])
        rest_wgts = [band_(wave_) * wave_ for band_, (wave_, _) in zip(rest_bands, rest_grids)]
        # SALT2: band integral; SALT3: band average
        weight = np.concatenate([wgt_ * dwave_ / HC_ERG_AA if relative_var else wgt_ / wgt_.sum()
                                 for wgt_, (_, dwave_) in zip(rest_wgts, rest_grids)])
        starts = np.cumsum([0] + [len(wave_) for wave_, _ in rest_grids[:-1]])  # first point per redshift

        for k, key in enumerate(("M0", "M1")):
            F[:, iband, k] = np.add.reduceat(surfaces[key].get_values(rest_wave) * weight, starts, axis=1).T
            Fe[:, iband, k] = np.add.reduceat(surfaces[key].get_values(rest_wave, edge, linear=True) * weight,
                                              starts, axis=1).T

        # variance surfaces at the band effective wavelength
        wave_eff = np.array([band_.wave_eff for band_ in rest_bands])
        for j, key in enumerate(variance_keys):
            V[:, iband, j] = surfaces[key].get_values(wave_eff).T
            Ve[:, iband, j] = surfaces[key].get_values(wave_eff, edge, linear=True).T
        if "errscale" not in surfaces:       # SALT3: no error scaling
            V[:, iband, 3] = Ve[:, iband, 3] = 1.

        CD[:, iband] = source._colordisp(wave_eff)

    return F, Fe, V, Ve, CD


def pad_band_grids(band_grids):
    """Pad the band integration grids to the same length (zero weight).

    Parameters
    ----------
    band_grids : list of dict
        Output of :func:`get_band_grids`.

    Returns
    -------
    band_wave : numpy.ndarray
        Wavelengths [A] (edge-padded), shape (nbands, nlambda).

    band_base : numpy.ndarray
        Integration weights (zero-padded), shape (nbands, nlambda).

    band_wgt : numpy.ndarray
        wave x transmission (zero-padded), shape (nbands, nlambda).

    band_range : numpy.ndarray
        Wavelength range of each band [A], shape (nbands, 2).
    """
    nlambda = max(len(grid_["wave"]) for grid_ in band_grids)
    band_wave = np.array([np.pad(grid_["wave"], (0, nlambda - len(grid_["wave"])), mode="edge")
                          for grid_ in band_grids])
    band_base, band_wgt = (np.array([np.pad(grid_[key], (0, nlambda - len(grid_[key]))) for grid_ in band_grids])
                           for key in ("base", "wgt"))
    band_range = np.array([[grid_["bp"].minwave(), grid_["bp"].maxwave()] for grid_ in band_grids])
    return band_wave, band_base, band_wgt, band_range


# ================== #
#                    #
#   Cache            #
#                    #
# ================== #
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
    clear_cache : Empty the cache.
    """
    zmin, zmax = np.floor(zmin * 100) / 100, np.ceil(zmax * 100) / 100 + 0.01
    # registered sources are cached by (name, version), others by instance
    key = (source.name, source.version) if source.name is not None else (source, None)
    modeldir = None if modeldir is None else os.path.abspath(modeldir)
    return _get_cached_tables_(tuple(bands), *key, modeldir, float(zmin), float(zmax),
                               float(c_max), float(dz))


@lru_cache(maxsize=16)
def _get_cached_tables_(bands, source, version, modeldir, zmin, zmax, c_max, dz):
    """Cached :func:`build_tables` (arguments must be hashable, see :func:`get_tables`)."""
    return build_tables(list(bands), zmin, zmax, source=source, version=version,
                        modeldir=modeldir, c_max=c_max, dz=dz)


def clear_cache():
    """Empty the cache of :func:`get_tables` (tables are rebuilt at the next call)."""
    _get_cached_tables_.cache_clear()


def cache_info():
    """Get the statistics of the cache of :func:`get_tables`.

    Returns
    -------
    functools._CacheInfo
        hits, misses (number of tables built), maxsize and currsize.
    """
    return _get_cached_tables_.cache_info()
