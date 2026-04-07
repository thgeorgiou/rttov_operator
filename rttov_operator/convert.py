"""
Extract and convert post-processed WRF fields into numpy arrays ready for pyrttov.

The input dataset is expected to have been through xWRF postprocessing, with dimensions
(t, z, y, x) and standard variable names (air_pressure, air_potential_temperature, etc).
"""

import logging
from dataclasses import dataclass

import numpy as np
import xarray as xr

from .config import RTTOVOperatorConfig

logger = logging.getLogger(__name__)

# Poisson constant for dry air: R/cp = 2/7
KAPPA = 2.0 / 7.0

# RTTOV minimum water vapor mixing ratio (hard limit)
QVAPOR_MIN = 1e-11


@dataclass
class RTTOVProfileData:
    """Extracted and reshaped profile data ready for pyrttov.

    All 3D fields have shape (nprofiles, nlevels) with z-axis oriented TOA-first.
    All 2D fields have shape (nprofiles,) or (nprofiles, 1).
    """

    # Dimensions
    ny: int
    nx: int
    nprofiles: int
    nlevels: int

    # 3D atmospheric profiles (nprofiles, nlevels), TOA-first
    p: np.ndarray
    """Pressure full-levels [hPa]"""

    p_half: np.ndarray
    """Pressure half-levels [hPa], shape (nprofiles, nlevels+1), TOA-first.
    The bottom half-level equals the surface pressure."""

    t: np.ndarray
    """Temperature [K]"""

    qv: np.ndarray
    """Water vapor mixing ratio [kg/kg]"""

    qc: np.ndarray
    """Cloud liquid water mixing ratio [kg/kg]"""

    qi: np.ndarray
    """Cloud ice water mixing ratio [kg/kg], includes kappa*QSNOW"""

    cfrac: np.ndarray
    """Cloud fraction [0-1]"""

    # 2D surface fields (nprofiles, 1)
    tsk: np.ndarray
    """Skin temperature [K]"""

    psfc: np.ndarray
    """Surface pressure [hPa]"""

    t2m: np.ndarray
    """2m temperature [K], shape (nprofiles,)"""

    q2m: np.ndarray
    """2m water vapor mixing ratio [kg/kg], shape (nprofiles,)"""

    u10: np.ndarray
    """10m U wind [m/s]"""

    v10: np.ndarray
    """10m V wind [m/s]"""

    # Per-profile geometry (nprofiles,)
    lat: np.ndarray
    """Latitude [degrees]"""

    lon: np.ndarray
    """Longitude [degrees]"""

    surface_altitude: np.ndarray
    """Surface altitude [km]"""

    surftype: np.ndarray
    """Surface type: 0=land, 1=sea"""

    aerosol_data: dict[int, np.ndarray]
    """Data for each aerosol species, indexed as in aertable"""


def _reshape_3d(arr: np.ndarray, nlevels: int) -> np.ndarray:
    """Reshape (z, y, x) -> (nprofiles, nlevels) and flip to TOA-first."""
    return np.flip(arr.reshape(nlevels, -1).T, axis=1)


def _reshape_2d(arr: np.ndarray) -> np.ndarray:
    """Reshape (y, x) -> (nprofiles, 1)."""
    return arr.reshape(-1, 1)


def extract_rttov_profiles(
    ds_t: xr.Dataset, config: RTTOVOperatorConfig
) -> RTTOVProfileData:
    """Extract and convert WRF fields for a single timestep into arrays for pyrttov.

    Args:
        ds_t: Single-timestep xarray Dataset (no 't' dimension, or t already selected).
        config: RTTOV operator configuration.

    Returns:
        RTTOVProfileData with all fields ready for pyrttov profile setup.
    """
    ny, nx = ds_t.sizes["y"], ds_t.sizes["x"]
    nlevels = ds_t.sizes["z"]
    nprofiles = ny * nx

    # --- 3D atmospheric profiles ---

    # Pressure full-levels: Pa -> hPa
    p_hpa = ds_t["air_pressure"].values / 100.0  # (z, y, x)
    p = _reshape_3d(p_hpa, nlevels)

    # Pressure half-levels: computed by sandwiching full levels between P_TOP and PSFC,
    # then averaging adjacent pairs to get nlevels+1 interface pressures.
    p_top_hpa = float(ds_t["P_TOP"].values) / 100.0  # scalar Pa -> hPa
    psfc_hpa = ds_t["PSFC"].values / 100.0  # (y, x)
    # Build extended array: (nlevels+2, y, x).
    # WRF z=0 is near surface, z=nlevels-1 is near TOA, so PSFC goes at index 0,
    # P_TOP at index nlevels+1. _reshape_3d will then flip to TOA-first order.
    p_top_bc = np.full((1, ny, nx), p_top_hpa)
    p_sfc_bc = psfc_hpa[np.newaxis, :, :]
    p_extended = np.concatenate(
        [p_sfc_bc, p_hpa, p_top_bc], axis=0
    )  # (nlevels+2, y, x)
    p_half_hpa = 0.5 * (p_extended[:-1] + p_extended[1:])  # (nlevels+1, y, x)
    p_half = _reshape_3d(p_half_hpa, nlevels + 1)

    # Enforce strict monotonic increase (TOA-first) required by RTTOV.
    # Non-monotonicity can occur near steep orography in sigma coordinates.
    p_half = np.maximum.accumulate(p_half, axis=1)
    eps = np.finfo(np.float64).eps * 1e6  # ~2e-10
    for i in range(1, p_half.shape[1]):
        p_half[:, i] = np.maximum(p_half[:, i], p_half[:, i - 1] + eps)

    # Check pressure monotonicity (should decrease from ground to TOA, i.e. increase
    # in our TOA-first array from left to right -- but after flip it should be
    # decreasing from index 0 to nlevels-1, i.e. TOA has lowest pressure)
    # After flip: index 0 = TOA (low pressure), index -1 = ground (high pressure)
    # So p should be increasing along axis=1
    non_monotonic = np.any(np.diff(p, axis=1) <= 0)
    if non_monotonic:
        logger.warning(
            "Non-monotonic pressure profiles detected. "
            "RTTOV's EnableInterp and ApplyRegLimits should handle this, "
            "but results may be degraded."
        )

    # Temperature: theta * (p_hPa / 1000)^(R/cp)
    theta = ds_t["air_potential_temperature"].values  # (z, y, x)
    temp = theta * (p_hpa / 1000.0) ** KAPPA
    t = _reshape_3d(temp, nlevels)

    # Water vapor
    qv_raw = ds_t["QVAPOR"].values
    qv_raw = np.maximum(qv_raw, QVAPOR_MIN)
    qv = _reshape_3d(qv_raw, nlevels)

    # Cloud liquid water
    qc = _reshape_3d(ds_t["QCLOUD"].values, nlevels)

    # Cloud ice: QICE + kappa * QSNOW
    qi_raw = ds_t["QICE"].values + config.clouds.kappa * ds_t["QSNOW"].values
    qi = _reshape_3d(qi_raw, nlevels)

    # Cloud fraction
    cfrac = _reshape_3d(ds_t["CLDFRA"].values, nlevels)

    # 2D surface fields
    tsk = _reshape_2d(ds_t["TSK"].values)
    psfc = _reshape_2d(ds_t["PSFC"].values / 100.0)  # Pa -> hPa
    t2m = ds_t["T2"].values.ravel()
    q2m = np.maximum(ds_t["Q2"].values.ravel(), QVAPOR_MIN)
    u10 = _reshape_2d(ds_t["U10"].values)
    v10 = _reshape_2d(ds_t["V10"].values)

    # Per-profile geometry
    lat = ds_t["latitude"].values.ravel()
    lon = ds_t["longitude"].values.ravel()

    # Surface altitude from geopotential_height at lowest model level, convert to km
    # After xWRF postprocessing, z index 0 is the lowest level (near surface)
    geopotential_height = ds_t["geopotential_height"].values  # (z, y, x)
    surface_alt_m = geopotential_height[0, :, :]  # lowest level
    surface_altitude = surface_alt_m.ravel() / 1000.0  # m -> km

    # Surface type from XLAND: 1=land->0, 2=water->1
    xland = ds_t["XLAND"].values.ravel()
    surftype = np.where(xland >= 2, 1, 0).astype(np.int32)

    # Aerosol data (apply unit_scale to convert WRF units to kg/kg for RTTOV)
    aerosol_data = {}
    scale = config.aerosols.unit_scale
    for idx, map in config.aerosols.species_map.items():
        aerosol_data[int(idx)] = _reshape_3d(
            scale * sum(ds_t[k].values * v for k, v in map.items()), nlevels
        )

    return RTTOVProfileData(
        ny=ny,
        nx=nx,
        nprofiles=nprofiles,
        nlevels=nlevels,
        p=p.astype(np.float64),
        p_half=p_half.astype(np.float64),
        t=t.astype(np.float64),
        qv=qv.astype(np.float64),
        qc=qc.astype(np.float64),
        qi=qi.astype(np.float64),
        cfrac=cfrac.astype(np.float64),
        tsk=tsk.astype(np.float64),
        psfc=psfc.astype(np.float64),
        t2m=t2m.astype(np.float64),
        q2m=q2m.astype(np.float64),
        u10=u10.astype(np.float64),
        v10=v10.astype(np.float64),
        lat=lat.astype(np.float64),
        lon=lon.astype(np.float64),
        surface_altitude=surface_altitude.astype(np.float64),
        surftype=surftype,
        aerosol_data=aerosol_data,
    )
