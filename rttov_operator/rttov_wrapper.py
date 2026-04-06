"""
pyrttov setup, profile building, and forward model execution.

Handles creating RTTOV instances, building atmospheric profiles from extracted
WRF data, computing solar angles, and running the RTTOV forward model.

Targets RTTOV v14 / pyrttov wrapper API.
"""

import datetime as dt
import logging
import sys
import time
from typing import Literal

import numpy as np
import pandas as pd
import pvlib
import xarray as xr

from .config import RTTOVOperatorConfig
from .convert import RTTOVProfileData, extract_rttov_profiles

logger = logging.getLogger(__name__)


def _ensure_pyrttov(config: RTTOVOperatorConfig):
    """Ensure pyrttov is importable, adding wrapper path to sys.path if configured."""
    if config.coefficients.python_wrapper_path:
        wrapper_path = config.coefficients.python_wrapper_path
        if wrapper_path not in sys.path:
            sys.path.insert(0, wrapper_path)


def create_rttov_instance(
    config: RTTOVOperatorConfig, mode: Literal["ir", "vis"]
) -> "pyrttov.Rttov":
    """Create and configure a pyrttov.Rttov instance.

    Args:
        config: Operator configuration.
        mode: "ir" for infrared channels, "vis" for visible channels.

    Returns:
        Configured and loaded pyrttov.Rttov instance.
    """
    _ensure_pyrttov(config)
    import pyrttov

    rttov = pyrttov.Rttov()
    coef = config.coefficients

    # Coefficient files
    rttov.FileCoef = coef.resolve(coef.coef_file)
    rttov.FileHydrotable = coef.resolve(coef.hydrotable_file)
    if mode == "vis":
        rttov.FileMfasisNN = coef.resolve(coef.mfasis_nn_file)

    # Options common to both modes
    rttov.Options.StoreRad = False
    rttov.Options.Nthreads = config.performance.nthreads
    rttov.Options.NprofsPerCall = config.performance.nprofs_per_call
    rttov.Options.EnableInterp = True       # was AddInterp
    rttov.Options.Hydrometeors = True       # was AddClouds
    rttov.Options.UserHydroOptParam = False  # was UserCldOptParam
    rttov.Options.ThermalSolver = 3         # 3=delta-Eddington (THERMAL_SOLVER_DELTA_EDD)
    rttov.Options.O3Data = False            # was OzoneData
    rttov.Options.ApplyRegLimits = True
    rttov.Options.VerboseWrapper = config.verbose
    rttov.Options.Verbose = config.verbose

    # Mode-specific options
    if mode == "ir":
        rttov.Options.Solar = False         # was AddSolar
        rttov.Options.SolarSolver = 1       # was VisScattModel
        chan_list = config.channels.ir_channel_id_list()
    else:
        rttov.Options.Solar = True          # was AddSolar
        rttov.Options.SolarSolver = 2       # 2=MFASIS-NN (SOLAR_SOLVER_MFASIS_NN)
        chan_list = config.channels.vis_channel_id_list()

    # Load instrument
    try:
        rttov.loadInst(chan_list)
    except pyrttov.RttovError as e:
        raise RuntimeError(f"Failed to load RTTOV instrument ({mode}): {e}") from e

    logger.info(
        "Loaded RTTOV %s instance with %d channels: %s",
        mode.upper(),
        len(chan_list),
        chan_list,
    )
    return rttov


def compute_solar_angles(
    lat: np.ndarray, lon: np.ndarray, time_utc: dt.datetime
) -> tuple[np.ndarray, np.ndarray]:
    """Compute solar zenith and azimuth angles for each profile using pvlib.

    Args:
        lat: Latitude array, shape (nprofiles,).
        lon: Longitude array, shape (nprofiles,).
        time_utc: UTC datetime for the computation.

    Returns:
        (sunzen, sunazi): Solar zenith and azimuth angles in degrees,
        each shape (nprofiles,).
    """
    nprofiles = len(lat)
    times = pd.DatetimeIndex([time_utc] * nprofiles, tz="UTC")
    solpos = pvlib.solarposition.get_solarposition(
        times, lat, lon, method="nrel_numpy"
    )
    sunzen = solpos["apparent_zenith"].values
    sunazi = solpos["azimuth"].values
    return sunzen.astype(np.float64), sunazi.astype(np.float64)


def build_profiles(
    data: RTTOVProfileData,
    config: RTTOVOperatorConfig,
    time_utc: dt.datetime,
) -> "pyrttov.Profiles":
    """Build a pyrttov.Profiles object from extracted WRF data.

    Args:
        data: Extracted profile data from convert.extract_rttov_profiles.
        config: Operator configuration.
        time_utc: UTC datetime for the profiles.

    Returns:
        Configured pyrttov.Profiles instance.
    """
    _ensure_pyrttov(config)
    import pyrttov

    nprofiles = data.nprofiles
    nlevels = data.nlevels
    nhydro = config.clouds.nhydro

    # Profiles(nprofiles, nlevels, nsurfaces): nlevels = half-level count = nlayers+1
    profiles = pyrttov.Profiles(nprofiles, nlevels + 1, 1)
    profiles.GasUnits = 1       # kg/kg for mixing ratios (gas_unit_kg_per_kg)
    profiles.MmrHydro = True    # kg/kg for hydrometeor mixing ratios

    # Pressure: half-levels required; full-levels omitted (RTTOV derives them from PHalf)
    profiles.PHalf = data.p_half   # (nprofiles, nlevels+1)

    # Atmospheric profiles (on nlayers = nlevels full levels)
    profiles.T = data.t
    profiles.Q = data.qv

    # Hydrometeors: set per-type using 1-based indices; shape (nprofiles, nlevels)
    liq = config.clouds.liq_hydro_idx + 1  # convert to 1-based
    ice = config.clouds.ice_hydro_idx + 1

    zeros = np.zeros((nprofiles, nlevels), dtype=np.float64)
    cfrac = data.cfrac.astype(np.float64)

    for n in range(1, nhydro + 1):
        profiles.setHydroN(zeros.copy(), n)
        profiles.setHydroFracN(zeros.copy(), n)
        profiles.setHydroDeffN(zeros.copy(), n)

    profiles.setHydroN(data.qc.astype(np.float64), liq)
    profiles.setHydroFracN(cfrac, liq)
    profiles.setHydroDeffN(np.full((nprofiles, nlevels), config.clouds.clwde, dtype=np.float64), liq)

    profiles.setHydroN(data.qi.astype(np.float64), ice)
    profiles.setHydroFracN(cfrac, ice)
    profiles.setHydroDeffN(np.full((nprofiles, nlevels), config.clouds.icede, dtype=np.float64), ice)

    # Near-surface: [t2m, q2m, u10, v10, wind_fetch]; shape (nprofiles, nsurfaces=1, 5)
    profiles.NearSurface = np.column_stack([
        data.t2m,                          # 2m temperature from T2 variable
        data.q2m,                          # 2m water vapor from Q2 variable
        data.u10[:, 0],
        data.v10[:, 0],
        np.full(nprofiles, 1e5),           # wind fetch [m]
    ]).astype(np.float64).reshape(nprofiles, 1, 5)

    # Datetime: (nprofiles, 6) = [year, month, day, hour, min, sec]
    profiles.DateTimes = np.tile(
        [time_utc.year, time_utc.month, time_utc.day,
         time_utc.hour, time_utc.minute, time_utc.second],
        (nprofiles, 1),
    ).astype(np.int32)

    # Solar angles (per-profile via pvlib)
    sunzen, sunazi = compute_solar_angles(data.lat, data.lon, time_utc)

    # Angles: [satzen, satazi, sunzen, sunazi]
    profiles.Angles = np.column_stack([
        np.full(nprofiles, config.satellite.satzen),
        np.full(nprofiles, config.satellite.satazi),
        sunzen,
        sunazi,
    ]).astype(np.float64)

    # Surface geometry: [lat, lon, elevation_km]
    profiles.SurfGeom = np.column_stack(
        [data.lat, data.lon, data.surface_altitude]
    ).astype(np.float64)

    # Surface type: [surftype, watertype]; shape (nprofiles, nsurfaces=1, 2)
    profiles.SurfType = np.column_stack(
        [data.surftype, np.zeros(nprofiles, dtype=np.int32)]
    ).astype(np.int32).reshape(nprofiles, 1, 2)

    # Skin: [skinT, salinity, snow_frac, foam_frac, fastem_coef1..5]; shape (nprofiles, nsurfaces=1, 9)
    sfc = config.surface
    skin = np.tile(
        [0.0, sfc.default_salinity, sfc.default_snow_fraction, sfc.default_foam_fraction,
         3.0, 5.0, 15.0, 0.1, 0.3],   # default FASTEM-6 coefficients
        (nprofiles, 1),
    ).astype(np.float64)
    skin[:, 0] = data.tsk[:, 0]
    profiles.Skin = skin.reshape(nprofiles, 1, 9)

    return profiles


def setup_surface_emis_refl(
    rttov_instance: "pyrttov.Rttov",
    config: RTTOVOperatorConfig,
    mode: Literal["ir", "vis"],
    nchan: int,
    nprofiles: int,
    time_utc: dt.datetime,
) -> None:
    """Set up surface emissivity and reflectance for RTTOV.

    Attempts to use atlas data if configured, falls back to RTTOV defaults
    (setting -1 triggers RTTOV's built-in emissivity models).

    Args:
        rttov_instance: The RTTOV instance to configure.
        config: Operator configuration.
        mode: "ir" or "vis".
        nchan: Number of channels.
        nprofiles: Number of profiles.
        time_utc: UTC datetime (month needed for atlas).
    """
    _ensure_pyrttov(config)
    import pyrttov

    # Shape: (5, nprofiles, nsurfaces=1, nchannels); -1 = use RTTOV built-in model
    surfemisrefl = -np.ones((5, nprofiles, 1, nchan), dtype=np.float64)

    coef = config.coefficients
    month = time_utc.month

    # Try IR emissivity atlas
    try:
        ir_atlas = pyrttov.Atlas()
        ir_atlas.AtlasPath = coef.resolve(coef.emis_atlas_path)
        ir_atlas.loadIrEmisAtlas(month, ang_corr=True)
        atlas_emis = ir_atlas.getEmisBrdf(rttov_instance)  # (nprofiles, nsurfaces, nchan)
        surfemisrefl[0] = atlas_emis  # emissivity
        surfemisrefl[1] = atlas_emis  # direct reflectance
        logger.debug("Loaded IR emissivity atlas for month %d", month)
    except Exception as e:
        logger.warning("Could not load IR emissivity atlas: %s. Using RTTOV defaults.", e)

    # Try BRDF atlas for VIS channels
    if mode == "vis":
        try:
            brdf_atlas = pyrttov.Atlas()
            brdf_atlas.AtlasPath = coef.resolve(coef.brdf_atlas_path)
            brdf_atlas.loadBrdfAtlas(month, rttov_instance)
            brdf_atlas.IncSea = False
            atlas_brdf = brdf_atlas.getEmisBrdf(rttov_instance)  # (nprofiles, nsurfaces, nchan)
            surfemisrefl[1] = atlas_brdf
            logger.debug("Loaded BRDF atlas for month %d", month)
        except Exception as e:
            logger.warning("Could not load BRDF atlas: %s. Using RTTOV defaults.", e)

    rttov_instance.SurfEmisRefl = surfemisrefl


def run_rttov(
    ds_t: xr.Dataset,
    config: RTTOVOperatorConfig,
    ir_rttov: "pyrttov.Rttov | None",
    vis_rttov: "pyrttov.Rttov | None",
) -> dict[str, np.ndarray]:
    """Run RTTOV for a single timestep.

    Args:
        ds_t: Single-timestep xarray Dataset.
        config: Operator configuration.
        ir_rttov: Pre-loaded IR RTTOV instance, or None if no IR channels.
        vis_rttov: Pre-loaded VIS RTTOV instance, or None if no VIS channels.

    Returns:
        Dict mapping channel names to (ny, nx) output arrays (BT or reflectance).
    """
    _ensure_pyrttov(config)
    import pyrttov

    data = extract_rttov_profiles(ds_t, config)

    time_val = ds_t["t"].values
    time_utc = pd.Timestamp(time_val).to_pydatetime().replace(tzinfo=dt.timezone.utc)
    logger.info("Processing timestep: %s", time_utc)

    profiles = build_profiles(data, config, time_utc)

    results: dict[str, np.ndarray] = {}

    # Run IR channels
    if ir_rttov is not None and config.channels.ir_channels:
        ir_rttov.Profiles = profiles
        nchan_ir = len(config.channels.ir_channels)
        setup_surface_emis_refl(
            ir_rttov, config, "ir", nchan_ir, data.nprofiles, time_utc
        )
        try:
            t0 = time.monotonic()
            ir_rttov.runDirect()
            logger.info("RTTOV IR forward model: %.1fs", time.monotonic() - t0)
        except pyrttov.RttovError as e:
            raise RuntimeError(f"RTTOV IR forward model failed: {e}") from e

        if ir_rttov.RadQuality is not None:
            n_issues = int(np.sum(ir_rttov.RadQuality > 0))
            if n_issues > 0:
                logger.warning("RTTOV IR quality issues: %d profiles", n_issues)

        for i, name in enumerate(config.channels.ir_channels):
            results[name] = (
                ir_rttov.BtRefl[:, i].reshape(data.ny, data.nx).astype(np.float32)
            )

    # Run VIS channels
    if vis_rttov is not None and config.channels.vis_channels:
        vis_rttov.Profiles = profiles
        nchan_vis = len(config.channels.vis_channels)
        setup_surface_emis_refl(
            vis_rttov, config, "vis", nchan_vis, data.nprofiles, time_utc
        )
        try:
            t0 = time.monotonic()
            vis_rttov.runDirect()
            logger.info("RTTOV VIS forward model: %.1fs", time.monotonic() - t0)
        except pyrttov.RttovError as e:
            raise RuntimeError(f"RTTOV VIS forward model failed: {e}") from e

        if vis_rttov.RadQuality is not None:
            n_issues = int(np.sum(vis_rttov.RadQuality > 0))
            if n_issues > 0:
                logger.warning("RTTOV VIS quality issues: %d profiles", n_issues)

        for i, name in enumerate(config.channels.vis_channels):
            results[name] = (
                vis_rttov.BtRefl[:, i].reshape(data.ny, data.nx).astype(np.float32)
            )

    return results
