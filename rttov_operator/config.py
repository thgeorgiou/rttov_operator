from dataclasses import dataclass, field
from pathlib import Path

from mashumaro.mixins.toml import DataClassTOMLMixin


@dataclass
class ChannelConfig:
    """Defines which channels to simulate and their instrument IDs."""

    ir_channels: list[str] = field(default_factory=list)
    """IR channel names to simulate (e.g. ["IR108", "WV62", "WV73"])"""

    vis_channels: list[str] = field(default_factory=list)
    """VIS channel names to simulate (e.g. ["VIS06", "VIS08"])"""

    channel_ids: dict[str, int] = field(default_factory=dict)
    """Maps channel names to RTTOV coefficient file channel indices.
    Must contain entries for all channels listed in ir_channels and vis_channels.
    Example for MSG-4 SEVIRI: {"VIS06": 1, "VIS08": 2, "NIR16": 3, "WV62": 5, "WV73": 6, "IR108": 9}
    """

    def __post_init__(self):
        all_channels = set(self.ir_channels) | set(self.vis_channels)
        missing = all_channels - set(self.channel_ids.keys())
        if missing:
            raise ValueError(
                f"channel_ids is missing entries for: {missing}. "
                f"All channels in ir_channels and vis_channels must have an ID mapping."
            )

    @property
    def all_channel_names(self) -> list[str]:
        return self.ir_channels + self.vis_channels

    def ir_channel_id_list(self) -> tuple[int, ...]:
        return tuple(self.channel_ids[name] for name in self.ir_channels)

    def vis_channel_id_list(self) -> tuple[int, ...]:
        return tuple(self.channel_ids[name] for name in self.vis_channels)


@dataclass
class CoefficientPaths:
    """Paths to RTTOV coefficient and data files."""

    rttov_base: Path
    """Base directory of the RTTOV installation"""

    coef_file: str = "rtcoef_rttov14/rttov14pred54L/rtcoef_msg_4_seviri_o3co2.dat"
    """Coefficient file path relative to rttov_base"""

    hydrotable_file: str = (
        "rtcoef_rttov14/hydrotable_visir/rttov_hydrotable_msg_4_seviri.dat"
    )
    """Hydrotable file path relative to rttov_base (VIS/IR cloud scattering)"""

    aertable_file: str = (
        "rtcoef_rttov14/aertable_visir/rttov_aertable_msg_4_seviri_cams.dat"
    )
    """Which aerosol table to use relative to rttov_base (used when Aerosols are enabled)"""

    mfasis_nn_file: str = (
        "rtcoef_rttov14/mfasis_nn/rttov_mfasis_nn_hydro_msg_4_seviri_v140.dat"
    )
    """MFASIS-NN coefficient file path relative to rttov_base (used for VIS channels)"""

    emis_atlas_path: str = "emis_data"
    """IR emissivity atlas directory relative to rttov_base"""

    brdf_atlas_path: str = "brdf_data"
    """BRDF atlas directory relative to rttov_base (used for VIS channels)"""

    python_wrapper_path: str | None = None
    """Optional path to add to sys.path for pyrttov import (e.g. /path/to/rttov/wrapper)"""

    def resolve(self, relative: str) -> str:
        return str(self.rttov_base / relative)


@dataclass
class SatelliteGeometry:
    """Satellite viewing geometry (constant across the domain)."""

    satzen: float = 45.0
    """Satellite zenith angle [degrees]"""

    satazi: float = 180.0
    """Satellite azimuth angle [degrees]"""


@dataclass
class CloudConfig:
    """Cloud parameterization settings for RTTOV."""

    kappa: float = 0.1
    """Fraction of QSNOW to add to QICE for ice cloud input"""

    nhydro: int = 6
    """Number of hydrometeor types in the scattering coefficient file"""

    liq_hydro_idx: int = 2
    """0-based index into the Hydro array for liquid water clouds.
    Corresponds to Cumulus Continental Clean (type 3, 1-based) in the standard RTTOV cloud scheme."""

    ice_hydro_idx: int = 5
    """0-based index into the Hydro array for ice clouds.
    Corresponds to Cirrus (type 6, 1-based) in the standard RTTOV cloud scheme."""

    clwde: float = 20.0
    """Effective diameter for liquid water clouds [microns]"""

    icede: float = 60.0
    """Effective diameter for ice clouds [microns]"""


@dataclass
class AerosolConfig:
    """Aerosols configuration for RRTOV"""

    enabled: bool = False
    """Whether to enable Aerosols"""

    naer_total: int = 9
    """How many species the aerosol coefficient files contains (count)"""

    species_map: dict[int, dict[str, float]] = field(default_factory=dict)
    """
    The linear coefficients to use for mapping between the species RTTOV uses and WRF.
    The outer dictionary keys refer to `aertable` species indices, while the inner
    dictionary contains a mapping between WRF-variable and the coefficient.
    So for example, the dictionary {3: {'DUST_1': 0.5, 'DUST_2': 0.2 }} would set the
    `Aer3` aerosol species to 0.5 * DUST_1 + 0.2 * DUST_2.

    Any species not specified here gets a default value of 0.0.
    """


@dataclass
class SurfaceDefaults:
    """Default surface property values."""

    default_salinity: float = 35.0
    default_snow_fraction: float = 0.0
    default_foam_fraction: float = 0.0


@dataclass
class PerformanceConfig:
    """Threading and batching configuration for RTTOV."""

    nthreads: int = 48
    """Number of OpenMP threads for RTTOV"""

    nprofs_per_call: int = 1024
    """Number of profiles per RTTOV call (internal batching)"""


@dataclass
class RTTOVOperatorConfig(DataClassTOMLMixin):
    """Top-level configuration for the RTTOV operator."""

    channels: ChannelConfig = field(default_factory=ChannelConfig)
    coefficients: CoefficientPaths = field(
        default_factory=lambda: CoefficientPaths(rttov_base=Path("/opt/rttov"))
    )
    satellite: SatelliteGeometry = field(default_factory=SatelliteGeometry)
    clouds: CloudConfig = field(default_factory=CloudConfig)
    aerosols: AerosolConfig = field(default_factory=AerosolConfig)
    surface: SurfaceDefaults = field(default_factory=SurfaceDefaults)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    verbose: bool = False
