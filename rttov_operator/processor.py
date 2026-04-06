"""
DataProcessor subclass for RTTOV forward model integration with WRF-Ensembly.
"""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from .config import RTTOVOperatorConfig
from .rttov_wrapper import create_rttov_instance, run_rttov

logger = logging.getLogger(__name__)


class RTTOVProcessor:
    """DataProcessor that runs RTTOV on post-processed WRF output.

    Generates synthetic satellite brightness temperatures (IR) and reflectances (VIS)
    and adds them as new variables to the dataset.

    RTTOV instances are loaded once at initialization and reused across all timesteps
    and process() calls for performance.
    """

    def __init__(self, config_file: str | Path, **kwargs):
        super().__init__(**kwargs)
        config_path = Path(config_file)
        self.config = RTTOVOperatorConfig.from_toml(config_path.read_text())

        # Pre-load RTTOV instances (coefficient loading is expensive)
        self.ir_rttov = None
        self.vis_rttov = None
        if self.config.channels.ir_channels:
            logger.info("Loading RTTOV IR instance...")
            self.ir_rttov = create_rttov_instance(self.config, mode="ir")
        if self.config.channels.vis_channels:
            logger.info("Loading RTTOV VIS instance...")
            self.vis_rttov = create_rttov_instance(self.config, mode="vis")

    def process(self, ds: xr.Dataset, context) -> xr.Dataset:
        """Run RTTOV for each timestep and add BT/reflectance fields to the dataset.

        Args:
            ds: Input xarray Dataset with dimensions (t, z, y, x).
            context: ProcessingContext from WRF-Ensembly pipeline.

        Returns:
            Dataset with added channel variables.
        """
        n_times = ds.sizes["t"]
        logger.info("Running RTTOV for %d timestep(s)", n_times)

        for t_idx in range(n_times):
            ds_t = ds.isel(t=t_idx)
            results = run_rttov(ds_t, self.config, self.ir_rttov, self.vis_rttov)

            for channel_name, data_2d in results.items():
                if channel_name not in ds:
                    # Initialize variable on first timestep
                    ds[channel_name] = (
                        ("t", "y", "x"),
                        np.full(
                            (n_times, ds.sizes["y"], ds.sizes["x"]),
                            np.nan,
                            dtype=np.float32,
                        ),
                    )
                    # Set attributes based on channel type
                    if channel_name in self.config.channels.vis_channels:
                        ds[channel_name].attrs = {
                            "units": "1",
                            "long_name": f"{channel_name} reflectance",
                        }
                    else:
                        ds[channel_name].attrs = {
                            "units": "K",
                            "long_name": f"{channel_name} brightness temperature",
                        }

                ds[channel_name].values[t_idx] = data_2d

        return ds

    @property
    def name(self) -> str:
        return "RTTOV Operator"
