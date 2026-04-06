"""
Command-line interface for running the RTTOV operator standalone.
"""

import logging
from pathlib import Path

import click
import numpy as np
import xarray as xr

from .config import RTTOVOperatorConfig
from .rttov_wrapper import create_rttov_instance, run_rttov


@click.command()
@click.argument(
    "input_path",
    type=click.Path(exists=True, readable=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "output_path",
    type=click.Path(writable=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--config",
    "config_file",
    required=True,
    type=click.Path(exists=True, readable=True, dir_okay=False, path_type=Path),
    help="Path to TOML configuration file.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose logging.",
)
def main(input_path: Path, output_path: Path, config_file: Path, verbose: bool):
    """Run RTTOV on a post-processed WRF output file.

    INPUT_PATH is the path to the post-processed WRF NetCDF file.
    OUTPUT_PATH is the path where the output file will be written.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    if output_path.exists():
        raise click.ClickException(f"Output file already exists: {output_path}")

    # Load config
    config = RTTOVOperatorConfig.from_toml(config_file.read_text())
    if verbose:
        config.verbose = True

    logger.info("Loading RTTOV instances...")
    ir_rttov = None
    vis_rttov = None
    if config.channels.ir_channels:
        ir_rttov = create_rttov_instance(config, mode="ir")
    if config.channels.vis_channels:
        vis_rttov = create_rttov_instance(config, mode="vis")

    logger.info("Opening input: %s", input_path)
    ds = xr.open_dataset(input_path)
    n_times = ds.sizes["t"]

    for t_idx in range(n_times):
        ds_t = ds.isel(t=t_idx)
        results = run_rttov(ds_t, config, ir_rttov, vis_rttov)

        for channel_name, data_2d in results.items():
            if channel_name not in ds:
                ds[channel_name] = (
                    ("t", "y", "x"),
                    np.full(
                        (n_times, ds.sizes["y"], ds.sizes["x"]),
                        np.nan,
                        dtype=np.float32,
                    ),
                )
                if channel_name in config.channels.vis_channels:
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

    logger.info("Writing output: %s", output_path)
    ds.to_netcdf(output_path, unlimited_dims=["t"])
    logger.info("Done.")


if __name__ == "__main__":
    main()
