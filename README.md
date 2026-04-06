# rttov-operator

A [WRF-Ensembly](https://github.com/NOA-ReACT/wrf_ensembly) postprocessing plugin that runs [RTTOV v14](https://nwp-saf.eumetsat.int/site/software/rttov/) on post-processed WRF ensemble output to generate synthetic satellite imagery (brightness temperatures and reflectances).

Requires a working RTTOV v14 installation with the pyrttov Python wrapper compiled and accessible.

---

## Installation

```bash
uv pip install -e .
# or
pip install -e .
```

The package exposes a `rttov-operator` CLI entry point.

---

## Prerequisites

1. **RTTOV v14** compiled with netCDF support. See the [NWPSAF RTTOV page](https://nwp-saf.eumetsat.int/site/software/rttov/) for download and build instructions.

2. **pyrttov wrapper** compiled against RTTOV v14. The wrapper lives in `<rttov_base>/wrapper/`. Either add it to `PYTHONPATH` or set `python_wrapper_path` in the config.

3. **Coefficient files** for your instrument, downloaded from the NWPSAF coefficients page:
   - Gas absorption coefficient file (`.dat`)
   - Cloud/hydrometeor scattering coefficient file
   - MFASIS-NN file (`.nc`) if simulating VIS channels
   - IR emissivity atlas and/or BRDF atlas (optional but recommended)

   > **Note:** RTTOV v13 coefficient files are incompatible with v14. HDF5 files (`.H5`) must be replaced with the v14 netCDF equivalents.

4. **Input data** must be post-processed WRF output (e.g. from the `XWRFPostProcessor` in the WRF-Ensembly pipeline) containing at minimum:

   | Variable | Description |
   |---|---|
   | `air_pressure` | Full-level pressure [Pa], dims `(t, z, y, x)` |
   | `air_potential_temperature` | Potential temperature [K] |
   | `QVAPOR` | Water vapor mixing ratio [kg/kg] |
   | `QCLOUD` | Cloud liquid water mixing ratio [kg/kg] |
   | `QICE` | Cloud ice mixing ratio [kg/kg] |
   | `QSNOW` | Snow mixing ratio [kg/kg] |
   | `CLDFRA` | Cloud fraction [0–1] |
   | `TSK` | Skin temperature [K] |
   | `PSFC` | Surface pressure [Pa] |
   | `T2` | 2 m temperature [K] |
   | `Q2` | 2 m water vapor mixing ratio [kg/kg] |
   | `U10`, `V10` | 10 m wind components [m/s] |
   | `geopotential_height` | Geopotential height [m] |
   | `latitude`, `longitude` | Grid coordinates [degrees] |
   | `XLAND` | Land mask (1=land, 2=water) |

---

## Configuration

Copy `config.example.toml` and edit it for your setup:

```toml
[coefficients]
rttov_base = "/opt/rttov14"
coef_file = "rtcoef_rttov14/rttov14pred54L/rtcoef_msg_4_seviri_o3co2.dat"
sccld_file = "rtcoef_rttov14/cldaer_visir/sccldcoef_msg_4_seviri.dat"
mfasis_cld_file = "rtcoef_rttov14/mfasis_nn/rttov_mfasis_nn_cld_msg_4_seviri.nc"
python_wrapper_path = "/opt/rttov14/wrapper"  # omit if already on PYTHONPATH
```

All coefficient file paths are relative to `rttov_base`.

### Key options

**`[channels]`** — which channels to simulate and their RTTOV coefficient indices:
```toml
ir_channels = ["IR108", "WV62", "WV73"]
vis_channels = ["VIS06"]

[channels.channel_ids]
VIS06 = 1
WV62 = 5
WV73 = 6
IR108 = 9
```
IR channels produce brightness temperatures [K]; VIS channels produce reflectances [dimensionless].

**`[clouds]`** — hydrometeor type mapping:
```toml
nhydro = 6           # number of hydrometeor types in the scattering coefficient file
liq_hydro_idx = 2    # 0-based index for liquid water clouds (Cumulus Continental Clean)
ice_hydro_idx = 5    # 0-based index for ice clouds (Cirrus)
clwde = 20.0         # effective diameter for liquid clouds [µm]
icede = 60.0         # effective diameter for ice clouds [µm]
```
The hydrometeor indices must match the ordering in `sccld_file`. For the standard 6-type RTTOV scheme: 0=Stco, 1=Stma, 2=Cucc, 3=Cucp, 4=Cuma, 5=Cirr.

**`[satellite]`** — constant viewing geometry across the domain:
```toml
satzen = 45.0   # satellite zenith angle [degrees]
satazi = 180.0  # satellite azimuth angle [degrees]
```
Solar angles are computed per grid point and timestep using pvlib.

**`[performance]`**:
```toml
nthreads = 48        # OpenMP threads inside RTTOV
nprofs_per_call = 1024  # profiles per internal RTTOV batch
```

---

## Usage

### As a WRF-Ensembly pipeline processor

Register `RTTOVProcessor` in your WRF-Ensembly postprocessing pipeline configuration:

```toml
[[postprocess.processors]]
processor = "rttov_operator.processor.RTTOVProcessor"
params = { config_file = "/path/to/rttov_config.toml" }
```

The processor adds one new variable per configured channel to the dataset. RTTOV instances are loaded once at pipeline startup and reused across all members and cycles.

### Standalone CLI

```bash
rttov-operator INPUT_PATH OUTPUT_PATH --config rttov_config.toml
```

Reads a post-processed WRF NetCDF file, runs RTTOV for all timesteps, and writes the input dataset with the new channel variables appended to `OUTPUT_PATH`.

```
Options:
  --config PATH   Path to TOML configuration file.  [required]
  -v, --verbose   Enable verbose logging.
```

Example:
```bash
rttov-operator forecast_mean_cycle_023.nc forecast_mean_cycle_023_rttov.nc \
    --config config.example.toml --verbose
```

---

## Output variables

For each configured channel, a new variable is added with dimensions `(t, y, x)`:

| Channel type | Variable name | Units | Long name |
|---|---|---|---|
| IR (`ir_channels`) | e.g. `IR108` | `K` | `IR108 brightness temperature` |
| VIS (`vis_channels`) | e.g. `VIS06` | `1` | `VIS06 reflectance` |

---

## Notes on the RTTOV v14 profile structure

RTTOV v14 changed how vertical profiles are specified. The key difference from v13:

- **Half-level pressures required**: The bottom half-level is the surface pressure. There is no longer a separate surface pressure field. The upstream pipeline must provide `air_pressure_hf` on the staggered (z+1) grid.
- **Unified hydrometeor interface**: Named cloud types (Cucc, Cirr, etc.) are replaced by a single `Hydro` array indexed by hydrometeor type. The `liq_hydro_idx` / `ice_hydro_idx` config options control the mapping.
- **MFASIS-NN**: The MFASIS lookup-table solver from v13 is replaced by MFASIS neural network files (`.nc`). Old `.H5` files are incompatible.
