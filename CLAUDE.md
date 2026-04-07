# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the standalone CLI (primary way to test)
uv run rttov-operator --config ./config.test.toml INPUT.nc OUTPUT.nc

# Install in editable mode
uv pip install -e .

# Rebuild after edits (uv run does this automatically)
uv run rttov-operator ...
```

There are no automated tests. Manual testing uses `config.test.toml` with a local WRF output file.

## Architecture

The tool is a postprocessing plugin for [WRF-Ensembly](https://github.com/NOA-ReACT/wrf_ensembly) that wraps RTTOV v14 to produce synthetic satellite imagery from WRF model output.

**Entry points:**
- `cli.py` — standalone `rttov-operator` command; opens a NetCDF file, iterates timesteps, writes results back
- `processor.py` — `RTTOVProcessor` class for use inside the WRF-Ensembly pipeline; RTTOV instances are loaded once at init and reused across all `process()` calls

**Data flow:**
1. `config.py` loads TOML config via `mashumaro` into typed dataclasses (`RTTOVOperatorConfig`)
2. `rttov_wrapper.create_rttov_instance()` loads RTTOV coefficient files and sets options; two separate instances are created for IR and VIS channels because they require different coefficient files and solver settings
3. For each timestep, `convert.extract_rttov_profiles()` reshapes WRF variables from `(z, y, x)` → `(nprofiles, nlevels)` in TOA-first order
4. `rttov_wrapper.build_profiles()` populates a `pyrttov.Profiles` object; `rttov_wrapper.run_rttov()` runs the forward model and reshapes output back to `(ny, nx)`

**pyrttov API (v14 wrapper conventions):**
- `pyrttov.Profiles(nprofiles, nlevels, nsurfaces)` where `nlevels` = number of **half** levels = nlayers+1
- `PHalf` shape: `(nprofiles, nlevels)` — required; `P` (full-level pressure) is optional
- `SurfEmisRefl` shape: `(5, nprofiles, nsurfaces, nchannels)` — set to -1 to use RTTOV's built-in emissivity models
- `Skin`, `NearSurface`, `SurfType` include the `nsurfaces` dimension: `(nprofiles, nsurfaces, leadingdim)`
- Hydrometeors are set per-type via `setHydroN/setHydroFracN/setHydroDeffN` with **1-based** indices
- Aerosols are set per-type via `setAerN` with **1-based** indices; enable with `Options.Aerosols = True` and `MmrAer = True`
- File attributes: `FileHydrotable` (not `FileSccld`), `FileAertable`, `FileMfasisNN` (not `FileMfasisCld`)
- `ThermalSolver = 3` = delta-Eddington; `SolarSolver = 2` = MFASIS-NN

**pyrttov import:**
`pyrttov` is not a standard pip package. Its path is set via `config.coefficients.python_wrapper_path` which is added to `sys.path` before import. Always call `_ensure_pyrttov(config)` before importing pyrttov in any function.

**Config field names** (easy to confuse with older docs):
- `hydrotable_file` — cloud/hydrometeor scattering coefficients
- `aertable_file` — aerosol scattering coefficients (CAMS scheme)
- `mfasis_nn_file` — MFASIS neural network file for VIS channels
- `liq_hydro_idx` / `ice_hydro_idx` — **0-based** in config; converted to **1-based** before calling `setHydroN`
- `aerosols.species_map` — outer keys are **1-based** aertable species indices; values are `{WRF_var: weight}` dicts; `convert.py` computes `AerN = Σ weight × WRF_var` and stores the result in `RTTOVProfileData.aerosol_data`
- `aerosols.naer_total` — total species count in the aertable file; species not in `species_map` are zeroed out
