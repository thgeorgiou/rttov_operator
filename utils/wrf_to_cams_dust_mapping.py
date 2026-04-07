"""
Compute linear remapping coefficients between GOCART dust bins (used by WRF-CHEM)
and CAMS dust bins (used by RTTOV's CAMS aerosol optical property tables).

The mapping assumes mass is distributed uniformly in log(radius) within each
GOCART bin. This is consistent with the lognormal size distributions typically
observed for atmospheric dust and weights mass appropriately toward the larger
end of each bin.

For each GOCART bin, the fraction of its mass falling into a given CAMS bin is
computed as the log-radius overlap divided by the GOCART bin's log-radius width.
Mass falling outside the CAMS size range (below the smallest CAMS bin) is
assigned to the nearest CAMS bin to preserve total mass.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DustBin:
    """A dust size bin defined by its radius edges in micrometers."""

    name: str
    r_min: float  # micrometers
    r_max: float  # micrometers

    @property
    def log_width(self) -> float:
        return math.log(self.r_max) - math.log(self.r_min)


# GOCART 5-bin dust scheme as used in WRF-CHEM (radius in micrometers).
# These bin edges follow the standard GOCART implementation.
GOCART_BINS: list[DustBin] = [
    DustBin("DUST_1", 0.02, 2.0),
    DustBin("DUST_2", 2.0, 3.6),
    DustBin("DUST_3", 3.6, 6.0),
    DustBin("DUST_4", 6.0, 12.0),
    DustBin("DUST_5", 12.0, 20.0),
]

# CAMS 3-bin dust scheme as represented in RTTOV's CAMS aerosol tables.
# The CAMS data on ADS are a lso listed with these ranges (2026-04)
CAMS_BINS: list[DustBin] = [
    DustBin("DUS1", 0.03, 0.55),
    DustBin("DUS2", 0.55, 0.9),
    DustBin("DUS3", 0.9, 20.0),
]


def log_overlap_fraction(source: DustBin, target: DustBin) -> float:
    """Fraction of `source` bin's log-mass-weighted extent that overlaps `target`.

    Returns 0.0 if the bins do not overlap.
    """
    r_low = max(source.r_min, target.r_min)
    r_high = min(source.r_max, target.r_max)
    if r_high <= r_low:
        return 0.0
    return (math.log(r_high) - math.log(r_low)) / source.log_width


def compute_gocart_to_cams_mapping(
    gocart_bins: list[DustBin] = GOCART_BINS,
    cams_bins: list[DustBin] = CAMS_BINS,
) -> dict[str, dict[str, float]]:
    """Compute the GOCART-to-CAMS dust bin mapping coefficients.

    For each GOCART bin, distributes its mass across CAMS bins based on
    log-radius overlap. Mass falling below the smallest CAMS bin is assigned
    to that smallest bin (preserves total mass conservation). Mass falling
    above the largest CAMS bin is similarly assigned to the largest bin,
    though for typical dust schemes this case does not arise.

    Returns:
        Nested dict: {cams_bin_name: {gocart_bin_name: coefficient}}.
        Only non-zero coefficients are included.
    """

    # Initialize result with empty dicts for each CAMS bin.
    result: dict[str, dict[str, float]] = {cb.name: {} for cb in cams_bins}

    cams_min = min(cb.r_min for cb in cams_bins)
    cams_max = max(cb.r_max for cb in cams_bins)
    smallest_cams = min(cams_bins, key=lambda cb: cb.r_min)
    largest_cams = max(cams_bins, key=lambda cb: cb.r_max)

    for gb in gocart_bins:
        coefficients: dict[str, float] = {cb.name: 0.0 for cb in cams_bins}

        # Mass below CAMS range -> smallest CAMS bin.
        if gb.r_min < cams_min:
            r_low = gb.r_min
            r_high = min(gb.r_max, cams_min)
            below_fraction = (math.log(r_high) - math.log(r_low)) / gb.log_width
            coefficients[smallest_cams.name] += below_fraction

        # Mass above CAMS range -> largest CAMS bin.
        if gb.r_max > cams_max:
            r_low = max(gb.r_min, cams_max)
            r_high = gb.r_max
            above_fraction = (math.log(r_high) - math.log(r_low)) / gb.log_width
            coefficients[largest_cams.name] += above_fraction

        # Mass within CAMS range -> distributed by overlap.
        for cb in cams_bins:
            coefficients[cb.name] += log_overlap_fraction(gb, cb)

        # Sanity check: coefficients for this GOCART bin should sum to 1.
        total = sum(coefficients.values())
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError(
                f"Mapping coefficients for {gb.name} sum to {total}, expected 1.0. "
                f"Check bin definitions for gaps or overlaps."
            )

        # Populate result, skipping zero coefficients for clarity.
        for cb_name, coef in coefficients.items():
            if coef > 0.0:
                result[cb_name][gb.name] = coef

    return result


def to_species_map(
    mapping: dict[str, dict[str, float]],
    cams_aertable_indices: dict[str, int],
) -> dict[int, dict[str, float]]:
    """Convert a name-keyed mapping to the aertable-index-keyed format.

    Args:
        mapping: Output of `compute_gocart_to_cams_mapping`, keyed by CAMS bin name.
        cams_aertable_indices: Mapping from CAMS bin name to its 1-based index
            in the RTTOV aertable file. For the standard CAMS aerosol table
            this is typically {"DUS1": 2, "DUS2": 3, "DUS3": 4}.

    Returns:
        Dict in the format expected by AerosolConfig.species_map.
    """

    return {
        cams_aertable_indices[cams_name]: gocart_coeffs
        for cams_name, gocart_coeffs in mapping.items()
        if gocart_coeffs  # skip empty entries
    }


if __name__ == "__main__":
    # Print the mapping for documentation / verification purposes.
    mapping = compute_gocart_to_cams_mapping()
    print("GOCART -> CAMS dust bin mapping (log-radius overlap method)")
    print("=" * 60)
    for cams_name, coeffs in mapping.items():
        print(f"\n{cams_name}:")
        if not coeffs:
            print("  (no contributions)")
            continue
        for gocart_name, coef in coeffs.items():
            print(f"  {gocart_name}: {coef:.6f}")

    # Round coefficients
    for cams_name, coeffs in mapping.items():
        for gocart_name, coef in coeffs.items():
            mapping[cams_name][gocart_name] = round(coef, 4)

    # Print out coefficients
    cams_indices = {"DUS1": 2, "DUS2": 3, "DUS3": 4}
    species_map = to_species_map(mapping, cams_indices)
    print("\n\nspecies_map format:")
    print("=" * 60)
    for idx, coeffs in species_map.items():
        print(f"{idx}: {coeffs}")
