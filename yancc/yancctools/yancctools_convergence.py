"""
This file includes AI generated code (Gemini/Claude)
"""

import logging
import math

import numpy as np
from scipy.constants import elementary_charge, proton_mass

from yancc.field import Field
from yancc.solve import solve_dke
from yancc.velocity_grids import MaxwellSpeedGrid, UniformPitchAngleGrid
from yancc.yancctools.yancctools import _merge_options, generate_runid

logger = logging.getLogger(__name__)


# Known species lookup by (mass_amu_rounded, charge_e_rounded).
_SPECIES_NAMES = {
    (0, -1): "Electron",
    (1, 1): "Hydrogen",
    (2, 1): "Deuterium",
    (3, 1): "Tritium",
    (4, 2): "Helium",
    (6, 3): "Lithium-6",
    (7, 3): "Lithium-7",
    (9, 4): "Beryllium",
    (10, 5): "Boron-10",
    (11, 5): "Boron-11",
    (14, 7): "Nitrogen",
    (16, 8): "Oxygen",
}


def identify_species_name(species) -> str:
    """
    Identify a species name from its mass and charge.

    This function was entirely generated through an AI tool (Claude).

    Parameters
    ----------
    species : Species or LocalMaxwellian or GlobalMaxwellian
        A species object with a `.species` attribute (Maxwellian) or
        direct `.mass` and `.charge` attributes (Species).

    Returns
    -------
    str
        Human-readable species name.
    """
    # Navigate to the underlying Species object
    sp = species
    if hasattr(sp, "species"):
        sp = sp.species

    mass_amu = round(float(sp.mass) / proton_mass)
    charge_e = round(float(sp.charge) / elementary_charge)
    key = (mass_amu, charge_e)

    return _SPECIES_NAMES.get(key, f"Z={charge_e},A={mass_amu}")


def run_convergence_scan(
    eq_type: str,
    eq_data,
    global_species: list,
    rho: float,
    erho: float,
    options: dict = None,
    **kwargs,
) -> dict:
    """
    Run a convergence scan over numerical resolution parameters.

    Parameters
    ----------
    eq_type : str
        Equilibrium type ("desc", "vmec", etc.)
    eq_data :
        Equilibrium data object or file path.
    global_species : list
        List of GlobalMaxwellian species.
    rho : float
        Radial surface label to scan at.
    erho : float
        Radial electric field to use for the scan.
    options : dict, optional
        Dictionary containing nominal configuration options.
    **kwargs :
        Additional options or overrides.

    Returns
    -------
    dict
        HDF5-compatible dictionary with scan results.
    """
    opts = _merge_options(options, **kwargs)

    runid = opts.get("runid")
    if runid is None:
        runid = generate_runid()
        opts["runid"] = runid

    logger.info(f"Starting convergence scan (runid: {runid}) at rho={rho:.4f}")

    # Extract nominal values
    nominal_nx = opts.get("nx", 5)
    nominal_na = opts.get("na", 65)
    nominal_nt = opts.get("nt", 17)
    nominal_nz = opts.get("nz", 33)

    nominal_values = {
        "nx": nominal_nx,
        "na": nominal_na,
        "nt": nominal_nt,
        "nz": nominal_nz,
    }

    scan_steps = opts.get("convergence_scan_steps", 3)

    # Identify species names for labeling
    species_names = [identify_species_name(s) for s in global_species]

    # Helper to generate odd integer scan ranges roughly -50% to +100%.
    # The nominal value is always included so that deviation plots have
    # an exact reference point.
    def generate_scan_range(nominal_val, steps):
        min_val = max(5, math.floor(nominal_val * 0.5))
        max_val = math.ceil(nominal_val * 2.0)

        # ensure min_val is odd
        if min_val % 2 == 0:
            min_val += 1

        # generate linear space then make them odd integers
        vals = np.linspace(min_val, max_val, steps)
        odd_vals = []
        for v in vals:
            int_v = int(round(v))
            if int_v % 2 == 0:
                int_v += 1
            if int_v < 5:
                int_v = 5
            if int_v not in odd_vals:
                odd_vals.append(int_v)

        # Always include the nominal value itself
        nominal_odd = nominal_val
        if nominal_odd % 2 == 0:
            nominal_odd += 1
        if nominal_odd not in odd_vals:
            odd_vals.append(nominal_odd)

        return sorted(list(set(odd_vals)))

    params = {
        "nx": generate_scan_range(nominal_nx, scan_steps),
        "na": generate_scan_range(nominal_na, scan_steps),
        "nt": generate_scan_range(nominal_nt, scan_steps),
        "nz": generate_scan_range(nominal_nz, scan_steps),
    }

    local_species = [s.localize(rho) for s in global_species]

    # Extract only the keys needed by solve_dke
    dke_keys = [
        "p1",
        "p2",
        "rtol",
        "atol",
        "m",
        "k",
        "maxiter",
        "print_every",
        "operator_weights",
        "nL",
        "quad",
        "skip_init_print",
        "potentials",
        "M",
        "B",
        "C",
        "U",
        "f1",
    ]
    dke_opts = {k: v for k, v in opts.items() if k in dke_keys}

    scan_results = {}

    for param_name, param_values in params.items():
        logger.info(f"Scanning parameter: {param_name} over {param_values}")
        results_for_param = []

        for val in param_values:
            # Set up the specific resolution for this run
            current_res = {
                "nx": nominal_nx,
                "na": nominal_na,
                "nt": nominal_nt,
                "nz": nominal_nz,
            }
            current_res[param_name] = val

            try:
                # 1. Reconstruct Field
                if eq_type == "desc":
                    field = Field.from_desc(
                        eq_data, rho, current_res["nt"], current_res["nz"]
                    )
                elif eq_type == "vmec":
                    field = Field.from_vmec(
                        eq_data, rho**2, current_res["nt"], current_res["nz"]
                    )
                elif eq_type == "booz_xform":
                    field = Field.from_booz_xform(
                        eq_data, rho**2, current_res["nt"], current_res["nz"]
                    )
                elif eq_type == "ipp_bc":
                    field = Field.from_ipp_bc(
                        eq_data, rho**2, current_res["nt"], current_res["nz"]
                    )
                else:
                    raise ValueError(f"Unknown equilibrium type: {eq_type}")

                # 2. Reconstruct Grids
                speedgrid = MaxwellSpeedGrid(current_res["nx"])
                pitchgrid = UniformPitchAngleGrid(current_res["na"])

                # 3. Solve DKE
                _, _, fluxes, _ = solve_dke(
                    field,
                    pitchgrid,
                    speedgrid,
                    local_species,
                    erho,
                    print_every=0,
                    **dke_opts,
                )

                particle_flux = [float(f) for f in fluxes["<particle_flux>"]]
                results_for_param.append(
                    {
                        "value": val,
                        "particle_flux": particle_flux,
                    }
                )

            except Exception as e:
                logger.error(
                    f"Failed convergence scan point {param_name}={val}: {e}"
                )

        scan_results[param_name] = results_for_param

    results = {
        "runid": runid,
        "rho": float(rho),
        "erho": float(erho),
        "species_names": species_names,
        "nominal_values": nominal_values,
        "options": opts,
        "scan_results": scan_results,
    }

    logger.info(f"Convergence scan complete (runid: {runid}).")
    return results
