import numpy as np
from scipy.optimize import brentq
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from yancc.solve import solve_dke

def calculate_flux_difference(erho: float, field, pitchgrid, speedgrid, species) -> float:
    """
    Runs the DKE solver for a given Erho and returns the net particle flux.
    Net Flux = Gamma_i - Gamma_e (assuming species[0] is ions, species[1] is electrons).
    """
    _, _, fluxes, _ = solve_dke(
        field, pitchgrid, speedgrid, species, erho, print_every=0
    )

    particle_flux = fluxes['<particle_flux>']
    return particle_flux[0] - particle_flux[1]


def find_ambipolar_roots(
        field,
        pitchgrid,
        speedgrid,
        species,
        erho_min: float = -5000.0,
        erho_max: float = 5000.0,
        n_points: int = 30,
        num_processors: int = 5,
        ):
    """
    Scans a range of Erho values in parallel to bracket and find all roots
    where the net particle flux is zero.

    Returns:
        erho_grid (np.ndarray): The Erho values evaluated during the coarse scan.
        flux_diffs (np.ndarray): The corresponding net fluxes from the scan.
        roots (list): The refined Erho roots where net flux is zero.
    """
    # 1. Define the coarse scan grid
    erho_grid = np.linspace(erho_min, erho_max, n_points)

    # 2. Freeze the static arguments so we only map 'erho' across the workers
    worker_func = partial(
        calculate_flux_difference,
        field=field,
        pitchgrid=pitchgrid,
        speedgrid=speedgrid,
        species=species
    )

    # 3. Execute the coarse scan in parallel
    print(f"Running coarse scan with {n_points} points in parallel...")
    with ProcessPoolExecutor(num_processors) as executor:
        flux_diffs = list(executor.map(worker_func, erho_grid))

    flux_diffs = np.array(flux_diffs)

    # 4. Detect brackets (sign changes) and refine roots
    roots = []
    print("Scanning for roots...")
    for i in range(len(erho_grid) - 1):
        # A sign change means the curve crossed zero between index i and i+1
        if flux_diffs[i] * flux_diffs[i + 1] <= 0:
            try:
                # Refine the exact zero-crossing point
                root = brentq(worker_func, erho_grid[i], erho_grid[i + 1], xtol=1.0)
                roots.append(root)
            except ValueError:
                # Failsafe in case the function is discontinuous within the bracket
                pass

    print(f"Found {len(roots)} ambipolar root(s).")
    return erho_grid, flux_diffs, sorted(roots)
