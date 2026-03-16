"""
Some tools to help with running performance scans in parameters:
 - Ambipolar solover for Er
 - Profile scans over rho

Code generation heavily utilized AI coding agenents including:
  Gemini
Please be careful in using this code and YMMV.
"""


import numpy as np
import datetime
from pathlib import Path
from scipy.optimize import brentq
from functools import partial
import logging

from yancc.solve import solve_dke
from yancc.field import Field

logger = logging.getLogger(__name__)


def generate_runid() -> str:
    """
    Generates a unique 20-character run ID based on Japanese era (Reiwa), 
    Zodiac symbol, date, and time.
    
    Format: Era-Zodiac-YYMMDD-HHMMSS (20 characters)
    Example: R8-PIS-260311-153045
    """
    now = datetime.datetime.now()
    
    # Japanese Era (Reiwa) (2 chars)
    reiwa_year = now.year - 2018
    era_str = f"R{reiwa_year}"
    
    # Zodiac (3 chars)
    day = now.day
    month = now.month
    if month == 1: zodiac = "CAP" if day < 20 else "AQU"
    elif month == 2: zodiac = "AQU" if day < 19 else "PIS"
    elif month == 3: zodiac = "PIS" if day < 21 else "ARI"
    elif month == 4: zodiac = "ARI" if day < 20 else "TAU"
    elif month == 5: zodiac = "TAU" if day < 21 else "GEM"
    elif month == 6: zodiac = "GEM" if day < 21 else "CAN"
    elif month == 7: zodiac = "CAN" if day < 23 else "LEO"
    elif month == 8: zodiac = "LEO" if day < 23 else "VIR"
    elif month == 9: zodiac = "VIR" if day < 23 else "LIB"
    elif month == 10: zodiac = "LIB" if day < 23 else "SCO"
    elif month == 11: zodiac = "SCO" if day < 22 else "SAG"
    else: zodiac = "SAG" if day < 22 else "CAP"
    
    # Date/Time (6+1+6 = 13 chars)
    dt_str = now.strftime("%y%m%d-%H%M%S")
    
    # 2 (era) + 1 (dash) + 3 (zodiac) + 1 (dash) + 13 (dt) = 20 chars
    return f"{era_str}-{zodiac}-{dt_str}"


def calculate_net_charge_flux(erho: float, field, pitchgrid, speedgrid, species, **kwargs):
    """
    Runs the DKE solver for a given Erho and returns the net radial charge flux.
    
    Net Charge Flux = Σ q_a * Γ_a
    where q_a is the charge of species 'a' and Γ_a is the particle flux.
    """
    _, _, fluxes, _ = solve_dke(
        field, pitchgrid, speedgrid, species, erho, print_every=0, **kwargs
    )

    particle_flux = fluxes['<particle_flux>']
    # sum(charge * particle_flux)
    net_charge_flux = sum(s.species.charge * f for s, f in zip(species, particle_flux))
    return float(net_charge_flux)


def _merge_options(options: dict = None, **kwargs) -> dict:
    """Merge options dictionary with keyword arguments."""
    opts = options.copy() if options else {}
    opts.update(kwargs)
    return opts


def find_ambipolar_roots(
        field,
        pitchgrid,
        speedgrid,
        species,
        options: dict = None,
        rho: float = None,
        **kwargs
        ):
    """
    Scans a range of Erho values (sequentially) to bracket and find all roots
    where the net charge flux is zero.
    
    Parameters
    ----------
    field, pitchgrid, speedgrid, species : ...
    options : dict, optional
        Dictionary containing:
        erho_min : float, default -10000.0
        erho_max : float, default 10000.0
        erho_num : int, default 21
        show_plot : bool, default False
        ... other DKE options
    rho : float, optional
        Radial label for logging/plotting.
    **kwargs : 
        Additional options to override or supplement the options dict.
    
    Returns
    -------
    erho_grid : np.ndarray
        The Erho values evaluated during the coarse scan.
    flux_diffs : np.ndarray
        The corresponding net charge fluxes from the scan.
    roots_detailed : list of dict
        List containing {"Er": float, "fluxes": dict} for each root found.
    """
    opts = _merge_options(options, **kwargs)
    erho_min = opts.get("erho_min", -10000.0)
    erho_max = opts.get("erho_max", 10000.0)
    erho_num = opts.get("erho_num", 21)
    show_plot = opts.get("show_plot", False)

    logger.info(f"Finding ambipolar roots for rho={rho if rho is not None else 'unknown'}")
    erho_grid = np.linspace(erho_min, erho_max, erho_num)

    # Filter out yancctools specific options before passing to solve_dke
    yancctools_keys = ["erho_min", "erho_max", "erho_num", "show_plot", "nt", "nz", "num_processors", "runid", "output_path"]
    dke_opts = {k: v for k, v in opts.items() if k not in yancctools_keys}

    worker_func = partial(
        calculate_net_charge_flux,
        field=field,
        pitchgrid=pitchgrid,
        speedgrid=speedgrid,
        species=species,
        **dke_opts
    )

    # Coarse scan (serial)
    logger.debug(f"Starting coarse scan with {erho_num} points...")
    flux_diffs = []
    for i, er in enumerate(erho_grid):
        val = worker_func(er)
        flux_diffs.append(val)
        if (i + 1) % 5 == 0 or (i + 1) == erho_num:
             logger.debug(f"Progress: {i+1}/{erho_num} points evaluated.")
    
    flux_diffs = np.array(flux_diffs)

    roots_detailed = []
    logger.debug("Scanning brackets for sign changes...")
    for i in range(len(erho_grid) - 1):
        if flux_diffs[i] * flux_diffs[i + 1] <= 0:
            logger.info(f"Found bracket at Erho = [{erho_grid[i]:.2f}, {erho_grid[i+1]:.2f}] V/m. Refining...")
            try:
                root_er = brentq(worker_func, erho_grid[i], erho_grid[i + 1], xtol=1.0)
                logger.info(f"Refined root found at Erho = {root_er:.2f} V/m. Retrieving full fluxes...")
                
                # Retrieve full fluxes at the final root
                _, _, fluxes, _ = solve_dke(
                    field, pitchgrid, speedgrid, species, root_er, print_every=0, **dke_opts
                )
                
                roots_detailed.append({
                    "Er": float(root_er),
                    "fluxes": fluxes
                })
            except ValueError as e:
                logger.warning(f"Brentq refinement failed in bracket {i}: {e}")
                pass

    # Sort detailed roots by Er value
    roots_detailed.sort(key=lambda x: x["Er"])
    
    logger.info(f"Finished root finding. Found {len(roots_detailed)} root(s).")
    
    return erho_grid, flux_diffs, roots_detailed


def _worker_rho_scan(rho, eq_type, eq_data, pitchgrid, speedgrid, global_species, options: dict):
    """
    Worker function for radial scan. 
    Constructs the field and localizes species at a specific rho, then finds roots.
    """
    nt = options.get("nt", 32)
    nz = options.get("nz", 32)

    logger.info(f"Worker starting for rho={rho:.4f}")
    # 1. Reconstruct Field at this rho
    try:
        if eq_type == "desc":
            field = Field.from_desc(eq_data, rho, nt, nz)
        elif eq_type == "vmec":
            field = Field.from_vmec(eq_data, rho**2, nt, nz) 
        elif eq_type == "booz_xform":
            field = Field.from_booz_xform(eq_data, rho**2, nt, nz)
        elif eq_type == "ipp_bc":
            field = Field.from_ipp_bc(eq_data, rho**2, nt, nz)
        else:
            raise ValueError(f"Unknown equilibrium type: {eq_type}")
    except Exception as e:
        logger.error(f"Failed to construct field for rho={rho}: {e}")
        return rho, []

    # 2. Localize species
    local_species = [s.localize(rho) for s in global_species]

    # 3. Find roots
    erho_grid, flux_diffs, roots_detailed = find_ambipolar_roots(
        field, pitchgrid, speedgrid, local_species, 
        options=options, rho=rho
    )
    
    logger.info(f"Worker finished for rho={rho:.4f}. Found {len(roots_detailed)} roots.")
    return rho, roots_detailed, erho_grid, flux_diffs


def scan_ambipolar_profile(
        rho_grid: np.ndarray,
        eq_type: str,
        eq_data,
        pitchgrid,
        speedgrid,
        global_species,
        options: dict = None,
        runid: str = None,
        **kwargs
        ):
    """
    Scans multiple radial surfaces sequentially to find ambipolar Erho profiles.
    
    Parameters
    ----------
    rho_grid : np.ndarray
        Array of radial surfaces to scan.
    eq_type : str
        Equilibrium type ("desc", "vmec", etc.)
    eq_data : 
        Equilibrium data object or file path.
    pitchgrid, speedgrid, global_species : ...
    options : dict, optional
        Dictionary containing configuration options.
    runid : str, optional
        User-defined ID for this scan. If not given, a unique 20-char ID 
        is generated automatically.
    **kwargs :
        Additional options to override or supplement the options dict.
    
    Returns
    -------
    results : dict
        A nested dictionary: 
        { 
            "rho_str": {
                "roots": [ {"Er": float, "fluxes": dict}, ... ],
                "erho_grid": np.ndarray,
                "flux_diffs": np.ndarray
            }, 
            "runid": str 
        }
    """
    opts = _merge_options(options, **kwargs)
    
    if runid is None:
        runid = opts.get("runid", generate_runid())
    
    # Save back to options dict
    opts["runid"] = runid
    if options is not None:
        options["runid"] = runid

    logger.info(f"Starting radial scan (runid: {runid}) over {len(rho_grid)} surfaces (sequential execution)...")
    
    worker = partial(
        _worker_rho_scan,
        eq_type=eq_type,
        eq_data=eq_data,
        pitchgrid=pitchgrid,
        speedgrid=speedgrid,
        global_species=global_species,
        options=opts
    )

    # Run sequentially to avoid pickling issues with JAX/lambdas in notebooks
    scan_results = [worker(rho) for rho in rho_grid]

    # Sort results by rho
    scan_results.sort(key=lambda x: x[0])
    
    results = {
        f"{rho:.2f}": {
            "roots": roots,
            "erho_grid": er_grid,
            "flux_diffs": fl_diffs
        } for rho, roots, er_grid, fl_diffs in scan_results
    }
    results["runid"] = runid
    logger.info(f"Radial scan complete (runid: {runid}).")
    return results
