import numpy as np
from scipy.optimize import brentq
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import plotly.graph_objects as go
import logging

from yancc.solve import solve_dke
from yancc.field import Field

logger = logging.getLogger(__name__)

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


def find_ambipolar_roots(
        field,
        pitchgrid,
        speedgrid,
        species,
        erho_min: float = -10000.0,
        erho_max: float = 10000.0,
        erho_num: int = 21,
        show_plot: bool = False,
        rho: float = None,
        **kwargs
        ):
    """
    Scans a range of Erho values (sequentially) to bracket and find all roots
    where the net charge flux is zero.
    
    Returns
    -------
    erho_grid : np.ndarray
        The Erho values evaluated during the coarse scan.
    flux_diffs : np.ndarray
        The corresponding net charge fluxes from the scan.
    roots_detailed : list of dict
        List containing {"Er": float, "fluxes": dict} for each root found.
    """
    logger.info(f"Finding ambipolar roots for rho={rho if rho is not None else 'unknown'}")
    erho_grid = np.linspace(erho_min, erho_max, erho_num)

    worker_func = partial(
        calculate_net_charge_flux,
        field=field,
        pitchgrid=pitchgrid,
        speedgrid=speedgrid,
        species=species,
        **kwargs
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
                    field, pitchgrid, speedgrid, species, root_er, print_every=0, **kwargs
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
    
    if show_plot:
        title = f"Ambipolar Search (rho={rho})" if rho is not None else "Ambipolar Search"
        plot_ambipolar_scan(erho_grid, flux_diffs, roots_detailed, title=title)

    return erho_grid, flux_diffs, roots_detailed


def _worker_rho_scan(rho, eq_type, eq_data, nt, nz, pitchgrid, speedgrid, global_species, erho_min, erho_max, erho_num, show_plots, **kwargs):
    """
    Worker function for radial scan. 
    Constructs the field and localizes species at a specific rho, then finds roots.
    """
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
    _, _, roots_detailed = find_ambipolar_roots(
        field, pitchgrid, speedgrid, local_species, 
        erho_min=erho_min, erho_max=erho_max, erho_num=erho_num,
        show_plot=show_plots, rho=rho, **kwargs
    )
    
    logger.info(f"Worker finished for rho={rho:.4f}. Found {len(roots_detailed)} roots.")
    return rho, roots_detailed


def scan_ambipolar_profile(
        rho_grid: np.ndarray,
        eq_type: str,
        eq_data,
        nt: int,
        nz: int,
        pitchgrid,
        speedgrid,
        global_species,
        erho_min: float = -10000.0,
        erho_max: float = 10000.0,
        erho_num: int = 21,
        num_processors: int = 4,
        show_plots: bool = True,
        **kwargs
        ):
    """
    Scans multiple radial surfaces in parallel to find ambipolar Erho profiles.
    
    Returns
    -------
    results : dict
        A nested dictionary: { "rho_str": [ {"Er": float, "fluxes": dict}, ... ] }
    """
    logger.info(f"Starting radial scan over {len(rho_grid)} surfaces with {num_processors} processors...")
    
    worker = partial(
        _worker_rho_scan,
        eq_type=eq_type,
        eq_data=eq_data,
        nt=nt,
        nz=nz,
        pitchgrid=pitchgrid,
        speedgrid=speedgrid,
        global_species=global_species,
        erho_min=erho_min,
        erho_max=erho_max,
        erho_num=erho_num,
        show_plots=show_plots,
        **kwargs
    )

    if num_processors > 1:
        with ProcessPoolExecutor(num_processors) as executor:
            scan_results = list(executor.map(worker, rho_grid))
    else:
        scan_results = [worker(rho) for rho in rho_grid]

    # Sort results by rho
    scan_results.sort(key=lambda x: x[0])
    
    results = {f"{rho:.2f}": roots for rho, roots in scan_results}
    logger.info("Radial scan complete.")
    return results


def plot_ambipolar_profile(results: dict):
    """
    Plots the radial profile of ambipolar electric fields in kV/m.
    """
    # Convert string keys back to floats for numerical plotting and sorting
    rhos_numeric = sorted([float(r) for r in results.keys()])
    
    root_sets = {0: [], 1: [], 2: []}
    rho_sets = {0: [], 1: [], 2: []}
    
    for r_num in rhos_numeric:
        r_str = f"{r_num:.2f}"
        # Extract Er values from the nested detailed roots
        detailed_roots = results[r_str]
        # Roots are already sorted by Er in find_ambipolar_roots
        er_values = [rd["Er"] for rd in detailed_roots]
        
        if len(er_values) == 1:
            rho_sets[0].append(r_num)
            root_sets[0].append(er_values[0] / 1e3)
        elif len(er_values) >= 3:
            for i in range(min(3, len(er_values))):
                rho_sets[i].append(r_num)
                root_sets[i].append(er_values[i] / 1e3)
        elif len(er_values) == 2:
            rho_sets[0].append(r_num)
            root_sets[0].append(er_values[0] / 1e3)
            rho_sets[2].append(r_num)
            root_sets[2].append(er_values[1] / 1e3)

    fig = go.Figure()

    colors = {0: "royalblue", 1: "black", 2: "crimson"}
    names = {0: "Ion Root", 1: "Unstable Root", 2: "Electron Root"}
    dashes = {0: "solid", 1: "dash", 2: "solid"}

    for i in range(3):
        if rho_sets[i]:
            fig.add_trace(go.Scatter(
                x=rho_sets[i],
                y=root_sets[i],
                mode='lines+markers',
                name=names[i],
                line=dict(color=colors[i], dash=dashes[i], width=2),
                marker=dict(size=6)
            ))

    fig.update_layout(
        title="Radial Profile of Ambipolar Electric Field",
        xaxis_title="Normalized Radius (rho)",
        yaxis_title="Ambipolar E_rho [kV/m]",
        template="plotly_white",
        hovermode="x unified"
    )

    fig.show()

def plot_ambipolar_scan(erho_grid: np.ndarray, flux_diffs: np.ndarray, roots: list, title: str = "Ambipolar Electric Field Search"):
    """
    Visualizes the coarse Erho scan and the refined ambipolar roots in kV/m.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=erho_grid / 1e3, 
        y=flux_diffs, 
        mode='lines+markers',
        name='Net Charge Flux',
        line=dict(shape='spline', smoothing=0.8, color='royalblue', width=2),
        marker=dict(size=6, color='lightblue')
    ))

    fig.add_hline(
        y=0, 
        line_width=2, 
        line_dash="dash", 
        line_color="black",
        annotation_text="Ambipolarity", 
        annotation_position="bottom right"
    )

    if roots:
        # Extract Er values from potential list of dicts (new format)
        er_values = []
        for r in roots:
            if isinstance(r, dict) and "Er" in r:
                er_values.append(r["Er"])
            else:
                er_values.append(r)
        
        root_y = [0.0] * len(er_values) 
        fig.add_trace(go.Scatter(
            x=np.array(er_values) / 1e3, 
            y=root_y, 
            mode='markers',
            name='Roots',
            marker=dict(size=14, color='crimson', symbol='x-thin', line=dict(width=3))
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Radial Electric Field E_rho [kV/m]",
        yaxis_title="Net Radial Charge Flux [A/m^2]",
        template="plotly_white",
        hovermode="x unified"
    )

    fig.show()
