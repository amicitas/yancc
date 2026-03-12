import numpy as np
from scipy.optimize import brentq
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import plotly.graph_objects as go

from yancc.solve import solve_dke
from yancc.field import Field

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
        n_points: int = 21,
        **kwargs
        ):
    """
    Scans a range of Erho values (sequentially) to bracket and find all roots
    where the net charge flux is zero.
    """
    erho_grid = np.linspace(erho_min, erho_max, n_points)

    worker_func = partial(
        calculate_net_charge_flux,
        field=field,
        pitchgrid=pitchgrid,
        speedgrid=speedgrid,
        species=species,
        **kwargs
    )

    # Coarse scan (serial)
    flux_diffs = [worker_func(er) for er in erho_grid]
    flux_diffs = np.array(flux_diffs)

    roots = []
    for i in range(len(erho_grid) - 1):
        if flux_diffs[i] * flux_diffs[i + 1] <= 0:
            try:
                root = brentq(worker_func, erho_grid[i], erho_grid[i + 1], xtol=1.0)
                roots.append(root)
            except ValueError:
                pass

    return erho_grid, flux_diffs, sorted(roots)


def _worker_rho_scan(rho, eq_type, eq_data, nt, nz, pitchgrid, speedgrid, global_species, erho_min, erho_max, n_points, **kwargs):
    """
    Worker function for radial scan. 
    Constructs the field and localizes species at a specific rho, then finds roots.
    """
    # 1. Reconstruct Field at this rho
    if eq_type == "desc":
        field = Field.from_desc(eq_data, rho, nt, nz)
    elif eq_type == "vmec":
        field = Field.from_vmec(eq_data, rho**2, nt, nz) # VMEC from_vmec takes s=rho^2
    elif eq_type == "booz_xform":
        field = Field.from_booz_xform(eq_data, rho**2, nt, nz)
    elif eq_type == "ipp_bc":
        field = Field.from_ipp_bc(eq_data, rho**2, nt, nz)
    else:
        raise ValueError(f"Unknown equilibrium type: {eq_type}")

    # 2. Localize species
    local_species = [s.localize(rho) for s in global_species]

    # 3. Find roots
    _, _, roots = find_ambipolar_roots(
        field, pitchgrid, speedgrid, local_species, 
        erho_min=erho_min, erho_max=erho_max, n_points=n_points, **kwargs
    )
    
    return rho, roots


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
        n_points: int = 21,
        num_processors: int = 4,
        **kwargs
        ):
    """
    Scans multiple radial surfaces in parallel to find ambipolar Erho profiles.

    Parameters
    ----------
    rho_grid : np.ndarray
        Array of rho (sqrt normalized toroidal flux) values to scan.
    eq_type : str
        One of "desc", "vmec", "booz_xform", "ipp_bc".
    eq_data : 
        The equilibrium object (for desc) or path to file (for others).
    nt, nz : int
        Field resolution.
    pitchgrid, speedgrid : 
        Velocity grid objects.
    global_species : list of GlobalMaxwellian
        Species definitions across radius.
    erho_min, erho_max : float
        Range for Erho search at each radius.
    n_points : int
        Number of points in the Erho scan at each radius.
    num_processors : int
        Number of parallel processes for the radial scan.
    **kwargs :
        Additional arguments for solve_dke.

    Returns
    -------
    results : dict
        A dictionary mapping rho to a list of found roots.
    """
    print(f"Starting radial scan over {len(rho_grid)} surfaces with {num_processors} processors...")
    
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
        n_points=n_points,
        **kwargs
    )

    with ProcessPoolExecutor(num_processors) as executor:
        scan_results = list(executor.map(worker, rho_grid))

    # Sort results by rho just in case
    scan_results.sort(key=lambda x: x[0])
    
    return {rho: roots for rho, roots in scan_results}


def plot_ambipolar_profile(results: dict):
    """
    Plots the radial profile of ambipolar electric fields.
    Handles multiple roots by categorizing them (Ion, Electron, Unstable).
    
    Parameters
    ----------
    results : dict
        Dictionary mapping rho to list of roots.
    """
    rhos = sorted(results.keys())
    
    # We'll organize roots by index.
    # Typically, if there are 3 roots, they are sorted:
    # 0: Ion root (usually most negative)
    # 1: Unstable root
    # 2: Electron root (usually most positive)
    
    root_sets = {0: [], 1: [], 2: []}
    rho_sets = {0: [], 1: [], 2: []}
    
    for r in rhos:
        roots = sorted(results[r])
        if len(roots) == 1:
            # If only one root, we need to decide if it's "ion-like" or "electron-like"
            # For simplicity, we'll just put it in the first set if it's alone,
            # or try to match continuity if we were being fancy.
            rho_sets[0].append(r)
            root_sets[0].append(roots[0])
        elif len(roots) >= 3:
            for i in range(min(3, len(roots))):
                rho_sets[i].append(r)
                root_sets[i].append(roots[i])
        elif len(roots) == 2:
            # Rare case, likely at a bifurcation point
            rho_sets[0].append(r)
            root_sets[0].append(roots[0])
            rho_sets[2].append(r)
            root_sets[2].append(roots[1])

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
                marker=dict(size=4)
            ))

    fig.update_layout(
        title="Radial Profile of Ambipolar Electric Field",
        xaxis_title="Normalized Radius (rho)",
        yaxis_title="Ambipolar E_rho [V/m]",
        template="plotly_white",
        hovermode="x unified"
    )

    fig.show()

def plot_ambipolar_scan(erho_grid: np.ndarray, flux_diffs: np.ndarray, roots: list):
    """
    Visualizes the coarse Erho scan and the refined ambipolar roots using Plotly.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=erho_grid, 
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
        root_y = [0.0] * len(roots) 
        fig.add_trace(go.Scatter(
            x=roots, 
            y=root_y, 
            mode='markers',
            name='Roots',
            marker=dict(size=14, color='crimson', symbol='x-thin', line=dict(width=3))
        ))

    fig.update_layout(
        title="Ambipolar Electric Field Search",
        xaxis_title="Radial Electric Field E_rho [V/m]",
        yaxis_title="Net Radial Charge Flux [A/m^2]",
        template="plotly_white",
        hovermode="x unified"
    )

    fig.show()
