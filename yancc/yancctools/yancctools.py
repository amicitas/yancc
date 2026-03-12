import numpy as np
from scipy.optimize import brentq
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import plotly.graph_objects as go

from yancc.solve import solve_dke

def calculate_net_charge_flux(erho: float, field, pitchgrid, speedgrid, species, **kwargs):
    """
    Runs the DKE solver for a given Erho and returns the net radial charge flux.
    
    Net Charge Flux = Σ q_a * Γ_a
    where q_a is the charge of species 'a' and Γ_a is the particle flux.
    
    Parameters
    ----------
    erho : float
        Radial electric field E_rho = -∂Φ /∂ρ [V/m].
    field : Field
        Magnetic field information.
    pitchgrid : UniformPitchAngleGrid
        Pitch angle grid data.
    speedgrid : MaxwellSpeedGrid
        Speed grid data.
    species : list of LocalMaxwellian
        Species information.
    **kwargs : dict
        Additional arguments passed to solve_dke (e.g., rtol, m, k, maxiter).
        
    Returns
    -------
    net_charge_flux : float
        Net radial charge flux in Amps/m^2.
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
        num_processors: int = 4,
        **kwargs
        ):
    """
    Scans a range of Erho values in parallel to bracket and find all roots
    where the net charge flux is zero (ambipolarity condition).

    Parameters
    ----------
    field : Field
        Magnetic field information.
    pitchgrid : UniformPitchAngleGrid
        Pitch angle grid data.
    speedgrid : MaxwellSpeedGrid
        Speed grid data.
    species : list of LocalMaxwellian
        Species information.
    erho_min : float, optional
        Minimum E_rho for the scan [V/m]. Defaults to -10000.
    erho_max : float, optional
        Maximum E_rho for the scan [V/m]. Defaults to 10000.
    n_points : int, optional
        Number of points in the coarse scan. Defaults to 21.
    num_processors : int, optional
        Number of parallel processes to use. Defaults to 4.
    **kwargs : dict
        Additional arguments passed to solve_dke.

    Returns
    -------
    erho_grid : np.ndarray
        The Erho values evaluated during the coarse scan.
    flux_diffs : np.ndarray
        The corresponding net charge fluxes from the scan.
    roots : list
        The refined Erho roots where net charge flux is zero.
    """
    # 1. Define the coarse scan grid
    erho_grid = np.linspace(erho_min, erho_max, n_points)

    # 2. Freeze the static arguments so we only map 'erho' across the workers
    worker_func = partial(
        calculate_net_charge_flux,
        field=field,
        pitchgrid=pitchgrid,
        speedgrid=speedgrid,
        species=species,
        **kwargs
    )

    # 3. Execute the coarse scan in parallel
    print(f"Running coarse scan with {n_points} points in parallel ({num_processors} processors)...")
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
                # We use the same worker_func which might be slow if solve_dke is slow,
                # but brentq is generally efficient.
                root = brentq(worker_func, erho_grid[i], erho_grid[i + 1], xtol=1.0)
                roots.append(root)
            except ValueError:
                # Failsafe in case the function is discontinuous or brentq fails
                pass

    print(f"Found {len(roots)} ambipolar root(s).")
    return erho_grid, flux_diffs, sorted(roots)


def plot_ambipolar_scan(erho_grid: np.ndarray, flux_diffs: np.ndarray, roots: list):
    """
    Visualizes the coarse Erho scan and the refined ambipolar roots using Plotly.
    
    Parameters
    ----------
    erho_grid : np.ndarray
        Erho values from the coarse scan.
    flux_diffs : np.ndarray
        Net charge fluxes from the coarse scan.
    roots : list
        Refined roots where net charge flux is zero.
    """
    fig = go.Figure()

    # Plot the coarse scan data
    fig.add_trace(go.Scatter(
        x=erho_grid, 
        y=flux_diffs, 
        mode='lines+markers',
        name='Net Charge Flux (Coarse Scan)',
        line=dict(shape='spline', smoothing=0.8, color='royalblue', width=2),
        marker=dict(size=6, color='lightblue')
    ))

    # Add a horizontal line at y=0 to indicate ambipolarity
    fig.add_hline(
        y=0, 
        line_width=2, 
        line_dash="dash", 
        line_color="black",
        annotation_text="Ambipolarity (Σ q_a Γ_a = 0)", 
        annotation_position="bottom right"
    )

    # Plot the highly refined roots
    if roots:
        # The y-value for roots is mathematically 0
        root_y = [0.0] * len(roots) 
        fig.add_trace(go.Scatter(
            x=roots, 
            y=root_y, 
            mode='markers',
            name='Ambipolar Roots',
            marker=dict(size=14, color='crimson', symbol='x-thin', line=dict(width=3))
        ))

    # Format the layout for scientific readability
    fig.update_layout(
        title="Ambipolar Electric Field Search",
        xaxis_title="Radial Electric Field E_rho [V/m]",
        yaxis_title="Net Radial Charge Flux [A/m^2]",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    fig.show()
