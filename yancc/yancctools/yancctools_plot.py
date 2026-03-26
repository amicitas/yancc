# This file includes AI generated code (Gemini/Claude)
"""
Plotting routines for yancctools.

All plot functions return a ``plotly.graph_objects.Figure`` and accept a
``show`` parameter (default ``True``).  When ``show=False`` the figure is
returned without calling ``fig.show()``, which is useful for embedding in
Dash applications.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_ambipolar_profile(results: dict, show: bool = True):
    """
    Plot the radial profile of ambipolar electric fields in kV/m.

    This method was entirely generated through an AI tool (Gemini/Claude).

    Parameters
    ----------
    results : dict
        Output dictionary from ``scan_ambipolar_profile``.
    show : bool, optional
        If True (default), call ``fig.show()``.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure object.
    """
    # Convert string keys back to floats for numerical plotting and sorting
    # Skip metadata keys like 'runid'
    scan_data = results.get("scan_results", [])
    # Sort by rho just in case
    scan_data.sort(key=lambda x: x["rho"])
    
    root_sets = {0: [], 1: [], 2: []}
    rho_sets = {0: [], 1: [], 2: []}
    
    for data in scan_data:
        r_num = data["rho"]
        # Extract Er values from the nested detailed roots (new structure)
        detailed_roots = data.get("roots", [])
        
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

    if show:
        fig.show()

    return fig


def plot_ambipolar_summary(
    results: dict, global_species: list = None, show: bool = True
):
    """
    Display a comprehensive summary page including input profiles, flux
    scans, and the radial Er profile.

    Profile data can come from either live ``global_species`` objects or
    from the ``results["profiles"]`` dictionary saved by
    ``scan_ambipolar_profile``.  If both are provided, ``global_species``
    takes precedence.

    This method was entirely generated through an AI tool (Gemini/Claude).

    Parameters
    ----------
    results : dict
        Output dictionary from ``scan_ambipolar_profile``.
    global_species : list, optional
        List of GlobalMaxwellian species objects.  When *None*, the
        function falls back to ``results["profiles"]``.
    show : bool, optional
        If True (default), call ``fig.show()``.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure object.
    """
    runid = results.get("runid", "unknown")
    scan_data = results.get("scan_results", [])
    scan_data.sort(key=lambda x: x["rho"])
    
    # Create subplots: 2 rows, 2 columns
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Temperature Profiles [keV]", 
            "Density Profiles [10^20 m^-3]",
            "Ambipolar Search (Flux vs Er)",
            "Radial Ambipolar Er Profile [kV/m]"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    # 1. Plot Input Profiles
    # Prefer live species objects; fall back to saved profile data.
    if global_species:
        rho_dense = np.linspace(0, 1, 100)
        for s in global_species:
            name = s.species.__class__.__name__ if hasattr(s.species, "__class__") else str(s.species)
            if "Electron" in str(s.species):
                name = "Electrons"
            elif "Hydrogen" in str(s.species):
                name = "Hydrogen"

            T_vals = np.array([float(s.temperature(r)) for r in rho_dense]) / 1e3
            fig.add_trace(go.Scatter(
                x=rho_dense, y=T_vals, name=f"T_{name}", legendgroup="profiles"
            ), row=1, col=1)

            n_vals = np.array([float(s.density(r)) for r in rho_dense]) / 1e20
            fig.add_trace(go.Scatter(
                x=rho_dense, y=n_vals, name=f"n_{name}", legendgroup="profiles"
            ), row=1, col=2)
    elif "profiles" in results:
        profiles = results["profiles"]
        rho_arr = np.array(profiles["rho"])
        for sp in profiles["species"]:
            name = sp["name"]
            T_vals = np.array(sp["temperature"]) / 1e3  # eV -> keV
            fig.add_trace(go.Scatter(
                x=rho_arr, y=T_vals, name=f"T_{name}", legendgroup="profiles"
            ), row=1, col=1)

            n_vals = np.array(sp["density"]) / 1e20  # m^-3 -> 10^20 m^-3
            fig.add_trace(go.Scatter(
                x=rho_arr, y=n_vals, name=f"n_{name}", legendgroup="profiles"
            ), row=1, col=2)

    # 2. Plot all flux scans
    # Use a colormap for different rhos
    for i, data in enumerate(scan_data):
        r_num = data["rho"]
        r_str = f"{r_num:.2f}"
        er_grid = data["erho_grid"]
        fl_diffs = data["flux_diffs"]
        roots = data["roots"]
        
        # Flux scan line
        fig.add_trace(go.Scatter(
            x=er_grid / 1e3, y=fl_diffs, 
            name=f"rho={r_str}",
            mode='lines',
            line=dict(width=1),
            opacity=0.6,
            legendgroup="scans",
            showlegend=False
        ), row=2, col=1)
        
        # Root markers
        er_roots = [rd["Er"] for rd in roots]
        fig.add_trace(go.Scatter(
            x=np.array(er_roots) / 1e3,
            y=[0.0] * len(er_roots),
            mode='markers',
            marker=dict(symbol='x', size=8),
            showlegend=False
        ), row=2, col=1)

    fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)

    # 3. Plot final Er profile
    root_sets = {0: [], 1: [], 2: []}
    rho_sets = {0: [], 1: [], 2: []}
    for data in scan_data:
        r_num = data["rho"]
        detailed_roots = data.get("roots", [])
        er_values = [rd["Er"] for rd in detailed_roots]
        
        if len(er_values) == 1:
            rho_sets[0].append(r_num); root_sets[0].append(er_values[0] / 1e3)
        elif len(er_values) >= 3:
            for i in range(min(3, len(er_values))):
                rho_sets[i].append(r_num); root_sets[i].append(er_values[i] / 1e3)
        elif len(er_values) == 2:
            rho_sets[0].append(r_num); root_sets[0].append(er_values[0] / 1e3)
            rho_sets[2].append(r_num); root_sets[2].append(er_values[1] / 1e3)

    colors = {0: "royalblue", 1: "black", 2: "crimson"}
    names = {0: "Ion Root", 1: "Unstable Root", 2: "Electron Root"}
    dashes = {0: "solid", 1: "dash", 2: "solid"}
    
    for i in range(3):
        if rho_sets[i]:
            fig.add_trace(go.Scatter(
                x=rho_sets[i], y=root_sets[i],
                mode='lines+markers', name=names[i],
                line=dict(color=colors[i], dash=dashes[i])
            ), row=2, col=2)

    # Layout updates
    fig.update_layout(
        height=900, width=1100,
        title_text=f"Ambipolar Summary (RunID: {runid})",
        template="plotly_white"
    )
    
    fig.update_xaxes(title_text="rho", row=1, col=1)
    fig.update_xaxes(title_text="rho", row=1, col=2)
    fig.update_xaxes(title_text="Erho [kV/m]", row=2, col=1)
    fig.update_xaxes(title_text="rho", row=2, col=2)
    
    fig.update_yaxes(title_text="T [keV]", row=1, col=1)
    fig.update_yaxes(title_text="n [10^20 m^-3]", row=1, col=2)
    fig.update_yaxes(title_text="Charge Flux [A/m^2]", row=2, col=1)
    fig.update_yaxes(title_text="Er [kV/m]", row=2, col=2)

    if show:
        fig.show()

    return fig


def plot_ambipolar_scan(
    erho_grid: np.ndarray,
    flux_diffs: np.ndarray,
    roots: list,
    title: str = "Ambipolar Electric Field Search",
    show: bool = True,
):
    """
    Visualize the coarse Erho scan and the refined ambipolar roots in kV/m.

    This method was entirely generated through an AI tool (Gemini/Claude).

    Parameters
    ----------
    erho_grid : np.ndarray
        Array of Er values from the coarse scan.
    flux_diffs : np.ndarray
        Net charge flux at each Er value.
    roots : list
        List of root dicts (with ``"Er"`` key) or raw float values.
    title : str, optional
        Plot title.
    show : bool, optional
        If True (default), call ``fig.show()``.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure object.
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

    if show:
        fig.show()

    return fig


def plot_convergence_scan(results: dict, show: bool = True):
    """
    Plot convergence scan results with absolute flux and percent deviation.

    For each scanned parameter (nx, na, nt, nz), two subplots are shown
    side by side: the left shows absolute particle flux values, the right
    shows percent deviation relative to the flux at the nominal resolution.
    A vertical dashed line marks the nominal value on each subplot.

    This function was entirely generated through an AI tool (Claude).

    Parameters
    ----------
    results : dict
        Output dictionary from ``run_convergence_scan``, containing keys
        ``scan_results``, ``runid``, ``species_names``, and
        ``nominal_values``.
    show : bool, optional
        If True (default), call ``fig.show()``. The figure is always
        returned regardless of this setting.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure object.
    """
    scan_results = results.get("scan_results", {})
    runid = results.get("runid", "Unknown Run")
    species_names = results.get("species_names", None)
    nominal_values = results.get("nominal_values", {})
    rho = results.get("rho", None)
    erho = results.get("erho", None)

    params = list(scan_results.keys())
    num_params = len(params)

    if num_params == 0:
        print("No convergence scan data to plot.")
        return None

    # Layout: one row per parameter, two columns (absolute | deviation)
    rows = num_params
    cols = 2

    subplot_titles = []
    for p in params:
        nom = nominal_values.get(p)
        nom_str = f" (nominal={nom})" if nom is not None else ""
        subplot_titles.append(f"{p} — Particle Flux{nom_str}")
        subplot_titles.append(f"{p} — Deviation from Nominal{nom_str}")

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.12,
        vertical_spacing=0.08,
    )

    # Default color sequence for species
    _colors = [
        "royalblue",
        "crimson",
        "seagreen",
        "darkorange",
        "mediumpurple",
        "goldenrod",
        "deeppink",
        "teal",
    ]

    for i, param in enumerate(params):
        row = i + 1

        data = scan_results[param]
        if not data:
            continue

        x_vals = [d["value"] for d in data]
        num_species = len(data[0]["particle_flux"])
        nominal_val = nominal_values.get(param)

        # Find the flux values at the nominal resolution for deviation calc.
        # Use exact match first; fall back to linear interpolation if the
        # nominal value was not included in the scan points.
        nominal_fluxes = None
        if nominal_val is not None:
            for d in data:
                if int(d["value"]) == int(nominal_val):
                    nominal_fluxes = d["particle_flux"]
                    break

            if nominal_fluxes is None and len(data) >= 2:
                # Interpolate flux at the nominal value
                x_arr = np.array([d["value"] for d in data], dtype=float)
                nominal_fluxes = []
                for sp_idx in range(num_species):
                    y_arr = np.array(
                        [d["particle_flux"][sp_idx] for d in data], dtype=float
                    )
                    nominal_fluxes.append(
                        float(np.interp(float(nominal_val), x_arr, y_arr))
                    )

        for sp_idx in range(num_species):
            # Determine species label
            if species_names and sp_idx < len(species_names):
                sp_name = species_names[sp_idx]
            else:
                sp_name = f"Species {sp_idx}"

            color = _colors[sp_idx % len(_colors)]
            y_vals = [d["particle_flux"][sp_idx] for d in data]

            # Left column: absolute flux values
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines+markers",
                    name=sp_name,
                    legendgroup=sp_name,
                    showlegend=(i == 0),
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                ),
                row=row,
                col=1,
            )

            # Right column: percent deviation from nominal
            if nominal_fluxes is not None:
                ref = nominal_fluxes[sp_idx]
                if ref != 0.0:
                    dev_vals = [(v - ref) / abs(ref) * 100.0 for v in y_vals]
                else:
                    dev_vals = [0.0] * len(y_vals)

                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=dev_vals,
                        mode="lines+markers",
                        name=sp_name,
                        legendgroup=sp_name,
                        showlegend=False,
                        line=dict(color=color, width=2),
                        marker=dict(size=6),
                    ),
                    row=row,
                    col=2,
                )

        # Add nominal value vertical line on both columns
        if nominal_val is not None:
            for col_idx in [1, 2]:
                fig.add_vline(
                    x=nominal_val,
                    line_dash="dash",
                    line_color="gray",
                    line_width=1.5,
                    row=row,
                    col=col_idx,
                )

        # Add zero reference line on deviation column
        fig.add_hline(
            y=0,
            line_dash="dot",
            line_color="gray",
            line_width=1,
            row=row,
            col=2,
        )

        # Axis labels
        fig.update_xaxes(title_text=param, row=row, col=1)
        fig.update_xaxes(title_text=param, row=row, col=2)
        fig.update_yaxes(title_text="Particle Flux", row=row, col=1)
        fig.update_yaxes(title_text="Deviation [%]", row=row, col=2)

    # Build title
    title_parts = [f"Convergence Scan (RunID: {runid})"]
    if rho is not None:
        title_parts.append(f"rho={rho:.4f}")
    if erho is not None:
        title_parts.append(f"Erho={erho:.2e} V/m")
    title = " | ".join(title_parts)

    fig.update_layout(
        title_text=title,
        height=max(400, 350 * rows),
        width=1100,
        showlegend=True,
        template="plotly_white",
        hovermode="x unified",
    )

    if show:
        fig.show()

    return fig


def plot_profiles(results: dict, show: bool = True):
    """
    Plot temperature and density profiles from saved profile data.

    This is a standalone two-panel plot (temperature and density vs rho)
    that works from the ``results["profiles"]`` dictionary saved by
    ``scan_ambipolar_profile``, without requiring live species objects.

    This function was entirely generated through an AI tool (Claude).

    Parameters
    ----------
    results : dict
        Output dictionary from ``scan_ambipolar_profile`` containing a
        ``"profiles"`` key.
    show : bool, optional
        If True (default), call ``fig.show()``.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure object, or *None* if no profile data is found.
    """
    profiles = results.get("profiles")
    if profiles is None:
        return None

    runid = results.get("runid", "unknown")
    rho_arr = np.array(profiles["rho"])

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Temperature [keV]", "Density [10^20 m^-3]"),
        horizontal_spacing=0.12,
    )

    _colors = [
        "royalblue",
        "crimson",
        "seagreen",
        "darkorange",
        "mediumpurple",
        "goldenrod",
    ]

    for ii, sp in enumerate(profiles["species"]):
        name = sp["name"]
        color = _colors[ii % len(_colors)]

        T_vals = np.array(sp["temperature"]) / 1e3  # eV -> keV
        fig.add_trace(
            go.Scatter(
                x=rho_arr,
                y=T_vals,
                mode="lines+markers",
                name=f"T_{name}",
                line=dict(color=color, width=2),
                marker=dict(size=5),
            ),
            row=1,
            col=1,
        )

        n_vals = np.array(sp["density"]) / 1e20  # m^-3 -> 10^20 m^-3
        fig.add_trace(
            go.Scatter(
                x=rho_arr,
                y=n_vals,
                mode="lines+markers",
                name=f"n_{name}",
                line=dict(color=color, width=2, dash="dash"),
                marker=dict(size=5),
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(title_text="rho", row=1, col=1)
    fig.update_xaxes(title_text="rho", row=1, col=2)
    fig.update_yaxes(title_text="T [keV]", row=1, col=1)
    fig.update_yaxes(title_text="n [10^20 m^-3]", row=1, col=2)

    fig.update_layout(
        title_text=f"Input Profiles (RunID: {runid})",
        height=450,
        width=1000,
        template="plotly_white",
        hovermode="x unified",
    )

    if show:
        fig.show()

    return fig
