# This file includes AI generated code: Gemini 3.5 Sonnet
"""
Plotting routines for yancctools.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_ambipolar_profile(results: dict):
    """
    Plots the radial profile of ambipolar electric fields in kV/m.
    
    This method was entirely generated through an AI tool (Gemini 3.5 Sonnet).
    """
    # Convert string keys back to floats for numerical plotting and sorting
    # Skip metadata keys like 'runid'
    rhos_numeric = sorted([float(r) for r in results.keys() if r != "runid"])
    
    root_sets = {0: [], 1: [], 2: []}
    rho_sets = {0: [], 1: [], 2: []}
    
    for r_num in rhos_numeric:
        r_str = f"{r_num:.2f}"
        # Extract Er values from the nested detailed roots (new structure)
        data = results[r_str]
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

    fig.show()


def plot_ambipolar_summary(results: dict, global_species: list = None):
    """
    Displays a comprehensive summary page including:
    1. Input Profiles (T, n)
    2. All individual net charge flux scans (flux vs Er)
    3. Final ambipolar Er profile
    
    This method was entirely generated through an AI tool (Gemini 3.5 Sonnet).
    """
    runid = results.get("runid", "unknown")
    rhos_numeric = sorted([float(r) for r in results.keys() if r != "runid"])
    
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

    # 1. Plot Input Profiles (if provided)
    if global_species:
        rho_dense = np.linspace(0, 1, 100)
        for s in global_species:
            name = s.species.__class__.__name__ if hasattr(s.species, "__class__") else str(s.species)
            # Find a friendly name
            if "Electron" in str(s.species): name = "Electrons"
            elif "Hydrogen" in str(s.species): name = "Hydrogen"
            
            # Temperature
            T_vals = np.array([float(s.temperature(r)) for r in rho_dense]) / 1e3 # to keV
            fig.add_trace(go.Scatter(
                x=rho_dense, y=T_vals, name=f"T_{name}", legendgroup="profiles"
            ), row=1, col=1)
            
            # Density
            n_vals = np.array([float(s.density(r)) for r in rho_dense]) / 1e20 # to 10^20
            fig.add_trace(go.Scatter(
                x=rho_dense, y=n_vals, name=f"n_{name}", legendgroup="profiles"
            ), row=1, col=2)

    # 2. Plot all flux scans
    # Use a colormap for different rhos
    for i, r_num in enumerate(rhos_numeric):
        r_str = f"{r_num:.2f}"
        data = results[r_str]
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
    for r_num in rhos_numeric:
        r_str = f"{r_num:.2f}"
        detailed_roots = results[r_str]["roots"]
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

    fig.show()


def plot_ambipolar_scan(erho_grid: np.ndarray, flux_diffs: np.ndarray, roots: list, title: str = "Ambipolar Electric Field Search"):
    """
    Visualizes the coarse Erho scan and the refined ambipolar roots in kV/m.
    
    This method was entirely generated through an AI tool (Gemini 3.5 Sonnet).
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
