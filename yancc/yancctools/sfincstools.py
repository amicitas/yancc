"""
Some tools to help with comparisons with previous/current sfincs runs, including
those generated with spincsscan and mirscan.

Code generation utilized AI tools including:
  Gemini 3.1 pro
"""
import numpy as np
import plotly.graph_objects as go
from io import StringIO


def _preprocess_profile_file(filepath):
    """
    Private helper function to read the file and strip comments.

    It removes lines starting with comment characters (#, !, %) and
    returns a single string buffer suitable for np.loadtxt.
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Filter out comments and empty lines
        # context [1] indicates comments start with #, but we include ! and % for robustness
        clean_lines = [
            line for line in lines
            if line.strip() and not line.strip()[0] in ['#', '!', '%']
        ]

        # We need at least 2 lines:
        # Line 1: The scheme integer (e.g., "1")
        # Line 2+: The coefficient matrix
        if len(clean_lines) < 2:
            return None

        # Skip the first line (scheme integer) and join the rest
        return "\n".join(clean_lines[1:])

    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def generate_profile_functions(filepath):
    """
    Reads a profiles file where rows correspond to polynomial coefficients.
    Returns a dictionary of callable functions.

    The input file is pre-processed to remove comments and headers.

    Column Mapping (0-based index):
    Col 0-2: Ignored (NErs, generalEr_min, generalEr_max)
    Col 3: Species 1 Density
    Col 4: Species 1 Temperature
    Col 5: Species 2 Density
    Col 6: Species 2 Temperature
    """

    output_functions = {}

    # --- 1. Preprocess File ---
    clean_data_str = _preprocess_profile_file(filepath)

    if clean_data_str is None:
        print("Error: File could not be pre-processed or contains insufficient data.")
        return {}

    # --- 2. Load Data ---
    try:
        data = np.loadtxt(StringIO(clean_data_str))
    except ValueError as e:
        print(f"Error parsing data matrix: {e}")
        return {}

    # Handle case where there is only one row of coefficients (1D array -> 2D array)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    # --- 3. Map Columns to Functions ---
    column_map = {
        3: 'species_1_density',
        4: 'species_1_temperature',
        5: 'species_2_density',
        6: 'species_2_temperature'
    }

    # Validate that the file has enough columns
    max_needed_col = max(column_map.keys())
    if data.shape[1] <= max_needed_col:
        print(f"Error: File has {data.shape[1]} columns, but we need index {max_needed_col}.")
        return {}

    for col_idx, name in column_map.items():
        # Extract the column. The rows are the coefficients c0, c1, c2...
        coeffs = data[:, col_idx]

        # Create closure for the polynomial function
        def make_poly_func(coefficients):
            # Capture coefficients in a local array to prevent late-binding issues
            local_coeffs = np.array(coefficients)

            def poly_func(rN):
                rN = np.atleast_1d(rN)
                # Evaluates: c0 + c1*x + c2*x^2 ...
                return np.polynomial.polynomial.polyval(rN, local_coeffs)

            return poly_func

        output_functions[name] = make_poly_func(coeffs)

    return output_functions


def plot_profiles_plotly(profile_dict):
    """
    Plots the generated profile functions using Plotly.
    """
    if not profile_dict:
        print("No profiles available to plot.")
        return

    rN = np.linspace(0, 1, 100)
    fig = go.Figure()

    for name, func in profile_dict.items():
        y_vals = func(rN)

        # Style logic
        line_dash = 'dash' if 'temperature' in name else 'solid'
        color = 'blue' if 'species_1' in name else 'red'

        fig.add_trace(go.Scatter(
            x=rN,
            y=y_vals,
            mode='lines',
            name=name,
            line=dict(dash=line_dash, color=color, width=3)
        ))

    fig.update_layout(
        title="Reconstructed Polynomial Profiles",
        xaxis_title="rN (Normalized Radius)",
        yaxis_title="Normalized Value",
        template="plotly_white",
        legend_title="Profile Type"
    )

    fig.show()
