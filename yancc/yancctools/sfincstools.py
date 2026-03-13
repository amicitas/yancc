"""
Some tools to help with comparisons with previous/current sfincs runs, including
those generated with spincsscan and mirscan.

Code generation utilized AI tools including:
  Gemini 3.1 pro
"""

import numpy as np
import plotly.graph_objects as go
from io import StringIO


def _read_and_clean_file(filepath):
    """
    Private helper function to read the file, strip comments,
    and return the data matrix as a numpy array.

    Skips lines starting with '#', '!', or '%'.
    Assumes the first valid line is a scalar header (e.g., '1')
    and skips it to return only the coefficient matrix.
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Filter out comments and empty lines
        clean_lines = [
            line for line in lines
            if line.strip() and not line.strip()[0] in ['#', '!', '%']
        ]

        if len(clean_lines) < 2:
            return None

        # The first clean line is the profile scheme integer (e.g., "1") [1].
        # We skip it to get the coefficient matrix.
        matrix_str = "".join(clean_lines[1:])

        # Parse the string into a numpy array
        data = np.loadtxt(StringIO(matrix_str))

        # Ensure 2D array even if there is only one row of coeffs
        if data.ndim == 1:
            data = data.reshape(1, -1)

        return data

    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def generate_profile_functions(filepath):
    """
    Generates a dictionary of polynomial functions from the input file.

    Column Mapping (0-based):
    Col 0-2: Ignored
    Col 3: Species 1 Density
    Col 4: Species 1 Temperature
    Col 5: Species 2 Density
    Col 6: Species 2 Temperature
    """

    output_functions = {}

    # 1. Get cleaned data matrix
    data = _read_and_clean_file(filepath)

    if data is None:
        return {}

    # 2. Define Mapping
    column_map = {
        3: 'species_1_density',
        4: 'species_1_temperature',
        5: 'species_2_density',
        6: 'species_2_temperature'
    }

    # Validate dimensions
    max_col = max(column_map.keys())
    if data.shape[1] <= max_col:
        print(f"Error: Data has {data.shape[1]} columns; expected at least {max_col + 1}.")
        return {}

    # 3. Create Functions
    for col_idx, name in column_map.items():
        coeffs = data[:, col_idx]

        # Closure to capture coefficients
        def make_poly_func(coefficients):
            # Enforce float type for safety
            local_coeffs = np.array(coefficients, dtype=float)

            def poly_func(rN):
                rN = np.atleast_1d(rN)
                # Evaluates c0 + c1*x + c2*x^2 ...
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

        # Determine style based on name
        is_temp = 'temperature' in name
        is_spec1 = 'species_1' in name

        line_dash = 'dash' if is_temp else 'solid'
        color = 'blue' if is_spec1 else 'red'

        fig.add_trace(go.Scatter(
            x=rN,
            y=y_vals,
            mode='lines',
            name=name,
            line=dict(dash=line_dash, color=color, width=3)
        ))

    fig.update_layout(
        title="Polynomial Profiles (Reconstructed)",
        xaxis_title="rN (Normalized Radius)",
        yaxis_title="Normalized Value",
        template="plotly_white",
        legend_title="Profile Type"
    )
    fig.show()


# --- Execution ---
target_file = '/u/npablant/analysis/w7x/171207006/stelltran/run07/sfincs/t2.2273/profiles'

profiles_map = generate_profile_functions(target_file)
plot_profiles_plotly(profiles_map)