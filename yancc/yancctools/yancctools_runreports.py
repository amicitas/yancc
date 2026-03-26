# This file includes AI generated code (Claude)
"""
Browser-based run report interface for yancctools.

Provides a Dash application that scans a directory of saved runs and
presents an interactive interface for reviewing results, options, and
plots.  Run directories are expected to follow the standard layout
produced by ``yancctools_io``::

    {output_path}/{runid}/
        options.json   (or options.yaml)
        results.h5
        inputs.h5      (optional)

This module was entirely generated through an AI tool (Claude).

Usage
-----
>>> from yancc.yancctools import launch_run_reports
>>> launch_run_reports("/path/to/runs/")
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from yancc.yancctools.yancctools_io import load_options, load_results

logger = logging.getLogger(__name__)

# Pattern matching the runid directory format: Era-Zodiac-YYMMDD-HHMMSS
# e.g. R8-ARI-260324-234924
_RUNID_PATTERN = re.compile(
    r"^R\d+-[A-Z]{3}-\d{6}-\d{6}$"
)


def discover_runs(directory: str) -> list[dict]:
    """
    Scan a directory for valid run subdirectories.

    Parameters
    ----------
    directory : str
        Path to the parent directory containing run folders.

    Returns
    -------
    list[dict]
        List of run metadata dicts sorted by runid (newest first).
        Each dict has keys: ``runid``, ``path``, ``has_options``,
        ``has_results``, ``has_inputs``.
    """
    root = Path(directory)
    if not root.is_dir():
        logger.warning(f"Directory does not exist: {directory}")
        return []

    runs = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        if not _RUNID_PATTERN.match(entry.name):
            continue

        has_results = (entry / "results.h5").exists()
        has_options = (
            (entry / "options.json").exists()
            or (entry / "options.yaml").exists()
        )
        has_inputs = (entry / "inputs.h5").exists()

        # Only include runs that have results
        if not has_results:
            logger.debug(f"Skipping {entry.name}: no results.h5")
            continue

        runs.append({
            "runid": entry.name,
            "path": entry,
            "has_options": has_options,
            "has_results": has_results,
            "has_inputs": has_inputs,
        })

    # Sort newest first (runid encodes date/time)
    runs.sort(key=lambda r: r["runid"], reverse=True)
    return runs


def detect_run_type(results: dict) -> str:
    """
    Determine whether a results dict is from an ambipolar scan or a
    convergence scan.

    Parameters
    ----------
    results : dict
        Loaded results dictionary.

    Returns
    -------
    str
        ``"ambipolar"`` or ``"convergence"`` or ``"unknown"``.
    """
    scan_results = results.get("scan_results")
    if scan_results is None:
        return "unknown"

    # Ambipolar: scan_results is a list of dicts with "rho" keys
    if isinstance(scan_results, list):
        return "ambipolar"

    # Convergence: scan_results is a dict with parameter name keys
    if isinstance(scan_results, dict):
        param_keys = {"nx", "na", "nt", "nz"}
        if param_keys & set(scan_results.keys()):
            return "convergence"

    return "unknown"


def _format_option_value(value) -> str:
    """Format an option value for human-readable display."""
    if isinstance(value, float):
        if abs(value) >= 1e4 or (abs(value) < 1e-2 and value != 0.0):
            return f"{value:.2e}"
        return f"{value:.4g}"
    if isinstance(value, (list, np.ndarray)):
        if len(value) > 8:
            return f"[{len(value)} elements]"
        return str(value)
    return str(value)


def _extract_run_info(options: dict, results: dict, run_type: str) -> list[dict]:
    """
    Extract human-readable key-value pairs from options and results.

    Parameters
    ----------
    options : dict or None
        Loaded options dictionary.
    results : dict
        Loaded results dictionary.
    run_type : str
        ``"ambipolar"`` or ``"convergence"``.

    Returns
    -------
    list[dict]
        List of ``{"label": str, "value": str}`` pairs.
    """
    info = []

    # RunID
    runid = results.get("runid", options.get("runid", "unknown") if options else "unknown")
    info.append({"label": "Run ID", "value": runid})
    info.append({"label": "Run Type", "value": run_type.title()})

    # Source options (prefer options dict, fall back to results["options"])
    opts = options if options else results.get("options", {})

    # Resolution
    for key in ("nx", "na", "nt", "nz"):
        if key in opts:
            info.append({"label": key, "value": str(opts[key])})

    # Solver settings
    for key in ("rtol", "atol"):
        if key in opts:
            info.append({"label": key, "value": _format_option_value(opts[key])})

    # Er scan range (ambipolar)
    if run_type == "ambipolar":
        erho_min = opts.get("erho_min")
        erho_max = opts.get("erho_max")
        erho_num = opts.get("erho_num")
        if erho_min is not None and erho_max is not None:
            info.append({
                "label": "Er Range",
                "value": f"{erho_min / 1e3:.1f} to {erho_max / 1e3:.1f} kV/m"
                         + (f" ({erho_num} pts)" if erho_num else ""),
            })

        # Rho grid
        scan_results = results.get("scan_results", [])
        if scan_results:
            rhos = [sr["rho"] for sr in scan_results]
            rho_strs = [f"{r:.2f}" for r in rhos]
            info.append({
                "label": "Rho Grid",
                "value": f"[{', '.join(rho_strs)}]  ({len(rhos)} surfaces)",
            })

    # Convergence-specific
    if run_type == "convergence":
        rho = results.get("rho")
        erho = results.get("erho")
        if rho is not None:
            info.append({"label": "rho", "value": f"{rho:.4f}"})
        if erho is not None:
            info.append({"label": "Erho", "value": f"{erho:.2e} V/m"})

        species_names = results.get("species_names", [])
        if species_names:
            info.append({"label": "Species", "value": ", ".join(species_names)})

        nominal = results.get("nominal_values", {})
        if nominal:
            nom_str = ", ".join(f"{k}={v}" for k, v in nominal.items())
            info.append({"label": "Nominal Resolution", "value": nom_str})

    # Profile info
    profiles = results.get("profiles")
    if profiles:
        species_list = profiles.get("species", [])
        names = [sp["name"] for sp in species_list]
        if names:
            info.append({"label": "Profile Species", "value": ", ".join(names)})

    # Output path
    output_path = opts.get("output_path")
    if output_path:
        info.append({"label": "Output Path", "value": str(output_path)})

    return info


def _build_ambipolar_results_table(results: dict) -> list[dict]:
    """
    Build a summary table for ambipolar scan results.

    Returns
    -------
    list[dict]
        List of row dicts with keys for table columns.
    """
    scan_results = results.get("scan_results", [])
    rows = []
    for sr in scan_results:
        rho = sr["rho"]
        roots = sr.get("roots", [])
        er_values = [r["Er"] / 1e3 for r in roots]  # kV/m

        row = {
            "rho": f"{rho:.3f}",
            "num_roots": str(len(roots)),
        }

        # Classify roots
        if len(er_values) == 1:
            row["ion_root"] = f"{er_values[0]:.2f}"
            row["unstable_root"] = "--"
            row["electron_root"] = "--"
        elif len(er_values) == 2:
            row["ion_root"] = f"{er_values[0]:.2f}"
            row["unstable_root"] = "--"
            row["electron_root"] = f"{er_values[1]:.2f}"
        elif len(er_values) >= 3:
            row["ion_root"] = f"{er_values[0]:.2f}"
            row["unstable_root"] = f"{er_values[1]:.2f}"
            row["electron_root"] = f"{er_values[2]:.2f}"
        else:
            row["ion_root"] = "--"
            row["unstable_root"] = "--"
            row["electron_root"] = "--"

        rows.append(row)

    return rows


def _build_convergence_results_table(results: dict) -> list[dict]:
    """
    Build a summary table for convergence scan results.

    Returns
    -------
    list[dict]
        List of row dicts with keys for table columns.
    """
    scan_results = results.get("scan_results", {})
    species_names = results.get("species_names", [])
    nominal_values = results.get("nominal_values", {})
    rows = []

    for param, data in scan_results.items():
        nominal = nominal_values.get(param)
        for point in data:
            val = point["value"]
            fluxes = point.get("particle_flux", [])
            row = {
                "parameter": param,
                "value": str(val),
                "is_nominal": "Yes" if nominal and int(val) == int(nominal) else "",
            }
            for ii, flux in enumerate(fluxes):
                sp_name = species_names[ii] if ii < len(species_names) else f"Species {ii}"
                row[f"flux_{sp_name}"] = f"{flux:.4e}"
            rows.append(row)

    return rows


def create_app(directory: str):
    """
    Construct the Dash application for browsing run reports.

    Parameters
    ----------
    directory : str
        Path to the parent directory containing run folders.

    Returns
    -------
    dash.Dash
        The configured Dash application (not yet started).
    """
    try:
        import dash
        from dash import dcc, html, dash_table
        from dash.dependencies import Input, Output, State
    except ImportError as exc:
        raise ImportError(
            "The 'dash' package is required for run reports. "
            "Install it with: pip install dash"
        ) from exc

    from yancc.yancctools.yancctools_plot import (
        plot_ambipolar_summary,
        plot_convergence_scan,
        plot_profiles,
    )

    app = dash.Dash(
        __name__,
        title="Yancc Run Reports",
        suppress_callback_exceptions=True,
    )

    # ------------------------------------------------------------------ #
    # Layout                                                              #
    # ------------------------------------------------------------------ #

    sidebar_style = {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "320px",
        "padding": "20px",
        "backgroundColor": "#f8f9fa",
        "overflowY": "auto",
        "borderRight": "1px solid #dee2e6",
    }

    content_style = {
        "marginLeft": "340px",
        "padding": "20px",
    }

    sidebar = html.Div(
        [
            html.H4("Yancc Run Reports", style={"marginBottom": "20px"}),
            html.Hr(),
            html.Div(
                [
                    html.Label(
                        "Directory:",
                        style={"fontWeight": "bold", "fontSize": "12px"},
                    ),
                    html.Div(
                        str(directory),
                        style={
                            "fontSize": "11px",
                            "wordBreak": "break-all",
                            "color": "#666",
                            "marginBottom": "10px",
                        },
                    ),
                    html.Button(
                        "Refresh",
                        id="refresh-button",
                        n_clicks=0,
                        style={
                            "width": "100%",
                            "marginBottom": "15px",
                            "padding": "6px",
                        },
                    ),
                ]
            ),
            html.Hr(),
            html.Label(
                "Runs:",
                style={"fontWeight": "bold", "marginBottom": "10px"},
            ),
            html.Div(id="run-list-container"),
        ],
        style=sidebar_style,
    )

    content = html.Div(
        [
            html.Div(id="main-content", children=[
                html.Div(
                    "Select a run from the sidebar to view its report.",
                    style={
                        "textAlign": "center",
                        "color": "#999",
                        "marginTop": "100px",
                        "fontSize": "18px",
                    },
                )
            ]),
        ],
        style=content_style,
    )

    # Hidden store for the currently selected runid
    app.layout = html.Div([
        dcc.Store(id="selected-runid", data=None),
        sidebar,
        content,
    ])

    # ------------------------------------------------------------------ #
    # Callbacks                                                           #
    # ------------------------------------------------------------------ #

    @app.callback(
        Output("run-list-container", "children"),
        Input("refresh-button", "n_clicks"),
    )
    def update_run_list(_n_clicks):
        """Populate the sidebar with discovered runs."""
        runs = discover_runs(directory)
        if not runs:
            return html.Div(
                "No runs found.",
                style={"color": "#999", "fontStyle": "italic"},
            )

        buttons = []
        for run in runs:
            runid = run["runid"]
            badges = []
            if run["has_options"]:
                badges.append(
                    html.Span(
                        "opts",
                        style={
                            "backgroundColor": "#28a745",
                            "color": "white",
                            "padding": "1px 4px",
                            "borderRadius": "3px",
                            "fontSize": "10px",
                            "marginLeft": "4px",
                        },
                    )
                )
            if run["has_inputs"]:
                badges.append(
                    html.Span(
                        "inp",
                        style={
                            "backgroundColor": "#17a2b8",
                            "color": "white",
                            "padding": "1px 4px",
                            "borderRadius": "3px",
                            "fontSize": "10px",
                            "marginLeft": "4px",
                        },
                    )
                )

            buttons.append(
                html.Div(
                    html.Button(
                        [html.Span(runid, style={"fontFamily": "monospace"})] + badges,
                        id={"type": "run-button", "index": runid},
                        n_clicks=0,
                        style={
                            "width": "100%",
                            "textAlign": "left",
                            "padding": "8px 10px",
                            "marginBottom": "4px",
                            "border": "1px solid #ccc",
                            "borderRadius": "4px",
                            "backgroundColor": "white",
                            "cursor": "pointer",
                            "fontSize": "13px",
                        },
                    ),
                )
            )

        return buttons

    @app.callback(
        Output("selected-runid", "data"),
        Input({"type": "run-button", "index": dash.ALL}, "n_clicks"),
        State({"type": "run-button", "index": dash.ALL}, "id"),
        prevent_initial_call=True,
    )
    def select_run(n_clicks_list, id_list):
        """Store the selected runid when a sidebar button is clicked."""
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update

        # Find which button was clicked
        triggered_id = ctx.triggered[0]["prop_id"]
        for btn_id in id_list:
            if btn_id["index"] in triggered_id:
                return btn_id["index"]

        return dash.no_update

    @app.callback(
        Output("main-content", "children"),
        Input("selected-runid", "data"),
        prevent_initial_call=True,
    )
    def display_run_report(runid):
        """Render the full report for the selected run."""
        if runid is None:
            return html.Div(
                "Select a run from the sidebar to view its report.",
                style={
                    "textAlign": "center",
                    "color": "#999",
                    "marginTop": "100px",
                },
            )

        run_path = Path(directory) / runid

        # Load results
        try:
            results = load_results(str(run_path / "results.h5"))
        except Exception as exc:
            return html.Div(
                f"Error loading results: {exc}",
                style={"color": "red"},
            )

        # Load options (optional)
        options = None
        for fname in ("options.json", "options.yaml"):
            opt_path = run_path / fname
            if opt_path.exists():
                try:
                    options = load_options(str(opt_path))
                except Exception:
                    pass
                break

        run_type = detect_run_type(results)

        # ----- Build report sections ----- #
        sections = []

        # Header
        sections.append(
            html.H3(
                f"Run Report: {runid}",
                style={"borderBottom": "2px solid #333", "paddingBottom": "10px"},
            )
        )

        # Run Info Panel
        info_rows = _extract_run_info(options, results, run_type)
        info_table = dash_table.DataTable(
            columns=[
                {"name": "Property", "id": "label"},
                {"name": "Value", "id": "value"},
            ],
            data=info_rows,
            style_cell={
                "textAlign": "left",
                "padding": "8px 12px",
                "fontFamily": "monospace",
                "fontSize": "13px",
            },
            style_header={
                "fontWeight": "bold",
                "backgroundColor": "#f0f0f0",
            },
            style_data_conditional=[
                {
                    "if": {"row_index": "odd"},
                    "backgroundColor": "#fafafa",
                }
            ],
            style_table={"marginBottom": "20px"},
        )
        sections.append(html.H4("Run Configuration"))
        sections.append(info_table)

        # Results Summary Table
        if run_type == "ambipolar":
            table_rows = _build_ambipolar_results_table(results)
            if table_rows:
                columns = [
                    {"name": "rho", "id": "rho"},
                    {"name": "# Roots", "id": "num_roots"},
                    {"name": "Ion Root [kV/m]", "id": "ion_root"},
                    {"name": "Unstable [kV/m]", "id": "unstable_root"},
                    {"name": "Electron Root [kV/m]", "id": "electron_root"},
                ]
                sections.append(html.H4("Ambipolar Roots Summary"))
                sections.append(
                    dash_table.DataTable(
                        columns=columns,
                        data=table_rows,
                        style_cell={
                            "textAlign": "center",
                            "padding": "8px",
                            "fontFamily": "monospace",
                            "fontSize": "13px",
                        },
                        style_header={
                            "fontWeight": "bold",
                            "backgroundColor": "#f0f0f0",
                        },
                        style_data_conditional=[
                            {
                                "if": {"row_index": "odd"},
                                "backgroundColor": "#fafafa",
                            }
                        ],
                        style_table={"marginBottom": "20px"},
                    )
                )

        elif run_type == "convergence":
            table_rows = _build_convergence_results_table(results)
            if table_rows:
                # Determine columns dynamically from first row
                col_ids = list(table_rows[0].keys())
                col_names = {
                    "parameter": "Parameter",
                    "value": "Value",
                    "is_nominal": "Nominal?",
                }
                columns = []
                for cid in col_ids:
                    name = col_names.get(cid, cid.replace("flux_", "Flux: "))
                    columns.append({"name": name, "id": cid})

                sections.append(html.H4("Convergence Scan Summary"))
                sections.append(
                    dash_table.DataTable(
                        columns=columns,
                        data=table_rows,
                        style_cell={
                            "textAlign": "center",
                            "padding": "8px",
                            "fontFamily": "monospace",
                            "fontSize": "13px",
                        },
                        style_header={
                            "fontWeight": "bold",
                            "backgroundColor": "#f0f0f0",
                        },
                        style_data_conditional=[
                            {
                                "if": {"row_index": "odd"},
                                "backgroundColor": "#fafafa",
                            },
                            {
                                "if": {
                                    "filter_query": '{is_nominal} eq "Yes"',
                                },
                                "backgroundColor": "#e8f5e9",
                                "fontWeight": "bold",
                            },
                        ],
                        style_table={
                            "marginBottom": "20px",
                            "overflowX": "auto",
                        },
                    )
                )

        # Plots
        sections.append(
            html.H4(
                "Plots",
                style={"borderTop": "1px solid #ccc", "paddingTop": "15px"},
            )
        )

        if run_type == "ambipolar":
            # Ambipolar summary (4-panel with profiles from saved data)
            try:
                fig_summary = plot_ambipolar_summary(results, show=False)
                sections.append(
                    dcc.Graph(
                        figure=fig_summary,
                        style={"marginBottom": "30px"},
                    )
                )
            except Exception as exc:
                sections.append(
                    html.Div(
                        f"Error generating summary plot: {exc}",
                        style={"color": "red"},
                    )
                )

            # Standalone profile plot (if profile data available)
            if "profiles" in results:
                try:
                    fig_profiles = plot_profiles(results, show=False)
                    if fig_profiles is not None:
                        sections.append(
                            dcc.Graph(
                                figure=fig_profiles,
                                style={"marginBottom": "30px"},
                            )
                        )
                except Exception as exc:
                    sections.append(
                        html.Div(
                            f"Error generating profile plot: {exc}",
                            style={"color": "red"},
                        )
                    )

        elif run_type == "convergence":
            try:
                fig_conv = plot_convergence_scan(results, show=False)
                if fig_conv is not None:
                    sections.append(
                        dcc.Graph(
                            figure=fig_conv,
                            style={"marginBottom": "30px"},
                        )
                    )
            except Exception as exc:
                sections.append(
                    html.Div(
                        f"Error generating convergence plot: {exc}",
                        style={"color": "red"},
                    )
                )

        return sections

    return app


def launch_run_reports(directory: str, config: dict | None = None) -> None:
    """
    Launch the browser-based run report interface.

    This function was entirely generated through an AI tool (Claude).

    Parameters
    ----------
    directory : str
        Path to the parent directory containing run folders.
    config : dict, optional
        Server configuration with optional keys:

        - ``host`` : str, default ``"127.0.0.1"``
        - ``port`` : int, default ``8050``
        - ``debug`` : bool, default ``True``
    """
    if config is None:
        config = {}

    host = config.get("host", "127.0.0.1")
    port = config.get("port", 8050)
    debug = config.get("debug", True)

    logger.info(f"Launching run reports for directory: {directory}")
    logger.info(f"Server: http://{host}:{port}/")

    app = create_app(directory)
    app.run(host=host, port=port, debug=debug)


def main(argv: list[str] | None = None) -> None:
    """
    Command-line entry point for the run reports server.

    This function was entirely generated through an AI tool (Claude).

    Can be invoked as::

        python -m yancc.yancctools.runreports /path/to/runs/
        python -m yancc.yancctools.runreports /path/to/runs/ --port 9000
        python -m yancc.yancctools.runreports .   # current directory

    Or, if installed via pip with console_scripts::

        yancc-runreports /path/to/runs/
    """
    parser = argparse.ArgumentParser(
        prog="yancc-runreports",
        description="Launch the Yancc browser-based run report interface.",
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory containing run folders (default: current directory).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address to bind to (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port number (default: 8050).",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable Dash debug mode (debug is on by default).",
    )

    args = parser.parse_args(argv)

    # Set up logging so messages are visible on the console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    directory = str(Path(args.directory).resolve())

    config = {
        "host": args.host,
        "port": args.port,
        "debug": not args.no_debug,
    }

    launch_run_reports(directory, config=config)


if __name__ == "__main__":
    main()
