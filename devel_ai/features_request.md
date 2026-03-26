<!-- This file includes AI generated code (Gemini) -->

# Feature Requests & Status

| Feature | Requested Date | Status   | Notes |
| :--- | :--- |:---------| :--- |
| **Better input, output and run handling** | 2026-03-15 | **Done** | Enhancing yancctools with robust I/O and run management. |
| - Create `yancctools_io` module | 2026-03-16 | Done     | Separate functions for options (JSON/YAML), inputs (HDF5), and results (HDF5). |
| - Standardized radial scan I/O | 2026-03-16 | Done     | Functions to write results and options to run-specific directories. |
| - Run management with `runid` | 2026-03-16 | Done     | Automatic directory creation using `Era-Zodiac-YYMMDD-HHMMSS` format inside `output_path`. |
| - Notebook "Part 3" development | 2026-03-16 | Done     | Focus on radial scans, ambipolar solutions, and file-based plotting (re-loading data). |
| **Improve scan_ambipolar_profile output** | 2026-03-21 | **Done** | Refactored results to be HDF5-compatible list of dicts (`scan_results`), including full options. |
| **AI assisted development tracking** | 2026-03-15 | **Done** | Created `devel_ai` directory and initial logs. |
| **Automated convergence scan tool** | 2026-03-22 | **Done** | Implement convergence scan tool iterating over numerical resolution parameters, including tracking particle fluxes, HDF5 output, and Plotly visualizations. |
| **Improved convergence scan plotting** | 2026-03-23 | **Done** | Species names in legends, nominal value markers, relative/percent deviation subplots, consistent plotly_white styling, return figure object. Also store species names and nominal values in results dict. |
| **Part 6 notebook: VMEC convergence + radial ambipolar scan** | 2026-03-24 | Pending | Unified notebook using VMEC equilibrium (not DESC) with SFINCS profiles: convergence scan at rho=0.2/Er=0.0, then radial ambipolar scan. Builds on Parts 4 & 5. |
| **Browser-based run reports (yancctools_runreports)** | 2026-03-25 | Pending | Dash-based web interface for reviewing saved runs. Sidebar lists runs by runid; selecting a run shows config, results tables, and embedded Plotly plots. Supports both ambipolar and convergence run types. CLI via `yancc-runreports` or `python -m yancc.yancctools`. Also: refactored plot functions to return figures, added profile data to ambipolar results dict, new `plot_profiles` function. |
