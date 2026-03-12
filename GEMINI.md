# Project Rules: Yancc

## Core Mandates
- **No Backwards Compatibility:** Prioritize clean, modern refactoring and optimal API design over maintaining compatibility with older versions or structures.
- **Options Dictionaries:** Prefer using an `options` dictionary for function inputs instead of long lists of keyword arguments.
-- Input Options Dictionaries should be compatible with json or yaml
- **Output Dictionaries** Prefer using a 'results' dictionary for function outputs.
--  Output dictionaries should be compatible with hdf5 when possible
- **Run Tracking:** All batch or radial scans must include a unique `runid` (20-character format: `Era-Zodiac-YYMMDD-HHMMSS`).
- **Logging:** Ensure `logging.INFO` is configured and visible in all execution environments, including Jupyter notebooks.

## Coding Standards
- Use Python 3.10+ features (e.g., modern type hinting).
- Maintain JSON/YAML compatibility for all input configuration and options dictionaries.
- Maintain hdf5/h5py compatibility for all output dictionaries.
- Visualization tools (Plotly) should handle metadata keys (like `runid`) gracefully.
