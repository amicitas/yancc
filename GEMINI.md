# Project Rules: Yancc

## Core Mandates
- **No Backwards Compatibility:** Prioritize clean, modern refactoring and optimal API design over maintaining compatibility with older versions or structures.
- **Input Dictionaries:** Prefer using an `options` or `config` dictionary for function inputs instead of long lists of keyword arguments.
    - Input Dictionaries should be compatible with json and yaml whenever possible
- **Output Dictionaries** Prefer using a 'results' dictionary for function outputs.
    -  Output dictionaries should be compatible with hdf5 whenever possible
- **Run Tracking:** All batch or radial scans must include a unique `runid` (20-character format: `Era-Zodiac-YYMMDD-HHMMSS`).
- **Logging:** Ensure `logging` is configured and visible in all execution environments, including Jupyter notebooks.
- **AI Usage Disclaimer**  At the top of every file that is modified by an AI tool, add a disclaimer saying 'This file includes AI generated code' 
    - Add the high-level AI tool identifier eg. Gemini 3.1 pro
    - When new functions or methods are entirely generated through an AI tool, put a similar appropriate disclaimer in the doc string.
- **Full Diffs:** When asking for permission to modify files, always show the full diff instead of only the first few lines.

## Coding Standards
- Use Python 3.10+ features (e.g., modern type hinting).
- Maintain JSON/YAML compatibility for all input configuration and options dictionaries.
- Maintain hdf5/h5py compatibility for all output dictionaries.
- Visualization tools (Plotly) should handle metadata keys (like `runid`) gracefully.
- I prefer spaces over tabs in all situations.