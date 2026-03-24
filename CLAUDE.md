# Project Rules: Yancc

## Core Mandates
- **No Backwards Compatibility:** Prioritize clean, modern refactoring and optimal API design over maintaining compatibility with older versions or structures.
- **Input Dictionaries:** Prefer using an `options` or `config` dictionary for function inputs instead of long lists of keyword arguments.
    - Input Dictionaries should be compatible with json and yaml whenever possible
- **Output Dictionaries** Prefer using a `results` dictionary for function outputs.
    -  Output dictionaries should be compatible with hdf5 whenever possible
- **Run Tracking:** All batch or radial scans must include a unique `runid` (20-character format: `Era-Zodiac-YYMMDD-HHMMSS`).
- **Logging:** Ensure `logging` is configured and visible in all execution environments, including Jupyter notebooks.
- **AI Usage Disclaimer**  At the top of every file that is modified by an AI tool, add a disclaimer saying 'This file includes AI generated code'
    - Add the high-level AI tool identifier eg. Gemini, Claude, Codex etc.
    - Example: `This file includes AI generated code (Gemini/Claude/Codex)`
    - When new functions or methods are entirely generated through an AI tool, put a similar appropriate disclaimer in the doc string.
- **Session Logging:** When the user asks to close a session write a descriptive log of the current session's changes and key findings to `devel_ai/devel_ai_log.txt`.
- **Feature Requests:**
    - **Creation:** Upon identifying a new feature or significant change request, immediately add it to `devel_ai/features_request.md` with the current date and "Pending" status.
    - **Alignment:** Before creating a plan, verify alignment with `devel_ai/features_request.md`. If the task is new, create a tracking entry first.
    - **Completion:** Upon successful implementation and verification, update the status to "Done" in `devel_ai/features_request.md` and include a brief note on the implementation details.

## Coding Standards
- Use Python 3.10+ features (e.g., modern type hinting with `list[...]` not `List[...]`).
- Maintain JSON/YAML compatibility for all input configuration and options dictionaries.
- Maintain hdf5/h5py compatibility for all output dictionaries.
- Visualization tools (Plotly) should handle metadata keys (like `runid`) gracefully.
- I prefer spaces over tabs in all situations.
- Follow NumPy-style docstrings (project uses `flake8-docstrings` with numpy convention).
- Format code with Black (line length 88) and sort imports with isort (profile: black).

## Architecture Notes
- Core physics objects (`Field`, `Species`, operators) are `equinox.Module` subclasses — treat them as immutable, JAX-traceable dataclasses. Use `eqx.field(static=True)` for non-traced fields.
- Linear operators follow the `lineax.AbstractLinearOperator` interface with `mv()` methods.
- Array types are annotated with `jaxtyping` (e.g., `Float[Array, "ntheta nzeta"]`).
- JAX 64-bit precision is enforced globally (`jax_enable_x64 = True`).

## Testing
- Run tests: `python -m pytest -v tests/`
- Skip slow tests: `pytest -m "not slow"`
- Pytest markers: `unit`, `regression`, `slow`, `fast`
- Linting: `black yancc/ tests/ && isort --profile black yancc/ tests/ && flake8 yancc/ tests/`
- Type checking: `pyright yancc/`

## Project Structure
- `yancc/` — Core solver (DKE/MDKE for neoclassical transport in fusion plasmas)
- `yancc/yancctools/` — Higher-level tooling (ambipolar Er solver, radial scans, I/O, plotting)
- `tests/` — Test suite with session-scoped fixtures in `conftest.py`
- `devel_ai/` — AI development tracking (session logs, feature requests)
