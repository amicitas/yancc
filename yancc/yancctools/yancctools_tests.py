# -*- coding: utf-8 -*-
"""
Test script for yancctools I/O and radial scan functionality.
Replicates the workflow from Part 3 notebook with minimal parameters.

This file includes AI generated code (Gemini)
"""

import numpy as np
import tempfile
import uuid
import logging
import pytest
from pathlib import Path
import desc
import yancc
from yancc.velocity_grids import MaxwellSpeedGrid, UniformPitchAngleGrid
from yancc.species import GlobalMaxwellian
from yancc.yancctools import yancctools, yancctools_io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_radial_scan_io():
    """
    Test radial scan execution and I/O operations.
    - Runs a minimal radial scan (1 point, low resolution).
    - Verifies runid generation.
    - Saves options, inputs, and results.
    - Loads results back.
    - Asserts data integrity.
    """
    
    # Setup temporary directory for test output
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 1. Minimal Options
        options = {
            "output_path": str(temp_path),
            "erho_min": -5000.0,
            "erho_max": 5000.0,
            "erho_num": 5,      # Minimal points for coarse scan
            "nt": 9,            # Minimal resolution
            "nz": 9,
            "rtol": 1e-3,
            "num_processors": 1 # Serial execution for test
        }
        
        # 2. Fake Profiles (Simple Parabolic)
        # Using W7-X like parameters scaled down or simple
        def density(r): return 1e20 * (1 - r**2)
        def temp_e(r): return 3e3 * (1 - r**2)
        def temp_i(r): return 1e3 * (1 - r**2)
        
        global_species = [
            GlobalMaxwellian(yancc.species.Electron, temperature=temp_e, density=density),
            GlobalMaxwellian(yancc.species.Hydrogen, temperature=temp_i, density=density)
        ]
        
        # 3. Equilibrium
        # Using DESC W7-X example (assuming cached/available as per notebook)
        try:
            eq = desc.examples.get("W7-X")
        except Exception as e:
            pytest.skip(f"Skipping test: Could not load W7-X equilibrium: {e}")
            return

        # 4. Minimal Grids
        rho_grid = np.array([0.5])
        speedgrid = MaxwellSpeedGrid(nx=5)
        pitchgrid = UniformPitchAngleGrid(nxi=17)
        
        # 5. Unique Test RunID
        runid = f"ai_test_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"Starting test run: {runid}")
        
        # 6. Execute Scan
        results = yancctools.scan_ambipolar_profile(
            rho_grid,
            eq_type="desc",
            eq_data=eq,
            pitchgrid=pitchgrid,
            speedgrid=speedgrid,
            global_species=global_species,
            options=options,
            runid=runid
        )
        
        # Assertions on RunID
        assert results['runid'] == runid
        assert '0.50' in results
        
        # 7. I/O Validation
        run_dir = temp_path / runid
        
        # Save
        yancctools_io.save_options(options, run_dir)
        
        inputs = {
            "rho_grid": rho_grid,
            "test_key": "test_value"
        }
        yancctools_io.save_inputs(inputs, run_dir)
        yancctools_io.save_results(results, run_dir)
        
        # Verify Files Exist
        assert (run_dir / "options.json").exists()
        assert (run_dir / "options.yaml").exists()
        assert (run_dir / "inputs.h5").exists()
        assert (run_dir / "results.h5").exists()
        
        # Load Back
        loaded_options = yancctools_io.load_options(run_dir / "options.json")
        loaded_inputs = yancctools_io.load_inputs(run_dir / "inputs.h5")
        loaded_results = yancctools_io.load_results(run_dir / "results.h5")
        
        # Verify Content
        assert loaded_options['erho_num'] == 5
        assert loaded_inputs['test_key'] == "test_value"
        assert loaded_results['runid'] == runid
        assert '0.50' in loaded_results
        
        # Verify numerical data (approx)
        # Check that erho_grid in loaded results matches original
        np.testing.assert_allclose(
            loaded_results['0.50']['erho_grid'], 
            results['0.50']['erho_grid']
        )
        
        logger.info("Test passed successfully.")

if __name__ == "__main__":
    test_radial_scan_io()
