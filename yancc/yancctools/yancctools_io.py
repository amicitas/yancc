# -*- coding: utf-8 -*-
"""
.. Authors
     Novimir Pablant <npablant@pppl.gov>

I/O tools for yancctools.
Handles saving/loading of options, inputs, and results.

This file includes AI generated code (Gemini)
"""

import json
import yaml
import logging
from pathlib import Path
from . import mirhdf5

logger = logging.getLogger(__name__)

def save_options(options: dict, directory: str, filename: str = 'options'):
    """
    Save options dictionary to both JSON and YAML files in the specified directory.
    
    Parameters
    ----------
    options : dict
        The options dictionary to save.
    directory : str
        The directory where files should be saved.
    filename : str, optional
        The base filename (without extension). Default is 'options'.
    """
    path = Path(directory)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        
    # Save as JSON
    json_path = path / f"{filename}.json"
    with open(json_path, 'w') as f:
        json.dump(options, f, indent=4)
    logger.info(f"Saved options to {json_path}")
    
    # Save as YAML
    yaml_path = path / f"{filename}.yaml"
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(options, f, default_flow_style=False)
    logger.info(f"Saved options to {yaml_path}")


def load_options(file_path: str) -> dict:
    """
    Load options from a JSON or YAML file.
    
    Parameters
    ----------
    file_path : str
        Path to the options file (.json or .yaml/.yml).
        
    Returns
    -------
    dict
        The loaded options dictionary.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Options file not found: {file_path}")
    
    with open(path, 'r') as f:
        if path.suffix in ('.yaml', '.yml'):
            return yaml.safe_load(f)
        elif path.suffix == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}. Use .json or .yaml")


def save_inputs(inputs: dict, directory: str, filename: str = 'inputs.h5'):
    """
    Save input data (profiles, equilibrium, etc.) to an HDF5 file using mirhdf5.
    
    Parameters
    ----------
    inputs : dict
        Dictionary containing input data to save.
    directory : str
        The directory where the file should be saved.
    filename : str, optional
        The filename (including extension). Default is 'inputs.h5'.
    """
    path = Path(directory)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        
    file_path = path / filename
    mirhdf5.dictToHdf5(inputs, str(file_path))
    logger.info(f"Saved inputs to {file_path}")


def load_inputs(file_path: str) -> dict:
    """
    Load input data from an HDF5 file using mirhdf5.
    
    Parameters
    ----------
    file_path : str
        Path to the HDF5 file.
        
    Returns
    -------
    dict
        The loaded inputs dictionary.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Inputs file not found: {file_path}")
        
    return mirhdf5.hdf5ToDict(str(path))


def save_results(results: dict, directory: str, filename: str = 'results.h5'):
    """
    Save simulation results to an HDF5 file using mirhdf5.
    
    Parameters
    ----------
    results : dict
        Dictionary containing simulation results.
    directory : str
        The directory where the file should be saved.
    filename : str, optional
        The filename (including extension). Default is 'results.h5'.
    """
    path = Path(directory)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        
    file_path = path / filename
    mirhdf5.dictToHdf5(results, str(file_path))
    logger.info(f"Saved results to {file_path}")


def load_results(file_path: str) -> dict:
    """
    Load simulation results from an HDF5 file using mirhdf5.
    
    Parameters
    ----------
    file_path : str
        Path to the HDF5 file.
        
    Returns
    -------
    dict
        The loaded results dictionary.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {file_path}")
        
    return mirhdf5.hdf5ToDict(str(path))
