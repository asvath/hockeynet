"""
Configuration utilities for loading project settings
"""
import json
from pathlib import Path


def load_paths() -> dict:
    """
    Load the directory paths configuration
    :return: dict containing directory paths
    """
    config_path = Path(__file__).parent.parent.parent / "config" / "paths.json"
    with open(config_path, 'r') as f:
        return json.load(f)


def load_dataset_metadata() -> dict:
    """
    Load the dataset metadata
    :return: dict containing dataset metadata
    """
    metadata_path =  PROJECT_ROOT / PATHS["DATASET_METADATA"]
    with open(metadata_path, 'r') as f:
        return json.load(f)


# Get project root directory (3 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load paths once at import time
PATHS = load_paths()
