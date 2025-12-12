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
    config_path = Path(__file__).parent.parent.parent / "config" / "dataset_metadata.json"
    with open(config_path, 'r') as f:
        return json.load(f)


# Get project root directory (3 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load paths once at import time
PATHS = load_paths()

# Convert to absolute paths relative to project root
RAW_VIDEO_DIR = PROJECT_ROOT / PATHS["RAW_VIDEO_DIR"]
RAW_ANNOTATIONS_DIR = PROJECT_ROOT / PATHS["RAW_ANNOTATIONS_DIR"]
PROCESSED_FRAMES_DIR = PROJECT_ROOT / PATHS["PROCESSED_FRAMES_DIR"]
PROCESSED_ANNOTATIONS_DIR = PROJECT_ROOT / PATHS["PROCESSED_ANNOTATIONS_DIR"]
