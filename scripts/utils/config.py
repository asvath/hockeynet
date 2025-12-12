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


# Load paths once at import time
PATHS = load_paths()

# individual paths
RAW_VIDEO_DIR = PATHS["RAW_VIDEO_DIR"]
RAW_ANNOTATIONS_DIR = PATHS["RAW_ANNOTATIONS_DIR"]
PROCESSED_FRAMES_DIR = PATHS["PROCESSED_FRAMES_DIR"]
PROCESSED_ANNOTATIONS_DIR = PATHS["PROCESSED_ANNOTATIONS_DIR"]
