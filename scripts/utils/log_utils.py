from scripts.utils.config import LOGS_DIR
from datetime import datetime
import os
import json
"""
Utility functions for logging various outputs during TTC delay data preprocessing.

Includes:
- Logging unique stations by category (passenger, non-passenger, unknown)
- Logging station names with directional phrases (to, toward, towards)
"""

def write_log(log_lines:list, prefix: str, log_dir = LOGS_DIR) -> str:
    """
    Writes log and saves to disk
    :param log_lines: list of log lines
    :param prefix: prefix of log file name, e.g. frame extraction
    :param log_dir: log directory
    :return: log_path: absolute path of log
    """
    date_str = datetime.now().strftime('%Y-%m-%d')
    date_folder = os.path.join(log_dir, date_str)
    os.makedirs(date_folder, exist_ok=True)  # Create folder if it doesn't exist

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_folder = os.path.join(date_folder, prefix)
    os.makedirs(log_folder, exist_ok=True)
    log_path = os.path.join(log_dir, f'{prefix}_{timestamp}.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        for line in log_lines:
            f.write(line + '\n')
    return log_path


def write_json_log(data: dict, log_name: str, log_dir = LOGS_DIR, run_timestamp: str = None) -> str:
    """
    Appends structured data as JSON to a JSONL log file.
    Each entry is on its own line for easy parsing.

    :param data: dictionary of data to log
    :param log_name: name of the log file (without extension), e.g. 'frame_extraction'
    :param log_dir: log directory
    :param run_timestamp: optional timestamp for the run (if None, generates new timestamp)
    :return: log_path: absolute path of log file
    """
    log_folder = os.path.join(log_dir, log_name)
    os.makedirs(log_folder, exist_ok=True)
    date_str = datetime.now().strftime('%Y-%m-%d')
    date_folder = os.path.join(log_folder, date_str)
    os.makedirs(date_folder, exist_ok=True)  # Create folder if it doesn't exist

    # Use provided timestamp or generate new one
    if run_timestamp is None:
        run_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Add timestamp to the data
    data['timestamp'] = datetime.now().isoformat()

    # Use JSONL format (one JSON object per line)
    log_path = os.path.join(date_folder, f'log_{run_timestamp}.jsonl')

    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data) + '\n')

    return log_path