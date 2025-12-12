"""
Utils for video processing
"""
import json
from pathlib import Path
def extract_frames_from_video(video_path: str, output_path: str, fps = None) -> None:
    """
    Extract frames from a video
    :param video_path: path containing raw video file
    :param output_path: path where frames will be saved
    :param fps: frames per second
    :return: None
    """
    pass

def load_dataset_metadata() -> dict:
    config_path = Path(__file__).parent.parent.parent / "config" / "dataset_metadata.json"
    with open(config_path, 'r') as f:
        return json.load(f)


