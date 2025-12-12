"""
Utils for video processing
"""
from pathlib import Path
import cv2


def extract_frames_from_video(video_path: str, output_path: str, fps = None) -> None:
    """
    Extract frames from a video
    :param video_path: path containing raw video file
    :param output_path: path where frames will be saved
    :param fps: frames per second (if None, extracts all frames)
    :return: None
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Get video name without extension
    video_name = video_path.stem

    # Open the video
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame interval
    if fps is None: # save every frame
        frame_interval = 1
    else: # calculate how many frames to skip
        frame_interval = int(video_fps / fps)
        if frame_interval < 1:
            frame_interval = 1

    frame_count = 0
    saved_count = 0

    while True:
        # read frane from video source
        # ret: bool, True if successful
        # frame is the image data as a np array
        ret, frame = cap.read()

        if not ret:
            break

        # Save frame at specified interval
        if frame_count % frame_interval == 0: # checks if current frame number is divisible by the interval
            # Format frame number as 4-digit string (e.g., 0001, 0002, ...)
            frame_filename = f"{video_name}_{saved_count:04d}.jpg"
            frame_path = output_path / frame_filename
            cv2.imwrite(str(frame_path), frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames from {video_name} to {output_path}")


def rename_video(game_name:str, video_path:Path) -> None:
    """
    Rename video file to: game_name_clip_number, e.g. allstar_2019_001
    :param game_name: name of the game
    :param video_path: path to video file
    """

    for clip in video_path.iterdir():
        if clip.is_file() and game_name not in clip.stem:
            new_name = clip.with_name(f"{game_name}_{clip.stem}{clip.suffix}")
            clip.rename(new_name)

def standardize_all_video_names(video_dir:Path) -> None:
    """
    Standardize names of all video files
    :param video_dir: directory containing all video files
    """
    for game_dir in video_dir.iterdir():
        game_name = game_dir.name
        rename_video(game_name, game_dir)

