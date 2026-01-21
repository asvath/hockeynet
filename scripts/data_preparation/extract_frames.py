from scripts.utils.video_utils import extract_frames_from_video
from scripts.utils.config import load_paths, PROJECT_ROOT

PATHS = load_paths()
RAW_VIDEOS = PROJECT_ROOT / PATHS["RAW_VIDEOS_DIR"]
PROCESSED_VIDEOS = PROJECT_ROOT / PATHS["PROCESSED_VIDEOS_DIR"]

GAME_NAMES = ['shl_hv71_v_tik', 'pwhl_sea_v_tor']
EXTENSION = ".mp4"

def extract_frames():
    for game_name in GAME_NAMES:
        input_path = RAW_VIDEOS / f"{game_name}{EXTENSION}"
        output_path = PROCESSED_VIDEOS / game_name

        if not input_path.exists():
            print(f"Warning: {input_path} not found, skipping")
            continue

        log = extract_frames_from_video(input_path, output_path)
        print(log)


if __name__ == "__main__":
    extract_frames()