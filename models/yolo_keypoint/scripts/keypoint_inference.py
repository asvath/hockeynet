import cv2
import torch

from models.yolo_keypoint.scripts.keypoint_utils import load_model, detect_keypoints
from scripts.utils.config import load_paths, PROJECT_ROOT

PATHS = load_paths()


def run_keypoint_detection(game_name: str, conf_threshold: float = 0.5):
    """
    Run keypoint detection on all frames for a game and save results as .pt file.
    :param game_name: name of the game
    :param conf_threshold: minimum confidence for a keypoint to be included
    """
    model = load_model()
    frames_dir = PROJECT_ROOT / PATHS["PROCESSED_VIDEOS_DIR"] / game_name

    frame_paths = sorted(frames_dir.glob(f"{game_name}_*.jpg"))
    print(f"Found {len(frame_paths)} frames")

    keypoint_results = {}

    for i, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path))
        result = detect_keypoints(model, frame, conf_threshold=conf_threshold)
        keypoint_results[i] = result

        if i % 100 == 0:
            print(f"Frame {i}/{len(frame_paths)} — detected {result['num_detected']} keypoints")

    output_dir = PROJECT_ROOT / PATHS["YOLO_KEYPOINT_OUTPUTS_DIR"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{game_name}_keypoints.pt"
    torch.save(keypoint_results, output_path)
    print(f"Saved {len(keypoint_results)} frames to {output_path}")


if __name__ == "__main__":
    game_name = "shl_hv71_v_tik"
    run_keypoint_detection(game_name, conf_threshold=0.8)