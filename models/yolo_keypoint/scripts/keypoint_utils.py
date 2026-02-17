"""
YOLO Keypoints Utils for Ice Rink Keypoints Detection
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Add project root to path for config import
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.config import PATHS, PROJECT_ROOT


def load_model(weights_path: Path = None) -> YOLO:
    """
    Load YOLO keypoint model.
    :param weights_path: path to weights file, defaults to config path
    :return: YOLO model
    """
    if weights_path is None:
        weights_dir = PROJECT_ROOT / PATHS["YOLO_KEYPOINT_WEIGHTS"]
        weights_path = weights_dir / "HockeyRink.pt"
    return YOLO(weights_path)


def detect_keypoints(model: YOLO, image_path: Path, conf_threshold: float = 0.8) -> dict:
    """
    Run keypoint detection on a single frame.
    :param model: loaded YOLO model
    :param image_path: path to image
    :param conf_threshold: minimum confidence for keypoint
    :return: dict with keypoint detections
    """
    results = model(image_path, verbose=False)

    if len(results) == 0 or results[0].keypoints is None:
        return {"keypoints": {}, "num_detected": 0}

    # Get keypoints: shape (num_detections, num_keypoints, 3) -> [x, y, conf]
    kpts_data = results[0].keypoints.data.cpu().numpy()

    # If no detections
    if len(kpts_data) == 0:
        return {"keypoints": {}, "num_detected": 0}

    kpts = kpts_data[0]  # shape: (56, 3)

    # Filter by confidence and build output dict
    detected = {}
    for i, (x, y, conf) in enumerate(kpts):
        if conf >= conf_threshold:
            detected[f"KP-{i}"] = {
                "x": float(x),
                "y": float(y),
                "confidence": float(conf)
            }

    return {
        "keypoints": detected,
        "num_detected": len(detected),
        "total_keypoints": len(kpts)
    }


def visualize_keypoints(frame: np.ndarray, detection: dict, radius: int = 5, color: tuple = (0, 255, 0)):
    """
    Display frame with detected keypoints drawn as circles with labels.
    :param frame: BGR image
    :param detection: output from detect_keypoints
    :param radius: circle radius in pixels
    :param color: BGR color for circles and text
    """
    vis = frame.copy()
    for name, data in detection["keypoints"].items():
        x, y = int(data["x"]), int(data["y"])
        cv2.circle(vis, (x, y), radius, color, -1)
        cv2.putText(vis, name, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.imshow("Keypoint Detection", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_keypoint_array(detection: dict) -> tuple[np.ndarray, list[str]]:
    """
    Convert detection dict to numpy array for homography.
    :param detection: output from detect_keypoints
    :return: (Nx2 array of pixel coords, list of keypoint names)
    """
    kpts = detection["keypoints"]
    names = list(kpts.keys())
    coords = np.array([[kpts[n]["x"], kpts[n]["y"]] for n in names])
    return coords, names


if __name__ == "__main__":
    game_name = "shl_hv71_v_tik"
    # Example usage
    model = load_model()

    i = 0
    # Test on a frame
    frame = cv2.imread(str(
        PROJECT_ROOT / PATHS["PROCESSED_VIDEOS_DIR"] / game_name / f"{game_name}_{i:04d}.jpg"
    ))

    result = detect_keypoints(model, frame, conf_threshold=0.8)
    print(f"Detected {result['num_detected']} / {result['total_keypoints']} keypoints")

    for name, data in result["keypoints"].items():
        print(f"  {name}: ({data['x']:.1f}, {data['y']:.1f}) conf={data['confidence']:.2f}")

    visualize_keypoints(frame, result)
