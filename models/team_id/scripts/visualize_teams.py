import numpy as np
import supervision as sv
import torch

from models.team_id.scripts.team_features import all_rle_to_masks
from scripts.utils.config import load_paths, PROJECT_ROOT

PATHS = load_paths()
from models.sam3.scripts.visualization_utils import sam_to_sv_rle

TEAM_COLOR = sv.ColorPalette.from_hex([
    "#1f77b4",  # team 0
    "#d62728",  # team 1
])

team_mask_annotator = sv.MaskAnnotator(
    color=TEAM_COLOR,
    color_lookup=sv.ColorLookup.CLASS,  # color by detections.class_id (team)
    opacity=0.6
)

_TEAM_LABEL_ANNOTATOR = None
_TEAM_LABEL_TEXT_SCALE = None


def annotate_teams(image: np.ndarray, detections: sv.Detections, text="player") -> np.ndarray:
    """
    Masks colored by team (detections.class_id), labels show tracker_id.
    """
    global _TEAM_LABEL_ANNOTATOR, _TEAM_LABEL_TEXT_SCALE

    annotated = image.copy()

    # masks: use TEAM colors based on class_id
    annotated = team_mask_annotator.annotate(annotated, detections)

    if not text:
        return annotated

    if _TEAM_LABEL_ANNOTATOR is None:
        h, w = image.shape[:2]
        _TEAM_LABEL_TEXT_SCALE = sv.calculate_optimal_text_scale(resolution_wh=(w, h))
        _TEAM_LABEL_ANNOTATOR = sv.LabelAnnotator(
            text_scale=_TEAM_LABEL_TEXT_SCALE,
            text_color=sv.Color.BLACK,
            text_position=sv.Position.TOP_CENTER,
            text_offset=(0, -30),
        )

    # label text uses tracker_id, but we can also include team if you want
    team_ids = detections.class_id.astype(int)
    labels = [
        f"#{tid} {text} T{team}"
        for tid, team in zip(detections.tracker_id, team_ids)
    ]

    annotated = _TEAM_LABEL_ANNOTATOR.annotate(annotated, detections, labels)
    return annotated


def make_callback_teams(frame_outputs, text:str):
  """
    Create a video callback that annotates frames using SAM3 outputs

    :param frame_outputs: Dictionary mapping frame index to SAM3 outputs (on CPU)
    :param text: Label text (e.g., "hockey player", "puck")
    :return: Callback function compatible with sv.process_video
  """
  def callback(frame: np.ndarray, index: int) -> np.ndarray:
    """
    Apply SAM3 annotations to a given frame and returns the annotated frame for rendering.
    :param frame: video frame
    :param index: frame index
    """
    img = frame.copy()

    output = frame_outputs.get(index)
    if output is None:
        return img
    # decode RLE masks
    rle_list = output["out_masks_rle"]
    masks = all_rle_to_masks(rle_list)
    output["decoded_masks"] = masks
    detections = sam_to_sv_rle(output)
    detections.class_id = np.asarray(output["team_ids"], dtype=int)
    return annotate_teams(img, detections, text)
  return callback


if __name__ == "__main__":
    game_name = "shl_hv71_v_tik"

    output_dir = PROJECT_ROOT / PATHS["TEAM_ID_OUTPUTS_DIR"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{game_name}_w_team_ids.pt"
    players_results = torch.load(output_path, map_location="cpu", weights_only=False)

    SOURCE_VIDEO = PROJECT_ROOT / PATHS["RAW_VIDEOS_DIR"] / f"{game_name}.mp4"
    TARGET_VIDEO = output_dir / f"{game_name}_w_team_ids.mp4"

    callback = make_callback_teams(players_results, "hockey player")
    sv.process_video(
        source_path=str(SOURCE_VIDEO),
        target_path=str(TARGET_VIDEO),
        callback=callback,
    )
