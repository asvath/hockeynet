import json
import sys
from pathlib import Path

import numpy as np
import supervision as sv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
from PIL import Image


def sam_to_sv(result: dict) -> sv.Detections:
  """
  Convert SAM3 output dictionary into a Supervision Detections object.
  Bounding boxes are derived from the segmentation masks rather than using
  detector-provided boxes, enabling more precise, mask-centric tracking.

  :param result: Dictionary containing SAM3 outputs for a single frame.
  :return: A Supervision Detections object containing bounding boxes, masks,
             confidence scores, and tracker IDs.
  """
  return sv.Detections(
      xyxy=sv.mask_to_xyxy(result["out_binary_masks"]),
      mask=result["out_binary_masks"],
      confidence=result["out_probs"],
      tracker_id=result["out_obj_ids"],
  )

COLOR = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
])

mask_annotator = sv.MaskAnnotator(
    color=COLOR,
    color_lookup=sv.ColorLookup.TRACK,
    opacity=0.6
)

_SINGLE_CLASS_LABEL_ANNOTATOR = None   # cached label annotator for single-class mode
_LABEL_TEXT_SCALE = None  # shared text scale (resolution is constant)


def annotate_single_class(image: np.ndarray, detections: sv.Detections, text = None) -> np.ndarray:

    """
    Annotate an image with segmentation masks and optional text labels.
    Labels are formatted as: "#<tracker_id> <text>".

    :param image: Input frame as a NumPy array of shape (H, W, C).
    :param detections: Supervision Detections object containing masks and
                       optional tracker IDs.
    :param text: Optional label suffix (e.g., "player", "puck"). If None,
                 only masks are drawn.
    :return: Annotated image with masks and optional labels.
    """
    global _SINGLE_CLASS_LABEL_ANNOTATOR, _LABEL_TEXT_SCALE

    # draw masks
    annotated_image = image.copy()
    annotated_image = mask_annotator.annotate(annotated_image, detections)

    # if no masks provided, return mask-only visualization
    if not text:
      return  annotated_image

    # initialize label annotator once (resolution is constant)
    if _SINGLE_CLASS_LABEL_ANNOTATOR is None:
      h, w = image.shape[:2]
      _LABEL_TEXT_SCALE = sv.calculate_optimal_text_scale(resolution_wh=(w, h))

      _SINGLE_CLASS_LABEL_ANNOTATOR = sv.LabelAnnotator(
          color=COLOR,
          color_lookup=sv.ColorLookup.TRACK,
          text_scale=_LABEL_TEXT_SCALE,
          text_color=sv.Color.BLACK,
          text_position=sv.Position.TOP_CENTER,
          text_offset=(0, -30),
      )

    # build per-object labels
    labels = [
        f"#{tracker_id} {text}"
        for tracker_id in detections.tracker_id
    ]

    # draw labels
    annotated_image = _SINGLE_CLASS_LABEL_ANNOTATOR.annotate(annotated_image, detections, labels)

    return annotated_image


def draw_xywh_bboxes(image_path: Path, prompts: dict, save_path: Path = None):
    """
    Draw xywh bounding boxes from SAM3 prompts format on an image.
    :param image_path: path to image
    :param prompts: dict with keys like 'puck_bboxes', 'faceoff_boxes', etc.
                    Each value is a list of [x, y, w, h] boxes
    :param save_path: optional path to save the figure
    :return: matplotlib figure
    """
    img = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)

    # Color map for each box type
    color_map = {
        'puck_bboxes': 'lime',
        'faceoff_boxes': 'red',
        'stick_bboxes': 'blue',
        'skate_bboxes': 'cyan',
        'blade_bboxes': 'magenta',
    }

    for box_type, boxes in prompts.items():
        if box_type == 'bbox_labels':
            continue
        color = color_map.get(box_type, 'yellow')
        for bbox in boxes:
            x, y, w, h = bbox
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x, y - 5, box_type.replace('_bboxes', '').replace('_boxes', ''),
                    bbox=dict(facecolor=color, alpha=0.5),
                    fontsize=8, color='black')

    # Legend
    legend_patches = [Patch(facecolor=c, label=k.replace('_bboxes', '').replace('_boxes', ''))
                      for k, c in color_map.items() if k in prompts]
    ax.legend(handles=legend_patches, loc='upper right')

    ax.axis('off')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig

# define colours for the object classes
# one colour per class
CLASS_COLOURS = {
    "hockey player": "#3399ff",   # blue
    "goalie": "#ff9b00",          # orange
    "referee": "#ffff00",         # yellow
    "hockey puck": "#ff0000",     # red
}

# cache annotators per class (multi-class mode)
_MULTI_CLASS_MASK_ANNOTATORS = {}
_MULTI_CLASS_LABEL_ANNOTATORS = {}

def get_annotators_for_class(class_name:str, image_shape: tuple[int, int, int]) -> tuple[sv.MaskAnnotator, sv.LabelAnnotator]:
  """
  Retrieve cached or create supervision annotators using class-specific colours for a given object class.
  The text scale for labels is computed once based on the video resolution and reused across all classes.

  :param class_name: Name of the object class (e.g., "hockey player", "puck").
  :param image_shape: Shape of the input image (H, W, C), used to determine
                  the video resolution for text scaling.
  :return: Tuple containing:
        - MaskAnnotator for drawing segmentation masks
        - LabelAnnotator for drawing text labels
  """
  global _LABEL_TEXT_SCALE

  # if annotators for class_name has already been created, reuse them
  if class_name in _MULTI_CLASS_MASK_ANNOTATORS:
      return _MULTI_CLASS_MASK_ANNOTATORS[class_name], _MULTI_CLASS_LABEL_ANNOTATORS[class_name]

  # create palette with single color
  colour = CLASS_COLOURS.get(class_name, "#ffffff") # default white

  palette = sv.ColorPalette.from_hex([colour])

  # create mask annotator: draws segmentation masks
  class_mask_annotator = sv.MaskAnnotator(
      color=palette,
      color_lookup=sv.ColorLookup.INDEX,  # choose colour based on class_id or NONE
      opacity=0.6
  )

  h, w = image_shape[:2]

  # initialize text scale once as resolution is constant
  if _LABEL_TEXT_SCALE is None:
      _LABEL_TEXT_SCALE = sv.calculate_optimal_text_scale(
          resolution_wh=(w, h)
      )

  # create the label annotator for this class; draws text labels near detections
  class_label_annotator = sv.LabelAnnotator(
      color=palette,
      color_lookup=sv.ColorLookup.INDEX,
      text_scale=_LABEL_TEXT_SCALE,
      text_color=sv.Color.BLACK,
      text_position=sv.Position.TOP_CENTER,
      text_offset=(0, -30),
  )

  _MULTI_CLASS_MASK_ANNOTATORS[class_name] = class_mask_annotator
  _MULTI_CLASS_LABEL_ANNOTATORS[class_name] = class_label_annotator

  return class_mask_annotator, class_label_annotator

def annotate_multi_class(image: np.ndarray, detections: sv.Detections, class_name=None) -> np.ndarray:
  """
  Annotate a video frame with segmentation masks and labels for a given class.
  Uses class-specific colors defined in CLASS_COLOURS.

  :param image: input video frame as a NumPy array (H, W, C).
  :param detections: detections for a given class
  :param class_name: name of object class (e.g, hockey player)
  :return: Annotated image with masks and labels drawn
  """

  annotated_image = image.copy()

  # if no class_name provided, return non-annotated image
  if not class_name:
      return annotated_image

  # get annotators for given class
  mask_annotator, label_annotator = get_annotators_for_class(
      class_name, image.shape
  )

  # Draw masks
  annotated_image = mask_annotator.annotate(
      annotated_image, detections
  )

  # Build labels for each object (e.g, #16 hockey player)
  labels = [
      f"#{tid} {class_name}"
      for tid in detections.tracker_id
  ]

  # Draw labels
  annotated_image = label_annotator.annotate(
      annotated_image, detections, labels
  )

  return annotated_image

def make_callback_multi(results_cpu: dict[str, dict[int, object]]):
    """
    Create a video callback that annotates frames using SAM3 outputs for all classes
    :param results_cpu: { class_name -> { frame_index -> SAM3_output } }
    :return results_cpu: dict[str, dict[int, object]]
    """
    def callback(frame: np.ndarray, index: int) -> np.ndarray:
      """
      Apply SAM3 annotations for all classes to a given frame and returns the annotated frame for rendering.
      :param frame: video frame as Numpy array (H,W,C)
      :param index: frame index
      :param text: label text (e.g, "hockey player")
      :return annotated frame for rendering
      """
      img = frame.copy()

      for class_name, per_class_outputs in results_cpu.items():
          output = per_class_outputs.get(index)
          if output is None:
              continue
          detections = sam_to_sv(output)
          img = annotate_multi_class(img, detections, class_name)  # important: feed back img

      return img

    return callback


if __name__ == "__main__":
    # Example usage
    # Add project root to path for config import
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.utils.config import PATHS
    prompts_dir = PROJECT_ROOT / PATHS["SAM3_PROMPTS_DIR"]
    prompts_path = Path(prompts_dir/"frame0_prompts.json")
    image_path = Path(prompts_dir/"frame0.jpg")

    with open(prompts_path, "r") as f:
        prompts = json.load(f)

    fig = draw_xywh_bboxes(image_path, prompts)
    plt.show()

