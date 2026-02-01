import numpy as np
import supervision as sv

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

_LABEL_ANNOTATOR = None   # cached label annotator, resolution is constant
_LABEL_TEXT_SCALE = None


def annotate(image: np.ndarray, detections: sv.Detections, text = None) -> np.ndarray:

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
    global _LABEL_ANNOTATOR, _LABEL_TEXT_SCALE

    # draw masks
    annotated_image = image.copy()
    annotated_image = mask_annotator.annotate(annotated_image, detections)

    # if no masks provided, return mask-only visualization
    if not text:
      return  annotated_image

    # initialize label annotator once (resolution is constant)
    if _LABEL_ANNOTATOR is None:
      h, w = image.shape[:2]
      _LABEL_TEXT_SCALE = sv.calculate_optimal_text_scale(resolution_wh=(w, h))

      _LABEL_ANNOTATOR = sv.LabelAnnotator(
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
    annotated_image = _LABEL_ANNOTATOR.annotate(annotated_image, detections, labels)

    return annotated_image