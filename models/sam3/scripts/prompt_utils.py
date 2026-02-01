"""
Utility functions for SAM3 prompt generation
"""
from typing import Tuple, List


def box_to_xywh(box: dict) -> Tuple[float, float, float, float]:
    """
    Convert a box dictionary to xywh format
    :param box: dict with 'x', 'y', 'width', 'height' keys (values may be strings)
    :return: tuple of (x, y, width, height) as floats
    """
    return (
        float(box["x"]),
        float(box["y"]),
        float(box["width"]),
        float(box["height"])
    )


def xywh_to_xyxy(x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
    """
    Convert xywh format to xyxy format
    :param x: x coordinate of top-left corner
    :param y: y coordinate of top-left corner
    :param w: width
    :param h: height
    :return: tuple of (x1, y1, x2, y2) representing top-left and bottom-right corners
    """
    return x, y, x + w, y + h


def xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
    """
    Convert xyxy format to xywh format
    :param x1: x coordinate of top-left corner
    :param y1: y coordinate of top-left corner
    :param x2: x coordinate of bottom-right corner
    :param y2: y coordinate of bottom-right corner
    :return: tuple of (x, y, width, height)
    """
    return x1, y1, x2 - x1, y2 - y1


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """
    Calculate Intersection over Union (IoU) for two boxes in xyxy format
    :param a: first box as (x1, y1, x2, y2)
    :param b: second box as (x1, y1, x2, y2)
    :return: IoU score between 0.0 and 1.0
    """
    ax1, ay1, ax2, ay2 = a # (x1, y1) : top-left corner, (x2, y2) : bottom-right corner
    bx1, by1, bx2, by2 = b
    # find intersection rectangle
    ix1, iy1 = max(ax1, bx1), max(ay1, by1) # left edge of overlap
    ix2, iy2 = min(ax2, bx2), min(ay2, by2) # right edge of overlap
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    if ax2 < ax1 or ay2 < ay1 or bx2 < bx1 or by2 < by1:
        print("Invalid box:", a, b)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center_xywh(x: float, y: float, w: float, h: float) -> Tuple[float, float]:
    """
    Calculate center point of a box in xywh format
    :param x: x coordinate of top-left corner
    :param y: y coordinate of top-left corner
    :param w: width
    :param h: height
    :return: tuple of (center_x, center_y)
    """
    return x + w / 2, y + h / 2


def closest_box(target_xywh: Tuple[float, float, float, float],
                candidates_xywh: List[Tuple[float, float, float, float]]) -> int:
    """
    Find the index of the closest box to a target based on center distance
    :param target_xywh: target box as (x, y, w, h)
    :param candidates_xywh: list of candidate boxes as (x, y, w, h)
    :return: index of the closest candidate, or None if candidates is empty
    """
    tx, ty, tw, th = target_xywh
    tcx, tcy = center_xywh(tx, ty, tw, th)
    best_i, best_d = None, float("inf")
    for i, (x, y, w, h) in enumerate(candidates_xywh):
        cx, cy = center_xywh(x, y, w, h)
        d = (cx - tcx) ** 2 + (cy - tcy) ** 2
        if d < best_d:
            best_d, best_i = d, i
    return best_i


def pad_xywh(box_xywh: Tuple[float, float, float, float],pad: float = 8.0) -> Tuple[float, float, float, float]:
    """
    Adds padding to a bbox in xywh format
    :param box_xywh: bbox as (x, y, w, h)
    :param pad: padding to add to bbox
    :return: expanded bbox as (x, y, w, h)
    """
    x, y, w, h = box_xywh
    return x - pad, y - pad, w + 2 * pad, h + 2 * pad
