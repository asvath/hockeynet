import numpy as np
from pathlib import Path
"""
Utility functions for annotations
"""

def load_annotation(path:Path) -> list:
    """
    Reads YOLO format annotation file
    :param path: path to annotation file
    :return: list of annotations
    """
    annotations = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:5])
                annotations.append(
                    {'class_id': class_id,
                     'x': x,
                     'y': y,
                     'w': w,
                     'h': h
                     })

    return annotations

def validate_annotation(annotation:list)->bool:
    """
    Checks if the annotation is valid:
     - are the coordinates of the bounding boxes in valid range [0-1]
     - are the class ids valid (0-6)
    :param annotation: list with annotations
    :return: boolean
    """
    for ann in annotation:
        if not( 0<= ann['x'] <= 1 and 0<= ann['y'] <= 1 and 0<= ann['w'] <= 1 and 0<= ann['h'] <= 1):
            return False
        if not (0 <= ann['class_id'] <= 6):
            return False
    return True

def yolo_to_absolute(x:float, y:float, w: float, h:float, img_width:int, img_height:int)->tuple:
    """
    Converts a YOLO formatted annotation into absolute, pixel corner coordinate
    :param x: normalized x coordinate of bounding box
    :param y: normalized y coordinate of bounding box
    :param w: normalized width of bounding box
    :param h: normalized height of bounding box
    :param img_width: image width
    :param img_height: image height
    :return: tuple of bounding box corner coordinates, absolute width and absolute height
    """
    x_min = (x - w/2) * img_width
    x_max = (x + w/2) * img_width
    y_min = (y - h/2) * img_height
    y_max = (y + h/2) * img_height
    w_abs = w * img_width
    h_abs = h * img_height

    return x_min, y_min, x_max, y_max, w_abs, h_abs

def yolo_to_absolute_viz(x:float, y:float, w: float, h:float, img_width:int, img_height:int)->tuple:

    """
    Converts a YOLO formatted annotation into absolute, pixel corner coordinates for visualization
    :param x: normalized x coordinate of bounding box
    :param y: normalized y coordinate of bounding box
    :param w: normalized width of bounding box
    :param h: normalized height of bounding box
    :param img_width: image width
    :param img_height: image height
    :return: tuple of bounding box corner coordinates
    """
    x_min, y_min, x_max, y_max, _, _ = yolo_to_absolute(x, y, w, h, img_width, img_height)
    x_min = np.floor(x_min).astype(int)
    x_max = np.ceil(x_max).astype(int)
    y_min = np.floor(y_min).astype(int)
    y_max = np.ceil(y_max).astype(int)

    return x_min, y_min, x_max, y_max

def get_annotation_stats(annotations: list)-> dict:
    """
    Get annotation statistics for a single frame
    :param annotations:
    :return: dict with statistics
    """

    total_annotations = len(annotations)
    class_counts = {}
    width = []
    height = []
    for ann in annotations:
        class_counts[ann['class_id']] = class_counts.get(ann['class_id'], 0) + 1
        width.append(ann['w'])
        height.append(ann['h'])

    return {'total_annotations': total_annotations,
            'class_counts': class_counts,
            'avg_bbox_width': sum(width) / total_annotations,
            'avg_bbox_height': sum(height) / total_annotations
            }

def load_all_annotations(annotations_dir: Path)-> dict:
    """
    Loads all annotation files from a directory
    :param annotations_dir: directory with annotations
    :return: dict with filename as key and annotations as value
    """
    dataset_annotations = {}
    files = list(annotations_dir.iterdir())
    for file in files:
        dataset_annotations[file.stem] = load_annotation(file)


    return dataset_annotations


def yolo_to_coco_bbox(x, y, w, h, img_width, img_height) -> list:
    """
    Converts a YOLO formatted bbox into coco bbox
    :param x: normalized x coordinate of bounding box
    :param y: normalized y coordinate of bounding box
    :param w: normalized width of bounding box
    :param h: normalized height of bounding box
    :param img_width: image width
    :param img_height: image height
    :return: list [x_min, y_min, width, height] in absolute pixel coordinates (COCO format)
    """
    x_min, y_min, _, _, w_abs, h_abs = yolo_to_absolute(x, y, w, h, img_width, img_height)

    return [x_min, y_min, w_abs, h_abs]


def convert_to_coco(annotations:list)->dict:
    """
    Converts a YOLO formatted annotation into COCO format
    :param annotations:
    :return:
    """
    pass