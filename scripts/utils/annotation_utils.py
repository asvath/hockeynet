import json
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
                category_id = int(parts[0])
                x, y, w, h = map(float, parts[1:5])
                annotations.append(
                    {
                        "category_id": category_id,
                        'x' : x,
                        'y' : y,
                        'w' : w,
                        'h': h,
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
        if not (0 <= ann['category_id'] <= 6):
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
        class_counts[ann['category_id']] = class_counts.get(ann['category_id'], 0) + 1
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


def yolo_to_coco_bbox(x, y, w, h, img_width, img_height) -> tuple:
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

    return x_min, y_min, w_abs, h_abs

def create_coco_annotation_per_bbox(category_id, x:float, y:float, w: float, h:float,
                           img_width:int, img_height:int) -> dict:
    """
    Creates a single COCO annotation
    :param category_id: class id
    :param x: normalized x coordinate of bounding box
    :param y: normalized y coordinate of bounding box
    :param w: normalized width of bounding box
    :param h: normalized height of bounding box
    :param img_width: image width
    :param img_height: image height
    :return: dict with annotation
    """
    x_min, y_min, w_abs, h_abs = yolo_to_coco_bbox(x, y, w, h, img_width, img_height)
    area =  w_abs * h_abs

    return {
        'category_id': category_id,
        'bbox': [x_min, y_min, w_abs, h_abs],
        'area': area,
        'iscrowd': 0,
    }

def create_coco_annotation_per_frame(path:Path, img_width: int, img_height: int) -> list:
    """
    Creates a COCO annotation for the whole frame
    :param path: path to annotation file
    :param img_width: image width
    :param img_height: image height
    :return: list of COCO annotations for the frame
    """

    yolo_annotations = load_annotation(path)
    annotations = []

    for ann in yolo_annotations:
        coco_ann = create_coco_annotation_per_bbox(category_id= ann['category_id'],
                                        x = ann['x'],
                                        y = ann['y'],
                                        w = ann['w'],
                                        h = ann['h'],
                                        img_width=img_width,
                                        img_height=img_height
                                        )
        annotations.append(coco_ann)

    return annotations


def convert_to_coco_format(annotations_dir: Path, dataset_metadata_path: Path, output_path: Path,
                           img_width = 1920, img_height = 1080) -> dict:
    """
    Converts entire YOLO dataset to COCO JSON format and writes it out
    :param annotations_dir: directory with annotations
    :param dataset_metadata_path: path to dataset metadata JSON file
    :param output_path: path to output JSON file
    :param img_width: image width
    :param img_height: image height
    :return: dict containing COCO formatted annotations, image information and categories
    """
    images = []
    annotations = []
    categories = []
    img_id = 0
    ann_id = 0
    ann_files = list(annotations_dir.iterdir())

    for file in ann_files:
        img_id += 1
        images.append({
            "id": img_id,
            "file_name": file.stem + ".jpg",
            "width": img_width,
            "height" : img_height
        })
        coco_annotations = create_coco_annotation_per_frame(file, img_width= img_width, img_height= img_height)
        for ann in coco_annotations:
            ann_id += 1
            ann['id'] = ann_id
            ann['image_id'] = img_id
            annotations.append(ann)

    with open(dataset_metadata_path, 'r') as f:
        metadata = json.load(f)
        for cat, name in metadata["classes"].items():
            categories.append({
                "id": int(cat),
                "name": name
            })
        info = {
            "dataset_name": metadata["dataset_name"],
            "source": metadata["source"],
            "total_frames": metadata["total_frames"],
            "total_annotations": metadata["total_annotations"],
            "image_resolution": metadata["image_resolution"],
            "image_format": "JPG",
        }

    coco_data = {
        "info": info,
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    # Save to file if output_path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=2)

    return coco_data

def load_coco_annotation(coco_json_path: Path) -> dict:
    """
    Load COCO format annotations
    :param coco_json_path: path to coco json file
    :return: dict with COCO formatted annotations
    """
    with open(coco_json_path, "r") as f:
        data = json.load(f)

    return data

def image_lookup(coco_annotation:dict) -> dict:
    """
    Creates dictionary with image name and image data
    :param coco_annotation: dict with COCO annotation
    :return: dictionary with image name and image data
    """
    image_dict = {}
    for img in coco_annotation['images']:
        image_dict[img['file_name']] = img

    return image_dict

def image_id_lookup(coco_annotation:dict) -> dict:
    """
    Creates dictionary with image id and image data
    :param coco_annotation: dict with COCO annotation
    :return: dictionary with image id and image data
    """
    image_id_dict = {}
    for img in coco_annotation['images']:
        image_id_dict[img['id']] = img

    return image_id_dict

def category_lookup(coco_annotation:dict) -> dict:
    """
    Creates dictionary with category id and category name
    :param coco_annotation: dict with COCO annotation
    :return: dictionary with category id and category name
    """
    cat_dict = {}
    for cat in coco_annotation['categories']:
        cat_dict[cat["id"]] = cat["name"]

    return cat_dict

def annotation_lookup(coco_annotation: dict) -> dict:
    """
    Creates dictionary mapping image_id to list of annotations
    :param coco_annotation: dict with COCO annotation
    :return: dictionary with image_id to list of annotations
    """
    annotation_dict = {}
    for ann in coco_annotation['annotations']:
        image_id = ann['image_id']
        if image_id not in annotation_dict:
            annotation_dict[image_id] = []
        annotation_dict[image_id].append(ann)

    return annotation_dict

def coco_to_absolute(x_min_float:float, y_min_float:float, w_float: float, h_float:float)->tuple:
    """
    Converts a coco formatted annotation into absolute, pixel coordinates
    :param x_min_float: normalized x_min coordinate of bounding box
    :param y_min_float: normalized y_min coordinate of bounding box
    :param w_float: normalized width of bounding box
    :param h_float: normalized height of bounding box
    :return: tuple of bounding box corner coordinates, absolute width and absolute height
    """
    x_min = int(np.floor(x_min_float))
    y_min = int(np.floor(y_min_float))
    x_max = int(np.ceil(x_min_float + w_float))
    y_max = int(np.ceil(y_min_float + h_float))

    width = x_max - x_min
    height = y_max - y_min

    return x_min, y_min, x_max, y_max, width, height

def find_images_with_multiples(coco_annotation: dict, category_id: int) -> list:
    """
    Find all images with multiple annotations for a given category (e.g. two pucks in an image)
    :param coco_annotation: dict with COCO annotation
    :param category_id: category id
    :return: list of images with multiple annotations for the given category
    """
    # Find images with multiple annotations for category
    counts = {}
    for ann in coco_annotation['annotations']:
        if ann['category_id'] == category_id:
            img_id = ann['image_id']
            counts[img_id] = counts.get(img_id, 0) + 1

    # Filter images with more than 1 annotation
    multiple_cat = {img_id: count for img_id, count in counts.items() if count > 1}

    image_id_dict = image_id_lookup(coco_annotation)
    multiple_cat_img = []
    for img_id, _ in multiple_cat.items():
        multiple_cat_img.append(image_id_dict[img_id]['file_name'])

    return multiple_cat_img
