"""
Utility functions for visualization
"""
from pathlib import Path
from annotation_utils import load_coco_annotation


def image_lookup(coco_annotation:dict) -> dict:
    """
    Creates dictionary with image name and image data
    :param coco_annotation: dict with COCO annotation
    :return: dictionary with image name and image data
    """
    image_dict = {}
    for img in coco_annotation['images']:
        image_dict[img['file_name'].stem] = img

    return image_dict

def annotation_lookup(image_name: str, coco_annotation: dict, image_dict: dict) -> dict:
    """
    Creates dictionary with image name and annotation data
    :param image_name: name of image
    :param coco_annotation: dict with COCO annotation
    :param image_dict: dict with image name and image data
    :return: dictionary with image name and annotation data
    """
    annotation_dict = {}
    annotation_list = []
    image_id = image_dict[image_name]['id']
    for ann in coco_annotation['annotations']:
        if ann['image_id'] == image_id:
            annotation_list.append(ann)
    annotation_dict[image_name] = annotation_list

    return annotation_dict


def draw_coco_bboxes(image, annotations, categories):
    """Draw bounding boxes on image"""
    pass


def visualize_random_samples(coco_data, images_dir, n_samples=5):
    """Visualize random samples from dataset"""
    pass