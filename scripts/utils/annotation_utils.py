"""
Utility functions for annotations
"""

def load_annotation(path:str) -> list:
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


def yolo_to_absolute(x, y, w, h, img_width, img_height)->tuple:
    pass

def get_annotation_stats(annotations: list)-> dict:
    pass
