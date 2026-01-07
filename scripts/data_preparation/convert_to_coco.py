"""
Converts YOLO annotations to COCO format
"""
from scripts.utils.annotation_utils import convert_to_coco_format
from scripts.utils.config import load_paths, load_dataset_metadata, PROJECT_ROOT

PATHS = load_paths()
metadata = load_dataset_metadata()
annotations_dir = PROJECT_ROOT / PATHS["RAW_ANNOTATIONS_DIR"]
dataset_metadata_path = PROJECT_ROOT / PATHS["DATASET_METADATA"]
output_path = PROJECT_ROOT / PATHS["COCO_ANNOTATIONS"]
img_width = metadata["image_resolution"]["width"]
img_height = metadata["image_resolution"]["height"]

def convert_to_coco():
    print("Converting YOLO annotations to COCO format...")
    coco_data = convert_to_coco_format(annotations_dir= annotations_dir,
                           dataset_metadata_path= dataset_metadata_path,
                           output_path= output_path,
                           img_width=img_width,
                           img_height=img_height
                           )
    print(f"Conversion complete")
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print(f" Output saved to: {output_path}")

if __name__=="__main__":
    convert_to_coco()