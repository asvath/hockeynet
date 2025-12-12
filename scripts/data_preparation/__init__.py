"""
Data preparation pipeline for HockeyNet dataset.

Modules:
- extract_frames: Extract frames from video files
- cvat_to_coco: Convert CVAT XML annotations to COCO format
- merge_annotations: Combine CVAT and Roboflow annotations
- visualize_annotations: Visualize bounding boxes on images
- create_splits: Create train/val/test dataset splits
"""
