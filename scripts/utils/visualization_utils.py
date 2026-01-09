"""
Utility functions for visualization
"""
from pathlib import Path

from matplotlib.figure import Figure

from scripts.utils.annotation_utils import coco_to_absolute
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def draw_coco_bboxes(image_path: Path, annotations: list, categories: dict) -> Figure:
    """
    Draw COCO bounding boxes on image using matplotlib
    :param image_path: path to image
    :param annotations: list of annotations
    :param categories: dict mapping category_id to category_name
    return figure with bounding boxes
    """
    # Load image
    img = Image.open(image_path)

    # Create figure
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)

    # Define colors
    colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'orange']

    # Draw each bbox
    for ann in annotations:
        x_min, y_min, width, height = ann['bbox']
        category_id = ann['category_id']

        color = colors[category_id % len(colors)]
        category_name = categories[category_id]

        # absolute pixel coordinates
        x_min, y_min, _, _, width, height = coco_to_absolute(x_min, y_min, width, height)

        # Create rectangle
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

        # Add label
        ax.text(x_min, y_min - 5, category_name,
                bbox=dict(facecolor=color, alpha=0.5),
                fontsize=10, color='white')

    ax.axis('off')
    plt.tight_layout()
    return fig

def visualize_random_samples(coco_data, images_dir, n_samples=5):
    """Visualize random samples from dataset"""
    pass