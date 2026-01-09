"""
Utility functions for visualization
"""
from pathlib import Path
import random

from matplotlib.figure import Figure

from scripts.utils.annotation_utils import coco_to_absolute, image_lookup, annotation_lookup, category_lookup
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

def visualize_random_samples(coco_data: dict, images_dir: Path, n_samples: int = 5) -> Figure:
    """
    Visualize random samples from dataset with bounding boxes in a grid layout
    :param coco_data: COCO data containing image info, annotations, category info
    :param images_dir: directory containing images
    :param n_samples: number of random samples to visualize
    :return: figure with grid of annotated images
    """
    # Create lookups
    img_lookup = image_lookup(coco_data)
    ann_lookup = annotation_lookup(coco_data)
    cat_lookup = category_lookup(coco_data)

    # Get all image names and randomly sample
    all_images = list(img_lookup.keys())
    sample_images = random.sample(all_images, min(n_samples, len(all_images)))

    # Determine grid layout (max 3 columns)
    n_cols = min(3, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols  # Ceiling division

    # Define colors for categories
    colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'orange']

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))

    # Handle single subplot case
    if n_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Draw each sample
    for idx, image_name in enumerate(sample_images):
        # Get image info and annotations
        img_data = img_lookup[image_name]
        image_id = img_data['id']
        annotations = ann_lookup.get(image_id, [])

        # Load image
        image_path = images_dir / image_name
        img = Image.open(image_path)

        # Display image
        axes[idx].imshow(img)

        # Draw bounding boxes
        for ann in annotations:
            x_min, y_min, width, height = ann['bbox']
            category_id = ann['category_id']

            color = colors[category_id % len(colors)]
            category_name = cat_lookup[category_id]

            # Convert to absolute pixel coordinates
            x_min, y_min, _, _, width, height = coco_to_absolute(x_min, y_min, width, height)

            # Create rectangle
            rect = patches.Rectangle(
                (x_min, y_min), width, height,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            axes[idx].add_patch(rect)

            # Add label
            axes[idx].text(x_min, y_min - 5, category_name,
                          bbox=dict(facecolor=color, alpha=0.5),
                          fontsize=8, color='white')

        axes[idx].set_title(f"{image_name}\n{len(annotations)} objects", fontsize=10)
        axes[idx].axis('off')

    # Hide unused subplots
    for idx in range(len(sample_images), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    return fig