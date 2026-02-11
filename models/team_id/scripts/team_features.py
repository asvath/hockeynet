"""
Extract features for splitting teams into two groups
"""
from typing import Tuple, List

import cv2
import numpy as np


def rle_to_mask(rle: dict) -> np.ndarray:
    """
    Convert rle mask of a single object into binary mask
    :param rle: mask store as run-length encoding
    :return:
    """
    h, w = rle["size"]
    counts = rle["counts"] # counts = [5, 3, 2...]: 5 zeroes, 3 ones, 2 zeroes etc

    flat = np.zeros(h * w, dtype=np.uint8) # 1D array of length h*w filled with zeros
    val = 0  # either 0 or 1, it s the value the current run represents
    idx = 0 # current position in the flattened array where we're writing
    for c in counts:
        if c and val == 1: # if val ==0, we don't need to write anything because flat is already zeroes
            flat[idx:idx+c] = 1  # fill the slice with zeroes
        idx += c
        val ^= 1 # flip val between 0 and 1, if val = 0, becomes 1 and vice versa

    return flat.reshape(h, w).astype(bool)

def all_rle_to_masks(rle_list:list) -> np.ndarray:
    """
    Convert all rle masks into binary mask
    :param rle_list: dict of RLEmasks
    :return: (N, H, W) boolean mask array
    """
    if not rle_list:
        return np.empty((0, 0, 0), dtype=bool)

    return np.stack([rle_to_mask(rle) for rle in rle_list])

def extract_torso(frame: np.ndarray, mask:np.ndarray, top_skip: float = 0.20, bottom_skip: float = 0.4,
                  min_pixels: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:

    """
    Extract torso pixels from a player mask by sampling a middle vertical of the player's bounding box
    :param frame: img
    :param mask: (H, W) boolean mask for one player
    :param top_skip: percentage of top skipped pixels (avoid helmet/head)
    :param bottom_skip: percentage of bottom skipped pixels (avoid pants/skates)
    :param min_pixels: minimum number of pixels required for submask else fallback to full mask
    :return: tuple containing torso image pixels and torso submask
    """

    # Use upper-body-ish region to avoid pants/ helmet
    ys, xs = np.where(mask) # row, column indices
    if ys.size == 0:
        empty_img = np.empty((0, 0, 3), dtype=frame.dtype)
        empty_mask = np.empty((0, 0), dtype=bool)
        empty_pixels = np.empty((0, 3), dtype=frame.dtype)
        return empty_img, empty_pixels, empty_mask, 0, 0

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    H = (y1 - y0 + 1) # height of mask, + 1 as zero-index
    y_start = y0 + int(top_skip * H)
    y_end = y1 - int(bottom_skip * H)

    # ensure at least 1 row
    if y_end <= y_start:
        y_start, y_end = y0, y1 + 1

    else:
        y_end = y_end + 1 # make end exclusive for slicing

    # make x end exclusive for slicing
    x1 = x1 + 1

    submask = mask[y_start: y_end, x0:x1]
    subimg = frame[y_start: y_end, x0:x1]
    torso_crop = subimg.copy()
    torso_crop[~submask] = 0
    pixels = subimg[submask]

    # fallback to full mask if band too small
    if pixels.shape[0] < min_pixels:
        pixels = frame[mask]

    return torso_crop, pixels, submask, y_start, x0

def extract_all_torsos(frame:np.ndarray, masks: np.ndarray, top_skip: float = 0.2, bottom_skip: float = 0.4,
                      min_pixels: int = 200) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[Tuple[int, int]]]:
    """
    Extract all torso pixels and torso submasks for all players in given frame
    :param frame: image
    :param masks: all player masks
    :param top_skip: percentage of top skipped pixels (avoid helmet/head)
    :param bottom_skip: percentage of bottom skipped pixels (avoid pants/skates_
    :param min_pixels: minimum number of pixels required for submask else fallback to full mask
    :return: tuple of lists of all players' torso pixels and all players torso submasks
    """
    all_pixels = []
    all_submasks = []
    all_offsets = []
    all_torso_crops = []

    for mask in masks:
        torso_crop, pixels, submask, y_off, x_off = extract_torso(frame, mask, top_skip, bottom_skip, min_pixels)
        all_torso_crops.append(torso_crop)
        all_pixels.append(pixels)
        all_submasks.append(submask)
        all_offsets.append((y_off, x_off))

    return all_torso_crops,all_pixels, all_submasks, all_offsets

def stack_submasks(all_submasks, all_offsets, frame_shape: Tuple[int,int]):
    """
    Reconstruct full-frame torso masks from cropped submasks
    :param all_submasks: list of (h, w) boolean submasks
    :param all_offsets: list of (y_off, x_off) offsets
    :param frame_shape: (H, W) of original frame
    :return: (N, H, W) boolean masks
    """
    H, W = frame_shape

    full_masks = []

    for submask, (y_off, x_off) in zip(all_submasks, all_offsets):

        # create full mask
        full = np.zeros((H, W), dtype=bool)

        h, w = submask.shape

        # fill out the full mask
        full[
            y_off : y_off + h,
            x_off : x_off + w
        ] = submask

        full_masks.append(full)

    return np.stack(full_masks)


def extract_colour_features(frame: np.ndarray, mask: np.ndarray, sat_thresh=25, min_keep=50) -> np.ndarray:
    """
    Extract colour features for a single player
    :param frame: image (H, W, 3)
    :param mask: boolean mask for a single player (H,W)
    :param sat_thresh
    :param min_keep
    :return: [hue, saturation, value] features for a given player
    """
    # Get every pixel (BGR) that belongs to the player
    pixels = frame[mask] # (N, 3), where N = number of pixels

    # convert pixels to HSV
    # pixels[None, :, :] adds a dimension (1, N, 3) to satisfy opencv, as it expects (H, W , 3)
    hsv = cv2.cvtColor(pixels[None, :, :], cv2.COLOR_BGR2HSV)[0]

    # get saturation (how, colourful is each pixel? i.e low sat is white/gray/black, high sat: "colourful")
    sat = hsv[:, 1]
    keep = sat >= sat_thresh # true, false

    # keep only saturated pixels if enough exist
    if keep.sum() >= min_keep:
        hsv = hsv[keep]

    return np.median(hsv, axis=0) # median colour


def extract_all_colour_features(frame, masks):
    """
    Extract colour features for all players
    :param frame: image (H, W, 3)
    :param masks: boolean masks for all players
    :return: colour featutres for all players
    """
    return np.stack([extract_colour_features(frame, mask) for mask in masks])

def canonicalize_teams(labels:np.ndarray, colour_features:np.ndarray):
    """
    Canonicalize labels for consistency across frames
    :param labels: Kmeans labels for team clustering
    :param colour_features: colour features for all players
    :return: consistent labels for teams
    """
    H = colour_features[:, 0]

    # median hue for cluster 0
    med0 = np.median(H[labels == 0])

    #m median hue for cluster 1
    med1 = np.median(H[labels == 1])

    # if cluster 0's hue is more than cluster 1's hue, switch label
    # i.e. cluster with higher hue is always cluster 1 and cluster with lower hue is always cluster 0
    if med0 > med1:
        labels = 1 - labels

    return labels
