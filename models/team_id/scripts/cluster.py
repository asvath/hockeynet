"""
Team clustering pipeline:
Load masks -> decode RLE -> extract torsos -> colour features -> k-means -> canonicalize
"""
import numpy as np
import torch
from sklearn.cluster import KMeans

from models.sam3.scripts.visualization_utils import load_frame_by_index
from models.team_id.scripts.team_features import (
    all_rle_to_masks,
    extract_all_colour_features,
    canonicalize_teams,
)
from scripts.utils.config import load_paths, PROJECT_ROOT

PATHS = load_paths()


def cluster_frame(frame: np.ndarray, masks: np.ndarray, n_clusters=2, sep_thresh = 5):
    """
    Run full team-clustering pipeline on a single frame.
    :param frame: (H, W, 3) BGR image
    :param masks: (N, H, W) decoded boolean masks
    :param n_clusters: number of teams
    :param sep_thresh: (optional) threshold for separating teams
    :return: canonicalized labels (N,)
    """
    # only 1 or no players
    N = masks.shape[0]
    if N < n_clusters:
        return np.zeros(N, dtype=int)

    # colour features from full masks
    feats = extract_all_colour_features(frame, masks)

    # k-means clustering into two teams
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0)
    labels = kmeans.fit_predict(feats)

    centers = kmeans.cluster_centers_
    sep = np.linalg.norm(centers[0] - centers[1])

    # only one team visible, assign all to 0, can be fixed with temporal smoothing later
    if sep < sep_thresh:
        return np.zeros(N, dtype=int)

    # canonicalize so labels are consistent across frames
    labels = canonicalize_teams(labels, feats)

    return labels


if __name__ == "__main__":
    game_name = "shl_hv71_v_tik"

    players_mask_track_path = (
        PROJECT_ROOT
        / PATHS["SAM3_OUTPUTS_MASK_TRACK_DIR"]
        / f"{game_name}_w_examples_filtered_players_2000px_sticks_removed_RLE.pt"
    )
    game_frames_path = PROJECT_ROOT / PATHS["PROCESSED_VIDEOS_DIR"] / f"{game_name}"

    players_results = torch.load(players_mask_track_path, map_location="cpu", weights_only=False)

    # cluster all frames

    for frame_idx in range(len(players_results)):

        frame = load_frame_by_index(game_frames_path, frame_idx)

        # decode RLE masks
        rle_list = players_results[frame_idx]["out_masks_rle"]
        masks = all_rle_to_masks(rle_list)
        # players_results[frame_idx]["decoded_masks"] = masks

        # cluster
        labels = cluster_frame(frame, masks)

        players_results[frame_idx]["team_ids"] = labels

    # save results with team_ids
    output_dir = PROJECT_ROOT / PATHS["TEAM_ID_OUTPUTS_DIR"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{game_name}_w_team_ids.pt"
    torch.save(players_results, output_path)
    print(f"Saved to {output_path}")
