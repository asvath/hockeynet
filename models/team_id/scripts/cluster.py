"""
Team clustering pipeline:
Load masks -> decode RLE -> extract torsos -> colour features -> k-means -> canonicalize
"""
import torch

from models.sam3.scripts.visualization_utils import load_frame_by_index
from models.team_id.scripts.team_features import (
    all_rle_to_masks,
)
from models.team_id.scripts.teamassigner import TeamAssigner
from models.team_id.scripts.teamdebouncer import TeamDebouncer
from scripts.utils.config import load_paths, PROJECT_ROOT

PATHS = load_paths()


def classify_players_into_teams(game_name: str, players_results: dict):

    # initialize team assigner
    teamassigner = TeamAssigner()

    # initialize smoother
    debouncer = TeamDebouncer(confirm_frames= 10, expire_after= 300)

    # cluster all frames

    for frame_idx in range(len(players_results)):

        frame = load_frame_by_index(game_frames_path, frame_idx)

        # decode RLE masks
        rle_list = players_results[frame_idx]["out_masks_rle"]
        masks = all_rle_to_masks(rle_list)
        # players_results[frame_idx]["decoded_masks"] = masks

        # cluster
        labels = teamassigner.step(frame, masks)
        players_results[frame_idx]["team_ids"] = debouncer.update(players_results[frame_idx]["out_obj_ids"], labels)


    # save results with team_ids
    output_dir = PROJECT_ROOT / PATHS["TEAM_ID_OUTPUTS_DIR"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{game_name}_w_team_ids.pt"
    torch.save(players_results, output_path)
    print(f"Saved to {output_path}")



if __name__ == "__main__":
    game_name = "shl_hv71_v_tik"

    players_mask_track_path = (
        PROJECT_ROOT
        / PATHS["SAM3_OUTPUTS_MASK_TRACK_DIR"]
        / f"{game_name}_w_examples_filtered_players_2000px_sticks_removed_RLE.pt"
    )
    game_frames_path = PROJECT_ROOT / PATHS["PROCESSED_VIDEOS_DIR"] / f"{game_name}"

    players_results = torch.load(players_mask_track_path, map_location="cpu", weights_only=False)

    classify_players_into_teams(game_name, players_results)


