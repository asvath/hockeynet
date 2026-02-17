import cv2
import torch

from models.team_id.scripts.jersey_recognition_llm_utils import (
    ROSTERS_DIR,
    load_roster,
    build_jersey_info,
    build_prompt,
    enhance_crop,
    np_bgr_to_data_url,
    read_jersey_number_llm,
)
from models.team_id.scripts.team_features import extract_all_torsos, all_rle_to_masks
from scripts.utils.config import load_paths, PROJECT_ROOT

PATHS = load_paths()


# tmp_dir = PROJECT_ROOT / "data" / "tmp" / "torso_crops"
# tmp_dir.mkdir(parents=True, exist_ok=True)

def detect_jersey_number(game_name: str, players_results: dict, prompt_text_lower_hue: str, prompt_text_upper_hue: str):
    """
    Detects jersey numbers for all players across all frames using an LLM.
    Appends jersey_ids and jersey_confidence to each frame's results and saves to .pt file.
    :param game_name: name of the game used for file paths
    :param players_results: dict of per-frame player tracking results
    :param prompt_text_lower_hue: LLM prompt for the lower hue team (team_id 0)
    :param prompt_text_upper_hue: LLM prompt for the upper hue team (team_id 1)
    """

    for i in range(len(players_results)):
        print(f"frame {i}")

        players_id = players_results[i]["out_obj_ids"] # object id
        player_team_ids = players_results[i]["team_ids"] # player's team id

        # create fields for jersey ids and confidence of id
        players_results[i]["jersey_ids"] = []
        players_results[i]["jersey_confidence"] = []
        rle_list = players_results[i]["out_masks_rle"]

        # convert rle masks to binary masks
        masks = all_rle_to_masks(rle_list)

        # load the frame
        frame = cv2.imread(str(
            PROJECT_ROOT / PATHS["PROCESSED_VIDEOS_DIR"] / game_name / f"{game_name}_{i:04d}.jpg"
        ))

        # crop out the players
        torso_crops, _, _, _ = extract_all_torsos(frame, masks, top_skip=0, bottom_skip=0)

        # check that we have equal number of players and crops
        assert len(torso_crops) == len(players_id)

        # for each player/crop, identify the jersey number
        for j, crop in enumerate(torso_crops):
            if crop is not None:
                # upscale the crop
                enhanced = enhance_crop(crop)

                # encode to image url for llm
                image_url = np_bgr_to_data_url(enhanced, ext=".png", quality=100)

                # read the jersey number
                result = read_jersey_number_llm(image_url, prompt_text_lower_hue) if player_team_ids[j] == 0 \
                    else read_jersey_number_llm(image_url, prompt_text_upper_hue)
                players_results[i]["jersey_ids"].append(result["number"])
                players_results[i]["jersey_confidence"].append(result["confidence"])

    torch.save(
        players_results,
        PROJECT_ROOT / PATHS["TEAM_ID_OUTPUTS_DIR"] / f"{game_name}_w_team_ids_jersey_ids_llm_fixed.pt",
    )

if __name__ == "__main__":
    game_name = "shl_hv71_v_tik"

    hv71_roster = load_roster(ROSTERS_DIR / "hv71.json")
    tik_roster = load_roster(ROSTERS_DIR / "tik.json")
    jersey_prompt = load_roster(ROSTERS_DIR / "jersey_prompt")

    team_lower_hue_info = build_jersey_info(hv71_roster) # lower hue
    team_upper_hue_info = build_jersey_info(tik_roster) # higher hue

    players_results = torch.load(
        PROJECT_ROOT / PATHS["TEAM_ID_OUTPUTS_DIR"] / "shl_hv71_v_tik_w_team_ids.pt",
        map_location="cpu",
        weights_only=False,
    )

    # build prompt for llm
    prompt_text_lower_hue = build_prompt(jersey_prompt, team_lower_hue_info)

    prompt_text_upper_hue = build_prompt(jersey_prompt, team_upper_hue_info)

    # detect jersey id
    detect_jersey_number(game_name, players_results, prompt_text_lower_hue, prompt_text_upper_hue)