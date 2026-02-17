import torch

from models.team_id.scripts.jerseyiddebouncer_parseq import JerseyIDDebouncer
from scripts.utils.config import load_paths, PROJECT_ROOT

PATHS = load_paths()


def assign_jersey_id(game_name: str, players_results: dict) -> dict:
    """
    Assigns smooth Jersery IDs
    :param game_name: name of the game
    :param players_results: players tracking info with raw jersey ids
    :return: player results with processed Jersey IDs
    """


    # initialize smoother
    debouncer = JerseyIDDebouncer(mode = "llm", init_frames = 10, confirm_frames=20, expire_after=300)

    for frame_idx in range(len(players_results)):


        players_results[frame_idx]["jersey_ids"] = debouncer.update(players_results[frame_idx]["out_obj_ids"],
                                                                    players_results[frame_idx]["jersey_ids"],
                                                                    players_results[frame_idx]["jersey_confidence"])


    # save results with jersey_ids
    output_dir = PROJECT_ROOT / PATHS["TEAM_ID_OUTPUTS_DIR"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{game_name}_w_team_ids_jersey_ids_w_smoothing_llm_fixed.pt"
    torch.save(players_results, output_path)
    print(f"Saved to {output_path}")

    return players_results

if __name__ == "__main__":
    game_name = "shl_hv71_v_tik"

    players_results = torch.load(
        PROJECT_ROOT / PATHS["TEAM_ID_OUTPUTS_DIR"] / "shl_hv71_v_tik_w_team_ids_jersey_ids_llm_fixed.pt",
        map_location="cpu",
        weights_only=False,
    )

    assign_jersey_id(game_name, players_results)