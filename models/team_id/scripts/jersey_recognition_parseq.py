import re

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

from strhub.data.module import SceneTextDataModule  # side-effect: registers charset/tokenizer
from strhub.models.parseq.system import PARSeq

from scripts.utils.config import load_paths, PROJECT_ROOT
from models.team_id.scripts.jersey_recognition_llm_utils import (
    ROSTERS_DIR,
    load_roster,
    build_jersey_info,
    enhance_crop,
)
from models.team_id.scripts.team_features import extract_all_torsos, all_rle_to_masks

PATHS = load_paths()

CKPT_PATH = r"C:\Users\Asha\Downloads\epoch=3-step=95-val_accuracy=98.7903-val_NED=99.3952.ckpt"


def load_parseq_model(ckpt_path: str = CKPT_PATH):
    """
    Load PARSeq model from checkpoint.
    :param ckpt_path: path to PARSeq checkpoint
    :return: (net, tokenizer, transform, device)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    hp = ckpt["hyper_parameters"]
    state = ckpt["state_dict"]

    system = PARSeq(**hp).to(device).eval()

    # add "model." prefix so it matches system.state_dict() keys
    state_with_prefix = {f"model.{k}": v for k, v in state.items()}
    system.load_state_dict(state_with_prefix, strict=False)

    net = system.model
    net.eval()
    tokenizer = system.tokenizer

    transform = T.Compose([
        T.Resize(hp["img_size"]),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])

    return net, tokenizer, transform, device


def parseq_predict_bgr(img_bgr: np.ndarray, net, tokenizer, transform, device):
    """
    Run PARSeq inference on a BGR image crop.
    :param img_bgr: BGR image
    :param net: PARSeq model
    :param tokenizer: PARSeq tokenizer
    :param transform: image transform pipeline
    :param device: torch device
    :return: (label string, confidence tensor)
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).convert("RGB")
    x = transform(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = net(images=x, tokenizer=tokenizer)

    probs = logits.softmax(-1)
    labels, confs = tokenizer.decode(probs)
    return labels[0], confs[0]


def detect_jersey_number(game_name: str, players_results: dict, lower_hue_info: dict, upper_hue_info: dict,
                         conf_threshold: float = 0.70):
    """
    Detects jersey numbers for all players across all frames using PARSeq.
    Appends jersey_ids and jersey_confidence to each frame's results and saves to .pt file.
    :param game_name: name of the game used for file paths
    :param players_results: dict of per-frame player tracking results
    :param lower_hue_info: jersey info for the lower hue team (team_id 0)
    :param upper_hue_info: jersey info for the upper hue team (team_id 1)
    :param conf_threshold: minimum character confidence to accept a prediction
    """
    net, tokenizer, transform, device = load_parseq_model()

    allowed_list_lower_hue = lower_hue_info["allowed_list"]
    allowed_list_upper_hue = upper_hue_info["allowed_list"]

    for i in range(len(players_results)):
        print(f"frame: {i}")
        frame = cv2.imread(str(
            PROJECT_ROOT / PATHS["PROCESSED_VIDEOS_DIR"] / game_name / f"{game_name}_{i:04d}.jpg"
        ))
        rle_list = players_results[i]["out_masks_rle"]
        masks = all_rle_to_masks(rle_list)
        torso_crops, _, _, _ = extract_all_torsos(frame =frame, masks= masks, top_skip=0.1, bottom_skip=0.4)
        players_id = players_results[i]["out_obj_ids"]
        player_team_ids = players_results[i]["team_ids"]

        players_results[i]["jersey_ids"] = []
        players_results[i]["jersey_confidence"] = []

        assert len(torso_crops) == len(players_id)



        tmp_dir = PROJECT_ROOT / "data" / "tmp" / "torso_crops"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        for j, crop in enumerate(torso_crops):
            team_id = player_team_ids[j]
            if crop is not None:
                enhanced = enhance_crop(crop, scale=8)

                label, conf = parseq_predict_bgr(enhanced, net, tokenizer, transform, device)

                char_confs = conf[:len(label)]




                if float(char_confs.min()) >= conf_threshold:


                    allowed = allowed_list_lower_hue if team_id == 0 else allowed_list_upper_hue
                    digits = re.sub(r'\D', '', label)

                    # filename = f"frame_{i:04d}_player_{players_id[j]:02d}_{digits}_{char_confs.min()}_before_{team_id}.jpg"
                    # cv2.imwrite(str(tmp_dir / filename), enhanced)

                    if digits and int(digits) in allowed:
                        players_results[i]["jersey_ids"].append(label)
                        players_results[i]["jersey_confidence"].append(float(char_confs.min()))
                    else:
                        players_results[i]["jersey_ids"].append(-1)
                        players_results[i]["jersey_confidence"].append(-1)
                else:
                    players_results[i]["jersey_ids"].append(-1)
                    players_results[i]["jersey_confidence"].append(-1)
                #
                # jersey_id = players_results[i]["jersey_ids"][j]
                # jersey_conf = round(players_results[i]["jersey_confidence"][j], 2)
                # filename = f"frame_{i:04d}_player_{players_id[j]:02d}_{jersey_id}_{jersey_conf}_{team_id}_after.jpg"
                # cv2.imwrite(str(tmp_dir / filename), enhanced)
            else:
                players_results[i]["jersey_ids"].append(-1)
                players_results[i]["jersey_confidence"].append(-1)

    torch.save(
        players_results,
        PROJECT_ROOT / PATHS["TEAM_ID_OUTPUTS_DIR"] / f"{game_name}_w_team_ids_jersey_ids_parseq_bigger_crop_top_bottom_conf_0.7.pt",
    )


if __name__ == "__main__":
    game_name = "shl_hv71_v_tik"

    hv71_roster = load_roster(ROSTERS_DIR / "hv71.json")
    tik_roster = load_roster(ROSTERS_DIR / "tik.json")

    team_lower_hue_info = build_jersey_info(hv71_roster)
    team_upper_hue_info = build_jersey_info(tik_roster)

    players_results = torch.load(
        PROJECT_ROOT / PATHS["TEAM_ID_OUTPUTS_DIR"] / "shl_hv71_v_tik_w_team_ids.pt",
        map_location="cpu",
        weights_only=False,
    )

    detect_jersey_number(game_name, players_results, team_lower_hue_info, team_upper_hue_info)

    # players_results = torch.load(
    #     PROJECT_ROOT / PATHS["TEAM_ID_OUTPUTS_DIR"] / "shl_hv71_v_tik_w_team_ids_jersey_ids_parseq.pt",
    #     map_location="cpu",
    #     weights_only=False,
    # )

    # for i in range(len(players_results)):
        # print("*****")
        # print(players_results[i].keys())
        # print(len(players_results[i]["out_obj_ids"]))
        # print(players_results[i]["team_ids"])
        # print(players_results[i]["jersey_ids"])
        # print(players_results[i]["jersey_confidence"])
