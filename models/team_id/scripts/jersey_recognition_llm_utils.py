import base64
import json
import os

import cv2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from scripts.utils.config import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

ROSTERS_DIR = PROJECT_ROOT / "data" / "rosters"


def load_roster(roster_path) -> dict:
    """
    Loads the hockey team's roster
    :param roster_path: path to roster
    :return: roster dict
    """
    with open(roster_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_jersey_info(roster: dict) -> dict:
    """
    Builds dict containing team's jersey info
    :param roster: roster dict
    :return: dict with jersey info
    """
    players = {p["number"]: p["name"] for p in roster["players"]}
    return {
        "jersey_fabric_colour": roster["jersey_fabric_colour"],
        "jersey_digit_colour": roster["jersey_digit_colour"],
        "additional_descriptions": roster["additional_jersey_descriptions"],
        "allowed_list": sorted(players.keys()),
    }


def enhance_crop(img: np.ndarray, scale: int = 4) -> np.ndarray:
    """
    Upsamples img to enhance the crop size
    :param img: image
    :param scale: upsampling factor
    :return: upsampled image
    """
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def np_bgr_to_data_url(img_bgr, ext=".jpg", quality=100) -> str:
    """
    Converts numpy array to base64 encoded url for Open AI
    :param img_bgr: bgr image
    :param ext: file extension
    :param quality: compression quality
    :return: image data as a base64 url
    """
    if ext.lower() in [".jpg", ".jpeg"]:
        ok, buf = cv2.imencode(ext, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        mime = "image/jpeg"
    else:
        ok, buf = cv2.imencode(ext, img_bgr)
        mime = "image/png"
    if not ok:
        raise ValueError("cv2.imencode failed")
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def build_prompt(jersey_prompt: dict, jersey_info: dict) -> str:
    """
    Build prompt text for Open AI to detect jersey number from the player crops
    :param jersey_prompt: text prompt template
    :param jersey_info: jersey info such as jersey colour to fill in the text prompt template
    :return: full prompt text
    """
    return (
        "Return ONLY valid JSON matching this output_format schema:\n"
        + json.dumps(jersey_prompt["output_format"], indent=2)
        + "\n\n"
        + "\n".join(jersey_prompt["prompt_template"]).format(**jersey_info)
    )


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def read_jersey_number_llm(
    image_data_url: str,
    prompt_text: str,
    model: str = "gpt-4.1-mini",
) -> dict:
    """
    Gets LLM to read the jersey number given a player's crop and text prompt
    :param image_data_url: player crop as a base64 encoded url
    :param prompt_text: prompt to read jersey number
    :param model: llm model to use
    :return: dict containing jersey number, confidence level, confidence percentage
    """
    resp = client.responses.create(
        model=model,
        temperature=0,
        input=[
            {
                "role": "system",
                "content": "You are a vision-based number detector.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_text},
                    {"type": "input_image", "image_url": image_data_url, "detail": "high"},
                ],
            },
        ],
        text={"format": {"type": "json_object"}},
    )
    return json.loads(resp.output_text)