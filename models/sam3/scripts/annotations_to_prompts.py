import json
import sys
from pathlib import Path

# Add project root to path for config import
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.config import PATHS, PROJECT_ROOT
from prompt_utils import box_to_xywh, pad_xywh

# --- read your frame annotation JSON --- This is roboflow SAM3 annotations output
prompts_dir = PROJECT_ROOT / PATHS["SAM3_PROMPTS_DIR"]
with open(prompts_dir / "frame0_annotations.json", "r") as f:
    data = json.load(f)

boxes = data["annotations"]

classes = ["big circle", "skate", "hockey stick", "stick blade", "hockey puck"]
# --- group boxes by label ---
pucks       = [b for b in boxes if b["label"] == "hockey puck"]
big_circles = [b for b in boxes if b["label"] == "big circle"]
sticks      = [b for b in boxes if b["label"] == "hockey stick"]
skates      = [b for b in boxes if b["label"] == "skate"]
blades      = [b for b in boxes if b["label"] == "stick blade"]

# assert only one puck
assert len(pucks) == 1

# --- build prompt bboxes---
puck_bboxes = []
faceoff_boxes = []
stick_bboxes = []
skate_bboxes = []
blade_bboxes = []
bbox_labels = []

# positive puck
puck_bboxes.append(list(box_to_xywh(pucks[0])))
bbox_labels.append(1)

# negatives: big circles
for b in big_circles:
    faceoff_boxes.append(list(box_to_xywh(b)))
    bbox_labels.append(0)

# negatives: all sticks
for b in sticks:
    stick_bboxes.append(list(box_to_xywh(b)))
    bbox_labels.append(0)

# negatives: all skates
for b in skates:
    skate_bboxes.append(list(box_to_xywh(b)))
    bbox_labels.append(0)

# negatives: all blades
for b in blades:
    blade_bboxes.append(list(pad_xywh(box_to_xywh(b)))) # add padding to blades
    bbox_labels.append(0)

# --- save to JSON ---
output = {
    "puck_bboxes": puck_bboxes,
    "faceoff_boxes": faceoff_boxes,
    "stick_bboxes": stick_bboxes,
    "skate_bboxes": skate_bboxes,
    "blade_bboxes": blade_bboxes,
    "bbox_labels": bbox_labels,
}

# write out positive and negative examples for "hockey puck".
# this will be used in sam3 video propagation
output_path = prompts_dir / "frame0_prompts.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"Saved prompts to {output_path}")