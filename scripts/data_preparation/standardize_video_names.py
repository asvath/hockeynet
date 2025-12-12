"""
Standardized video names:
gamename_clipnumber,e.g. allstar_2019_001
"""
from scripts.utils.config import RAW_VIDEO_DIR
from scripts.utils.video_utils import standardize_all_video_names

def preprocess_videos():
    standardize_all_video_names(RAW_VIDEO_DIR)

if __name__ =="__main__":
    preprocess_videos()