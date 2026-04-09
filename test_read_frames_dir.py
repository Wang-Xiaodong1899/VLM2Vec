IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
import os
import re
import numpy as np



def load_frames(frames_dir, filter_func=None):
    """
    Load image frames from a directory, with an optional filter function.
    """
    def natural_sort_key(filename):
        """Extract numbers from filenames for correct sorting."""
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

    results = []
    if not os.path.exists(frames_dir) or not os.path.isdir(frames_dir):
        return []
    frame_names = os.listdir(frames_dir)
    frame_names = sorted(frame_names, key=natural_sort_key)
    for frame_name in frame_names:
        ext = os.path.splitext(frame_name)[-1].lower()
        if ext.lower() in IMAGE_EXTENSIONS:
            if filter_func is None or filter_func(frame_name):
                image_path = f"{frames_dir}/{frame_name}"
                results.append(image_path)
    return results

frames = load_frames("test_parsed/LLaVA-Video-178K_1_2_m_youtube_v0_1_liwei_youtube_videos_videos_youtube_video_2024_ytb_PJY98RKW-tA/")
print(frames)
