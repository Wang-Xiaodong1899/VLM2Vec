import os
from typing import Optional


def _resolve_frame_dir(frame_basedir: str, video: str) -> Optional[str]:
    video_noext = os.path.splitext(video)[0]
    normalized = video_noext.replace("/", "_")
    print(normalized)
    candidates = [
        os.path.join(frame_basedir, normalized),
        os.path.join(frame_basedir, video_noext),
        os.path.join(frame_basedir, video.replace("/", "_")),
        os.path.join(frame_basedir, os.path.basename(video_noext)),
    ]
    for cand in candidates:
        if os.path.isdir(cand):
            return cand
    return None

frame_basedir = "test_parsed"
video = "LLaVA-Video-178K/1_2_m_youtube_v0_1/liwei_youtube_videos/videos/youtube_video_2024/ytb_PJY98RKW-tA.mp4"

frame_dir = _resolve_frame_dir(frame_basedir, video)
print(frame_dir)