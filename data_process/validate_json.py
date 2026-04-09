import json
import os
from tqdm import tqdm

file = "video_itg_data_single_clip_55884.json"
with open(file, "r") as f:
    data = json.load(f)

root = "/mnt/bn/wxd-video-understanding/wangxd/data"
frame_basedir = "Video_Frames_single_clip"
save_data = []
for item in tqdm(data):
    video = item["video"]
    # video_path = os.path.join(root, video)
    video_noext = os.path.splitext(video)[0]
    normalized = video_noext.replace("/", "_")
    candidates = [
        os.path.join(frame_basedir, normalized),
        os.path.join(frame_basedir, video_noext),
        os.path.join(frame_basedir, video.replace("/", "_")),
        os.path.join(frame_basedir, os.path.basename(video_noext)),
    ]
    # import pdb; pdb.set_trace()
    for cand in candidates:
        if os.path.isdir(cand):
            save_data.append(item)
            continue

# write save_data to json
with open(f"video_itg_data_single_clip_55884_save_frames_{len(save_data)}_qa.json", "w") as f:
    json.dump(save_data, f, ensure_ascii=False, indent=4)