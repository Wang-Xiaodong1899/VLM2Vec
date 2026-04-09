import json
import os
from tqdm import tqdm

file = "video_itg_data.json"
with open(file, "r") as f:
    data = json.load(f)

root = "/mnt/bn/wxd-video-understanding/wangxd/data"
save_data = []
# single_clip_2_3_m_data = []
for item in tqdm(data):
    video = item["video"]
    video_path = os.path.join(root, video)
    if not os.path.exists(video_path):
        continue
    clip_num = item["clip_num"]
    if len(clip_num) != 1:
        continue
    # if len(clip_num) == 1:
    #     continue
    # if "2_3_m" in video_path:
    #     single_clip_2_3_m_data.append(item)
    save_data.append(item)

# print(len(single_clip_2_3_m_data))
# print(len(save_data))
with open(f"video_itg_data_single_clip_{len(save_data)}.json", "w") as f:
    json.dump(save_data, f, ensure_ascii=False, indent=4)
