import json
from tqdm import tqdm

with open("VideoITG-40K/video_itg_data.json", 'r') as f:
    data = json.load(f)

save_items = []
for item in tqdm(data):
    video = item['video']
    subdir = video.split('/')[1]
    clip_num = item['clip_num']
    frame_num = item['frame_num']
    if subdir == "2_3_m_youtube_v0_1":
        if len(clip_num) == 1:
            save_items.append(len(frame_num))

import numpy as np
# 统计save_items均值、中位数、最大值、最小值
mean_clip_num = np.mean(save_items)
median_clip_num = np.median(save_items)
max_clip_num = np.max(save_items)
min_clip_num = np.min(save_items)
print(f"均值: {mean_clip_num}")
print(f"中位数: {median_clip_num}")
print(f"最大值: {max_clip_num}")
print(f"最小值: {min_clip_num}")
print(len(save_items))