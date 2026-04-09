import json
from tqdm import tqdm

with open("VideoITG-40K/video_itg_data.json", 'r') as f:
    data = json.load(f)

total_subsets = set()
subsets_count = {}
for item in tqdm(data):
    video = item['video']
    subset = video.split('/')[1]
    total_subsets.add(subset)
    if subset not in subsets_count:
        subsets_count[subset] = 0
    subsets_count[subset] += 1

print(total_subsets)
print(subsets_count)

# 绘制条形图
import matplotlib.pyplot as plt
import numpy as np

# 对子集按数量从大到小排序
sorted_items = sorted(subsets_count.items(), key=lambda x: x[1], reverse=True)
sorted_subsets = [item[0] for item in sorted_items]
sorted_counts = [item[1] for item in sorted_items]

# 创建图形
plt.figure(figsize=(12, 6))

# 绘制条形图
bars = plt.bar(range(len(sorted_subsets)), sorted_counts, color='skyblue', edgecolor='black')

# 添加数量标签
for bar, count in zip(bars, sorted_counts):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + max(sorted_counts)*0.01,
             f'{count}', ha='center', va='bottom', fontsize=9)

# 设置坐标轴
plt.xticks(range(len(sorted_subsets)), sorted_subsets, rotation=45, ha='right')
plt.xlabel('Subset')
plt.ylabel('Number')
# plt.title('子集数量分布（从大到小排序）')

# 调整布局
plt.tight_layout()

plt.savefig('video_itg_data.png')
