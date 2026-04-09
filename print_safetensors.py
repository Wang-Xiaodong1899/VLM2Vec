from safetensors import safe_open
from safetensors.torch import save_file
import torch

# 加载原始文件
input_file = "/mnt/bn/wxd-video-understanding/wangxd/wm-project/Qwen2-VL-Finetune/output/test_train/checkpoint-10/model.safetensors"
# 创建一个字典来存储修改后的tensors
new_tensors = {}

# 读取并修改键名
with safe_open(input_file, framework="pt", device="cpu") as f:
    for key in f.keys():
        print(key)