from safetensors import safe_open
from safetensors.torch import save_file
import torch

# 加载原始文件
input_file = "/mnt/bn/wxd-video-understanding/wangxd/VLM2Vec/outputs/Qwen2vl_2B_dec_train_only.videoitg.autoresize.lora16.BS1024.IB64.GCq8p8.NormTemp002.lr5e5.step5kwarm100.4H20-2-Fix-LoRA-3/checkpoint-100/model.safetensors"

# 创建一个字典来存储修改后的tensors
new_tensors = {}

# 读取并修改键名
with safe_open(input_file, framework="pt", device="cpu") as f:
    for key in f.keys():
        # 加载tensor
        tensor = f.get_tensor(key)
        
        # 修改键名
        if key.startswith("layers."):
            new_key = "post_decoder." + key
        else:
            new_key = key
            
        new_tensors[new_key] = tensor
        print(f"重命名: {key} -> {new_key}")

# 保存到新文件
output_file = "/mnt/bn/wxd-video-understanding/wangxd/VLM2Vec/outputs/Qwen2vl_2B_dec_train_only.videoitg.autoresize.lora16.BS1024.IB64.GCq8p8.NormTemp002.lr5e5.step5kwarm100.4H20-2-Fix-LoRA-3/checkpoint-100/new_model.safetensors"
save_file(new_tensors, output_file)

print(f"\n完成！共处理 {len(new_tensors)} 个tensors")
print(f"新文件已保存到: {output_file}")