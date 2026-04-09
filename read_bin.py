import torch

# 使用 mmap 模式，不立即加载所有数据
state_dict = torch.load("VLM2Vec-Qwen2VL-2B/adapter_model.bin", 
                        map_location="cpu", 
                        mmap=True)

keys = list(state_dict.keys())
print(f"Found {len(keys)} keys:")
for key in keys:
    print(f"  - {key}")