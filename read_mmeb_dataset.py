#!/usr/bin/env python3

import sys
import os
sys.path.append('/mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_6/user_seandonwang/projects/VLM2Vec')

from src.data.dataset.mmeb_dataset import load_mmeb_dataset
from src.arguments import ModelArguments, DataArguments, TrainingArguments

# 简单的参数设置，单卡训练
model_args = ModelArguments(
    model_name='/mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_6/user_seandonwang/models/Qwen2-VL-2B-Instruct',  # 需要指定模型名称
    model_backbone='qwen2_vl',  # 或其他
)

dataset_list = [
    'ImageNet_1K',
    'N24News',
    'HatefulMemes',
    'VOC2007',
    'SUN397',
    'OK-VQA',
    'A-OKVQA',
    'DocVQA',
    'InfographicsVQA',
    'ChartQA',
    'Visual7W',
    'VisDial',
    'CIRR',
    'VisualNews_t2i',
    'VisualNews_i2t',
    'MSCOCO_t2i',
    'MSCOCO_i2t',
    'NIGHTS',
    'WebQA',
    'MSCOCO'
]

data_args = DataArguments(
    dataset_name='TIGER-Lab/MMEB-train',
    subset_name=dataset_list,  # 示例子集
    dataset_split='original',
    image_dir='/mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_6/user_seandonwang/data/vlm2vec_train/MMEB-train/images',  # 替换为实际路径
    image_resolution='low',
    num_sample_per_subset=100,  # 限制样本数以快速测试
)

training_args = TrainingArguments(
    dataloader_num_workers=0,  # 单卡，设为0
    # 其他必要参数
)

# 加载数据集
kwargs = {
    'dataset_name': 'TIGER-Lab/MMEB-train',
    'dataset_split': 'original',
    'num_sample_per_subset': 100,
    'image_dir': '/mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_6/user_seandonwang/data/vlm2vec_train/MMEB-train/images',
}

for subset in dataset_list:
    kwargs['subset_name'] = subset
    try:
        dataset = load_mmeb_dataset(model_args, data_args, training_args, **kwargs)
        print(f"Loaded {subset} with {dataset.num_rows} samples")
        
        # 读取前1个样本
        count = 0
        for sample in dataset:
            print(f"Sample 0 for {subset}:")
            print(f"  Query Text: {sample['query_text'][:1]}")
            print(f"  Pos Text: {sample['pos_text'][:1]}")
            print(f"  Query Image: {type(sample['query_image'])}")
            print(f"  Pos Image: {type(sample['pos_image'])}")
            count += 1
            if count >= 1:
                break
    except Exception as e:
        print(f"Error loading {subset}: {e}")

print("Done reading all datasets.")