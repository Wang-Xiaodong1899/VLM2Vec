#!/bin/bash
# NOTE: replace ... with actual paths
# export LD_LIBRARY_PATH=...
# export PATH=...
# echo "conda location: $(which conda)"
# echo "Python location: $(which python)"
# echo "Python version: $(python --version)"

# export HF_DATASETS_CACHE=...
# export HF_HOME=...
# export WANDB_DISABLED=false
# export WANDB_PROJECT=...
# export WANDB_API_KEY=...
# export HUGGING_FACE_HUB_TOKEN=...
# export WANDB_PROJECT=...
# export WANDB_RUN_GROUP=...
export EXP_NAME=Qwen2vl_2B_dec_train_only.videoitg.autoresize.lora16.BS1024.IB64.GCq8p8.NormTemp002.lr5e5.step5kwarm100.8H20x2-55k-Fix-LoRA-DecLayer8

export WANDB_NAME=$EXP_NAME
export EXP_DIR=outputs/${EXP_NAME}
export WANDB_DIR=$EXP_DIR
echo $EXP_DIR

mkdir -p $EXP_DIR/wandb
# rm -rf $EXP_DIR/wandb/*
nnodes=3
nproc_per_node=8
node_rank=$1
port=2207
master_addr=26.43.176.66

# cd PATH_TO_VLM2VEC_REPO
torchrun --nproc_per_node=$nproc_per_node --nnodes=$nnodes --node_rank=$node_rank --master_port=$port --master_addr=$master_addr --max_restarts=0 train_dec.py --model_name /mnt/bn/wxd-video-understanding/wangxd/models/Qwen2-VL-2B-Instruct --lora_init_path VLM2Vec-Qwen2VL-2B --lora_init_merge True --bf16 --emb_dec_layer 8 --pooling eos --normalize True --temperature 0.02 --dataloader_num_workers 4 --dataset_config experiments/public/train/videoitg.yaml --run_name $EXP_NAME --output_dir $EXP_DIR --grad_cache True --per_device_train_batch_size 4 --gc_q_chunk_size 1 --gc_p_chunk_size 1 --interleave_batch_size 2 --lr_scheduler_type linear --learning_rate 5e-5 --max_steps 5000 --warmup_steps 10 --save_steps 500 --logging_steps 1 --save_safetensors True --remove_unused_columns False --resume_from auto --report_to wandb
