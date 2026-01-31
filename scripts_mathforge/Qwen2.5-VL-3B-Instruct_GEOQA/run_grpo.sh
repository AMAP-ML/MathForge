#!/bin/bash

RUN_NAME=Qwen2.5-VL-3B-Instruct_GEOQA-R1V-revised_GRPO

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes 8 \
    src/open_r1/grpo_vlm.py \
    --config recipes/Qwen2.5-VL-3B-Instruct/grpo/config_grpo.yaml \
    --run_name $RUN_NAME \
    --output_dir checkpoints/$RUN_NAME