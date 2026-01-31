#!/bin/bash

RUN_NAME=Qwen2.5-3B_MATH-augmented_DGPO

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes 8 \
    src/open_r1/grpo.py \
    --config recipes/Qwen2.5-3B/grpo/config_grpo_augmented.yaml \
    --enable_dgpo True --enable_dgpo_dqw True --dgpo_dqw_temp 2.0 \
    --run_name $RUN_NAME \
    --output_dir checkpoints/$RUN_NAME