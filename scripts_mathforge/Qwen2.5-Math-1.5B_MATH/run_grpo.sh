#!/bin/bash

RUN_NAME=Qwen2.5-Math-1.5B_MATH-lighteval_GRPO

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes 8 \
    src/open_r1/grpo.py \
    --config recipes/Qwen2.5-Math-1.5B/grpo/config_grpo.yaml \
    --run_name $RUN_NAME \
    --output_dir checkpoints/$RUN_NAME