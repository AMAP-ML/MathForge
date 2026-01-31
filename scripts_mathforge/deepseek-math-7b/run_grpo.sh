#!/bin/bash

RUN_NAME=deepseek-math-7b-warm_MATH-lighteval_GRPO

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes 8 \
    src/open_r1/grpo.py \
    --config recipes/deepseek-math-7b-warm/grpo/config_grpo.yaml \
    --run_name $RUN_NAME \
    --output_dir checkpoints/$RUN_NAME