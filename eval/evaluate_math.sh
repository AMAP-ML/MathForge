#!/bin/bash

export VLLM_WORKER_MULTIPROC_METHOD=spawn # Required for vLLM
MODEL=$1
MODEL_NAME=${MODEL//\//_}
MODEL_ARGS="model_name=$MODEL_NAME,dtype=bfloat16,max_model_length=4096,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95}"
TIME_STAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR=logs/eval/${MODEL_NAME}/$TIME_STAMP

TASKS="aime24 aime25 amc23 math_500 minerva olympiadbench"
for TASK in $TASKS; do
    lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
        --system-prompt "Please reason step by step, and put your final answer within \\boxed{}." \
        --custom-tasks eval/custom_math_tasks.py \
        --use-chat-template \
        --output-dir $OUTPUT_DIR
done