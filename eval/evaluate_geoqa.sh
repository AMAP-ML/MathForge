#!/bin/bash

MODEL=$1
MODEL_NAME=${MODEL//\//_}

python eval/test_qwen2vl_geoqa.py --model_path $MODEL --output_dir logs/eval/$MODEL_NAME