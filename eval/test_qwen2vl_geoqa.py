from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
from math_verify import parse, verify
import argparse
import os

from typing import Optional
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


def accuracy_reward(contents: list[str], solution: list[str], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is the same as the ground truth."""
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(sol, extraction_mode="first_match")
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            try:
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = 0.0    # None
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = 0.0    # None
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)
    return rewards



parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
parser.add_argument("--output_dir", type=str, default="logs/eval/Qwen_Qwen2.5-VL-3B-Instruct")
parser.add_argument("--prompt_path", type=str, default="eval/geoqa_test_prompts_formatted.jsonl")
args = parser.parse_args()


MODEL_PATH=args.model_path # qwen2vl model or grpoed model on geoqa train
BSZ=64 # reduce it if GPU OOM
OUTPUT_PATH=os.path.join(args.output_dir, os.path.basename(MODEL_PATH) + "_results.json")
PROMPT_PATH=args.prompt_path

#We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="cuda",
)

# default processer
processor = AutoProcessor.from_pretrained(MODEL_PATH, padding_side="left")

data = []
with open(PROMPT_PATH, "r") as f:
    for line in f:
        data.append(json.loads(line))

messages = []
for item in data:
    message = [
        {
        "role": "system",
        "content": "Please reason step by step, and put your final answer without units in \\boxed{}."
        },
        {
        "role": "user",
        "content": [
            {
                "type": "image", 
                "image": f"/mnt/workspace/daiyanqi/data/{item['image_path']}"
            },
            {
                "type": "text",
                "text": item['question']
            }
        ]
    }]
    messages.append(message)


all_outputs = []  # List to store all answers

# Process data in batches
for i in tqdm(range(0, len(messages), BSZ)):
    batch_messages = messages[i:i + BSZ]
    
    # Preparation for inference
    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
    
    image_inputs, video_inputs = process_vision_info(batch_messages)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    batch_output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    all_outputs.extend(batch_output_text)
    print(f"Processed batch {i//BSZ + 1}/{(len(messages) + BSZ - 1)//BSZ}")



final_output = []
correct_number = 0

for input_example, model_output in zip(data, all_outputs):
    original_output = model_output
    ground_truth = input_example['ground_truth']
    reward = accuracy_reward([original_output], [ground_truth])[0]

    # Count correct answers
    if reward == 1.0:
        correct_number += 1
        is_correct = True
    else:
        is_correct = False
    
    result = {
        'question': input_example,
        'ground_truth': ground_truth,
        'model_output': original_output,
        'is_correct':is_correct
    }
    final_output.append(result)


# Calculate and print accuracy
accuracy = correct_number / len(data) * 100
print(f"\nAccuracy: {accuracy:.2f}%")

# Save results to a JSON file
output_path = OUTPUT_PATH
with open(output_path, "w") as f:
    json.dump({
        'accuracy': accuracy,
        'results': final_output
    }, f, indent=4, ensure_ascii=False)

print(f"Results saved to {output_path}")





