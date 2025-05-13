#!/usr/bin/env python3
import os
import json
import glob
import time
import argparse
from vllm import LLM
from utils.metrics_calculator import calculate_tokens_per_sec
from utils.results_saver import save_output_and_metrics

def load_prompts(system_path, user_path, user_input):
    with open(system_path, 'r') as f:
        system_template = f.read()
    with open(user_path, 'r') as f:
        user_template = f.read()
    system_prompt = system_template.replace('{system_message}', '')
    user_prompt = user_template.replace('{user_input}', user_input)
    return f"{system_prompt}\n{user_prompt}"

def test_vllm_model(config_path, prompt):
    config = json.load(open(config_path))
    model_id = config['model_id']
    num_gpus = config.get('num_gpus', 1)
    generation_params = config.get('generation_params', {})
    # Initialize vLLM client
    client = LLM(model=model_id, tensor_parallel_size=num_gpus)
    # Generate output
    start = time.time()
    outputs = client.generate(prompt=prompt, **generation_params)
    # Collect generated text
    result_text = ''
    for output in outputs:
        result_text += output.text
    elapsed = time.time() - start
    metrics = {
        'input_tokens': len(prompt.split()),
        'output_tokens': len(result_text.split()),
        'total_time_sec': elapsed,
        'tokens_per_sec': calculate_tokens_per_sec(prompt, result_text, elapsed)
    }
    # Use model_id and GPU count as folder name
    folder_name = f"{model_id.replace('/', '_')}_{num_gpus}gpu"
    save_output_and_metrics(result_text, metrics, config['deployment_type'], folder_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test vLLM models')
    parser.add_argument('--prompt', type=str, required=True, help='User input prompt')
    args = parser.parse_args()
    full_prompt = load_prompts('prompts/system_prompt_template.txt', 'prompts/user_prompt_template.txt', args.prompt)
    cfgs = glob.glob('configs/model_specific/vllm/*.json')
    for cfg in cfgs:
        print(f"Testing vLLM model config: {cfg}")
        test_vllm_model(cfg, full_prompt) 