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
    local_path = config.get('local_path', '').replace('~', os.path.expanduser('~'))
    num_gpus = config.get('num_gpus', 1)
    generation_params = config.get('generation_params', {})
    
    # Initialize vLLM client with fallback to local path
    try:
        print(f"Attempting to load model {model_id} from Hugging Face...")
        client = LLM(model=model_id, tensor_parallel_size=num_gpus)
    except Exception as e:
        if local_path and os.path.exists(local_path):
            print(f"Model not found online. Using local model at: {local_path}")
            client = LLM(model=local_path, tensor_parallel_size=num_gpus)
        else:
            print(f"Error: Could not load model {model_id} online or locally. {str(e)}")
            if local_path:
                print(f"Local path {local_path} does not exist or is invalid.")
            return
    
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
    parser.add_argument('--model', type=str, help='Specific model to test (optional)')
    parser.add_argument('--list-models', action='store_true', help='List available models and exit')
    args = parser.parse_args()
    
    full_prompt = load_prompts('prompts/system_prompt_template.txt', 'prompts/user_prompt_template.txt', args.prompt)
    cfgs = glob.glob('configs/model_specific/vllm/*.json')
    
    # List models if requested
    if args.list_models:
        print("Available vLLM models:")
        for i, cfg_path in enumerate(sorted(cfgs), 1):
            with open(cfg_path, 'r') as f:
                cfg = json.load(f)
                print(f"{i}. {cfg['model_id']} ({cfg_path})")
        exit(0)
    
    # Filter models if specific model requested
    if args.model:
        model_cfgs = []
        for cfg_path in cfgs:
            with open(cfg_path, 'r') as f:
                cfg = json.load(f)
                if args.model.lower() in cfg['model_id'].lower():
                    model_cfgs.append(cfg_path)
        if not model_cfgs:
            print(f"Error: No models found matching '{args.model}'")
            exit(1)
        cfgs_to_test = model_cfgs
    else:
        cfgs_to_test = cfgs
    
    for cfg in cfgs_to_test:
        print(f"Testing vLLM model config: {cfg}")
        test_vllm_model(cfg, full_prompt) 