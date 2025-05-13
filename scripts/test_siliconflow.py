#!/usr/bin/env python3
import os
import time
import json
import glob
import argparse
import requests
from pathlib import Path
from dotenv import load_dotenv
from utils.metrics_calculator import calculate_tokens_per_sec
from utils.results_saver import save_output_and_metrics

# Load environment variables from .env file
load_dotenv()

def load_prompts(system_path, user_path):
    with open(system_path, 'r') as f:
        system_prompt = f.read()
    with open(user_path, 'r') as f:
        user_prompt = f.read()
    return system_prompt, user_prompt

def call_siliconflow_api(api_key, model_id, system_prompt, user_prompt, generation_params):
    base_url = "https://api.siliconflow.cn/v1"
    headers = {'Authorization': f"Bearer {api_key}", 'Content-Type': 'application/json'}
    
    endpoint = f"{base_url}/chat/completions"
    payload = {
        'model': model_id,
        'messages': [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        **generation_params
    }
    
    print(f"Calling {endpoint} for model {model_id}")
    start = time.time()
    try:
        resp = requests.post(endpoint, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        elapsed = time.time() - start
        
        output = data.get('choices', [{}])[0].get('message', {}).get('content', '')
        print(f"Success! Response received in {elapsed:.2f} seconds")
        
        # Save results
        metrics = {
            'input_tokens': len(system_prompt.split()) + len(user_prompt.split()),
            'output_tokens': len(output.split()),
            'total_time_sec': elapsed,
            'tokens_per_sec': calculate_tokens_per_sec(system_prompt + user_prompt, output, elapsed)
        }
        save_output_and_metrics(output, metrics, "siliconflow", model_id.replace('/', '_'))
        
        return output, elapsed
            
    except Exception as e:
        elapsed = time.time() - start
        print(f"Error calling API: {str(e)}")
        if 'resp' in locals() and resp:
            print(f"Response status: {resp.status_code}")
            print(f"Response: {resp.text}")
        return f"Error: {str(e)}", elapsed

def test_model(model_id, system_prompt, user_prompt, generation_params):
    print(f"\n=== Testing {model_id} ===")
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        print("Error: SILICONFLOW_API_KEY environment variable not found.")
        print("Please create a .env file with your API key or set it in your environment.")
        return 0
    output, elapsed = call_siliconflow_api(api_key, model_id, system_prompt, user_prompt, generation_params)
    
    if not output.startswith("Error:"):
        print(f"\nResponse (truncated to 200 chars):\n{output[:200]}...\n")
        print(f"Total time: {elapsed:.2f} seconds")
    return elapsed

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Test SiliconFlow models with configurable token length')
    parser.add_argument('--max-tokens', type=int, default=4096, help='Maximum tokens to generate')
    parser.add_argument('--model', type=str, help='Specific model to test (optional)')
    parser.add_argument('--list-models', action='store_true', help='List available models and exit')
    args = parser.parse_args()

    # Load all model configs
    model_configs = {}
    for config_file in glob.glob('configs/model_specific/siliconflow/*.json'):
        with open(config_file, 'r') as f:
            config = json.load(f)
            model_id = config['model_id']
            model_configs[model_id] = config['generation_params']
    
    # List models if requested
    if args.list_models:
        print("Available SiliconFlow models:")
        for i, model_id in enumerate(sorted(model_configs.keys()), 1):
            print(f"{i}. {model_id}")
        exit(0)

    # Load prompts
    system_prompt, user_prompt = load_prompts('prompts/system_prompt_template.txt', 'prompts/user_prompt_template.txt')
    
    # Override max_tokens if specified
    if args.max_tokens:
        for config in model_configs.values():
            config['max_tokens'] = args.max_tokens
    
    # Filter models if specific model requested
    if args.model:
        matching_models = [m for m in model_configs.keys() if args.model.lower() in m.lower()]
        if not matching_models:
            print(f"Error: No models found matching '{args.model}'")
            exit(1)
        models_to_test = matching_models
    else:
        models_to_test = list(model_configs.keys())
    
    print(f"Starting API tests with SiliconFlow models (max_tokens: {args.max_tokens})...")
    all_times = {}
    
    for model_id in models_to_test:
        try:
            elapsed = test_model(model_id, system_prompt, user_prompt, model_configs[model_id])
            all_times[model_id] = elapsed
        except Exception as e:
            print(f"Error testing {model_id}: {str(e)}")
    
    # Summary
    print("\n=== Summary ===")
    for model, time in all_times.items():
        print(f"{model}: {time:.2f} seconds")
    print("\nTests completed. Check results directory for full outputs and metrics.") 