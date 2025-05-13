#!/usr/bin/env python3
import os
import json
import glob
import time
import argparse
import fnmatch
import sys
import psutil
import torch
import traceback
from datetime import datetime
from vllm import LLM, SamplingParams
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

def get_gpu_memory_usage():
    """Get GPU memory usage in MiB for each GPU"""
    if torch.cuda.is_available():
        memory_stats = {}
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 2)  # MiB
            memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)    # MiB
            memory_stats[f"gpu_{i}"] = {
                "allocated_MiB": memory_allocated,
                "reserved_MiB": memory_reserved
            }
        return memory_stats
    return {"error": "CUDA not available"}

def get_system_metrics():
    """Get system-wide CPU and memory usage"""
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_used_GB": psutil.virtual_memory().used / (1024 ** 3)
    }

def test_vllm_model(config_path, prompt, repeats=1):
    try:
        config = json.load(open(config_path))
        model_id = config['model_id']
        local_path = config.get('local_path', '').replace('~', os.path.expanduser('~'))
        num_gpus = config.get('num_gpus', 1)
        generation_params = config.get('generation_params', {})
        
        # Initialize vLLM client with fallback to local path
        model_load_start = time.time()
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
                return False
        
        model_load_time = time.time() - model_load_start
        print(f"Model loaded in {model_load_time:.2f} seconds")
        
        # Prepare advanced metrics collection
        metrics = {
            'model_id': model_id,
            'num_gpus': num_gpus,
            'model_load_time_sec': model_load_time,
            'input_tokens': len(prompt.split()),
            'runs': []
        }
        
        # Get GPU info
        metrics['gpu_info'] = get_gpu_memory_usage()
        
        # Multiple runs for more reliable benchmarking
        for run in range(repeats):
            print(f"Run {run+1}/{repeats}...")
            run_metrics = {}
            
            # System metrics before generation
            run_metrics['pre_generation_system'] = get_system_metrics()
            run_metrics['pre_generation_gpu'] = get_gpu_memory_usage()
            
            # Generate output with timestamps for token-by-token timing
            start_time = time.time()
            sampling_params = SamplingParams(**generation_params)
            outputs = client.generate(prompt, sampling_params=sampling_params, use_tqdm=True)
            
            # Collect generated text
            result_text = ''
            for output in outputs:
                result_text += output.text
            
            generation_time = time.time() - start_time
            
            # System metrics after generation
            run_metrics['post_generation_system'] = get_system_metrics()
            run_metrics['post_generation_gpu'] = get_gpu_memory_usage()
            
            # Token-specific metrics
            run_metrics['output_tokens'] = len(result_text.split())
            run_metrics['output_chars'] = len(result_text)
            run_metrics['total_time_sec'] = generation_time
            run_metrics['tokens_per_sec'] = run_metrics['output_tokens'] / generation_time
            run_metrics['time_per_token_ms'] = (generation_time * 1000) / run_metrics['output_tokens'] if run_metrics['output_tokens'] > 0 else 0
            
            # For the first run, save the output text
            if run == 0:
                metrics['output_text'] = result_text
            
            metrics['runs'].append(run_metrics)
        
        # Calculate average metrics across all runs
        if repeats > 1:
            metrics['avg_generation_time_sec'] = sum(run['total_time_sec'] for run in metrics['runs']) / repeats
            metrics['avg_tokens_per_sec'] = sum(run['tokens_per_sec'] for run in metrics['runs']) / repeats
            metrics['avg_time_per_token_ms'] = sum(run['time_per_token_ms'] for run in metrics['runs']) / repeats
            metrics['std_dev_generation_time'] = (sum((run['total_time_sec'] - metrics['avg_generation_time_sec'])**2 for run in metrics['runs']) / repeats)**0.5
        
        # Use model_id and GPU count as folder name
        folder_name = f"{model_id.replace('/', '_')}_{num_gpus}gpu"
        
        # Save output and detailed metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = os.path.join('results', 'vllm', folder_name, f"run_{timestamp}_detailed_metrics.json")
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Also save a simplified version for compatibility
        simplified_metrics = {
            'input_tokens': metrics['input_tokens'],
            'output_tokens': metrics['runs'][0]['output_tokens'],
            'total_time_sec': metrics['runs'][0]['total_time_sec'],
            'tokens_per_sec': metrics['runs'][0]['tokens_per_sec'],
            'model_load_time_sec': metrics['model_load_time_sec'],
            'gpu_memory_usage_MiB': {k: v['reserved_MiB'] for k, v in metrics['gpu_info'].items()} if 'error' not in metrics['gpu_info'] else "N/A"
        }
        
        save_output_and_metrics(metrics['output_text'], simplified_metrics, config['deployment_type'], folder_name)
        
        print(f"Model: {model_id} with {num_gpus} GPUs")
        print(f"Load time: {metrics['model_load_time_sec']:.2f}s")
        print(f"Generation time: {metrics['runs'][0]['total_time_sec']:.2f}s")
        print(f"Tokens/sec: {metrics['runs'][0]['tokens_per_sec']:.2f}")
        print(f"Time per token: {metrics['runs'][0]['time_per_token_ms']:.2f}ms")
        
        return True
    except Exception as e:
        print(f"Failed to test model config {config_path}: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test vLLM models')
    parser.add_argument('--prompt', type=str, required=True, help='User input prompt')
    parser.add_argument('--model', type=str, help='Specific model to test (supports glob patterns like Qwen3-*)')
    parser.add_argument('--list-models', action='store_true', help='List available models and exit')
    parser.add_argument('--repeats', type=int, default=1, help='Number of times to repeat each test for more reliable results')
    args = parser.parse_args()
    
    full_prompt = load_prompts('prompts/system_prompt_template.txt', 'prompts/user_prompt_template.txt', args.prompt)
    cfgs = glob.glob('configs/model_specific/vllm/*.json')
    
    # List models if requested
    if args.list_models:
        print("Available vLLM models:")
        for i, cfg_path in enumerate(sorted(cfgs), 1):
            try:
                with open(cfg_path, 'r') as f:
                    cfg = json.load(f)
                    print(f"{i}. {cfg['model_id']} ({cfg_path})")
            except:
                print(f"{i}. [Error parsing] {cfg_path}")
        exit(0)
    
    # Filter models if specific model requested
    if args.model:
        model_cfgs = []
        pattern = args.model.lower().replace('*', '.*')
        for cfg_path in cfgs:
            try:
                with open(cfg_path, 'r') as f:
                    cfg = json.load(f)
                    model_id = cfg['model_id'].lower()
                    if fnmatch.fnmatch(model_id, args.model.lower()) or args.model.lower() in model_id:
                        model_cfgs.append(cfg_path)
            except Exception as e:
                print(f"Warning: Could not parse config {cfg_path}: {str(e)}")
                continue
                
        if not model_cfgs:
            print(f"Error: No models found matching '{args.model}'")
            exit(1)
        cfgs_to_test = model_cfgs
    else:
        cfgs_to_test = cfgs
    
    # Track results
    successful_tests = 0
    failed_tests = 0
    
    print(f"Found {len(cfgs_to_test)} model configuration(s) to test")
    print(f"Each test will be repeated {args.repeats} time(s) for more reliable results")
    
    results_summary = []
    
    for idx, cfg in enumerate(cfgs_to_test, 1):
        print(f"\n[{idx}/{len(cfgs_to_test)}] Testing vLLM model config: {cfg}")
        success = test_vllm_model(cfg, full_prompt, args.repeats)
        if success:
            successful_tests += 1
            # Add basic info to results summary
            try:
                with open(cfg, 'r') as f:
                    config = json.load(f)
                    results_summary.append({
                        "model_id": config['model_id'],
                        "num_gpus": config.get('num_gpus', 1),
                        "status": "Successful"
                    })
            except:
                results_summary.append({"config": cfg, "status": "Successful but couldn't parse config"})
        else:
            failed_tests += 1
            results_summary.append({"config": cfg, "status": "Failed"})
    
    print(f"\nTesting complete. {successful_tests} successful, {failed_tests} failed.")
    
    # Save summary of all tested models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join('results', 'vllm', f"summary_{timestamp}.json")
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2) 