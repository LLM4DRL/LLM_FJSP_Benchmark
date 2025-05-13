#!/usr/bin/env python3
import os
import time
import json
import glob
import argparse
from openai import OpenAI
from dotenv import load_dotenv
from utils.metrics_calculator import calculate_tokens_per_sec
from utils.results_saver import save_output_and_metrics

# Load environment variables from .env file
load_dotenv()

def load_prompts(system_path, user_path, user_input):
    with open(system_path, 'r') as f:
        system_template = f.read()
    with open(user_path, 'r') as f:
        user_template = f.read()
    # system_prompt = system_template.replace('{system_message}', '')
    # user_prompt = user_template.replace('{user_input}', user_input)
    return system_template, user_template

def call_api(provider_info, model_id, system_prompt, user_prompt, generation_params):
    base_url = provider_info['base_url']
    api_key = os.getenv(provider_info['api_key_env_var'])
    
    # Initialize the OpenAI client with the base URL and API key
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    
    # Extract thinking parameter if it exists
    thinking = generation_params.pop('thinking', False)
    
    # Determine if using chat completion or regular completion
    if 'gpt' in model_id.lower() or 'claude' in model_id.lower():
        start = time.time()
        try:
            print(f"Calling ChatCompletion for model {model_id}")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                **generation_params,
                extra_body={"thinking": thinking} if thinking is not None else {}
            )
            
            elapsed = time.time() - start
            output = response.choices[0].message.content
            
            return output, elapsed
        except Exception as e:
            print(f"Error calling API: {str(e)}")
            return f"Error: {str(e)}", time.time() - start
    else:
        start = time.time()
        try:
            print(f"Calling Completion for model {model_id}")
            prompt = f"{system_prompt}\n{user_prompt}"
            
            response = client.completions.create(
                model=model_id,
                prompt=prompt,
                **generation_params,
                extra_body={"thinking": thinking} if thinking is not None else {}
            )
            
            elapsed = time.time() - start
            output = response.choices[0].text
            
            return output, elapsed
        except Exception as e:
            print(f"Error calling API: {str(e)}")
            return f"Error: {str(e)}", time.time() - start

def test_model(config_path, system_prompt, user_prompt):
    config = json.load(open(config_path))
    model_id = config['model_id']
    provider_key = config['provider']
    generation_params = config.get('generation_params', {})
    api_providers = json.load(open('configs/api_providers.json'))
    provider_info = api_providers.get(provider_key)
    if provider_info is None:
        print(f"Unknown provider '{provider_key}' in {config_path}")
        return
    output, elapsed = call_api(provider_info, model_id, system_prompt, user_prompt, generation_params)
    metrics = {
        'input_tokens': len(system_prompt.split()) + len(user_prompt.split()),
        'output_tokens': len(output.split()),
        'total_time_sec': elapsed,
        'tokens_per_sec': calculate_tokens_per_sec(system_prompt + user_prompt, output, elapsed)
    }
    save_output_and_metrics(output, metrics, config['deployment_type'], model_id)

if __name__ == '__main__':
    system_prompt, user_prompt = load_prompts('prompts/system_prompt_template.txt', 'prompts/user_prompt_template.txt')
    # Gather all online model configs
    online_cfgs = glob.glob('configs/model_specific/online_vendor/*.json')
    online_cfgs += glob.glob('configs/model_specific/siliconflow/*.json')
    for cfg in online_cfgs:
        print(f"Testing online model config: {cfg}")
        test_model(cfg, system_prompt, user_prompt) 