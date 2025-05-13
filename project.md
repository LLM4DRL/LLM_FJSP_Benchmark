# Project Folder Structure: LLM Performance Testing

This document outlines a suggested folder structure for a project focused on testing the performance of various Large Language Models (LLMs), including online APIs and locally deployed models with vLLM (and potentially TensorRT-LLM).

## Root Directory Structure

llm_performance_testing/├── README.md├── requirements.txt├── prompts/│   ├── system_prompt_template.txt│   └── user_prompt_template.txt├── configs/│   ├── api_providers.json│   ├── output_format_template.json  # Or described in README│   └── model_specific/│       ├── online_vendor/│       │   ├── proprietary_model_A.json│       │   └── proprietary_model_B.json│       ├── siliconflow/│       │   ├── qwen1.5-7b-chat_sf.json│       │   └── llama3-8b-instruct_sf.json│       └── vllm/│           ├── qwen1.5-7b-chat_1gpu.json│           ├── qwen1.5-7b-chat_4gpu.json│           ├── qwen1.5-7b-chat_8gpu.json│           ├── llama3-8b-instruct_1gpu.json│           ├── llama3-8b-instruct_2gpu.json│           ├── llama3-8b-instruct_4gpu.json│           ├── # ... other open-source models and GPU configurations│       └── tensorrt_llm/ # Optional, if distinct from vLLM setup│           └── # ... model_trt_config.json├── scripts/│   ├── test_online_api.py│   ├── test_vllm.py│   ├── test_tensorrt_llm.py # Optional│   └── utils/│       ├── init.py│       ├── metrics_calculator.py│       └── results_saver.py└── results/├── README.md # Explains how results are stored, naming conventions├── online_vendor/│   └── proprietary_model_A/│       ├── run_YYYYMMDD_HHMMSS_output.txt│       └── run_YYYYMMDD_HHMMSS_metrics.json├── siliconflow/│   └── qwen1.5-7b-chat_sf/│       ├── run_YYYYMMDD_HHMMSS_output.txt│       └── run_YYYYMMDD_HHMMSS_metrics.json├── vllm/│   └── qwen1.5-7b-chat_4gpu/│       ├── run_YYYYMMDD_HHMMSS_output.txt│       └── run_YYYYMMDD_HHMMSS_metrics.json└── summary_report.csv # Aggregated key metrics from all tests
## Description of Components

### 1. `README.md` (Root)
* **Purpose:** Main project documentation.
* **Contents:**
    * Project overview and goals.
    * Setup instructions (installing dependencies, configuring API keys, vLLM server setup).
    * How to run the different test scripts.
    * Explanation of the configuration files.
    * Description of the expected output format (if not using `output_format_template.json`).
    * Notes on how results are stored and naming conventions.

### 2. `requirements.txt`
* **Purpose:** Lists all Python dependencies for the project.
* **Example Contents:**
    ```
    openai
    vllm
    pandas
    # tensorrt_llm (if used)
    # other necessary libraries (e.g., for specific API clients, YAML parsing)
    ```

### 3. `prompts/`
* **Purpose:** Stores the prompt templates.
* **Files:**
    * `system_prompt_template.txt`: The template for the system prompt.
    * `user_prompt_template.txt`: The template for the user prompt.
    * *(Optional)*: You can add more specific prompt files if tests require different prompt structures for different models or scenarios.

### 4. `configs/`
* **Purpose:** Contains all configuration files for the tests.
* **Files & Subdirectories:**
    * `api_providers.json`:
        * **Purpose:** Stores base URLs and API key environment variable names for the online API providers.
        * **Example:**
            ```json
            {
              "second_hand_vendor": {
                "base_url": "[https://api.vendor.com/v1](https://api.vendor.com/v1)",
                "api_key_env_var": "VENDOR_API_KEY"
              },
              "siliconflow": {
                "base_url": "[https://api.siliconflow.com/v1](https://api.siliconflow.com/v1)",
                "api_key_env_var": "SILICONFLOW_API_KEY"
              }
            }
            ```
    * `output_format_template.json` (or described in `README.md`):
        * **Purpose:** Defines the structure of the expected output from the LLMs if it's a structured format like JSON. If it's plain text, this might just be a description.
    * `model_specific/`:
        * **Purpose:** Contains individual configuration files for each model and deployment type being tested.
        * **Subdirectories:**
            * `online_vendor/`: Configs for models accessed via the "second hand api vendor".
            * `siliconflow/`: Configs for models accessed via SiliconFlow.
            * `vllm/`: Configs for models tested with vLLM.
            * `tensorrt_llm/` (Optional): Configs for TensorRT-LLM if the setup or parameters are significantly different from vLLM.
        * **Example `model_specific/vllm/qwen1.5-7b-chat_4gpu.json`:**
            ```json
            {
              "model_id": "Qwen/Qwen1.5-7B-Chat", // Path or identifier for vLLM
              "deployment_type": "vllm",
              "num_gpus": 4,
              "dtype": "auto", // or bfloat16, float16
              "max_model_len": null, // or specific value
              // Other vLLM specific parameters (e.g., quantization, tensor_parallel_size if not just num_gpus)
              "generation_params": {
                "max_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9
                // other generation parameters
              }
            }
            ```
        * **Example `model_specific/online_vendor/proprietary_model_A.json`:**
            ```json
            {
              "model_id": "vendor-model-name-v1", // Model name as per vendor's API
              "deployment_type": "online",
              "provider": "second_hand_vendor", // Key from api_providers.json
              "generation_params": {
                "max_tokens": 512,
                "temperature": 0.7
                // other generation parameters
              }
            }
            ```

### 5. `scripts/`
* **Purpose:** Contains the Python scripts to run the performance tests.
* **Files & Subdirectories:**
    * `test_online_api.py`:
        * **Function:** Reads model configs from `configs/model_specific/online_vendor/` and `configs/model_specific/siliconflow/`. Uses `api_providers.json` for endpoint details. Loads prompts. Iterates through specified online models, sends requests, measures time, calculates tokens/second, and saves the raw output and metrics using `utils/`.
    * `test_vllm.py`:
        * **Function:** Reads model configs from `configs/model_specific/vllm/`. Loads prompts. Initializes vLLM engine (or connects to a running server) based on config (e.g., `model_id`, `num_gpus`). Sends requests, measures time, calculates tokens/second, and saves results using `utils/`.
    * `test_tensorrt_llm.py` (Optional):
        * **Function:** Similar to `test_vllm.py`, but tailored for TensorRT-LLM specific SDK or serving framework.
    * `utils/`:
        * **Purpose:** Shared utility functions.
        * `__init__.py`: Makes `utils` a Python package.
        * `metrics_calculator.py`: Functions to calculate tokens per second, total time, potentially first token latency, etc.
        * `results_saver.py`: Functions to save the generated text output and performance metrics to the `results/` directory in a structured way.

### 6. `results/`
* **Purpose:** Stores all outputs and performance metrics from the test runs.
* **Files & Subdirectories:**
    * `README.md`: Explains the structure within this directory, naming conventions for files and folders (e.g., timestamping runs).
    * Subdirectories named after the provider/deployment type (e.g., `online_vendor/`, `siliconflow/`, `vllm/`).
    * Within each provider/deployment directory, further subdirectories for each specific model configuration tested (e.g., `proprietary_model_A/`, `qwen1.5-7b-chat_4gpu/`).
    * Inside each model-config specific directory:
        * `run_YYYYMMDD_HHMMSS_output.txt`: The raw text output generated by the LLM for a specific run.
        * `run_YYYYMMDD_HHMMSS_metrics.json`: A JSON file containing performance metrics for that run (e.g., total generation time, number of input tokens, number of output tokens, tokens per second, configuration used).
    * `summary_report.csv`:
        * **Purpose:** An aggregated CSV file for easy comparison of key metrics across all tests. This can be generated by a separate script or appended to by the test scripts.
        * **Example Columns:** `timestamp`, `model_id`, `deployment_type`, `num_gpus`, `input_tokens`, `output_tokens`, `total_time_sec`, `tokens_per_sec`.

## Workflow

1.  **Setup:**
    * Clone the repository (if applicable).
    * Install dependencies from `requirements.txt`.
    * Configure API keys as environment variables (referenced in `api_providers.json`).
    * Prepare prompt templates in `prompts/`.
    * Create/verify model configuration files in `configs/model_specific/`.
2.  **Testing:**
    * Run `scripts/test_online_api.py` to test models from online providers.
    * Run `scripts/test_vllm.py` to test locally deployed vLLM models (ensure vLLM server is running or the script handles engine initialization).
3.  **Review:**
    * Check raw outputs in the respective `results/` subdirectories for quality and adherence to the expected output template.
    * Analyze metrics in the `.json` files and the `summary_report.csv`.

This structure should provide a good starting point and can be adapted as your project evolves.
