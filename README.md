# LLM FJSP Benchmark

This project benchmarks the performance of various Large Language Models (LLMs) on Flexible Job Shop Scheduling Problems (FJSP), both via online APIs and local deployments using vLLM and TensorRT-LLM.

GitHub Repository: [https://github.com/LLM4DRL/LLM_FJSP_Benchmark](https://github.com/LLM4DRL/LLM_FJSP_Benchmark)

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set API keys:
   ```bash
   # Option 1: Copy the template and add your API keys
   cp .env.template .env
   # Edit .env to add your API keys
   
   # Option 2: Set environment variables directly
   export AIGCBEST_API_KEY=your_aigcbest_api_key
   export SILICONFLOW_API_KEY=your_siliconflow_api_key
   ```

## Running Tests
- Online API tests: `python scripts/test_online_api.py --prompt "Run scheduling test"`
- GPT-4.1 series test: `python scripts/single_test_online.py --max-tokens 4096`
- SiliconFlow models: `python scripts/test_siliconflow.py [--max-tokens 4096] [--model MODEL_NAME] [--list-models]`
- vLLM tests: `python scripts/test_vllm.py --prompt "Run scheduling test"`

## Available Models

### SiliconFlow Models
- DeepSeek models
  - Pro/deepseek-ai/DeepSeek-V3
  - Pro/deepseek-ai/DeepSeek-R1
- Qwen3 models
  - Qwen/QwQ-32B (new)
  - Qwen/Qwen3-30B-A3B
  - Qwen/Qwen3-14B
  - Qwen/Qwen3-8B
- GLM models
  - THUDM/GLM-Z1-32B-0414
  - THUDM/GLM-4-32B-0414

### Online Models (via aigcbest API)
- GPT models
  - gpt-4.1
  - gpt-4.1-mini
  - gpt-4.1-nano
- Claude models
  - claude-3-5-sonnet
  - claude-3-7-sonnet
  - claude-3-7-sonnet-20250219-thinking
- Gemini models
  - gemini-2.5-pro
- Other models
  - o4-mini
  - llama3-8b-instruct
  - llama3-70b-instruct

## Project Structure
See `project.md` for the full folder layout and descriptions.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
MIT 