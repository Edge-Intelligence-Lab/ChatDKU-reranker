VISIBLE_CUDA_DEVICES=0 HF_HOME=/datapool/huggingface/ HF_ENDPOINT=https://hf-mirror.com ./.venv/bin/python3 \
  -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-VL-Reranker-8B \
  --trust-remote-code \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 6767

# Uses default sampling params from model for both chat and completions {'temperature': 0.7, 'top_k': 20, 'top_p': 0.8}
