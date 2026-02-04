VISIBLE_CUDA_DEVICES=0 HF_HOME=/datapool/huggingface/ HF_ENDPOINT=https://hf-mirror.com ./.venv/bin/python3 \
  -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-VL-Reranker-8B \
  --trust-remote-code \
  --task score \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 6767

