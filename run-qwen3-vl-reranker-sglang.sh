VISIBLE_CUDA_DEVICES=4 HF_HOME=/datapool/huggingface/ HF_ENDPOINT=https://hf-mirror.com ./.venv/bin/python3 -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 30000 \
  --model-path Qwen/Qwen3-VL-Reranker-8B \
  --trust-remote-code \
  --dtype auto \
  --tp-size 1 \
  --disable-cuda-graph
  # --chat-template none

