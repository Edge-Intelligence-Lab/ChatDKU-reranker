CUDA_VISIBLE_DEVICES=6 vllm serve Qwen/Qwen3-VL-Reranker-8B \
  --runner pooling \
  --host 0.0.0.0 \
  --port 6767 \
  --trust-remote-code \
  --dtype bfloat16
