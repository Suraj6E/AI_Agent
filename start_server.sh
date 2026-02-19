#!/bin/bash
# This file can be removed for windows — replaced by start_server.py (Windows-compatible)

# Start vLLM server for GLM-4-9B
#
# --trust-remote-code    : required for GLM-4 (custom model code from HuggingFace)
# --served-model-name    : gives the model a clean name in the API
# --max-model-len 8192   : limits context window to stay within 16GB VRAM
# --dtype float16        : full half-precision, no quantization quality loss
# --port 8000            : default port, change if needed

python -m vllm.entrypoints.openai.api_server \
    --model ./models/glm-4-9b-chat \
    --served-model-name glm-4-9b \
    --trust-remote-code \
    --max-model-len 8192 \
    --dtype float16 \
    --port 8000