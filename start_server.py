"""
Start vLLM server for GLM-4-9B.

Flags explained:
  --trust-remote-code    : required for GLM-4 (custom model code from HuggingFace)
  --served-model-name    : gives the model a clean name in the API
  --max-model-len 8192   : limits context window to stay within 16GB VRAM
  --dtype float16        : full half-precision, no quantization quality loss
  --port 8000            : default port, change in .env if needed

Usage: python start_server.py
"""

import os
import subprocess
import sys
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "./models/glm-4-9b-chat")
MODEL_NAME = os.getenv("MODEL_NAME", "glm-4-9b")
PORT = os.getenv("VLLM_PORT", "8000")
MAX_MODEL_LEN = os.getenv("VLLM_MAX_MODEL_LEN", "8192")

cmd = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL_PATH,
    "--served-model-name", MODEL_NAME,
    "--trust-remote-code",
    "--max-model-len", MAX_MODEL_LEN,
    "--dtype", "float16",
    "--port", PORT,
]

print(f"Starting vLLM server...")
print(f"Model: {MODEL_PATH}")
print(f"Port: {PORT}")
print(f"Command: {' '.join(cmd)}\n")

subprocess.run(cmd)