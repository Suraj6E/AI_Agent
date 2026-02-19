"""
Download GLM-4-9B-Chat from HuggingFace to local models folder.
Uses safetensors format only — skips .bin and .pt files to save disk space.

Run this once before starting the vLLM server.
Usage: python download_model.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()

MODEL_ID = os.getenv("HF_MODEL_ID", "THUDM/glm-4-9b-chat")
MODEL_PATH = os.getenv("MODEL_PATH", "./models/glm-4-9b-chat")


def download():
    target = Path(MODEL_PATH)
    target.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {MODEL_ID} to {target.resolve()}")
    print("This will take a while (~18GB). Safetensors format only.\n")

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(target),
        ignore_patterns=["*.bin", "*.pt", "consolidated*"],
    )

    print(f"\nDone. Model saved to: {target.resolve()}")
    print("Next step: start the vLLM server with start_server.py or the command in start_server.sh")


if __name__ == "__main__":
    download()