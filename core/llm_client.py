# ============================================================
# FILE: core/llm_client.py
# ============================================================
# Talks to Ollama's OpenAI-compatible API.
# Simplified: no tools parameter. We handle tool calling ourselves
# by parsing the model's text output. This works with ANY model.
#
# Three functions:
#   chat()         - send messages, get text back
#   health_check() - verify Ollama is running and model is available
# ============================================================
#
# Changes:
#   - Timeout is now configurable via LLM_TIMEOUT in .env (default: 300s).
#     DeepSeek-R1's <think> phase can exceed 120s on complex specialist prompts.
#   - Retry logic added via LLM_MAX_RETRIES in .env (default: 2).
#     A single timeout no longer kills the whole multi-agent run.
# ============================================================

import os
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-r1:8b")
TIMEOUT = int(os.getenv("LLM_TIMEOUT", "300"))
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))


def chat(messages, temperature=0.7, max_tokens=2048):
    url = f"{BASE_URL}/chat/completions"

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(url, json=payload, timeout=TIMEOUT)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

        except requests.exceptions.ConnectionError:
            return (
                f"[ERROR] Cannot connect to Ollama at {BASE_URL}. "
                "Is Ollama running? Launch the Ollama app or run 'ollama serve'."
            )
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                print(f"  [LLM] Timeout on attempt {attempt}/{MAX_RETRIES}, retrying...")
                continue
            return f"[ERROR] Request timed out after {TIMEOUT}s ({MAX_RETRIES} attempts)."
        except requests.exceptions.HTTPError as e:
            return f"[ERROR] HTTP {e.response.status_code}: {e.response.text}"
        except (KeyError, IndexError) as e:
            return f"[ERROR] Unexpected response format: {e}"


def health_check():
    try:
        response = requests.get(f"{BASE_URL}/models", timeout=10)
        response.raise_for_status()
        models = response.json().get("data", [])
        model_ids = [m["id"] for m in models]

        if MODEL_NAME in model_ids:
            return True, f"Ollama is running. Model '{MODEL_NAME}' is available."

        return False, (
            f"Ollama is running but model '{MODEL_NAME}' not found. "
            f"Available: {model_ids}\n"
            f"Pull it with: ollama pull {MODEL_NAME}"
        )

    except requests.exceptions.ConnectionError:
        return False, (
            f"Cannot connect to Ollama at {BASE_URL}. "
            "Is Ollama running? Launch the Ollama app or run 'ollama serve'."
        )
    except Exception as e:
        return False, f"Health check failed: {e}"