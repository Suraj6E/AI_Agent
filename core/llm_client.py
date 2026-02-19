# ============================================================
# FILE: core/llm_client.py
# ============================================================
# Talks to Ollama's OpenAI-compatible API.
# Three functions:
#   chat()           - send messages, get response
#   extract_response() - parse the response into text or tool_calls
#   health_check()   - verify Ollama is running and model is available
# ============================================================

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-r1:8b")


def chat(messages, tools=None, temperature=0.7, max_tokens=2048):
    url = f"{BASE_URL}/chat/completions"

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.ConnectionError:
        return {
            "error": True,
            "message": (
                f"Cannot connect to Ollama at {BASE_URL}. "
                "Is Ollama running? It should start automatically on Windows. "
                "If not, launch the Ollama app or run 'ollama serve' in a terminal."
            ),
        }
    except requests.exceptions.Timeout:
        return {
            "error": True,
            "message": "Request timed out after 120 seconds. The model may be loading or overloaded.",
        }
    except requests.exceptions.HTTPError as e:
        return {
            "error": True,
            "message": f"HTTP error from Ollama: {e.response.status_code} — {e.response.text}",
        }


def extract_response(api_result):
    if api_result.get("error"):
        return {"type": "error", "content": api_result["message"]}

    try:
        choice = api_result["choices"][0]
        message = choice["message"]

        if message.get("tool_calls"):
            return {"type": "tool_calls", "content": message["tool_calls"]}

        return {"type": "text", "content": message.get("content", "")}

    except (KeyError, IndexError) as e:
        return {
            "type": "error",
            "content": f"Unexpected response format: {e}\nRaw: {json.dumps(api_result, indent=2)[:500]}",
        }


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