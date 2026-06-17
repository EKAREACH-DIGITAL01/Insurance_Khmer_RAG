from pathlib import Path
import subprocess
import time
import requests

MODELS = {
    "1": "SeaLLMs-v3-7B-Chat-Q4_K_M.gguf",
    "2": "SeaLLMs-v3-7B-Chat-Q5_K_M.gguf",
    "3": "SeaLLMs-v3-7B-Chat-Q6_K.gguf",
    "4": "SeaLLMs-v3-7B-Chat-Q8_0.gguf",
}

print("""
========================================
Khmer Legal Chatbot
========================================

Choose model:

1. SeaLLMs-v3-7B-Chat-Q4_K_M.gguf
2. SeaLLMs-v3-7B-Chat-Q5_K_M.gguf
3. SeaLLMs-v3-7B-Chat-Q6_K.gguf
4. SeaLLMs-v3-7B-Chat-Q8_0.gguf
""")

choice = input("Choice: ").strip()

if choice not in MODELS:
    raise SystemExit("Invalid choice.")

model = MODELS[choice]

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "generator" / model

if not MODEL_PATH.exists():
    raise SystemExit(f"Model not found:\n{MODEL_PATH}")

print(f"\nStarting {model}...")

llama_process = subprocess.Popen(
    [
        "llama-server",
        "-m",
        str(MODEL_PATH),
        "-ngl",
        "999",
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
    ]
)

print("Waiting for llama-server...")

while True:
    try:
        response = requests.get(
            "http://localhost:8080/v1/models",
            timeout=5,
        )

        if response.status_code == 200:
            break

    except Exception:
        pass

    time.sleep(1)

print("Model loaded.")
print("Launching Streamlit...")

try:
    subprocess.run(
        [
            "streamlit",
            "run",
            str(PROJECT_ROOT / "app.py"),
        ]
    )
finally:
    llama_process.terminate()
