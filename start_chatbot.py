from pathlib import Path
import subprocess
import time
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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

MODEL_PATH = PROJECT_ROOT / "models" / "generator" / model

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
        requests.get("http://localhost:8080/v1/models")
        break
    except:
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
