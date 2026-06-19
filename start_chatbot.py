import subprocess
import time
import requests
import platform
import shutil
from pathlib import Path

OS_NAME = platform.system()

print(f"Detected OS: {OS_NAME}")

LLAMA_SERVER = shutil.which("llama-server")

if LLAMA_SERVER is None:

    if OS_NAME == "Windows":
        raise SystemExit(
            "llama-server.exe not found. "
            "Please install llama.cpp and add it to PATH."
        )

    raise SystemExit(
        "llama-server not found. " "Please install llama.cpp and add it to PATH."
    )

print(f"llama-server: {LLAMA_SERVER}")

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
        LLAMA_SERVER,
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

start_time = time.time()

while True:

    if time.time() - start_time > 120:
        llama_process.kill()
        raise SystemExit("llama-server failed to start within 120 seconds.")

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

subprocess.run(
    [
        "streamlit",
        "cache",
        "clear",
    ]
)

try:
    subprocess.run(
        [
            "streamlit",
            "run",
            str(PROJECT_ROOT / "app.py"),
            "--client.toolbarMode=minimal",
            "--browser.gatherUsageStats=false",
        ]
    )
finally:
    llama_process.terminate()
