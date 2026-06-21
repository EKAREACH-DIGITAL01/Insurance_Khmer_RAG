import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"

HF_CACHE_DIR = MODELS_DIR / ".hf_cache"

HF_CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

os.environ["HF_HOME"] = str(HF_CACHE_DIR)

from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

EMBEDDING_DIR = MODELS_DIR / "retriever" / "BAAI-BGE-M3"

LLM_DIR = MODELS_DIR / "generator"

LLM_MODELS = {
    "1": {
        "hf_filename": "SeaLLM3-7B-Chat.Q4_K_M.gguf",
        "local_filename": "SeaLLMs-v3-7B-Chat-Q4_K_M.gguf",
    },
    "2": {
        "hf_filename": "SeaLLM3-7B-Chat.Q5_K_M.gguf",
        "local_filename": "SeaLLMs-v3-7B-Chat-Q5_K_M.gguf",
    },
    "3": {
        "hf_filename": "SeaLLM3-7B-Chat.Q6_K.gguf",
        "local_filename": "SeaLLMs-v3-7B-Chat-Q6_K.gguf",
    },
    "4": {
        "hf_filename": "SeaLLM3-7B-Chat.Q8_0.gguf",
        "local_filename": "SeaLLMs-v3-7B-Chat-Q8_0.gguf",
    },
}

# CREATE DIRECTORIES
EMBEDDING_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LLM_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# MODEL SELECTION
print("""
        ========================================
        Khmer Legal Chatbot Model Installer
        ========================================

        Retriever Model: BAAI/BGE-M3 (~2.1 GB)

        Choose a Generator Model:

        1. SeaLLMs-v3-7B Q4_K_M (~4.8 GB) - Fast / Recommended
        2. SeaLLMs-v3-7B Q5_K_M (~5.5 GB) - Best Balance
        3. SeaLLMs-v3-7B Q6_K   (~6.3 GB) - Higher Quality
        4. SeaLLMs-v3-7B Q8_0   (~8.1 GB) - Highest Quality / Slower

        """)

choice = input("Enter choice: ").strip()

if choice not in LLM_MODELS:
    raise SystemExit("Invalid choice.")

model_info = LLM_MODELS[choice]

hf_filename = model_info["hf_filename"]
local_filename = model_info["local_filename"]

selected_llm = LLM_DIR / local_filename

print(f"\nSelected model: {local_filename}")
print("\nStarting installation...")

# INSTALL BGE-M3
print("\n========== BGE-M3 ==========")

if (EMBEDDING_DIR / "modules.json").exists():

    print("BGE-M3 already installed.")
    print(f"Location: {EMBEDDING_DIR}")

else:

    print("Installing BGE-M3...")

    model = SentenceTransformer("BAAI/bge-m3")

    model.save(str(EMBEDDING_DIR))

    print("BGE-M3 installed successfully.")
    print(f"Location: {EMBEDDING_DIR}")


# INSTALL SEALLM
print("\n========== SeaLLM ==========")

llm_file = selected_llm

if llm_file.exists():

    print("SeaLLM already installed.")
    print(f"Location: {llm_file}")

else:

    print(f"Downloading {local_filename}...")

    downloaded_file = hf_hub_download(
        repo_id="mradermacher/SeaLLM3-7B-Chat-GGUF",
        filename=hf_filename,
        local_dir=str(LLM_DIR),
    )

    downloaded_path = Path(downloaded_file)

    if downloaded_path.name != local_filename:
        downloaded_path.rename(llm_file)

    print("SeaLLM installed successfully.")
    print(f"Location: {llm_file}")

# COMPLETE
print("\n========== COMPLETE ==========")

print("Installed models:")

print(f"- Embedding Model : {EMBEDDING_DIR}")

print("\nInstalled LLMs:")

for model_info in LLM_MODELS.values():

    llm_file = LLM_DIR / model_info["local_filename"]

    if llm_file.exists():
        print(f"- {llm_file}")

if not (EMBEDDING_DIR / "modules.json").exists():
    raise SystemExit("\nBGE-M3 installation failed.")

if not selected_llm.exists():
    raise SystemExit(f"\nSeaLLM installation failed: {selected_llm.name}")

print("\nSetup completed successfully.")
