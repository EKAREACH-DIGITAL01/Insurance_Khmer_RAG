from pathlib import Path
import os

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

LLM_MODELS = [
    {
        "hf_filename": "SeaLLM3-7B-Chat.Q8_0.gguf",
        "local_filename": "SeaLLMs-v3-7B-Chat-Q8_0.gguf",
    },
    {
        "hf_filename": "SeaLLM3-7B-Chat.Q6_K.gguf",
        "local_filename": "SeaLLMs-v3-7B-Chat-Q6_K.gguf",
    },
]

# ============================================================
# CREATE DIRECTORIES
# ============================================================

EMBEDDING_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LLM_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ============================================================
# INSTALL BGE-M3
# ============================================================

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

# ============================================================
# INSTALL SEALLM
# ============================================================

print("\n========== SeaLLM ==========")

for model_info in LLM_MODELS:

    hf_filename = model_info["hf_filename"]
    local_filename = model_info["local_filename"]

    llm_file = LLM_DIR / local_filename

    print(f"\nChecking: {local_filename}")

    if llm_file.exists():

        print("Already installed.")
        print(f"Location: {llm_file}")

        continue

    print("Downloading...")

    downloaded_file = hf_hub_download(
        repo_id="mradermacher/SeaLLM3-7B-Chat-GGUF",
        filename=hf_filename,
        local_dir=str(LLM_DIR),
    )

    downloaded_path = Path(downloaded_file)

    if downloaded_path.name != local_filename:
        downloaded_path.rename(llm_file)

    print("Installed successfully.")
    print(f"Location: {llm_file}")

# ============================================================
# COMPLETE
# ============================================================

print("\n========== COMPLETE ==========")

print("Installed models:")

print(f"- Embedding Model : {EMBEDDING_DIR}")

print("\nInstalled LLMs:")

for model_info in LLM_MODELS:

    llm_file = LLM_DIR / model_info["local_filename"]

    if llm_file.exists():
        print(f"- {llm_file}")
    else:
        print(f"- MISSING: {llm_file}")

if not (EMBEDDING_DIR / "modules.json").exists():
    raise SystemExit("\nBGE-M3 installation failed.")

for model_info in LLM_MODELS:

    llm_file = LLM_DIR / model_info["local_filename"]

    if not llm_file.exists():
        raise SystemExit(f"\nSeaLLM installation failed: {llm_file.name}")

print("\nSetup completed successfully.")
