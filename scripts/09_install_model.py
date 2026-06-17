from pathlib import Path
import os

from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"

HF_CACHE_DIR = MODELS_DIR / ".hf_cache"

HF_CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

os.environ["HF_HOME"] = str(HF_CACHE_DIR)

EMBEDDING_DIR = MODELS_DIR / "retriever" / "BAAI-BGE-M3"

LLM_DIR = MODELS_DIR / "generator"

HF_LLM_FILENAME = "SeaLLM3-7B-Chat.Q8_0.gguf"

LOCAL_LLM_FILENAME = "SeaLLMs-v3-7B-Chat-Q8_0.gguf"

LLM_FILE = LLM_DIR / LOCAL_LLM_FILENAME

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

if LLM_FILE.exists():

    print("SeaLLM already installed.")
    print(f"Location: {LLM_FILE}")

else:

    print("Downloading SeaLLM GGUF...")

    downloaded_file = hf_hub_download(
        repo_id="mradermacher/SeaLLM3-7B-Chat-GGUF",
        filename=HF_LLM_FILENAME,
        local_dir=str(LLM_DIR),
    )

    downloaded_path = Path(downloaded_file)

    if downloaded_path.name != LOCAL_LLM_FILENAME:
        downloaded_path.rename(LLM_FILE)

    print("SeaLLM installed successfully.")
    print(f"Location: {LLM_FILE}")

# ============================================================
# COMPLETE
# ============================================================

print("\n========== COMPLETE ==========")

print("Installed models:")

print(f"- Embedding Model : {EMBEDDING_DIR}")

if LLM_FILE.exists():
    print(f"- LLM Model       : {LLM_FILE}")
else:
    print("- LLM Model       : NOT INSTALLED")

if not (EMBEDDING_DIR / "modules.json").exists():
    raise SystemExit("\nBGE-M3 installation failed.")

if not LLM_FILE.exists():
    raise SystemExit("\nSeaLLM installation failed.")

print("\nSetup completed successfully.")
