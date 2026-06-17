from pathlib import Path

from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"

EMBEDDING_DIR = MODELS_DIR / "embeddings" / "bge-m3"

LLM_DIR = MODELS_DIR / "llm"

LLM_FILENAME = "SeaLLMs-v3-7B-Chat-Q8_0.gguf"

LLM_FILE = LLM_DIR / LLM_FILENAME

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

if any(EMBEDDING_DIR.iterdir()):

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

    snapshot_download(
        repo_id="tensorblock/SeaLLMs-v3-7B-Chat-GGUF",
        allow_patterns=[LLM_FILENAME],
        local_dir=str(LLM_DIR),
    )

    print("SeaLLM installed successfully.")
    print(f"Location: {LLM_FILE}")

# ============================================================
# COMPLETE
# ============================================================

print("\n========== COMPLETE ==========")

print("Installed models:")

print(f"- Embedding Model : {EMBEDDING_DIR}")

print(f"- LLM Model       : {LLM_FILE}")

print("\nSetup completed successfully.")
