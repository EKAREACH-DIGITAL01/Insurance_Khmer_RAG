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

# Comparison-only retriever model -- NOT used by the live app (app.py only
# ever loads EMBEDDING_DIR/BGE-M3). Used solely by
# scripts/09_compare_embeddings.py. Opt-in (see below): a previous version
# of this script installed it unconditionally on every run, which meant a
# person just re-checking their SeaLLM quantizations could get stuck
# behind an unrequested ~470MB download with no way to skip it.
COMPARISON_EMBEDDING_DIR = MODELS_DIR / "retriever" / "MiniLM-paraphrase-multilingual-L12-v2"
COMPARISON_EMBEDDING_HF_ID = "paraphrase-multilingual-MiniLM-L12-v2"

LLM_DIR = MODELS_DIR / "generator"

LLM_MODELS = {
    "1": {
        "hf_filename": "SeaLLM3-7B-Chat.Q4_K_M.gguf",
        "local_filename": "SeaLLMs-v3-7B-Chat-Q4_K_M.gguf",
        "label": "SeaLLMs-v3-7B Q4_K_M (~4.8 GB) - Fast / Recommended",
    },
    "2": {
        "hf_filename": "SeaLLM3-7B-Chat.Q5_K_M.gguf",
        "local_filename": "SeaLLMs-v3-7B-Chat-Q5_K_M.gguf",
        "label": "SeaLLMs-v3-7B Q5_K_M (~5.5 GB) - Best Balance",
    },
    "3": {
        "hf_filename": "SeaLLM3-7B-Chat.Q6_K.gguf",
        "local_filename": "SeaLLMs-v3-7B-Chat-Q6_K.gguf",
        "label": "SeaLLMs-v3-7B Q6_K   (~6.3 GB) - Higher Quality",
    },
    "4": {
        "hf_filename": "SeaLLM3-7B-Chat.Q8_0.gguf",
        "local_filename": "SeaLLMs-v3-7B-Chat-Q8_0.gguf",
        "label": "SeaLLMs-v3-7B Q8_0   (~8.1 GB) - Highest Quality / Slower",
    },
}

# CREATE DIRECTORIES
EMBEDDING_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

COMPARISON_EMBEDDING_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LLM_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("""
        ========================================
        Khmer Legal Chatbot Model Installer
        ========================================
        """)

print("Starting installation...")

# INSTALL BGE-M3
# Always automatic, no prompt -- this is a hard requirement for the live
# app to function at all, there is nothing to "choose" about it.
print("\n========== BGE-M3 (required) ==========")

if (EMBEDDING_DIR / "modules.json").exists():

    print("BGE-M3 already installed.")
    print(f"Location: {EMBEDDING_DIR}")

else:

    print("Installing BGE-M3 (~2.1 GB)...")

    model = SentenceTransformer("BAAI/bge-m3")

    model.save(str(EMBEDDING_DIR))

    print("BGE-M3 installed successfully.")
    print(f"Location: {EMBEDDING_DIR}")


# INSTALL COMPARISON EMBEDDING MODEL (MiniLM) -- OPT-IN
# Only used by scripts/09_compare_embeddings.py, never by the live
# chatbot, so this asks first rather than downloading unconditionally.
# Pressing Enter (empty input) skips it -- the safe default for someone
# who just wants to check on their generator models, like the case that
# motivated this change.
print("\n========== MiniLM (optional, comparison only) ==========")

if (COMPARISON_EMBEDDING_DIR / "modules.json").exists():

    print("MiniLM already installed.")
    print(f"Location: {COMPARISON_EMBEDDING_DIR}")

else:

    print(
        "Used only by scripts/09_compare_embeddings.py to benchmark "
        "BGE-M3 against an alternative embedding model -- the live "
        "chatbot does not need this."
    )
    install_minilm = (
        input("Install MiniLM now (~470 MB)? [y/N]: ").strip().lower()
    )

    if install_minilm == "y":
        try:
            comparison_model = SentenceTransformer(COMPARISON_EMBEDDING_HF_ID)
            comparison_model.save(str(COMPARISON_EMBEDDING_DIR))

            print("MiniLM installed successfully.")
            print(f"Location: {COMPARISON_EMBEDDING_DIR}")

        except Exception as e:
            # Non-fatal: a failed comparison-model download should never
            # block the rest of this installer or the live chatbot.
            print(f"MiniLM installation failed, continuing anyway: {e}")
            print("Re-run this installer later to retry.")
    else:
        print("Skipped. Re-run this installer anytime to install it later.")


# INSTALL SEALLM GENERATOR
# Only prompts for a choice if there's actually something left to choose
# -- if every quantization is already on disk, asking the person to pick
# one before the script will even check what's installed is pure friction
# with no purpose, since nothing will be downloaded either way.
print("\n========== SeaLLM Generator ==========")

installed_status = {
    key: (LLM_DIR / info["local_filename"]).exists()
    for key, info in LLM_MODELS.items()
}

selected_llm = None

if all(installed_status.values()):

    print("All SeaLLM quantizations are already installed -- nothing to choose.")
    for info in LLM_MODELS.values():
        print(f"  - {info['local_filename']}")

else:

    print("\nChoose a Generator Model:\n")
    for key, info in LLM_MODELS.items():
        tag = "  [already installed]" if installed_status[key] else ""
        print(f"        {key}. {info['label']}{tag}")
    print()

    choice = input("Enter choice: ").strip()

    if choice not in LLM_MODELS:
        raise SystemExit("Invalid choice.")

    model_info = LLM_MODELS[choice]
    hf_filename = model_info["hf_filename"]
    local_filename = model_info["local_filename"]
    selected_llm = LLM_DIR / local_filename

    print(f"\nSelected model: {local_filename}")

    if selected_llm.exists():

        print("Already installed.")
        print(f"Location: {selected_llm}")

    else:

        print(f"Downloading {local_filename}...")

        downloaded_file = hf_hub_download(
            repo_id="mradermacher/SeaLLM3-7B-Chat-GGUF",
            filename=hf_filename,
            local_dir=str(LLM_DIR),
        )

        downloaded_path = Path(downloaded_file)

        if downloaded_path.name != local_filename:
            downloaded_path.rename(selected_llm)

        print("SeaLLM installed successfully.")
        print(f"Location: {selected_llm}")

# COMPLETE
print("\n========== COMPLETE ==========")

print("Installed models:")

print(f"- Embedding Model : {EMBEDDING_DIR}")

if (COMPARISON_EMBEDDING_DIR / "modules.json").exists():
    print(f"- Comparison Embedding Model : {COMPARISON_EMBEDDING_DIR}")
else:
    print("- Comparison Embedding Model : not installed (optional)")

print("\nInstalled LLMs:")

any_llm_installed = False
for model_info in LLM_MODELS.values():

    llm_file = LLM_DIR / model_info["local_filename"]

    if llm_file.exists():
        print(f"- {llm_file}")
        any_llm_installed = True

if not (EMBEDDING_DIR / "modules.json").exists():
    raise SystemExit("\nBGE-M3 installation failed.")

if selected_llm is not None and not selected_llm.exists():
    raise SystemExit(f"\nSeaLLM installation failed: {selected_llm.name}")

if not any_llm_installed:
    raise SystemExit("\nNo SeaLLM generator model installed.")

print("\nSetup completed successfully.")