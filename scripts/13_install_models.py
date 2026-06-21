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

# ----------------------------------------------------------------------------
# Embedding models -- three genuine alternatives, not "main + comparison-only"
# ----------------------------------------------------------------------------
# Either can be used to actually build the production vector_db/ index (see
# scripts/07_generate_embeddings.py, which now lets you choose). Having
# BOTH installed also lets you run scripts/09_compare_embeddings.py.
#
# BGE-M3 (~568M params) and jina-embeddings-v3 (~572M params) are
# comparably sized -- a much fairer apples-to-apples comparison than the
# original MiniLM (~118M params, ~400MB), which was closer to comparing a
# Ferrari against a bicycle than two real alternatives.
#
# jina-embeddings-v3 needs `trust_remote_code=True` to load (its
# task-specific LoRA adapters are implemented as custom modeling code, not
# a standard SentenceTransformer architecture) and the `einops` package --
# add `einops` to requirements.txt if it isn't already there. BGE-M3 needs
# neither.
EMBEDDING_MODELS = {
    "1": {
        "label": "BAAI/bge-m3",
        "hf_id": "BAAI/bge-m3",
        "local_dir": MODELS_DIR / "retriever" / "BAAI-BGE-M3",
        "trust_remote_code": False,
        "task_aware": False,
        "note": "~568M params",
    },
    "2": {
        "label": "jinaai/jina-embeddings-v3",
        "hf_id": "jinaai/jina-embeddings-v3",
        "local_dir": MODELS_DIR / "retriever" / "jinaai-jina-embeddings-v3",
        "trust_remote_code": True,
        "task_aware": True,
        "note": "~572M params, requires `einops` and trust_remote_code=True",
    },
    "3": {
        "label": "paraphrase-multilingual-MiniLM-L12-v2",
        "hf_id": "paraphrase-multilingual-MiniLM-L12-v2",
        "local_dir": MODELS_DIR / "retriever" / "MiniLM-paraphrase-multilingual-L12-v2",
        "trust_remote_code": False,
        "task_aware": False,
        "note": "~118M params, ~470 MB -- noticeably smaller than the other two",
    },
}

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
for info in EMBEDDING_MODELS.values():
    info["local_dir"].mkdir(parents=True, exist_ok=True)

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

# INSTALL EMBEDDING MODEL(S)
# Only prompts if there's actually a choice left to make -- if every
# model is already installed, there's nothing to ask.
print("\n========== Embedding Model(s) ==========")

installed_embed_status = {
    key: (info["local_dir"] / "modules.json").exists()
    for key, info in EMBEDDING_MODELS.items()
}

if all(installed_embed_status.values()):

    print("All embedding models are already installed -- nothing to choose.")
    for info in EMBEDDING_MODELS.values():
        print(f"  - {info['label']} -> {info['local_dir']}")

else:

    print(
        f"{len(EMBEDDING_MODELS)} embedding models are available. Any of "
        "them can be used to build the production vector_db/ index "
        "(scripts/07_generate_embeddings.py lets you pick). Installing "
        "more than one also enables scripts/09_compare_embeddings.py.\n"
    )
    for key, info in EMBEDDING_MODELS.items():
        tag = "  [already installed]" if installed_embed_status[key] else ""
        print(f"        {key}. {info['label']} ({info['note']}){tag}")
    print()

    raw_choice = input(
        "Install which model(s)? Comma-separated (e.g. 1,3), or 'all': "
    ).strip().lower()

    if raw_choice == "all":
        to_install = list(EMBEDDING_MODELS.keys())
    else:
        to_install = [
            token.strip()
            for token in raw_choice.split(",")
            if token.strip() in EMBEDDING_MODELS
        ]

    if not to_install:
        print("No valid model(s) selected -- skipping embedding model installation.")

    for key in to_install:
        info = EMBEDDING_MODELS[key]

        if installed_embed_status[key]:
            print(f"{info['label']} already installed.")
            print(f"Location: {info['local_dir']}")
            continue

        print(f"\nInstalling {info['label']}...")

        try:
            kwargs = (
                {"trust_remote_code": True} if info["trust_remote_code"] else {}
            )
            embed_model = SentenceTransformer(info["hf_id"], **kwargs)
            embed_model.save(str(info["local_dir"]))

            print(f"{info['label']} installed successfully.")
            print(f"Location: {info['local_dir']}")

        except Exception as e:
            # Non-fatal per model -- a failed download of one embedding
            # model shouldn't block the other, or the SeaLLM install below.
            print(f"{info['label']} installation failed, continuing anyway: {e}")
            print("Re-run this installer later to retry.")


# INSTALL SEALLM GENERATOR
# Only prompts for a choice if there's actually something left to choose
# -- if every quantization is already on disk, asking the person to pick
# one before the script will even check what's installed is pure friction
# with no purpose, since nothing will be downloaded either way.
print("\n========== SeaLLM Generator ==========")

installed_llm_status = {
    key: (LLM_DIR / info["local_filename"]).exists()
    for key, info in LLM_MODELS.items()
}

selected_llm = None

if all(installed_llm_status.values()):

    print("All SeaLLM quantizations are already installed -- nothing to choose.")
    for info in LLM_MODELS.values():
        print(f"  - {info['local_filename']}")

else:

    print("\nChoose a Generator Model:\n")
    for key, info in LLM_MODELS.items():
        tag = "  [already installed]" if installed_llm_status[key] else ""
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

print("Installed embedding models:")

any_embed_installed = False
for info in EMBEDDING_MODELS.values():
    if (info["local_dir"] / "modules.json").exists():
        print(f"- {info['label']} : {info['local_dir']}")
        any_embed_installed = True

if not any_embed_installed:
    print("- none")

print("\nInstalled LLMs:")

any_llm_installed = False
for model_info in LLM_MODELS.values():

    llm_file = LLM_DIR / model_info["local_filename"]

    if llm_file.exists():
        print(f"- {llm_file}")
        any_llm_installed = True

if not any_embed_installed:
    raise SystemExit(
        "\nNo embedding model installed. scripts/07_generate_embeddings.py "
        "needs at least one."
    )

if selected_llm is not None and not selected_llm.exists():
    raise SystemExit(f"\nSeaLLM installation failed: {selected_llm.name}")

if not any_llm_installed:
    raise SystemExit("\nNo SeaLLM generator model installed.")

print("\nSetup completed successfully.")