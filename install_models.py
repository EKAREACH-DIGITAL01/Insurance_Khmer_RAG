import json
import os
from pathlib import Path

# ============================================================
# CRITICAL: os.environ["HF_HOME"] MUST be set before any HF
# import -- including sentence_transformers and transformers,
# which both transitively import huggingface_hub. huggingface_hub
# reads HF_HOME exactly once, during its own module __init__,
# and freezes the result into a module-level constant. Any
# os.environ assignment that happens after the first HF import
# (even an indirect one) is silently ignored. This is why the
# previous version still wrote to ~/.cache/huggingface despite
# appearing to set HF_HOME correctly -- the assignment came
# AFTER the imports on the very next lines.
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

# All HF assets live inside the project, under retriever/.cache
# (not models/.cache at the top level, since GGUF generator
# files have nothing to do with the HF cache -- keeping it under
# retriever/ makes the layout honest about what's actually there).
RETRIEVER_DIR = MODELS_DIR / "retriever"
HF_CACHE_DIR = RETRIEVER_DIR / ".cache"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_CACHE_DIR)

# All HF imports come AFTER the env var is set.
from huggingface_hub import hf_hub_download, snapshot_download
from sentence_transformers import SentenceTransformer

# ============================================================
# Architecture: ONE copy per model, no duplication
# ============================================================
# Previous design: set HF_HOME → import HF (too late) → download
# to ~/.cache → .save() to models/retriever/ → two copies on disk.
#
# New design: set HF_HOME correctly → import HF → download goes
# directly to models/retriever/.cache → load from there by
# Hub ID at runtime → one copy, fully self-contained.
#
# For native_hf models (jina-embeddings-v3-hf), the same applies
# automatically since they're always loaded by Hub ID and were
# never saved locally.
#
# "installed" is signalled by a lightweight INSTALLED.json marker
# in a per-model subdirectory under retriever/. The marker holds
# the Hub ID so any script can look up which model it represents
# without re-downloading anything. The actual weights live in
# retriever/.cache/ (managed by HF, never duplicated).
EMBEDDING_MODELS = {
    "1": {
        "label": "BAAI/bge-m3",
        "hf_id": "BAAI/bge-m3",
        "marker_dir": RETRIEVER_DIR / "BAAI-BGE-M3",
        "trust_remote_code": False,
        "task_aware": False,
        "note": "~568M params",
    },
    "2": {
        "label": "intfloat/multilingual-e5-large",
        "hf_id": "intfloat/multilingual-e5-large",
        "marker_dir": RETRIEVER_DIR / "multilingual-e5-large",
        "trust_remote_code": False,
        "task_aware": False,
        "doc_prefix": "passage: ",
        "query_prefix": "query: ",
        "note": "~560M params, Microsoft multilingual retrieval model",
    },
    "3": {
        "label": "paraphrase-multilingual-MiniLM-L12-v2",
        "hf_id": "paraphrase-multilingual-MiniLM-L12-v2",
        "marker_dir": RETRIEVER_DIR / "MiniLM-paraphrase-multilingual-L12-v2",
        "trust_remote_code": False,
        "task_aware": False,
        "note": "~118M params, ~470 MB -- noticeably smaller than the other two",
    },
}

LLM_DIR = MODELS_DIR / "generator"
LLM_DIR.mkdir(parents=True, exist_ok=True)

LLM_MODELS = {
    "1": {
        "repo_id": "mradermacher/SeaLLM3-7B-Chat-GGUF",
        "hf_filename": "SeaLLM3-7B-Chat.Q4_K_M.gguf",
        "local_filename": "SeaLLMs-v3-7B-Chat-Q4_K_M.gguf",
        "label": "SeaLLM-v3-7B-Q4_K_M (4.68 GB)",
    },
    "2": {
        "repo_id": "unsloth/Qwen3.5-9B-GGUF",
        "hf_filename": "Qwen3.5-9B-Q4_K_M.gguf",
        "local_filename": "Qwen3.5-9B-Q4_K_M.gguf",
        "label": "Qwen-3.5-9B-Q4_K_M (5.68 GB)",
    },
    "3": {
        # Sailor2-L ("L" = long): the 128K-context variant. The plain
        # sail/Sailor2-8B-Chat GGUF has max_position_embeddings=4096 baked in
        # (llama.cpp clamps --ctx-size to it), too small for our 5k-13k-token
        # RAG prompts. sail/Sailor2-L-8B-Chat is natively 131072 ctx (no rope
        # scaling needed), so it runs the pipeline like the other models.
        "repo_id": "mradermacher/Sailor2-L-8B-Chat-GGUF",
        "hf_filename": "Sailor2-L-8B-Chat.Q4_K_M.gguf",
        "local_filename": "Sailor2-L-8B-Chat-Q4_K_M.gguf",
        "label": "Sailor2-L-8B-Chat-Q4_K_M (5.24 GB, 128K ctx)",
    },
    "4": {
        "repo_id": "unsloth/Llama-3.1-8B-Instruct-GGUF",
        "hf_filename": "Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "local_filename": "Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "label": "Llama-3.1-8B-Instruct-Q4_K_M (4.92 GB)",
    },
}


def marker_path(info):
    return info["marker_dir"] / "INSTALLED.json"


def is_model_installed(info):
    return marker_path(info).exists()


def write_marker(info, extra=None):
    info["marker_dir"].mkdir(parents=True, exist_ok=True)
    payload = {"hf_id": info["hf_id"], "hf_cache": str(HF_CACHE_DIR)}
    if extra:
        payload.update(extra)
    with open(marker_path(info), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


print("""
        ========================================
        Khmer Legal Chatbot Model Installer
        ========================================

        HF cache location  : models/retriever/.cache/
        Generator location : models/generator/
        (No ~/.cache/huggingface writes -- project is fully self-contained)
        """)

print("Starting installation...")

# INSTALL EMBEDDING MODELS
print("\n========== Embedding Model(s) ==========")

installed_status = {
    key: is_model_installed(info)
    for key, info in EMBEDDING_MODELS.items()
}

if all(installed_status.values()):
    print("All embedding models already installed -- nothing to do.")
    for info in EMBEDDING_MODELS.values():
        print(f"  - {info['label']}")
else:
    print(
        f"\n{len(EMBEDDING_MODELS)} embedding models available. "
        "Any can be used to build the production vector_db/ index.\n"
    )
    for key, info in EMBEDDING_MODELS.items():
        print(f"        {key}. {info['label']} ({info['note']})")
    print()

    raw = input(
        "Install which model(s)? Comma-separated (e.g. 1, 2, 3), or 'all': "
    ).strip().lower()

    if raw == "all":
        to_install = list(EMBEDDING_MODELS.keys())
    else:
        to_install = [
            t.strip() for t in raw.split(",") if t.strip() in EMBEDDING_MODELS
        ]

    if not to_install:
        print("No valid selection -- skipping embedding model installation.")

    for key in to_install:
        info = EMBEDDING_MODELS[key]

        if installed_status[key]:
            print(f"\n{info['label']} already installed.")
            continue

        print(f"\nInstalling {info['label']}...")

        try:
            # All models use SentenceTransformer loaded by Hub ID.
            # HF_HOME set at top means weights go into models/retriever/.cache
            # -- one copy, no duplication.
            kwargs = (
                {"trust_remote_code": True}
                if info["trust_remote_code"]
                else {}
            )
            SentenceTransformer(info["hf_id"], **kwargs)
            write_marker(info)

            print(f"{info['label']} installed successfully.")
            print(f"  Cache  : {HF_CACHE_DIR}")
            print(f"  Marker : {marker_path(info)}")

        except Exception as e:
            print(f"{info['label']} installation failed, continuing anyway: {e}")
            print("Re-run this installer later to retry.")


# INSTALL GENERATOR MODEL(S)
print("\n========== Generator Model(s) ==========")

installed_llm = {
    key: (LLM_DIR / info["local_filename"]).exists()
    for key, info in LLM_MODELS.items()
}

if all(installed_llm.values()):
    print("All generator models already installed -- nothing to do.")
    for info in LLM_MODELS.values():
        print(f"  - {info['local_filename']}")
else:
    print(
        f"\n{len(LLM_MODELS)} generator models available. "
        "Any can be used by start_chatbot.py.\n"
    )
    for key, info in LLM_MODELS.items():
        print(f"        {key}. {info['label']}")
    print()

    raw = input(
        "Install which model(s)? Comma-separated (e.g. 1, 2, 3, 4), or 'all': "
    ).strip().lower()

    if raw == "all":
        to_install = list(LLM_MODELS.keys())
    else:
        to_install = [
            t.strip() for t in raw.split(",") if t.strip() in LLM_MODELS
        ]

    if not to_install:
        print("No valid selection -- skipping generator model installation.")

    for key in to_install:
        info = LLM_MODELS[key]
        target_path = LLM_DIR / info["local_filename"]

        if installed_llm[key]:
            print(f"\n{info['label']} already installed.")
            continue

        print(f"\nInstalling {info['label']}...")

        try:
            downloaded = hf_hub_download(
                repo_id=info["repo_id"],
                filename=info["hf_filename"],
                local_dir=str(LLM_DIR),
            )
            downloaded_path = Path(downloaded)
            if downloaded_path.name != info["local_filename"]:
                downloaded_path.rename(target_path)

            print(f"{info['label']} installed successfully.")
            print(f"  Location: {target_path}")

        except Exception as e:
            print(f"{info['label']} installation failed, continuing anyway: {e}")
            print("Re-run this installer later to retry.")


# SUMMARY
print("\n========== COMPLETE ==========")
print(f"\nProject layout:")
print(f"  models/retriever/.cache/  ← all HF model weights (one copy)")
print(f"  models/retriever/<name>/     ← INSTALLED.json markers only")
print(f"  models/generator/            ← GGUF files")

print("\nInstalled embedding models:")
any_embed = False
for info in EMBEDDING_MODELS.values():
    if is_model_installed(info):
        print(f"  - {info['label']}")
        any_embed = True
if not any_embed:
    print("  - none")

print("\nInstalled generator models:")
any_llm = False
for info in LLM_MODELS.values():
    llm_file = LLM_DIR / info["local_filename"]
    if llm_file.exists():
        print(f"  - {llm_file.name}")
        any_llm = True

if not any_embed:
    raise SystemExit("\nNo embedding model installed.")
if not any_llm:
    raise SystemExit("\nNo generator model installed.")

print("\nSetup completed successfully.")
print(f"\nVerify no files leaked to ~/.cache/huggingface:")
print(f"  ls ~/.cache/huggingface  (should be empty or not exist)")