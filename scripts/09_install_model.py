# from pathlib import Path
# from sentence_transformers import SentenceTransformer
# from huggingface_hub import snapshot_download

# MODELS_DIR = Path("models")

# EMBEDDING_DIR = MODELS_DIR / "embeddings" / "bge-m3"
# LLM_DIR = MODELS_DIR / "llm"

# EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
# LLM_DIR.mkdir(parents=True, exist_ok=True)

# print("Installing BGE-M3...")

# model = SentenceTransformer("BAAI/bge-m3")
# model.save(str(EMBEDDING_DIR))

# print("BGE-M3 installed.")

# print("Downloading SeaLLM GGUF...")

# snapshot_download(
#     repo_id="tensorblock/SeaLLMs-v3-7B-Chat-GGUF",
#     allow_patterns=["*Q8_0.gguf"],
#     local_dir=str(LLM_DIR),
# )

# print("SeaLLM installed.")
