"""Process setup, called before importing the UI or machine-learning libraries."""

import logging
import os
from pathlib import Path


def configure_runtime() -> None:
    """Load local settings and configure the model cache and console logging."""
    from dotenv import load_dotenv

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    for name, value in {
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "OMP_NUM_THREADS": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TQDM_DISABLE": "1",
        "TRANSFORMERS_VERBOSITY": "error",
        "HF_HOME": str(root / "models" / "retriever" / ".cache"),
    }.items():
        os.environ.setdefault(name, value)

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
