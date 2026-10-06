"""Assets UI components."""

from khmer_rag.config import PROJECT_ROOT

UI_ASSET_DIR = PROJECT_ROOT / "assets" / "ui"


def script_html(filename):
    """Load a checked-in browser script for a Streamlit component."""
    script = (UI_ASSET_DIR / filename).read_text(encoding="utf-8")
    return f"<script>{script}</script>"
