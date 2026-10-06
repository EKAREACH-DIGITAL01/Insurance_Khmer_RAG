import ast
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ArchitectureTests(unittest.TestCase):
    def test_backend_import_does_not_load_ui_or_models(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import app; import khmer_rag.pipeline; import sys; "
                "assert not ({'streamlit', 'torch', 'sentence_transformers'} & sys.modules.keys())",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_backend_does_not_depend_on_ui_or_scripts(self):
        for path in (ROOT / "khmer_rag").rglob("*.py"):
            if "ui" in path.relative_to(ROOT / "khmer_rag").parts:
                continue
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                for module in modules:
                    self.assertFalse(
                        module.startswith(("streamlit", "khmer_rag.ui", "scripts")),
                        f"{path}: {module}",
                    )

    def test_python_files_compile(self):
        for directory in ("khmer_rag", "scripts"):
            for path in (ROOT / directory).rglob("*.py"):
                compile(path.read_text(), str(path), "exec")
