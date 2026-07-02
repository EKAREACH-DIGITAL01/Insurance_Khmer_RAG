import subprocess
import sys
from pathlib import Path

# ============================================================================
# Generator Evaluation Rescoring -- thin wrapper around scripts/13 --rescore
# ============================================================================
# DEPRECATED AS A STANDALONE IMPLEMENTATION.
#
# This script used to carry its OWN copy of the scoring thresholds and re-bucket
# the values already stored in each answer's `detail` block (the cosine
# similarities, the faithfulness grounded/total counts, the completeness length
# ratio). That design had a fatal limitation: it could only change how a STORED
# raw measurement was bucketed into 1-5 -- it could NOT recompute the raw
# measurement itself. So once the faithfulness number-grounding logic changed
# (Khmer-word/Arabic-digit normalisation, list-ordinal exclusion), this script
# could no longer produce correct faithfulness scores, because it never re-reads
# the answer text or the context -- only the stale counts.
#
# The canonical rescorer is now `scripts/13_compare_generator_models.py
# --rescore`. It owns the scoring functions (single source of truth), reloads
# each answer, REBUILDS its retrieval context from the saved context_chunk_ids,
# and recomputes every metric from scratch with the embedding model -- without
# ever starting llama-server or regenerating an answer. It then rewrites the
# JSONL, every summary, and the final comparison report.
#
# This file is kept only so the familiar `scripts/14_...` entry point still
# works; it simply delegates. Re-run scripts/14_diagnose_generator_scores.py
# afterwards to refresh the diagnostics.
#
# USAGE:
#   python scripts/15_rescore_generator_scores.py
# ============================================================================

SCRIPT_13 = Path(__file__).resolve().parent / "13_compare_generator_models.py"


def main():
    print(
        "scripts/15 now delegates to scripts/13 --rescore (the canonical "
        "rescorer that owns the scoring functions).\n"
    )
    raise SystemExit(
        subprocess.call([sys.executable, str(SCRIPT_13), "--rescore"])
    )


if __name__ == "__main__":
    main()
