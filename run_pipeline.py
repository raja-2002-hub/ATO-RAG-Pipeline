"""
Run the full pipeline: prep_docs → build_index.

Assumes you already have data/chunks.jsonl from process_pages.py.

Usage:
    python run_pipeline.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

steps = [
    ("Step 1: Prep docs", [sys.executable, str(ROOT / "processing" / "prep_docs.py")]),
    ("Step 2: Build index", [sys.executable, str(ROOT / "indexing" / "build_index.py")]),
]

for name, cmd in steps:
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}\n")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"\n[ERROR] {name} failed!")
        sys.exit(1)

print(f"\n{'='*50}")
print(f"  Pipeline complete!")
print(f"  Start the API:")
print(f"  uvicorn api.app:app --port 8000")
print(f"{'='*50}")
