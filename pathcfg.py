# pathcfg.py
"""
Central place for all project paths.
Everything is expressed **relative to the repository root**, so
no one ever has to edit code when they clone the repo.
"""
from pathlib import Path

# The repo root is the directory that contains this file (../ from here)
PROJECT_ROOT = Path(__file__).resolve().parents[0]
print(PROJECT_ROOT)

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
DIST_DIR = OUTPUT_DIR / "distances"
FIG_DIR  = OUTPUT_DIR / "Fig2"
MODEL_DIR = OUTPUT_DIR / "models"

# Make sure the directories exist
for _p in [DATA_DIR, OUTPUT_DIR, DIST_DIR, FIG_DIR, MODEL_DIR]:
    _p.mkdir(parents=True, exist_ok=True)
