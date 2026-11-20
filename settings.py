"""Central settings. Tune values here or through command-line flags."""
from __future__ import annotations

import json
import os
from pathlib import Path

try:  # optional: read LLM_* variables from a local .env file
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs"
DATA_DIR = ROOT / "data"
PAGES_PATH = DATA_DIR / "pages.json"      # raw extracted text per page (slow step, cached)
CHUNKS_PATH = DATA_DIR / "chunks.json"    # chunks used by the app (fast step)
CONFIG_PATH = DATA_DIR / "config.json"    # best method + refusal threshold, written by evaluate.py
EVAL_PATH = ROOT / "eval_questions.csv"

# Set SCHOLARLOAN_EMBED_MODEL=none to disable the dense model (TF-IDF only).
EMBED_MODEL = os.getenv("SCHOLARLOAN_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_WORDS = 180
OVERLAP = 30
MIN_WORDS = 20   # pieces/pages with fewer words are treated as empty
TOP_K = 5


def load_config(path: Path | None = None) -> dict:
    path = Path(path) if path else CONFIG_PATH
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_config(cfg: dict, path: Path | None = None) -> None:
    path = Path(path) if path else CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
