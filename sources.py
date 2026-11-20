"""Load docs/sources.csv: title, publisher, URL and date for each PDF."""
from __future__ import annotations

import csv
from pathlib import Path

import settings


def load_sources(path: Path | None = None) -> dict[str, dict]:
    path = Path(path) if path else settings.DOCS_DIR / "sources.csv"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {
            row["file"].strip(): {k: (v or "").strip() for k, v in row.items() if k}
            for row in csv.DictReader(f)
            if (row.get("file") or "").strip()
        }
