from __future__ import annotations

# core/paths.py

import os
import sys
from pathlib import Path

# Project root (repo root)
ROOT = Path(__file__).resolve().parents[1]

# Ensure ROOT is on sys.path once
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Data dir is env-driven
DATA_DIR = os.getenv("DATA_DIR", "data")
DATA = ROOT / DATA_DIR

# Config + reference
CFG = ROOT / "config"
REF = ROOT / "ref"
