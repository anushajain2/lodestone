"""Paths and settings. Everything can be overridden with environment variables."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("LODESTONE_HOME", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CATALOG_PATH = DATA_DIR / "catalog.json.gz"
FEATURES_PATH = DATA_DIR / "features.parquet"
VAR_DIR = Path(os.environ.get("LODESTONE_VAR", ROOT / "var"))
DB_PATH = Path(os.environ.get("LODESTONE_DB", VAR_DIR / "lodestone.db"))
REPORTS_DIR = ROOT / "reports"

# Constraints a partner gets unless they say otherwise.
DEFAULT_E_HULL_MAX = 0.05  # eV/atom
TOXIC_ELEMENTS = ["Cd", "Hg", "Pb", "Tl", "As", "U", "Th"]
RADIOACTIVE_ELEMENTS = ["Tc", "Pm", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Pa", "U", "Th", "Np", "Pu"]
# Scarce or supply-risk elements a partner may optionally exclude.
SCARCE_ELEMENTS = ["Re", "Os", "Ir", "Ru", "Rh", "Pd", "Pt", "Au", "Te", "In", "Ge", "Sc", "Ta"]
RARE_EARTHS = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Y", "Sc"]

# Reference materials for context (approximate textbook values).
REFERENCE_MATERIALS = {
    "SiO2 (reference)": {"k": 3.9, "band_gap": 9.0},
    "HfO2 (reference)": {"k": 22.0, "band_gap": 5.8},
}

ANTHROPIC_MODEL = os.environ.get("LODESTONE_MODEL", "claude-opus-5-5")


def db_path() -> Path:
    """Read lazily so tests can point LODESTONE_DB at a temp file."""
    return Path(os.environ.get("LODESTONE_DB", DB_PATH))
