"""Ponto de entrada do EXP-07; pode ser chamado a partir da raiz do projeto."""
from __future__ import annotations

import sys
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parents[1]
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from src.exp07_pipeline import main  # noqa: E402


if __name__ == "__main__":
    main()
