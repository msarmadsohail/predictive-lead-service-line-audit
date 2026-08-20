"""Filesystem layout.

Everything resolves from the repository root. Set LSL_ROOT if the scripts are run
from somewhere other than a checkout, for example the ancillary bundle attached to
the arXiv submission.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get('LSL_ROOT') or Path(__file__).resolve().parents[1])
DATA = ROOT / 'ny-lsl-inventory'
WORK = ROOT / 'work'
FIG = ROOT / 'paper' / 'fig'

WORK.mkdir(parents=True, exist_ok=True)
