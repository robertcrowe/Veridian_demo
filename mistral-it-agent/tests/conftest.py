"""
Shared pytest fixtures for the mistral-it-agent test suite.

Adds the mistral-it-agent root to sys.path so the `agents` package is
importable regardless of which directory pytest is invoked from.

All data files are loaded once per session (scope="session") to avoid
repeated disk I/O across test modules.
"""

import json
import sys
from pathlib import Path

import pytest

# Make `agents` importable when running pytest from any working directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

_DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture(scope="session")
def data_files() -> dict:
    """
    Load all static JSON data files once per test session.

    Returns a dict with keys:
        raw_tickets        list[dict]  — labelled Veridian IT tickets
        kb                 dict        — generic IT knowledge base
        internal_kb        dict        — Veridian-specific knowledge base
        escalation_policy  dict        — multi-tier escalation paths
    """
    return {
        "raw_tickets": json.loads((_DATA_DIR / "raw_tickets.json").read_text()),
        "kb": json.loads((_DATA_DIR / "kb.json").read_text()),
        "internal_kb": json.loads((_DATA_DIR / "internal_kb.json").read_text()),
        "escalation_policy": json.loads((_DATA_DIR / "escalation_policy.json").read_text()),
    }
