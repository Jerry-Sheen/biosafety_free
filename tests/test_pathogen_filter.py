import pytest
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from user import get_pathogen_keys_from_vector_store, filter_pathogen_keys


def test_get_pathogen_keys():
    """Verify that pathogen keys can be extracted from the persistent vector store."""
    keys = get_pathogen_keys_from_vector_store("병원균(위험균) 종류 v3.json")
    assert len(keys) > 0
    # Must include key sample pathogens
    assert "Abnormal prion protein(PrPCWD)" in keys
    assert "Abnormal prion protein(PrPSc)" in keys
    assert "Abrin (아브린)" in keys


def test_filter_pathogen_keys_ab():
    """Verify filtering with 'ab'."""
    keys = get_pathogen_keys_from_vector_store("병원균(위험균) 종류 v3.json")
    filtered = filter_pathogen_keys("ab", keys)
    assert len(filtered) > 0
    assert "Abnormal prion protein(PrPCWD)" in filtered
    assert "Abnormal prion protein(PrPSc)" in filtered
    assert "Abrin (아브린)" in filtered


def test_filter_pathogen_keys_abr():
    """Verify filtering with 'abr'."""
    keys = get_pathogen_keys_from_vector_store("병원균(위험균) 종류 v3.json")
    filtered = filter_pathogen_keys("abr", keys)
    assert "Abrin (아브린)" in filtered
    assert "Abnormal prion protein(PrPCWD)" not in filtered


def test_filter_pathogen_keys_abrc():
    """Verify filtering with 'abrc' returning empty list."""
    keys = get_pathogen_keys_from_vector_store("병원균(위험균) 종류 v3.json")
    filtered = filter_pathogen_keys("abrc", keys)
    assert filtered == []


def test_filter_pathogen_keys_case_insensitive():
    """Verify case-insensitivity."""
    sample = [
        "Abnormal prion protein(PrPCWD)",
        "Abnormal prion protein(PrPSc)",
        "Abrin (아브린)"
    ]
    assert filter_pathogen_keys("ab", sample) == sample
    assert filter_pathogen_keys("AB", sample) == sample
    assert filter_pathogen_keys("Ab", sample) == sample
    assert filter_pathogen_keys("abr", sample) == ["Abrin (아브린)"]
    assert filter_pathogen_keys("ABR", sample) == ["Abrin (아브린)"]
    assert filter_pathogen_keys("abrc", sample) == []
    assert filter_pathogen_keys("'ab'", sample) == sample
    assert filter_pathogen_keys('"ab"', sample) == sample
    assert filter_pathogen_keys("'abr'", sample) == ["Abrin (아브린)"]
    assert filter_pathogen_keys("'abrc'", sample) == []
    assert filter_pathogen_keys("a", sample) == sample
    assert filter_pathogen_keys("e", sample) == sample[:2]
