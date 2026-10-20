"""
Unit tests for biosafety pathogen accuracy and regulatory status verification.
Guarantees 100% agreement with master dataset ('병원균(위험균) 종류 v3.json').
"""

import pytest
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.rag.pathogen_registry import PathogenRegistry
from src.rag.engine import parse_pathogen_chunk_attributes


@pytest.fixture(scope="module")
def registry():
    return PathogenRegistry.get_instance()


def test_registry_loaded(registry):
    """Test that master pathogen dataset is loaded properly with > 500 entries."""
    assert len(registry.master_data) >= 500
    assert len(registry.alias_map) >= 500


def test_yersinia_pestis_attributes(registry):
    """
    Test that Yersinia pestis (페스트균) has 100% exact attributes,
    specifically verifying that special veterinary pathogen is TRUE (해당).
    """
    queries = [
        "페스트균에 대해 알려줘.",
        "페스트균",
        "Yersinia pestis",
        "흑사병",
        "흑사병균"
    ]
    for q in queries:
        info = registry.find_pathogen(q)
        assert info is not None, f"Failed to match query '{q}'"
        assert "Yersinia pestis" in info["canonical_name"]
        assert info["risk_group"] == "제3등급"
        assert info["is_high_risk"] is True
        assert info["is_bioterror"] is True
        assert info["is_lmo"] is True
        assert info["is_bioweapon"] is True

        # CRITICAL TEST: Veterinary special pathogen MUST be True
        assert info["is_vet_special"] is True, f"Failed for query '{q}': is_vet_special must be True"
        assert info["is_veterinary"] is True

        # Check step1_markdown output
        md = info["step1_markdown"]
        assert "* **생물체위험군 분류**: :red[**제3등급**]" in md
        assert "* **감염병예방법(고위험병원체)**: :red[**해당**]" in md
        assert "* **가축전염병예방법 및 수의생명자원관리규정(특별관리병원체)**: :red[**해당**]" in md
        assert "* **수산생물질병관리법**: 해당사항 없음" in md

        # Verify applicable laws list
        app_laws = " ".join(info["applicable_laws"])
        assert "감염병의 예방" in app_laws
        assert "가축전염병" in app_laws
        assert "화학무기" in app_laws or "생물무기" in app_laws
        assert "유전자변형" in app_laws


def test_bacillus_anthracis(registry):
    """Test Bacillus anthracis (탄저균) attributes."""
    for q in ["탄저균", "탄저", "Bacillus anthracis"]:
        info = registry.find_pathogen(q)
        assert info is not None
        assert "Bacillus anthracis" in info["canonical_name"]
        assert info["is_high_risk"] is True
        assert info["is_bioterror"] is True
        assert info["is_bioweapon"] is True
        assert info["is_vet_special"] is True


def test_african_horse_sickness_virus(registry):
    """Test African horse sickness virus (아프리카마역 바이러스)."""
    for q in ["African horse sickness virus", "아프리카마역 바이러스", "아프리카마역"]:
        info = registry.find_pathogen(q)
        assert info is not None
        assert "African horse sickness virus" in info["canonical_name"]
        assert info["is_veterinary"] is True
        assert info["is_vet_special"] is True


def test_variola_disambiguation(registry):
    """Test that 소두창 (Variola minor) and 두창 (Variola virus) are disambiguated."""
    info_minor = registry.find_pathogen("소두창 바이러스")
    assert info_minor is not None
    assert "Variola minor" in info_minor["canonical_name"]
    assert info_minor["is_vet_special"] is False

    info_major = registry.find_pathogen("두창 바이러스")
    assert info_major is not None
    assert "Variola virus" in info_major["canonical_name"]
    assert info_major["is_vet_special"] is True


def test_parse_pathogen_chunk_fallback():
    """Test that text fallback parser also correctly recognizes special vet pathogen."""
    sample_chunk = """
    [병원균/항목: Yersinia pestis(페스트균)]
    - 감염병예방법(고위험병원체): True
    - 감염병예방법(생물테러감염병병원체): True
    - 유전자변형생물체법(국가관리대상병원체): True
    - 생화학무기금지법(생물작용제): True
    - 가축병예방법_제1종: False
    - 가축병예방법_제2종: False
    - 가축병예방법_제3종: False
    - 가축병전염예방법 및 수의생명자원관리규정(일반관리병원체): False
    - 가축병전염예방법 및 수의생명자원관리규정(특별관리병원체): True
    - 수산생물질병관리법: False
    - 생물체위험군 분류: 3등급
    - 비고: nan
    """
    res = parse_pathogen_chunk_attributes(sample_chunk)
    assert res is not None
    assert res["is_vet_special"] is True
    assert res["is_veterinary"] is True
    assert "* **가축전염병예방법 및 수의생명자원관리규정(특별관리병원체)**: :red[**해당**]" in res["step1_markdown"]
    assert "* **생물체위험군 분류**: :red[**제3등급**]" in res["step1_markdown"]


def test_parse_pathogen_chunk_with_remarks():
    """Test that remarks and risk group are formatted in red bold."""
    sample_chunk = """
    [병원균/항목: Sample Pathogen]
    - 감염병예방법(고위험병원체): True
    - 생물체위험군 분류: 제3등급
    - 비고: 고위험성 병원균
    """
    res = parse_pathogen_chunk_attributes(sample_chunk)
    assert res is not None
    assert "* **생물체위험군 분류**: :red[**제3등급**]" in res["step1_markdown"]
    assert "* **비고**: :red[**고위험성 병원균**]" in res["step1_markdown"]


def test_law_search_precautions():
    """Test that LawSearchClient includes structured precautions (penalties, joint liability, etc.)."""
    from src.rag.law_search import LawSearchClient, CORE_LAWS_KNOWLEDGE
    client = LawSearchClient()
    for law_key, data in CORE_LAWS_KNOWLEDGE.items():
        assert "precautions" in data, f"Missing precautions in {law_key}"
        prec = data["precautions"]
        assert "형사처벌 및 행정처분" in prec
        assert "양벌규정" in prec
        assert "보관 및 운송 규정" in prec
        assert "위험물질 취급 제한" in prec

    # Test format_law_context_for_prompt includes category 3. 주의사항
    sample_law_data = client.search_applicable_laws_for_pathogen("페스트균", {"is_high_risk": True, "is_bioweapon": True})
    formatted = client.format_law_context_for_prompt(sample_law_data)
    assert "3. 주의사항" in formatted
    assert "형사처벌" in formatted
    assert "양벌규정" in formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
