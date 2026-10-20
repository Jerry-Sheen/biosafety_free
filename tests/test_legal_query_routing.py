import pytest
from src.rag.engine import get_system_prompt, PATHOGEN_SYSTEM_PROMPT, LEGAL_EXPERT_SYSTEM_PROMPT
from src.rag.law_search import LawSearchClient
from src.rag.pathogen_registry import PathogenRegistry


def test_system_prompt_selection():
    """Verify system prompt is dynamically routed based on query type."""
    # When query is a pathogen
    prompt_pathogen = get_system_prompt(is_pathogen_query=True)
    assert prompt_pathogen == PATHOGEN_SYSTEM_PROMPT
    assert "8대 필수 항목별 규제 요건" in prompt_pathogen

    # When query is a general legal / procedural question
    prompt_legal = get_system_prompt(is_pathogen_query=False)
    assert prompt_legal == LEGAL_EXPERT_SYSTEM_PROMPT
    assert "법률전문가" in prompt_legal
    assert "법학대학 교수" in prompt_legal
    assert "변호사" in prompt_legal
    assert "특정 병원체 8대 항목 템플릿 사용 절대 금지" in prompt_legal


def test_general_law_search_and_format():
    """Verify law search provides objective statutory context without 8-item template."""
    client = LawSearchClient()
    res = client.search_laws_for_general_query("생물작용제 제조신고 절차")
    assert "applicable_laws" in res
    assert any(law["short_name"] == "생화학무기금지법" for law in res["applicable_laws"])

    formatted = client.format_general_law_context_for_prompt(res)
    assert "법제처(국가법령정보센터)" in formatted
    assert "주요 법정 규정 및 조문별 요건" in formatted
    # 8-item fixed category heading must NOT be forced
    assert "8대 필수 항목별 규제 요건:" not in formatted


def test_pathogen_vs_general_query_classification():
    """Verify registry correctly separates specific pathogens from general legal questions."""
    registry = PathogenRegistry.get_instance()

    # Specific pathogens
    assert registry.find_pathogen("St. Louis encephalitis virus") is not None
    assert registry.find_pathogen("탄저균") is not None or registry.find_pathogen("Bacillus anthracis") is not None

    # General legal / procedural queries
    assert registry.find_pathogen("생물작용제 제조신고 절차와 요건") is None
    assert registry.find_pathogen("LMO 연구시설 안전관리 기준") is None
    assert registry.find_pathogen("고위험병원체 수입허가 관련 법률") is None


def test_verbatim_article_25_extraction():
    """Verify Article 25 verbatim text from law.go.kr is extracted accurately."""
    client = LawSearchClient()
    res = client.search_laws_for_general_query("법 제25조의 내용을 정확하게 알려주고 비교 검토해줘")
    formatted = client.format_general_law_context_for_prompt(res)

    # Must contain official verbatim text of Article 25 from law.go.kr
    assert "제25조(벌칙)" in formatted
    assert "사형, 무기 또는 7년 이상의 징역" in formatted
    assert "무기(無期) 또는 5년 이상의 징역이나 1억원 이하의 벌금" in formatted

