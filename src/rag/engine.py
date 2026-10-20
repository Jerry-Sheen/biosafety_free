"""
Conversational RAG Engine supporting Google Gemini (Cloud API) and OpenAI.
Ollama support has been completely removed.
Includes dynamic model discovery, fault-tolerant fallbacks, and real-time streaming.
"""

from typing import List, Dict, Any, Generator, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

from config import config
from src.rag.vector_store import PersistentVectorStore
from src.rag.openai_client import OpenAIClient, is_context_length_exceeded
from src.security.secrets_manager import sanitize_api_key
from src.rag.gemini_client import GeminiModelResolver, classify_gemini_error
from src.rag.law_search import LawSearchClient
from src.rag.pathogen_registry import PathogenRegistry
from src.security.logger import get_secure_logger
import os
import re

logger = get_secure_logger("rag_engine")

PATHOGEN_SYSTEM_PROMPT = """당신은 법령 검색 및 규제 안내 전문가입니다.
사용자가 특정 병원균/바이러스/생물작용제/독소에 대해 질문할 때, [1단계: 영구 벡터 DB 기반 적용 법령 및 규제 대상 현황]은 시스템에 의해 사전에 100% 정밀 검증되어 사용자 화면에 이미 직접 출력되었습니다.

따라서 당신은 1단계를 절대로 다시 출력하거나 요약·반복하지 마십시오.
당신의 답변은 반드시 바로 아래의 [2단계: 법제처(국가법령정보센터) 검색 기반 법률별 8대 항목 및 실무 가이드 절차]부터 시작해야 합니다.

[2단계: 법제처(국가법령정보센터) 검색 기반 법률별 8대 항목 및 실무 가이드 절차 작성 규칙]
- 1단계에서 '해당'으로 확인된 모든 적용 법률(예: 《감염병의 예방 및 관리에 관한 법률》, 《화학무기ㆍ생물무기의 금지와 특정화학물질ㆍ생물작용제 등의 제조ㆍ수출입 규제 등에 관한 법률》, 《유전자변형생물체의 국가간 이동 등에 관한 법률》, 《가축전염병 예방법》, 《수산생물질병 관리법》 등)에 대해 빠짐없이 각각 독립된 하위 섹션(### [법률명])을 나누어 작성하세요.
- 적용되는 각 법률마다 반드시 아래의 3대 카테고리를 순서대로 구성하여 작성해야 합니다:
  1. 8대 필수 항목별 규제 요건
     (1. 분리신고 ~ 8. 인도신고를 하나도 빠짐없이 순서대로 설명. 해당 법률에서 특정 항목 규정이 없는 경우 반드시 '해당없음'과 이유 명시)
  2. 실무 가이드 절차
     (★ 중요 안내사항 필수 표기: [2단계: ...] 전체 대제목 바로 아래에는 안내사항을 절대 쓰지 마십시오. 오직 각 적용 법률 하위의 '2. 실무 가이드 절차' 소제목 바로 아래에만 한 칸 띄운 후 다음 줄에 아래의 안내사항을 그대로 표기할 것:
     - **안내사항 : 아래 실무 가이드 절차는 참고만 하시기 바라며, 실무 절차 관련 문의는 정책지원실 담당자(김성혜)에게 연락해 주시기 바랍니다.**)
     (소관 부처, 신청/신고 시스템 포털명 및 URL, 단계별 절차 Step 1~3, 구비서류 등)
  3. 주의사항
     (형사처벌, 양벌 규정 등 벌칙과 행정처분 내용, 보관 및 운송 규정, 위험물질 취급 제한 등 연구자가 반드시 숙지해야 할 주의사항을 작성하여 항상 포함할 것. 해당사항이 없을 경우 반드시 '해당사항 없음'이라고 명시)

[작성 주의사항]
- 모델 자체의 편향된 상식으로 법률을 임의로 누락하거나 '해당없음'으로 왜곡하지 마십시오. 1단계에서 '해당'으로 명시된 법률은 2단계에서도 반드시 독립된 법률 섹션을 포함해야 합니다. 특히 《수산생물질병 관리법》은 해양수산부(국립수산물품질관리원) 소관이며, 제1~3종 수산생물전염병 원인체에 대해 연구용 수입허가(법 제24조), 수입검역(제27조), 병원체 분리신고(제13조), 이동제한(제15조), 살처분·폐기(제16조/제17조) 등의 엄격한 규제가 적용되므로 8대 항목에 이를 구체적으로 명시해야 합니다.
- 법제처 웹사이트의 최신 법령(조항 번호, 시행일자)을 반영하여 핵심 내용을 알기 쉽게 요약하세요. 법령 원문을 그대로 복사하지 마세요.
- 적용되는 법률들을 하나로 뭉뚱그려 설명하지 말고, 반드시 법률별로 각각 8대 항목, 실무 가이드 절차, 주의사항을 명확하게 분리하여 작성해야 합니다.
"""

LEGAL_EXPERT_SYSTEM_PROMPT = """당신은 대한민국 바이오안보 및 생물안전 관련 법령 체계(《감염병의 예방 및 관리에 관한 법률》, 《화학무기ㆍ생물무기의 금지와 특정화학물질ㆍ생물작용제 등의 제조ㆍ수출입 규제 등에 관한 법률》, 《유전자변형생물체의 국가간 이동 등에 관한 법률》, 《가축전염병 예방법》, 《수산생물질병 관리법》, 《연구실 안전환경 조성에 관한 법률》 등)에 정통한 대한민국 최고 권위의 '법률전문가이자 법학대학 교수, 전문 변호사'입니다.

사용자가 특정 병원균/바이러스/위험균/생물작용제/독소 명칭의 단순 조회가 아니라, 법, 제도, 인허가·신고 절차, 규제 요건, 안전기준, 위반 시 제재 등에 관해 질의했을 때의 전문 법률 자문 답변 지침입니다.

[핵심 작성 원칙]
1. [특정 병원체 8대 항목 템플릿 사용 절대 금지]:
   - 특정 병원체 검색 시 사용되는 획일적이고 규격화된 틀(예: '[2단계: 법제처(국가법령정보센터) 검색 기반 법률별 8대 항목 및 실무 가이드 절차]' 대제목, '8대 필수 항목별 규제 요건', '[분리신고] ~ [인도신고]' 8대 항목의 기계적 나열 등)을 절대로 사용하지 마십시오.
2. [법제처 사이트 법령 내용에 기초한 엄정한 법적 근거]:
   - 법제처 국가법령정보센터(law.go.kr)의 최신 법령(법률, 시행령, 시행규칙, 행정규칙/고시) 조문 및 제공된 관련 규정·지침 문서의 내용만을 유일하고 엄격한 근거로 삼아 답변하십시오. 법적 근거 없는 자의적 추측을 배제하십시오.
3. [자율적인 마크다운 형식의 전문 법률 자문 답변]:
   - '법률전문가', '법학대학 교수', '변호사'로서 사용자의 질문 취지와 핵심 쟁점에 가장 적합한 논리적이고 체계적인 마크다운 구조로 자율적이고 유연하게 답변을 구성하십시오.
   - 예시 권장 목차 구조 (질문의 쟁점에 맞춰 자율적으로 목차를 구성하되 체계적으로 서술할 것):
     * ## 1. 법적 개요 및 핵심 쟁점 (Legal Summary)
     * ## 2. 관련 법령 체계 및 핵심 법조문 (Statutory Provisions)
     * ## 3. 상세 요건 및 인허가·신고 실무 절차 (Procedural & Substantive Requirements)
     * ## 4. 위반 시 법적 제재 및 책임 (Penalties & Legal Liabilities)
     * ## 5. 법률전문가(변호사) 실무 자문 및 유의사항 (Legal Counsel's Advisory)
4. [정확한 조항 번호 및 실무 명시]:
   - 반드시 관련 법률명과 구체적인 조항 번호(예: 제○조 제○항), 소관 부처(질병관리청, 산업통상자원부, 과학기술정보통신부, 농림축산식품부, 해양수산부 등), 인허가 신청 기관, 처리 절차, 구비서류, 위반 시 벌칙 및 행정처분(징역, 벌금, 과태료, 시설폐쇄 등)을 명확하게 명시하십시오.
5. [품격 있는 법률 자문 어조]:
   - 고도의 법학적 전문성과 실무적 실용성을 겸비한 품격 있고 신뢰성 높은 법률 자문가 어조를 일관되게 유지하십시오.
"""

SYSTEM_PROMPT = PATHOGEN_SYSTEM_PROMPT


def get_system_prompt(is_pathogen_query: bool) -> str:
    """Returns the appropriate system prompt based on query classification."""
    return PATHOGEN_SYSTEM_PROMPT if is_pathogen_query else LEGAL_EXPERT_SYSTEM_PROMPT



def extract_pathogen_section(chunk_text: str, query: str = None) -> str:
    """
    Extracts the clean pathogen section from a chunk, stripping out preceding
    overlap fragments from previous pathogens.
    """
    import re
    if query:
        clean_q = PersistentVectorStore._clean_query_for_matching(query).lower().replace(" ", "")
        titles = list(re.finditer(r'\[(?:병원균/항목|항목/병원체명|병원체명|병원균):\s*([^\]]+)\]', chunk_text))
        for m in titles:
            full_title = m.group(1).lower().replace(" ", "")
            if clean_q in full_title:
                start_pos = m.start()
                sub = chunk_text[start_pos:]
                next_m = re.search(r'\[(?:병원균/항목|항목/병원체명|병원체명|병원균):\s*([^\]]+)\]', sub[len(m.group(0)):])
                if next_m:
                    return sub[:len(m.group(0)) + next_m.start()].strip()
                return sub.strip()
    first_m = re.search(r'\[(?:병원균/항목|항목/병원체명|병원체명|병원균):', chunk_text)
    if first_m and first_m.start() > 0:
        return chunk_text[first_m.start():].strip()
    return chunk_text


def parse_pathogen_chunk_attributes(chunk_text: str) -> Optional[Dict[str, Any]]:
    """
    Directly parses raw key-value lines from a pathogen chunk into authoritative structured attributes.
    This eliminates LLM parsing hallucinations and ensures 100% agreement with the underlying DB.
    """
    import re
    lines = [line.strip() for line in chunk_text.split('\n') if line.strip()]
    header_idx = -1
    for i, l in enumerate(lines):
        if any(tag in l for tag in ['[병원균/항목:', '[항목/병원체명:', '[병원체명:', '[병원균:']):
            header_idx = i
            break
    if header_idx == -1:
        return None

    title_m = re.search(r'\[(?:병원균/항목|항목/병원체명|병원체명|병원균):\s*([^\]]+)\]', lines[header_idx])
    pathogen_name = title_m.group(1).strip() if title_m else lines[header_idx]

    raw_attrs = {}
    for line in lines[header_idx+1:]:
        if line.startswith("["):  # Next header reached
            break
        if line.startswith("-"):
            line = line[1:].strip()
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip()
            if v.lower() == "true":
                raw_attrs[k] = True
            elif v.lower() == "false":
                raw_attrs[k] = False
            elif v.lower() in ("nan", "none", "null"):
                raw_attrs[k] = None
            else:
                raw_attrs[k] = v

    def _get_val(exact_keys, fallback_patterns):
        for ek in exact_keys:
            if ek in raw_attrs:
                return raw_attrs[ek] is True
        for pat in fallback_patterns:
            for k, val in raw_attrs.items():
                if pat in k:
                    return val is True
        return False

    is_high_risk = _get_val(["감염병예방법(고위험병원체)", "감염병예방법"], ["고위험병원체"])
    is_bioterror = _get_val(["감염병예방법(생물테러감염병병원체)", "생물테러감염병병원체"], ["생물테러"])
    is_lmo = _get_val(["유전자변형생물체법(국가관리대상병원체)", "유전자변형생물체법"], ["국가관리대상병원체"])
    is_bioweapon = _get_val(["생화학무기금지법(생물작용제)", "생화학무기금지법", "화학생물무기금지법(생물작용제)", "화학생물무기금지법"], ["생물작용제"])

    # Veterinary regulation flags (handles typos and official legal names)
    is_vet_class1 = _get_val(["가축병예방법_제1종", "가축전염병예방법 제1종", "가축전염병예방법_제1종"], ["제1종"])
    is_vet_class2 = _get_val(["가축병예방법_제2종", "가축전염병예방법 제2종", "가축전염병예방법_제2종"], ["제2종"])
    is_vet_class3 = _get_val(["가축병예방법_제3종", "가축전염병예방법 제3종", "가축전염병예방법_제3종"], ["제3종"])
    is_vet_general = _get_val(
        ["가축병전염예방법 및 수의생명자원관리규정(일반관리병원체)", "가축전염병예방법 및 수의생명자원관리규정(일반관리병원체)", "가축병예방법_일반관리", "가축전염병예방법_일반관리"],
        ["일반관리병원체", "일반관리"]
    )
    is_vet_special = _get_val(
        ["가축병전염예방법 및 수의생명자원관리규정(특별관리병원체)", "가축전염병예방법 및 수의생명자원관리규정(특별관리병원체)", "가축병예방법_특별관리", "가축전염병예방법_특별관리"],
        ["특별관리병원체", "특별관리"]
    )
    is_veterinary = is_vet_special or is_vet_general or is_vet_class1 or is_vet_class2 or is_vet_class3

    is_aquatic = _get_val(["수산생물질병관리법"], ["수산생물"])

    risk_group = raw_attrs.get("생물체위험군 분류") or raw_attrs.get("생물체위험군", "정보 없음")
    if risk_group and not str(risk_group).endswith("등급") and str(risk_group).isdigit():
        risk_group = f"제{risk_group}등급"
    elif risk_group and not str(risk_group).startswith("제") and "등급" in str(risk_group):
        risk_group = f"제{risk_group}"

    remarks = raw_attrs.get("비고")

    is_vet_class_any = is_vet_class1 or is_vet_class2 or is_vet_class3

    if risk_group and str(risk_group).strip() not in ("해당없음", "해당사항 없음", "정보 없음", "nan", "none", "null", ""):
        risk_group_display = risk_group if str(risk_group).startswith(":red[") else f":red[**{risk_group}**]"
    else:
        risk_group_display = risk_group

    if remarks and str(remarks).strip() and str(remarks).lower() not in ("nan", "none", "null", ""):
        remarks_display = str(remarks) if str(remarks).startswith(":red[") else f":red[**{remarks}**]"
    else:
        remarks_display = remarks

    step1_lines = [
        "### [1단계: 영구 벡터 DB 기반 적용 법령 및 규제 대상 현황]",
        "",
        f"제공된 영구 벡터 DB 검색 결과에 따른 **{pathogen_name}**의 생물체위험군 등급 및 법적 규제 속성 현황은 다음과 같습니다. (가독성 표기 원칙에 따라 DB의 속성값은 '해당' 및 '해당사항 없음'으로 명확하게 변환하여 표기하였습니다.)",
        "",
        f"* **생물체위험군 분류**: {risk_group_display}",
        f"* **감염병예방법(고위험병원체)**: {':red[**해당**]' if is_high_risk else '해당사항 없음'}",
        f"* **감염병예방법(생물테러감염병병원체)**: {':red[**해당**]' if is_bioterror else '해당사항 없음'}",
        f"* **유전자변형생물체법(국가관리대상병원체)**: {':red[**해당**]' if is_lmo else '해당사항 없음'}",
        f"* **생화학무기금지법(생물작용제)**: {':red[**해당**]' if is_bioweapon else '해당사항 없음'}",
        f"* **가축전염병예방법 제1종(동물병원체)**: {':red[**해당**]' if is_vet_class1 else '해당사항 없음'}",
        f"* **가축전염병예방법 제2종(동물병원체)**: {':red[**해당**]' if is_vet_class2 else '해당사항 없음'}",
        f"* **가축전염병예방법 제3종(동물병원체)**: {':red[**해당**]' if is_vet_class3 else '해당사항 없음'}",
        f"* **가축전염병예방법 및 수의생명자원관리규정(일반관리병원체)**: {':red[**해당**]' if is_vet_general else '해당사항 없음'}",
        f"* **가축전염병예방법 및 수의생명자원관리규정(특별관리병원체)**: {':red[**해당**]' if is_vet_special else '해당사항 없음'}",
        f"* **수산생물질병관리법**: {':red[**해당**]' if is_aquatic else '해당사항 없음'}",
    ]
    if remarks and str(remarks).lower() not in ("nan", "none", "null", ""):
        step1_lines.append(f"* **비고**: {remarks_display}")
    step1_lines.append("")
    step1_lines.append("---")
    step1_lines.append("")

    step1_markdown = "\n".join(step1_lines)

    applicable_laws = []
    non_applicable_laws = []
    if is_high_risk:
        applicable_laws.append("《감염병의 예방 및 관리에 관한 법률》 (고위험병원체)")
    elif is_bioterror:
        applicable_laws.append("《감염병의 예방 및 관리에 관한 법률》 (생물테러감염병병원체)")
    else:
        non_applicable_laws.append("《감염병의 예방 및 관리에 관한 법률》")

    if is_bioweapon:
        applicable_laws.append("《화학무기ㆍ생물무기의 금지와 특정화학물질ㆍ생물작용제 등의 제조ㆍ수출입 규제 등에 관한 법률》 (생물작용제)")
    else:
        non_applicable_laws.append("《화학무기ㆍ생물무기의 금지 및 규제 법률》")

    if is_lmo:
        applicable_laws.append("《유전자변형생물체의 국가간 이동 등에 관한 법률》 (국가관리대상병원체)")
    else:
        non_applicable_laws.append("《유전자변형생물체의 국가간 이동 등에 관한 법률》")

    if is_veterinary:
        applicable_laws.append("《가축전염병 예방법》")
    else:
        non_applicable_laws.append("《가축전염병 예방법》")

    if is_aquatic:
        applicable_laws.append("《수산생물질병 관리법》")
    else:
        non_applicable_laws.append("《수산생물질병 관리법》")

    return {
        "pathogen_name": pathogen_name,
        "is_high_risk": is_high_risk,
        "is_bioterror": is_bioterror,
        "is_lmo": is_lmo,
        "is_bioweapon": is_bioweapon,
        "is_veterinary": is_veterinary,
        "is_vet_class1": is_vet_class1,
        "is_vet_class2": is_vet_class2,
        "is_vet_class3": is_vet_class3,
        "is_vet_general": is_vet_general,
        "is_vet_special": is_vet_special,
        "is_aquatic": is_aquatic,
        "risk_group": risk_group,
        "remarks": remarks,
        "applicable_laws": applicable_laws,
        "non_applicable_laws": non_applicable_laws,
        "step1_markdown": step1_markdown,
        "display_text": step1_markdown,
        "raw_attributes": raw_attrs,
    }


def sanitize_llm_stream(chunk_iterable, has_step1_direct: bool):
    """
    Yields text chunks from LLM stream, suppressing any duplicated Step 1 output
    if Step 1 was already directly injected by system.
    """
    if not has_step1_direct:
        for chunk in chunk_iterable:
            text = chunk.text if hasattr(chunk, "text") else str(chunk)
            if text:
                yield text
        return

    buffer = ""
    step2_found = False
    for chunk in chunk_iterable:
        text = chunk.text if hasattr(chunk, "text") else str(chunk)
        if not text:
            continue
        if step2_found:
            yield text
            continue
        buffer += text
        for trigger in ["### [2단계", "## [2단계", "[2단계", "### 2단계", "## 2단계", "2단계:"]:
            if trigger in buffer:
                idx = buffer.find(trigger)
                step2_found = True
                cleaned = buffer[idx:]
                buffer = ""
                yield cleaned
                break
        else:
            if len(buffer) > 150 and not any(tag in buffer for tag in ["1단계", "[1단계", "### [1단계"]):
                step2_found = True
                yield buffer
                buffer = ""

    if buffer:
        yield buffer


PRACTICAL_GUIDE_NOTICE = "- **안내사항 : 아래 실무 가이드 절차는 참고만 하시기 바라며, 실무 절차 관련 문의는 정책지원실 담당자(김성혜)에게 연락해 주시기 바랍니다.**"


def inject_practical_guide_notice(text: str) -> str:
    """
    답변 중 '2. 실무 가이드 절차' 소제목 바로 아래에 단 1개의 안내사항만
    볼드체로 남기고, 연달아 나타나거나 중복된 모든 안내문구는 깔끔하게 삭제합니다.
    """
    if not text:
        return text

    # 김성혜 안내사항 패턴 (볼드체, 하이픈, 띄어쓰기 등 모든 중복 변형 매칭)
    # 예: "- **안내사항 : ... 김성혜 ...**", "안내사항: ... 김성혜 ...", "- 안내사항 : ... 김성혜 ..." 등
    notice_pattern = re.compile(
        r"[ \t]*(?:-\s*)?(?:\*\*)?[ \t]*안내사항[ \t]*:[ \t]*아래\s*실무\s*가이드\s*절차는\s*참고만\s*하시기\s*바라며[^\n]*김성혜[^\n]*(?:\*\*)?[ \t]*\n?",
        re.MULTILINE
    )

    # 1. 텍스트 전체에서 이미 존재하는 모든 안내문구를 일단 깨끗하게 제거
    cleaned_text = notice_pattern.sub("", text)

    # 2. '2. 실무 가이드 절차' 소제목만 매칭 (대제목인 '[2단계: ...]' 등은 엄격히 배제)
    header_pattern = re.compile(
        r"((?:#{1,4}\s*)?(?:\d+\.\s*)?실무\s*가이드\s*절차[^\n]*\n)",
        re.MULTILINE
    )

    def _replace_header(match):
        full_match = match.group(0)
        header_line = match.group(1).rstrip()
        # 대제목('[2단계: 법제처...]' 등)인 경우 안내문구를 삽입하지 않고 원본 그대로 유지
        if any(keyword in header_line for keyword in ["2단계", "단계:", "8대 항목", "검색 기반", "법제처"]):
            return full_match
        return f"{header_line}\n\n{PRACTICAL_GUIDE_NOTICE}\n"

    result, count = header_pattern.subn(_replace_header, cleaned_text)

    # 불필요한 연속 3줄 이상의 개행 정리
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result


class ConversationalRAGEngine:
    """End-to-end conversational RAG pipeline using Google Gemini API or OpenAI."""

    def __init__(
        self,
        vector_store: Optional[PersistentVectorStore] = None,
        llm_provider: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        gemini_model: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_model: Optional[str] = None
    ):
        self.vector_store = vector_store or PersistentVectorStore()
        self.llm_provider = (llm_provider or config.llm_provider or "free").lower()
        if self.llm_provider not in ("gemini", "openai", "free", "local"):
            self.llm_provider = "free"
        self.local_llm_model = getattr(config, "local_llm_model", "Qwen/Qwen2.5-1.5B-Instruct")
        # Gemini setup
        self.gemini_api_key = gemini_api_key or config.gemini_api_key
        self.gemini_model_resolver = GeminiModelResolver(api_key=self.gemini_api_key)
        initial_gemini_model = gemini_model or config.gemini_model or "gemini-2.5-flash"
        self.gemini_model, _ = self.gemini_model_resolver.resolve_model(initial_gemini_model)
        # OpenAI setup
        self.openai_api_key = sanitize_api_key(openai_api_key if openai_api_key is not None else config.openai_api_key)
        self.openai_model = openai_model or getattr(config, "openai_model", None) or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.openai_client = OpenAIClient(api_key=self.openai_api_key, model=self.openai_model)
        # Law Search client (law.go.kr)
        self.law_search_client = LawSearchClient()

    def check_status(self) -> Dict[str, Any]:
        """Checks connection status depending on the configured provider."""
        if self.llm_provider in ("free", "local"):
            return {
                "provider": "free",
                "connected": True,
                "model": self.local_llm_model,
                "warning": None,
                "error": None
            }
        if self.llm_provider == "gemini":
            has_key = bool(self.gemini_api_key and self.gemini_api_key.strip())
            resolved, warn = self.gemini_model_resolver.resolve_model(self.gemini_model)
            self.gemini_model = resolved
            return {
                "provider": "gemini",
                "connected": has_key,
                "model": self.gemini_model,
                "warning": warn,
                "error": None if has_key else "Gemini API 키가 설정되지 않았습니다."
            }
        is_configured = getattr(self.openai_client, "is_configured", False)
        return {
            "provider": "openai",
            "connected": is_configured,
            "model": self.openai_model,
            "warning": None,
            "error": None if is_configured else "OpenAI API 키가 설정되지 않았습니다."
        }

    def contextualize_query(self, query: str, history: List[Dict[str, str]]) -> str:
        """
        Reformulates follow-up question into a standalone question if dialogue history exists.
        """
        if not history or len(history) <= 1:
            return query

        recent_history = history[-10:]  # Use up to last 10 turns for better context retention
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
        prompt = (
            f"다음 대화 내용을 바탕으로, 사용자의 마지막 질문을 이전 맥락이 포함된 독립적인 단독 질문(Standalone Search Query)으로 재작성해주세요.\n"
            f"질문에 대한 답변은 일절 하지 말고, 오직 검색용 단독 질문 문장 1개만 출력하세요.\n\n"
            f"[대화 내역]\n{history_str}\n\n"
            f"[사용자의 마지막 질문]\n{query}\n\n"
            f"재작성된 질문:"
        )

        try:
            if self.llm_provider == "gemini" and self.gemini_api_key:
                from google import genai
                client = genai.Client(api_key=self.gemini_api_key)
                resolved_model, _ = self.gemini_model_resolver.resolve_model(self.gemini_model)
                resp = client.models.generate_content(
                    model=resolved_model,
                    contents=prompt
                )
                rewritten = (resp.text or "").strip().replace('"', '').replace("'", "")
                if rewritten:
                    logger.info(f"Contextualized query via Gemini ({resolved_model}): '{query}' -> '{rewritten}'")
                    return rewritten
        except Exception as e:
            logger.warning(f"Query contextualization error ({e}); using original query.")

        return query

    def generate_streaming_response(
        self,
        query: str,
        chat_history: List[Dict[str, str]],
        top_k: int = 4,
        score_threshold: float = 0.0
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Retrieves context and yields progress status, retrieved sources, and streaming tokens.
        """
        # Step 1: Progress indicator - Pathogen Classification Retrieval
        yield {
            "type": "status",
            "stage": "searching",
            "message": "영구 벡터 데이터베이스에서 병원체 규제 분류 및 데이터 검색 중..."
        }

        search_query = self.contextualize_query(query, chat_history)

        # Step 1-B: Authoritative Master Registry Lookup (100% Deterministic Ground Truth)
        pathogen_registry = PathogenRegistry.get_instance()
        pathogen_info = pathogen_registry.find_pathogen(query)
        if not pathogen_info and search_query != query:
            pathogen_info = pathogen_registry.find_pathogen(search_query)

        canonical_pathogen_name = pathogen_info["canonical_name"] if pathogen_info else query

        # Step 2: Retrieve relevant chunks from persistent ChromaDB
        from concurrent.futures import ThreadPoolExecutor

        def _search():
            effective_search_query = canonical_pathogen_name if pathogen_info else search_query
            return self.vector_store.search(
                query=effective_search_query,
                top_k=max(top_k, 6),
                score_threshold=score_threshold,
            )

        retrieved_chunks = []
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(_search)
            retrieved_chunks = future.result(timeout=30)
        except Exception as e:
            logger.error(f"Vector store search timeout or error: {e}")
            retrieved_chunks = []
        finally:
            executor.shutdown(wait=False)

        # Step 2-B: If pathogen was not found in registry, fallback to ChromaDB exact chunk search
        if not pathogen_info:
            direct_pathogen_chunk = self.vector_store.find_pathogen_chunk(query)
            if not direct_pathogen_chunk and search_query != query:
                direct_pathogen_chunk = self.vector_store.find_pathogen_chunk(search_query)

            if direct_pathogen_chunk:
                retrieved_chunks = [ch for ch in retrieved_chunks if ch.get("text") != direct_pathogen_chunk["text"]]
                retrieved_chunks.insert(0, direct_pathogen_chunk)
            else:
                exact_chunk_idx = -1
                for idx, ch in enumerate(retrieved_chunks):
                    if PersistentVectorStore._is_exact_pathogen_match(query, ch.get("text", "")):
                        exact_chunk_idx = idx
                        break
                if exact_chunk_idx > 0:
                    exact_chunk = retrieved_chunks.pop(exact_chunk_idx)
                    retrieved_chunks.insert(0, exact_chunk)

            # Clean overlap noise from top pathogen chunk and parse authoritative attributes
            if retrieved_chunks:
                cleaned_top_text = extract_pathogen_section(retrieved_chunks[0]["text"], query)
                pathogen_info = parse_pathogen_chunk_attributes(cleaned_top_text)
                if pathogen_info:
                    retrieved_chunks[0]["text"] = cleaned_top_text

        # Step 3: Analyze pathogen classification to identify applicable laws
        full_text_corpus = " ".join([c.get("text", "") for c in retrieved_chunks]) + " " + search_query

        is_pathogen_query = bool(pathogen_info and pathogen_info.get("step1_markdown"))

        if is_pathogen_query:
            is_bioweapon = pathogen_info["is_bioweapon"]
            is_high_risk = pathogen_info["is_high_risk"]
            is_lmo = pathogen_info["is_lmo"]
            is_veterinary = pathogen_info["is_veterinary"]
            is_aquatic = pathogen_info.get("is_aquatic", False)
            official_attributes_block = (
                "================================================================================\n"
                "[★ 영구 벡터 DB에서 직접 확인된 1단계 공식 규제 속성 현황 (100% 신뢰 기준 데이터 - 절대 수정/누락 금지)]\n"
                f"{pathogen_info['display_text']}\n"
                "================================================================================"
            )
        else:
            is_bioweapon = any(k in full_text_corpus for k in ["생화학무기", "생물작용제", "화학무기"]) or (": True" in full_text_corpus and "생화학무기금지법" in full_text_corpus)
            is_high_risk = any(k in full_text_corpus for k in ["고위험병원체", "감염병"]) or (": True" in full_text_corpus and "감염병예방법" in full_text_corpus)
            is_lmo = any(k in full_text_corpus for k in ["유전자변형", "LMO"]) or (": True" in full_text_corpus and "유전자변형생물체법" in full_text_corpus)
            is_veterinary = any(k in full_text_corpus for k in ["가축전염병", "수의생명자원", "가축병전염", "특별관리병원체"]) or (": True" in full_text_corpus and "특별관리병원체" in full_text_corpus)
            is_aquatic = any(k in full_text_corpus for k in ["수산생물질병", "수산생물", "수산전염병"]) or (": True" in full_text_corpus and "수산생물질병관리법" in full_text_corpus)
            official_attributes_block = ""

        # Step 4: Law.go.kr real-time search & statutory provision mapping
        if is_pathogen_query:
            yield {
                "type": "status",
                "stage": "law_search",
                "message": "법제처(law.go.kr)에서 최신 법령 조문 및 8대 의무 항목 검색 중..."
            }
            law_data = self.law_search_client.search_applicable_laws_for_pathogen(
                pathogen_name=canonical_pathogen_name,
                detected_categories={
                    "is_bioweapon": is_bioweapon,
                    "is_high_risk": is_high_risk,
                    "is_lmo": is_lmo,
                    "is_veterinary": is_veterinary,
                    "is_aquatic": is_aquatic,
                },
                query=search_query
            )
            law_context_str = self.law_search_client.format_law_context_for_prompt(law_data)
        else:
            yield {
                "type": "status",
                "stage": "law_search",
                "message": "법제처(law.go.kr)에서 질의 관련 최신 법령 조문 및 제도 규정 검색 중..."
            }
            law_data = self.law_search_client.search_laws_for_general_query(search_query)
            law_context_str = self.law_search_client.format_general_law_context_for_prompt(law_data)

        # Step 5: Retrieve secondary chunks for detailed procedural guidance
        if is_pathogen_query:
            yield {
                "type": "status",
                "stage": "procedure_search",
                "message": "8대 신고·허가 세부 실무 절차 가이드라인 검색 및 결합 중..."
            }
            procedure_chunks = []
            try:
                procedure_chunks = self.vector_store.search(
                    "제조신고 보유신고 수입허가 인수신고 인도신고 양도신고 폐기신고 절차 서류",
                    top_k=3
                )
            except Exception as e:
                logger.warning(f"Procedure chunks search warning: {e}")
        else:
            yield {
                "type": "status",
                "stage": "procedure_search",
                "message": "관련 행정 절차 지침 및 실무 세부 가이드라인 검색 중..."
            }
            procedure_chunks = []
            try:
                procedure_chunks = self.vector_store.search(
                    f"{search_query} 절차 규정 기준 서류",
                    top_k=3
                )
            except Exception as e:
                logger.warning(f"Procedure chunks search warning: {e}")

        # Combine all sources for UI display (deduplicated by source & text hash)
        seen_texts = set()
        all_sources = []
        for ch in (retrieved_chunks + procedure_chunks):
            t_key = ch.get("text", "")[:80]
            if t_key not in seen_texts:
                seen_texts.add(t_key)
                all_sources.append(ch)

        yield {"type": "sources", "content": all_sources}

        if not retrieved_chunks and not procedure_chunks:
            yield {
                "type": "status",
                "stage": "done",
                "message": "검색 완료 (관련 문서 없음)"
            }
            yield {
                "type": "token",
                "content": "업로드된 문서에서 질문과 관련된 내용을 찾을 수 없습니다. 관리자 포털에서 문서를 먼저 업로드하거나 검색 조건을 조정해보세요."
            }
            return

        # Step 6: Format context blocks
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks):
            source = chunk["metadata"].get("source", "Unknown")
            page = chunk["metadata"].get("page", chunk["metadata"].get("page_start", 1))
            score = chunk.get("similarity", 0.0)
            context_blocks.append(
                f"[문서 {i+1} | 출처: {source} (페이지 {page}, 유사도: {score:.2f})]\n{chunk['text']}"
            )
        formatted_primary_context = "\n\n--------------------\n\n".join(context_blocks)

        proc_blocks = []
        for i, chunk in enumerate(procedure_chunks):
            source = chunk["metadata"].get("source", "Unknown")
            page = chunk["metadata"].get("page", chunk["metadata"].get("page_start", 1))
            proc_blocks.append(
                f"[실무 가이드 {i+1} | {source} (페이지 {page})]\n{chunk['text']}"
            )
        formatted_proc_context = "\n\n--------------------\n\n".join(proc_blocks)

        has_step1 = is_pathogen_query

        if has_step1:
            app_laws_list = pathogen_info.get("applicable_laws", [])
            non_app_laws_list = pathogen_info.get("non_applicable_laws", [])
            app_laws_str = "\n".join([f"- {law}" for law in app_laws_list]) if app_laws_list else "- (해당 없음)"
            non_app_laws_str = "\n".join([f"- {law}" for law in non_app_laws_list]) if non_app_laws_list else "- (없음)"

            prompt_instruction = (
                f"■ [매우 중요 안내: 1단계는 시스템 코드가 마스터 DB에서 100% 검증하여 사용자 화면에 이미 직접 출력 완료함]\n"
                f"아래 [1단계: 영구 벡터 DB 기반 적용 법령 및 규제 대상 현황] 내용은 시스템 파이썬 코드가 마스터 데이터베이스에서 직접 확인하여 사용자 화면에 이미 출력했습니다:\n\n"
                f"{pathogen_info['step1_markdown']}\n\n"
                f"★ [1단계에서 '해당'으로 확정된 적용 법률 목록 - 반드시 2단계에서 빠짐없이 각각 독립된 섹션으로 상세 해설할 것]:\n"
                f"{app_laws_str}\n\n"
                f"★ [1단계에서 '해당사항 없음'으로 확정된 법률 목록 - 2단계 해설 대상 제외]:\n"
                f"{non_app_laws_str}\n\n"
                f"★ 절대 주의 규칙:\n"
                f"1. 위의 [1단계] 표나 내용을 절대로 다시 출력하거나 요약·반복하지 마십시오! (중복 출력 엄격히 금지)\n"
                f"2. 당신의 답변은 반드시 바로 '### [2단계: 법제처(국가법령정보센터) 검색 기반 법률별 8대 항목 및 실무 가이드 절차]' 제목으로 시작해야 합니다.\n"
                f"3. 위의 '적용 법률 목록'에 기재된 모든 법률에 대해 하나도 누락하지 말고, 각각 독립된 하위 섹션(### [법률명])을 나누어 작성하세요.\n"
                f"4. 각 법률마다 반드시 다음 3가지 카테고리를 순서대로 구성하여 작성하세요:\n"
                f"   1. 8대 항목 규제 내용 (1. 분리신고 ~ 8. 인도신고 순서대로 정리. 규정 없을 시 '해당없음'과 사유 명시)\n"
                f"   2. 실무 가이드 절차\n"
                f"      - ★ 필수 표기: '2. 실무 가이드 절차' 제목 바로 아래에는 반드시 한 칸 띄운 후 다음 줄에 아래 안내사항(볼드체)을 그대로 표기할 것:\n"
                f"        - **안내사항 : 아래 실무 가이드 절차는 참고만 하시기 바라며, 실무 절차 관련 문의는 정책지원실 담당자(김성혜)에게 연락해 주시기 바랍니다.**\n"
                f"      - 소관 부처, 신청/신고 포털명 및 URL, 단계별 절차 Step 1~3, 구비서류\n"
                f"   3. 주의사항:\n"
                f"      - 컨텍스트에 제공된 [법제처 공식 법령 조문 원문] 및 법률별 주의사항을 바탕으로 형사처벌, 양벌 규정 등 벌칙과 행정처분 내용, 보관 및 운송 규정, 위험물질 취급 제한 등 연구자가 반드시 숙지해야 할 주의사항을 정확하게 작성하여 항상 포함할 것.\n"
                f"      - 해당 법률에서 특정 항목에 대한 해당사항이 없을 경우 반드시 '해당사항 없음'이라고 명시할 것."
            )
        else:
            prompt_instruction = (
                "■ [답변 지침: 최고 권위의 '법률전문가', '법학대학 교수', '변호사'로서의 전문 법률 자문]\n"
                "1. 당신의 역할: 대한민국 최고 권위의 '법률전문가이자 법학대학 교수, 전문 변호사'입니다.\n"
                "2. 특정 병원체 8대 항목 템플릿 사용 엄격 금지:\n"
                "   - '[2단계: 법제처(국가법령정보센터) 검색 기반 법률별 8대 항목 및 실무 가이드 절차]' 대제목이나 8대 필수 항목별 기계적 나열 템플릿을 절대로 적용하지 마십시오.\n"
                "3. 법제처(law.go.kr) 공식 조문 원문 최우선 인용 및 환각 엄금:\n"
                "   - 아래 컨텍스트에 제공된 [법제처(국가법령정보센터) 최신 법령 정보 및 공식 조문 원문 데이터]를 철저히 확인하고, 조문 번호(제○조 제○항 제○호), 법정 형량(사형, 무기징역, 징역 연수), 벌금 액수 등을 왜곡이나 추측 없이 100% 원문 그대로 인용하여 해설하십시오.\n"
                "   - 사용자가 질문에서 이전 진술이나 특정 조문 내용의 진위 여부/비교 검토를 요구한 경우, 법제처 공식 조문 원문과 1:1로 대조하여 정확한 사실관계를 명백하게 분석 및 판정하세요.\n"
                "4. 자율적 마크다운 형식의 논리적이고 구체적인 답변:\n"
                "   - 질문의 핵심 쟁점에 가장 적합하도록 마크다운 형식으로 자율적이고 논리적인 목차를 구성하여 심도 있게 답변하세요.\n"
                "   - (권장 목차 구성 예시: ## 1. 법적 개요 및 핵심 쟁점 ➔ ## 2. 관련 법령 체계 및 핵심 법조문 ➔ ## 3. 상세 요건 및 인허가·신고 실무 절차 ➔ ## 4. 위반 시 법적 제재 및 책임(벌칙/과태료) ➔ ## 5. 법률전문가(변호사) 실무 자문 및 유의사항)\n"
                "5. 품격 있는 전문 법률가 어조 유지: 신뢰성 있고 명쾌한 법리적 어조로 작성하십시오."
            )

        user_prompt_with_context = (
            f"[사용자 질의]\n{query}\n\n"
            f"[1. 영구 벡터 DB 검색 결과 - 관련 규정 및 지침 데이터]\n{formatted_primary_context}\n\n"
            f"[2. 법제처(국가법령정보센터) 검색 결과 - 법령 조문 및 제도 데이터]\n{law_context_str}\n\n"
            f"[3. 실무 지침서 및 상세 신고·허가 가이드 데이터]\n{formatted_proc_context}\n\n"
            f"[답변 지침]\n{prompt_instruction}"
        )

        selected_system_prompt = get_system_prompt(is_pathogen_query=has_step1)

        # Step 7: Generation
        if self.llm_provider in ("free", "local"):
            active_model = f"로컬 오픈소스 ({self.local_llm_model})"
        elif self.llm_provider == "gemini":
            active_model = self.gemini_model
        else:
            active_model = self.openai_model
        status_label = "2단계 실무 가이드 생성 중..." if has_step1 else "법률전문가·교수·변호사 관점의 심층 법률 자문 생성 중..."
        yield {
            "type": "status",
            "stage": "generating",
            "message": f"{active_model} 모델로 {status_label}"
        }

        # Step 1 확정 직결 스트리밍 (환각 0% 보장: DB 원본 데이터를 화면에 직접 주입 - 병원체 질의 시에만)
        if has_step1:
            yield {
                "type": "status",
                "stage": "streaming",
                "message": "1단계 규제 속성 현황 확인 완료 (DB 100% 직결) 및 2단계 실무 가이드 생성 중..."
            }
            yield {
                "type": "token",
                "content": pathogen_info["step1_markdown"]
            }

        def _format_citations() -> str:
            citation_lines = ["\n\n---\n**📚 참고 법령 및 공식 출처**"]
            for law in law_data.get("applicable_laws", []):
                citation_lines.append(f"- 법제처: [{law['full_name']} ({law['effective_date']} 시행)]({law['url']})")
            for i, ch in enumerate(retrieved_chunks[:3], 1):
                src = ch["metadata"].get("source", "Unknown")
                page = ch["metadata"].get("page", ch["metadata"].get("page_start", 1))
                citation_lines.append(f"- 내부 규제 DB {i}: {src} (p.{page})")
            return "\n".join(citation_lines)

        # Step 8-A: Google Gemini Cloud API Streaming
        if self.llm_provider == "gemini":
            if not self.gemini_api_key or not self.gemini_api_key.strip():
                yield {
                    "type": "error",
                    "content": (
                        "⚠️ **Google Gemini API 키가 설정되지 않았습니다.**\n\n"
                        "다수 사용자에게 빠르고 안정적인 실시간 스트리밍 답변을 제공하려면 Gemini API 키가 필요합니다.\n\n"
                        "1. 사이드바의 **'🔑 Gemini API 키'** 입력창에 API 키를 입력하거나,\n"
                        "2. 프로젝트의 `.env` 파일에 `GEMINI_API_KEY=your_key_here`를 등록해주세요.\n\n"
                        "👉 **[Google AI Studio에서 무료 API 키 즉시 발급받기](https://aistudio.google.com/app/apikey)**"
                    )
                }
                return
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=self.gemini_api_key)

                # Dynamically resolve and verify model
                resolved_model, warn = self.gemini_model_resolver.resolve_model(self.gemini_model)
                self.gemini_model = resolved_model
                if warn:
                    logger.info(f"Model resolution notice: {warn}")

                # Format conversation turns for multi-turn Gemini chat
                contents = []
                for m in chat_history[-4:]:
                    role = "model" if m["role"] == "assistant" else "user"
                    contents.append(types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=m["content"])]
                    ))
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=user_prompt_with_context)]
                ))

                response_stream = client.models.generate_content_stream(
                    model=self.gemini_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=selected_system_prompt,
                        temperature=0.2,
                        top_p=0.95
                    )
                )

                first_token = True
                stream_msg = "2단계 실시간 법령 및 실무 절차 스트리밍 중..." if has_step1 else "법률전문가·교수·변호사 관점의 심층 법률 자문 스트리밍 중..."
                for chunk_text in sanitize_llm_stream(response_stream, has_step1_direct=has_step1):
                    if first_token:
                        first_token = False
                        yield {"type": "status", "stage": "streaming", "message": stream_msg}
                    yield {"type": "token", "content": chunk_text}

                # 스트리밍 완료 후 법제처 및 문서 출처 추가 출력
                try:
                    yield {"type": "token", "content": _format_citations()}
                except Exception as cite_err:
                    logger.warning(f"Failed to format citations: {cite_err}")

                yield {"type": "status", "stage": "done", "message": "답변 완료"}

            except Exception as e:
                diag = classify_gemini_error(e)
                logger.error(f"Gemini streaming error ({diag['error_type']}): {e}")
                yield {
                    "type": "error",
                    "content": f"{diag['user_message']}\n\n*상세 오류: {str(e)}*"
                }
            return

        # Step 8-B: OpenAI Streaming
        if self.llm_provider == "openai":
            # Ensure API key is sanitized
            self.openai_api_key = sanitize_api_key(self.openai_api_key or getattr(self.openai_client, "api_key", ""))
            if self.openai_client and getattr(self.openai_client, "api_key", "") != self.openai_api_key:
                self.openai_client.set_api_key(self.openai_api_key)

            if not getattr(self.openai_client, "is_configured", False):
                yield {
                    "type": "error",
                    "content": "⚠️ **OpenAI API 키가 설정되지 않았습니다.**\n\n"
                               "OpenAI 모델을 사용하려면 사이드바에서 API 키를 입력하거나 .env 파일에 OPENAI_API_KEY를 설정해주세요."
                }
                return

            # Keep client model in sync with engine
            if hasattr(self.openai_client, "model") and self.openai_client.model != self.openai_model:
                self.openai_client.model = self.openai_model

            # Proactive handling for low-context models (e.g. gpt-3.5-turbo with 16k limit)
            is_low_context = "3.5" in str(self.openai_model)
            history_slice = -1 if is_low_context else -4
            messages = []
            for m in chat_history[history_slice:]:
                messages.append({"role": m["role"], "content": m["content"]})
            messages.append({"role": "user", "content": user_prompt_with_context})

            try:
                stream = self.openai_client.chat_stream(
                    messages=messages,
                    system=selected_system_prompt
                )
                first_token = True
                stream_msg = "2단계 실시간 법령 및 실무 절차 스트리밍 중..." if has_step1 else "법률전문가·교수·변호사 관점의 심층 법률 자문 스트리밍 중..."
                for chunk_text in sanitize_llm_stream(stream, has_step1_direct=has_step1):
                    if first_token:
                        first_token = False
                        yield {"type": "status", "stage": "streaming", "message": stream_msg}
                    yield {"type": "token", "content": chunk_text}
                try:
                    yield {"type": "token", "content": _format_citations()}
                except Exception as cite_err:
                    logger.warning(f"Failed to format citations: {cite_err}")
                yield {"type": "status", "stage": "done", "message": "답변 완료"}
            except Exception as e:
                logger.error(f"Error during OpenAI streaming: {e}")
                err_str = str(e)
                if isinstance(e, UnicodeEncodeError) or "codec can't encode" in err_str:
                    user_msg = (
                        "⚠️ **입력된 OpenAI API 키에 잘못된 문자(한글, 특수 유니코드, 보이지 않는 공백 등)가 포함되어 있습니다.**\n\n"
                        "OpenAI API 키는 영문 알파벳, 숫자, 하이픈(-)으로만 구성되어야 합니다.\n\n"
                        "👉 **해결 방법**:\n"
                        "1. 좌측 사이드바의 **'OpenAI API 키 입력'**란을 모두 지워주세요.\n"
                        "2. [OpenAI 플랫폼](https://platform.openai.com/api-keys)에서 발급받은 `sk-...` 키를 직접 복사하여 붙여넣어 주세요.\n"
                        "3. 키 앞뒤나 중간에 한글 또는 따옴표가 들어가지 않도록 확인해 주세요."
                    )
                elif is_context_length_exceeded(e):
                    user_msg = (
                        "⚠️ **OpenAI 모델의 최대 컨텍스트(토큰) 한도를 초과했습니다.**\n\n"
                        f"현재 선택된 `{self.openai_model}` 모델은 처리 가능한 토큰 한도(16,385 토큰)가 작아 방대한 법령·지침 데이터를 모두 담지 못했습니다.\n\n"
                        "👉 **해결 방법**: 좌측 사이드바의 **'OpenAI 모델 선택'**에서 128k 토큰을 지원하는 **`gpt-4o-mini`** (추천) 또는 **`gpt-4o`** 모델을 선택해 주세요."
                    )
                else:
                    user_msg = (
                        "⚠️ **OpenAI API 연결에 실패했습니다.**\n\n"
                        "1) API 키가 올바르게 설정되었는지 확인하세요.\n"
                        "2) 네트워크가 정상인지 확인하고 방화벽 등에서 차단되지 않았는지 확인하십시오.\n"
                        "3) `.env` 파일에 `OPENAI_API_KEY=your_key` 를 추가하거나, 사이드바에 키를 입력하십시오."
                    )
                yield {"type": "error", "content": f"{user_msg}\n\n*상세 오류: {err_str}*"}
                return

        # Step 8-C: Free Local Open-Source LLM Streaming (CPU Inference, Zero Cost)
        if self.llm_provider in ("free", "local"):
            from src.rag.local_llm import get_local_llm_provider
            local_provider = get_local_llm_provider()

            if not local_provider.is_ready:
                yield {
                    "type": "status",
                    "stage": "loading_model",
                    "message": f"로컬 오픈소스 AI 모델({self.local_llm_model}) 준비 및 로딩 중... (최초 1회 자동 다운로드)"
                }

            # High-efficiency prompt tailoring for Local CPU inference
            if has_step1:
                local_system_prompt = (
                    "당신은 대한민국 생물안전 및 법률 규제 전문가입니다. "
                    "1단계 규제 현황(생물체위험군 및 적용 법령)은 이미 확인되어 화면에 표시되었으므로, "
                    "2단계로서 해당 법률에 따른 실무 신고 절차(소관 부처, 신청 포털, 단계별 절차, 구비서류, 핵심 주의사항)를 "
                    "가독성 높은 마크다운 목록 형식으로 상세하고 명확하게 작성해주세요."
                )
                concise_primary = "\n---\n".join([c.get("text", "") for c in retrieved_chunks[:2]])
                concise_law = law_context_str[:1200] if law_context_str else ""
                local_user_content = (
                    f"[질문 대상 병원체/독소]\n{query}\n\n"
                    f"[1단계 확정 적용 법률]\n{app_laws_str}\n\n"
                    f"[관련 규제 지침 발췌]\n{concise_primary}\n\n"
                    f"[핵심 법조문 요약]\n{concise_law}\n\n"
                    f"[지침]\n위 적용 법률에 대한 2단계 실무 가이드 절차(신고 기관 및 포털, 단계별 실무 절차, 구비서류, 연구자 핵심 주의사항)를 작성하세요."
                )
            else:
                local_system_prompt = "당신은 대한민국 생물안전 전문 법률 자문가입니다. 질문에 대해 관련 법령에 기반하여 명확하고 신뢰성 있는 법률 자문을 제공하세요."
                concise_primary = "\n---\n".join([c.get("text", "") for c in retrieved_chunks[:2]])
                concise_law = law_context_str[:1500] if law_context_str else ""
                local_user_content = (
                    f"[사용자 질의]\n{query}\n\n"
                    f"[관련 규제 데이터]\n{concise_primary}\n\n"
                    f"[관련 법조문 발췌]\n{concise_law}\n\n"
                    f"[지침]\n위 데이터를 근거로 질문에 대해 상세히 답변하세요."
                )

            messages = [{"role": "system", "content": local_system_prompt}]
            for m in chat_history[-2:]:
                messages.append({"role": m["role"], "content": m["content"]})
            messages.append({"role": "user", "content": local_user_content})

            try:
                raw_stream = local_provider.stream_generate(messages=messages, max_new_tokens=600)
                first_token = True
                stream_msg = "2단계 실시간 법령 및 실무 절차 스트리밍 중 (로컬 무료 AI)..." if has_step1 else "법률전문가·교수·변호사 관점의 심층 법률 자문 스트리밍 중 (로컬 무료 AI)..."
                for chunk_text in sanitize_llm_stream(raw_stream, has_step1_direct=has_step1):
                    if first_token:
                        first_token = False
                        yield {"type": "status", "stage": "streaming", "message": stream_msg}
                    yield {"type": "token", "content": chunk_text}
                try:
                    yield {"type": "token", "content": _format_citations()}
                except Exception as cite_err:
                    logger.warning(f"Failed to format citations: {cite_err}")
                yield {"type": "status", "stage": "done", "message": "답변 완료"}
            except Exception as e:
                logger.error(f"Error during local LLM streaming: {e}", exc_info=True)
                yield {
                    "type": "error",
                    "content": f"⚠️ **로컬 오픈소스 AI 모델 추론 중 오류가 발생했습니다.**\n\n상세 오류: {str(e)}"
                }
                return

