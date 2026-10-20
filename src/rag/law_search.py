"""
Law Search & Regulatory Information Client for Biosafety RAG.
Fetches up-to-date statutory provisions from Ministry of Government Legislation (국가법령정보센터 law.go.kr)
and provides structured regulatory mappings for the 8 mandatory reporting items and practical guides for EACH applicable law:
1. 분리신고
2. 이동(분양)신고
3. 보유·폐기신고
4. 제조신고 및 개발실험승인
5. 수입(반입)신고
6. 인수신고
7. 수출신고
8. 인도신고
"""

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional
from src.security.logger import get_secure_logger

logger = get_secure_logger("law_search")

DRF_SEARCH_URL = "https://www.law.go.kr/DRF/lawSearch.do"
DRF_SERVICE_URL = "https://www.law.go.kr/DRF/lawService.do"
DEFAULT_OC = "test"

CORE_LAWS_KNOWLEDGE = {
    "감염병예방법": {
        "full_name": "감염병의 예방 및 관리에 관한 법률",
        "short_name": "감염병예방법",
        "ministry": "질병관리청 (의료안전예방국 생물안전과)",
        "law_url": "https://www.law.go.kr/법령/감염병의예방및관리에관한법률",
        "items": {
            "분리신고": "제21조(고위험병원체의 분리, 분양ㆍ이동 및 이동신고) 제1항 - 환자 검체나 자연환경 등에서 고위험병원체를 분리한 자는 지체 없이 질병관리청장에게 신고하여야 함.",
            "이동(분양)신고": "제21조 제2항 및 시행규칙 제16조 - 고위험병원체를 국내 다른 기관 등으로 이동(분양 포함)하려는 자는 이동 전 질병관리청장에게 사전 이동신고를 하여야 함.",
            "보유·폐기신고": "보유: 제23조(고위험병원체의 안전관리 등) - 고위험병원체 취급시설 허가·등록 및 보유현황 기록·관리 의무(생물테러감염병병원체는 제23조의3에 따라 별도 보유허가 필요) / 폐기: 제23조 및 시행규칙 제19조 - 고위험병원체를 폐기(불활화 또는 멸균)하려는 경우 사전에 폐기 신고서를 작성하여 질병관리청장에게 제출해야 함.",
            "제조신고 및 개발실험승인": "제조신고: 해당없음 (고위험병원체 자체는 '제조'가 아닌 보존·배양으로 취급되며 취급시설 등급에 따라 관리) / 개발실험승인: 제23조의2(고위험병원체의 개발실험 승인) - 고위험병원체를 변형하거나 위해성이 높은 유전자재조합 등의 개발실험을 하려는 자는 질병관리청장의 사전 승인을 받아야 함.",
            "수입(반입)신고": "제22조(고위험병원체의 반입 허가 등) - 국내로 고위험병원체를 들여오려는 자는 질병관리청장의 사전 반입허가를 받아야 함(반입허가제).",
            "인수신고": "제22조 및 시행규칙 제18조 - 반입허가를 받아 세관 통관 후 국내 연구시설로 최종 인수한 경우 인수 확인 및 현황 보고를 질병관리청에 이행.",
            "수출신고": "대외무역법 및 질병관리청 생물안전 지침 연계 - 고위험병원체 국외 반출 시 질병관리청 생물안전 승인 및 전략물자 수출허가 요건 준수.",
            "인도신고": "분양·이동신고 시 인계·인수 내역을 상호 확인하여 질병보건통합관리시스템에 인도/인수 내역을 등록."
        },
        "practical_guide": {
            "system_name": "질병관리청 질병보건통합관리시스템 (고위험병원체 관리시스템)",
            "system_url": "https://is.kdca.go.kr",
            "steps": [
                "Step 1 (사전 준비 및 연구시설 허가): 병원체 위해 등급(제2위험군/제3위험군)에 맞는 생물안전 연구시설(BL2 또는 BL3) 신고·허가 완료",
                "Step 2 (신고서 작성 및 제출): 질병보건통합관리시스템 접속 ➜ 분리신고(분리 즉시 지체 없이), 이동신고(이동 전 사전신고), 반입허가신청(국외 도입 전 사전신청) 서류 온라인 제출",
                "Step 3 (사후 관리 및 대장 기록): 고위험병원체 관리대장 비치, 이동/반입 후 인수확인 등록, 불활화/멸균 후 폐기신고서 제출"
            ]
        },
        "precautions": {
            "형사처벌 및 행정처분": "제77조(벌칙): 고위험병원체 무허가 반입(제22조제1항), 생물테러감염병병원체 무허가 보유(제23조의3제1항) 시 5년 이하의 징역 또는 5천만원 이하의 벌금. 제78조: 고위험병원체 분리 미신고 또는 거짓 신고(제21조제1항), 이동 미신고(제21조제2항), 폐기 미신고(제23조제3항), 안전관리기준 위반(제23조제1항) 시 3년 이하의 징역 또는 3천만원 이하의 벌금. 제23조제2항 위반 시 허가 취소 또는 시설 폐쇄 처분.",
            "양벌규정": "제82조(양벌규정): 법인의 대표자나 대리인, 사용인, 종업원이 위반행위를 한 경우 행위자를 처벌하는 외에 그 법인 또는 개인에게도 해당 조문의 벌금형 병과.",
            "보관 및 운송 규정": "허가·신고된 생물안전 연구시설(BL2/BL3) 내 2중 잠금장치 보관고에 보관 필수. 이동 및 수송 시 질병관리청 생물안전 수송지침(3중 밀폐 포장용기, 라벨 표기, 비상연락처 동봉) 엄격 준수.",
            "위험물질 취급 제한": "비인가자의 단독 보관시설 출입 및 취급 전면 금지, 고위험병원체 관리대장 상시 비치 및 기록 후 5년간 의무 보존."
        }
    },
    "화학생물무기금지법": {
        "full_name": "화학무기ㆍ생물무기의 금지와 특정화학물질ㆍ생물작용제 등의 제조ㆍ수출입 규제 등에 관한 법률",
        "short_name": "생화학무기금지법",
        "ministry": "산업통상자원부 (위탁/접수: 한국화학융합시험연구원 KTR)",
        "law_url": "https://www.law.go.kr/법령/화학무기ㆍ생물무기의금지와특정화학물질ㆍ생물작용제등의제조ㆍ수출입규제등에관한법률",
        "items": {
            "분리신고": "해당없음 (생화학무기금지법령상 분리신고라는 별도 명칭의 규정은 없으나, 질병 진단/검사 목적 외에 분리하여 증식·보유할 경우 제조신고 또는 보유신고 대상이 됨)",
            "이동(분양)신고": "양도신고(법 제10조의2, 시행규칙 제6조) - 생물작용제등을 양도하려는 자는 양도 전 사전에 양도물질, 수량, 상대방을 산업통상자원부장관에게 신고해야 함.",
            "보유·폐기신고": "보유신고(법 제13조, 시행령 제9조의2) - 생물작용제를 보유하게 된 날부터 30일 이내에 보유량 및 보유경위를 산업통상자원부장관에게 신고해야 하며, 매년 2월 말까지 보유현황 정기보고 / 폐기신고(법 제14조, 시행령 제11조) - 생물작용제등을 폐기하려는 경우 사전에 폐기물질, 수량, 폐기방법 등을 산업통상자원부장관에게 신고해야 함.",
            "제조신고 및 개발실험승인": "제조신고: 제5조의2(생물작용제 또는 독소의 제조신고) - 생물작용제 또는 독소를 제조(증식·배양·추출 포함)하려는 자는 미리 제조 목적과 제조량 등을 산업통상자원부장관에게 신고해야 함 / 개발실험승인: 해당없음 (연구·개발 활동 중 제조에 해당하는 경우 제조신고 의무 적용, 별도 개발실험승인은 감염병예방법/LMO법 소관)",
            "수입(반입)신고": "수입허가(법 제12조, 시행규칙 제8조) - 생물작용제등을 수입하려는 자는 산업통상자원부장관의 사전 허가를 받아야 함(국내 수입대행사를 통해 구매하더라도 사용자 본인의 수입허가 필요).",
            "인수신고": "시행령 제13조의2(인수ㆍ인도 신고) - 생물작용제등의 수입허가를 받은 자가 세관 통관 후 물품을 최종 인수하려는 경우 관할 기관에 인수신고서를 제출해야 함(인수일은 통관예정일).",
            "수출신고": "수출허가(법 제11조, 시행규칙 제7조) - 생물작용제등을 수출하려는 자는 산업통상자원부장관의 사전 수출허가를 받아야 함(대외무역법 전략물자 수출허가 연계).",
            "인도신고": "시행령 제13조의2(인수ㆍ인도 신고) - 수입 또는 양도한 자가 생물작용제등을 상대방에게 실제로 넘겨줄 때 인도신고를 이행해야 함."
        },
        "practical_guide": {
            "system_name": "산업통상자원부 생물작용제 및 독소 관리시스템 / 한국화학융합시험연구원(KTR)",
            "system_url": "https://www.ktr.or.kr",
            "steps": [
                "Step 1 (사전 신고·허가): 제조 활동 시작 전 제조신고서 제출, 해외 도입 전 수입허가신청서(물품매도확약서, 사용계획서 등 첨부) 제출",
                "Step 2 (통관 및 인수·인도): 세관 통관 후 물품 수령 시 인수신고서 제출, 국내 타 기관 양도 시 사전 양도신고 및 인도신고서 제출",
                "Step 3 (보유 관리 및 대장 기록): 보유일로부터 30일 이내 보유신고 완료, 취급관리대장 상시 비치 및 매년 2월 말 보유현황 정기보고",
                "Step 4 (폐기): 불활화/소각 폐기 전 사전에 폐기신고서 제출 및 승인 후 폐기"
            ]
        },
        "precautions": {
            "형사처벌 및 행정처분": "제25조(벌칙): ① 제4조의2제1항을 위반하여 화학무기ㆍ생물무기를 개발ㆍ제조ㆍ획득ㆍ보유ㆍ비축ㆍ이전ㆍ운송 또는 사용하거나 이를 지원 또는 권유한 자는 무기(無期) 또는 5년 이상의 징역이나 1억원 이하의 벌금에 처함. ② 화학무기ㆍ생물무기를 사용하여 사람의 생명ㆍ신체 또는 재산에 해를 끼치거나 그 밖에 공공의 안녕을 문란하게 한 자는 사형, 무기 또는 7년 이상의 징역에 처함. ③ 미수범 처벌. 제26조(벌칙): 무기 개발·제조 등의 목적으로 생물작용제등을 제조·보유·사용한 자(제4조의2제2항)는 7년 이하의 징역 또는 7천만원 이하의 벌금(미수범 처벌). 제27조(벌칙): 생물작용제등 제조 미신고(제5조의2제1항), 폐기명령 위반(제10조제1항), 무허가 수출입(제11조제1항, 제12조제1항) 위반자는 5년 이하의 징역 또는 5천만원 이하의 벌금. 제28조(벌칙): 보유 미신고(제13조), 인수·인도 미신고(제13조의2) 시 2년 이하의 징역 또는 2천만원 이하의 벌금.",
            "양벌규정": "제29조(양벌규정): 법인의 대표자나 대리인, 종업원이 업무에 관하여 제25조제1항(미수범 포함), 제26조~제28조 위반행위를 하면 행위자를 벌하는 외에 그 법인 또는 개인에게도 해당 조문의 벌금형 병과.",
            "보관 및 운송 규정": "생물보안(Biosecurity) 구역 설정, 2인 이상 입회하 출입, 24시간 CCTV 영상 감시 및 6개월간 녹화본 보존, 2중 잠금장치. 운송 시 사전 운송계획서 승인 및 보안 전용 운반 절차 준수.",
            "위험물질 취급 제한": "보안관리책임자 의무 지정, 결격사유자(전과자 등)의 생물작용제 취급 전면 배제, 도난·분실 방지 대책 의무화."
        }
    },
    "유전자변형생물체법": {
        "full_name": "유전자변형생물체의 국가간 이동 등에 관한 법률",
        "short_name": "유전자변형생물체법(LMO법)",
        "ministry": "과학기술정보통신부 / 질병관리청 / 산업통상자원부",
        "law_url": "https://www.law.go.kr/법령/유전자변형생물체의국가간이동등에관한법률",
        "items": {
            "분리신고": "해당없음 (LMO법은 유전자변형생물체에 적용되므로 자연계 분리 병원체는 직접 대상이 아님)",
            "이동(분양)신고": "제20조 및 통합고시 - 시험·연구용 LMO 이동 시 밀폐 운송용기 포장 및 생물안전 취급표시 준수",
            "보유·폐기신고": "보유: 제22조의4 - 시험·연구용 LMO 연구시설 내 취급관리대장 기록 및 5년간 보관 / 폐기: 연구시설 내 고압증기멸균 등 완전 불활화 후 폐기",
            "제조신고 및 개발실험승인": "제조신고: 해당없음 / 개발실험승인: 제22조(개발ㆍ실험의 승인) - 국가관리대상병원체 또는 고위험 병원체를 이용한 LMO 개발·실험 시 관계 중앙행정기관의 장(질병관리청 등)의 사전 승인 필요.",
            "수입(반입)신고": "제8조 및 제9조 - 시험·연구용 LMO 수입 시 관계 중앙행정기관의 장에게 수입승인을 받거나 수입신고 이행.",
            "인수신고": "해당없음 (수입승인/신고 후 통관 확인으로 갈음)",
            "수출신고": "제20조 - 국외 수출 시 수출통보서 제출",
            "인도신고": "해당없음 (연구시설 간 양도 관리대장 기록으로 갈음)"
        },
        "practical_guide": {
            "system_name": "LMO 국가연구안전정보시스템 (국가연구안전관리본부)",
            "system_url": "https://www.lmosafety.or.kr",
            "steps": [
                "Step 1 (시설 신고): LMO 연구시설(1~4등급) 설치·운영 신고 및 허가",
                "Step 2 (개발·실험 또는 수입 승인): 고위험/국가관리대상병원체 유전자변형 시 개발실험승인신청서 제출 ➜ 심의 후 승인서 발급",
                "Step 3 (사후 관리): LMO 보관관리대장 기록 및 정기 안전점검 이행"
            ]
        },
        "precautions": {
            "형사처벌 및 행정처분": "제39조(벌칙): 승인 없이 유전자변형생물체를 수입하거나 개발·실험한 자(제8조, 제22조)는 5년 이하의 징역 또는 7천만원 이하의 벌금. 제40조: 허가 없이 연구시설을 설치·운영한 자(제22조)는 3년 이하의 징역 또는 5천만원 이하의 벌금. 제41조: 수입승인 위반, 폐기명령 위반 시 2년 이하의 징역 또는 3천만원 이하의 벌금. 제42조: 연구시설 미신고 운영 시 1년 이하의 징역 또는 2천만원 이하의 벌금.",
            "양벌규정": "제43조(양벌규정): 법인의 대표자, 대리인, 종업원이 위반행위를 한 경우 행위자를 처벌하는 외에 그 법인 또는 개인에게도 해당 조의 벌금형 병과.",
            "보관 및 운송 규정": "LMO 보관설비에 생물안전 취급표시(픽토그램) 부착, 보관·관리대장 5년간 보관. 이동 시 견고한 밀폐 2중 용기 포장 및 취급주의 라벨 부착.",
            "위험물질 취급 제한": "기관생물안전위원회(IBC) 사전 심의 필수, 생물안전관리책임자 미임명 또는 안전교육 미이수 시 취급 제한."
        }
    },
    "가축전염병예방법": {
        "full_name": "가축전염병 예방법",
        "short_name": "가축전염병예방법",
        "ministry": "농림축산식품부 (농림축산검역본부)",
        "law_url": "https://www.law.go.kr/법령/가축전염병예방법",
        "items": {
            "분리신고": "제13조(가축전염병 병원체의 분리신고) - 가축전염병 병원체(특별관리병원체 등)를 분리한 자는 농림축산검역본부장에게 지체 없이 신고하여야 함.",
            "이동(분양)신고": "제14조의2(병원체의 이동신고 등) - 관리병원체를 이동하려는 자는 농림축산검역본부장에게 사전 이동신고를 하여야 함.",
            "보유·폐기신고": "보유: 관리병원체 보유현황 관리대장 기록·비치 / 폐기: 불활화 검증 후 사후 폐기보고",
            "제조신고 및 개발실험승인": "해당없음 (수의생명자원 보존·관리 지침 준수)",
            "수입(반입)신고": "제31조 및 제32조 - 지정검역물 및 동물병원체의 수입허가 및 검역신청.",
            "인수신고": "해당없음 (검역통관 확인으로 갈음)",
            "수출신고": "수출검역 신청 및 전략물자 승인",
            "인도신고": "해당없음 (이동신고 인계 확인으로 갈음)"
        },
        "practical_guide": {
            "system_name": "농림축산검역본부 동물검역 통합시스템",
            "system_url": "https://www.qia.go.kr",
            "steps": [
                "Step 1 (분리신고): 특별관리병원체 분리 즉시 검역본부에 분리신고서 제출",
                "Step 2 (이동신고): 타 기관 이동 전 사전 이동신고서 제출",
                "Step 3 (수입/검역): 국외 도입 시 수입허가 및 지정검역 절차 이행"
            ]
        },
        "precautions": {
            "형사처벌 및 행정처분": "제56조(벌칙): 허가 없이 수입금지물건(가축전염병 병원체)을 수입한 자(제32조)는 3년 이하의 징역 또는 3천만원 이하의 벌금. 제57조: 특별관리병원체 분리 미신고(제13조), 이동 미신고(제14조의2) 시 1년 이하의 징역 또는 1천만원 이하의 벌금. 시설 폐쇄명령 또는 사용정지 처분.",
            "양벌규정": "제61조(양벌규정): 법인의 대표자나 대리인, 종업원이 위반행위를 하면 행위자를 처벌하는 외에 그 법인 또는 개인에게도 해당 조의 벌금형 병과.",
            "보관 및 운송 규정": "수의생명자원 보존·관리 규정에 적합한 차폐시설 내 보관, 이동 시 사전 이동신고필증 수령 및 검역 규격 전용 용기 포장 운송.",
            "위험물질 취급 제한": "미신고 병원체의 외부 유출·무단 양도 전면 금지, 방역 관리자의 입회 및 취급대장 상시 기록."
        }
    },
    "수산생물질병관리법": {
        "full_name": "수산생물질병 관리법",
        "short_name": "수산생물질병관리법",
        "ministry": "해양수산부 (국립수산물품질관리원 수산물방역과/검역과)",
        "law_url": "https://www.law.go.kr/법령/수산생물질병관리법",
        "items": {
            "분리신고": "제13조(수산생물전염병 발생 등의 신고) 및 『수산생물질병의 병원체 관리 규정』(국립수산물품질관리원 고시) 제4조(병원체 분리신고 등) - 병성감정실시기관, 연구기관 등에서 수산생물전염병 병원체를 분리·확인한 경우 국가수산생물방역통합정보시스템(AQUABRE) 또는 서면을 통해 국립수산물품질관리원장 등 방역기관의 장에게 지체 없이 분리신고하여야 함.",
            "이동(분양)신고": "제15조(격리 및 이동제한 등) 및 『수산생물질병의 병원체 관리 규정』 - 전염병 확산 방지를 위한 감염 생물 및 물건의 격리·이동제한·출하제한 명령 준수. 연구기관 간 병원체를 분양·이동하려는 경우 병원체 분양신청서 및 안전관리계획서를 제출하여 방역기관의 사전 승인 필요.",
            "보유·폐기신고": "보유: 수산생물전염병 병원체 보존 현황 및 관리대장 상시 비치·기록 관리 의무 / 폐기: 제16조(살처분명령) 및 제17조(사체의 처분제한 등) - 감염된 수산생물 및 사체는 임의 이동이 금지되며 소각·매몰 등 폐기 처분 의무. 실험실 보존 병원체 폐기 시 고압증기멸균 등 완전 불활화 조치 후 폐기 기록 관리.",
            "제조신고 및 개발실험승인": "제조신고: 제24조 제1항 단서 및 동물용 의약품 등 취급규칙(해양수산부령) - 수산생물질병의 진료·예방을 위한 의약품(백신 등) 제조 시 해양수산부장관의 허가 및 수산용 동물용의약품 제조업·품목허가 필수 / 개발실험승인: 시험·연구조사 계획서 및 안전관리시설 적합성 확인.",
            "수입(반입)신고": "수입금지 및 수입허가(제24조 제1항 제2호, 제23조, 제27조) - 수산생물전염병의 병원체에 감염된 수산생물은 원칙적으로 수입이 전면 금지됨. 단, 시험·연구조사 또는 의약품 제조 목적에 한하여 해양수산부장관(국립수산물품질관리원)의 사전 '수입금지물건 수입허가'를 받아야 하며, 입항 시 즉시 지정검역물 수입검역 신청 필수.",
            "인수신고": "제27조(수입검역) 및 검역통관 절차 - 수입허가 물품의 통관 검역 합격 후 연구시설 인수 확인 및 국가 병원체 자원 인수증 작성.",
            "수출신고": "제31조(수출검역) - 수산생물 및 관련 물건을 국외로 수출하려는 경우 상대국 요구에 따라 국립수산물품질관리원에 수출검역을 신청하여 수산생물전염병 병원체 무감염 증명(검역증명서)을 발급받아야 함.",
            "인도신고": "병원체 분양 및 이동 시 상호 인수·인도 확인서 작성 및 방역통합정보시스템 관리."
        },
        "practical_guide": {
            "system_name": "국가수산생물방역통합정보시스템 (AQUABRE) / 국립수산물품질관리원",
            "system_url": "https://www.nfqs.go.kr",
            "steps": [
                "Step 1 (사전 수입허가 및 연구계획 승인): 병원체 또는 감염물 국외 도입 전 국립수산물품질관리원에 수입금지물건 수입허가신청서(시험연구계획서, 시설·장비 현황, 안전관리계획서 첨부) 제출 및 허가 취득",
                "Step 2 (검역 신청 및 분리신고): 입항 시 지정검역물 수입검역 신청 및 검역증명서 발급. 국내 시료에서 병원체 분리 시 지체 없이 방역기관에 분리신고서 제출",
                "Step 3 (사후 관리 및 이동·폐기): 취급·보존 대장 기록 및 3년간 보존. 타 기관 분양 시 사전 승인, 연구 완료 후 고압멸균 폐기"
            ]
        },
        "precautions": {
            "형사처벌 및 행정처분": "제53조(벌칙): 허가 없이 수입금지물건(전염병 병원체 감염 생물 등)을 수입한 자는 3년 이하의 징역 또는 3천만원 이하의 벌금. 제54조: 살처분 명령(제16조) 또는 사체 처분제한(제17조) 위반 시 1년 이하의 징역 또는 1천만원 이하의 벌금. 제57조: 분리신고 미이행 또는 거짓 신고 시 500만원 이하의 과태료. 시설 폐쇄 또는 업무정지 처분.",
            "양벌규정": "제55조(양벌규정): 법인의 대표자나 대리인, 종업원이 업무에 관하여 위반행위를 하면 행위자를 처벌하는 외에 그 법인 또는 개인에게도 해당 조의 벌금형 병과.",
            "보관 및 운송 규정": "병원체 보존시설 차폐 및 2중 잠금 관리, 외부 유출 방지 설비 구비. 이동 시 밀폐 누출 방지 전용 용기 포장 및 방역관 확인.",
            "위험물질 취급 제한": "감염 의심 수산생물의 임의 방류·폐기·유통 전면 금지, 방역 관리책임자 지정 및 취급 일지 상시 기록."
        }
    }
}


DEFAULT_CORE_ARTICLES = {
    "화학생물무기금지법": ["4의2", "5의2", "10의2", "11", "12", "13", "13의2", "14", "25", "26", "27", "28", "29"],
    "감염병예방법": ["21", "22", "23", "23의2", "23의3", "77", "78", "79", "82"],
    "유전자변형생물체법": ["8", "9", "20", "22", "39", "40", "41", "42", "43"],
    "가축전염병예방법": ["13", "14의2", "31", "32", "56", "57", "61"],
    "수산생물질병관리법": ["13", "15", "16", "17", "24", "27", "31", "53", "54", "55", "57"],
}


class LawSearchClient:
    """Client for querying law.go.kr and generating grounded statutory contexts."""

    _law_articles_cache: Dict[str, List[Dict[str, Any]]] = {}

    def __init__(self, oc: str = DEFAULT_OC):
        self.oc = oc

    def fetch_latest_law_metadata(self, law_query: str) -> Optional[Dict[str, Any]]:
        """Queries law.go.kr DRF API for current law metadata (MST, enactment date, url)."""
        try:
            url = f"{DRF_SEARCH_URL}?OC={self.oc}&target=law&type=XML&query={urllib.parse.quote(law_query)}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                xml_data = resp.read().decode("utf-8", errors="ignore")
                root = ET.fromstring(xml_data)
                law_node = root.find("law")
                if law_node is not None:
                    law_name = law_node.findtext("법령명한글") or law_query
                    law_mst = law_node.findtext("법령일련번호") or ""
                    ef_yd = law_node.findtext("시행일자") or ""
                    law_id = law_node.findtext("법령ID") or ""
                    formatted_date = f"{ef_yd[:4]}-{ef_yd[4:6]}-{ef_yd[6:]}" if len(ef_yd) == 8 else ef_yd
                    web_link = f"https://www.law.go.kr/법령/{urllib.parse.quote(law_name)}"
                    return {
                        "name": law_name,
                        "mst": law_mst,
                        "effective_date": formatted_date,
                        "law_id": law_id,
                        "link": web_link,
                    }
        except Exception as e:
            logger.warning(f"law.go.kr real-time lookup failed for '{law_query}': {e}")
        return None

    def fetch_law_articles(self, mst: str) -> List[Dict[str, Any]]:
        """
        Retrieves full statutory articles from law.go.kr DRF lawService.do by MST,
        parsing each article number, title, and verbatim clauses (항, 호).
        Results are cached in memory for high-performance zero-latency subsequent queries.
        """
        if not mst:
            return []
        if mst in self._law_articles_cache:
            return self._law_articles_cache[mst]

        try:
            url = f"{DRF_SERVICE_URL}?OC={self.oc}&target=law&MST={mst}&type=XML"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=6) as resp:
                xml_data = resp.read().decode("utf-8", errors="ignore")
                root = ET.fromstring(xml_data)
                articles = []
                for art in root.iter("조문단위"):
                    no = (art.findtext("조문번호") or "").strip()
                    title = (art.findtext("조문제목") or "").strip()
                    content = (art.findtext("조문내용") or "").strip()

                    lines = []
                    if content:
                        lines.append(content)
                    for hang in art.findall("항"):
                        h_c = (hang.findtext("항내용") or "").strip()
                        if h_c:
                            lines.append("  " + h_c)
                        for ho in hang.findall("호"):
                            ho_c = (ho.findtext("호내용") or "").strip()
                            if ho_c:
                                lines.append("    " + ho_c)
                    full_text = "\n".join(lines).strip()
                    if no and full_text:
                        articles.append({
                            "no": no,
                            "title": title,
                            "text": full_text
                        })
                if articles:
                    self._law_articles_cache[mst] = articles
                    return articles
        except Exception as e:
            logger.warning(f"Failed to fetch full statutory articles for MST {mst}: {e}")
        return []

    def get_relevant_articles(
        self,
        mst: str,
        query: str = "",
        law_key: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Extracts verbatim law articles matching specific article numbers
        (e.g. '제25조', '25조', '제4조의2') or keywords (벌칙, 허가, 신고 등) from query or defaults.
        """
        articles = self.fetch_law_articles(mst)
        if not articles:
            return []

        import re
        # 1) Detect explicit article numbers mentioned in query (e.g. '제25조', '25조', '제4조의2')
        explicit_nos = set(re.findall(r'제?\s*(\d+(?:의\d+)?)\s*조', query))

        matched = []
        seen_nos = set()

        # Prioritize explicitly queried articles (e.g., if user asks about '제25조', ensure it is returned first)
        for a in articles:
            if a["no"] in explicit_nos:
                matched.append(a)
                seen_nos.add(a["no"])

        # Check default core article numbers for this law
        default_nos = DEFAULT_CORE_ARTICLES.get(law_key, [])
        for a in articles:
            if a["no"] in default_nos and a["no"] not in seen_nos:
                matched.append(a)
                seen_nos.add(a["no"])

        # If user asks about '벌칙', '처벌', '과태료', include matching penalty articles
        q_lower = query.lower()
        if any(w in q_lower for w in ["벌칙", "처벌", "형벌", "징역", "벌금", "양벌", "과태료"]):
            for a in articles:
                if a["no"] not in seen_nos and ("벌칙" in a["title"] or "양벌" in a["title"] or "과태료" in a["title"] or "벌칙" in a["text"]):
                    matched.append(a)
                    seen_nos.add(a["no"])

        return matched

    def search_applicable_laws_for_pathogen(
        self,
        pathogen_name: str,
        detected_categories: Dict[str, Any],
        query: str = ""
    ) -> Dict[str, Any]:
        """
        Determines applicable laws based on pathogen classification
        and compiles grounded statutory details for the 8 mandatory reporting items and exact articles.
        """
        applicable_laws = []
        is_bioweapon = detected_categories.get("is_bioweapon", True)
        is_high_risk = detected_categories.get("is_high_risk", True)
        is_lmo = detected_categories.get("is_lmo", False)
        is_veterinary = detected_categories.get("is_veterinary", False)
        is_aquatic = detected_categories.get("is_aquatic", False)

        if is_high_risk:
            applicable_laws.append("감염병예방법")
        if is_bioweapon:
            applicable_laws.append("화학생물무기금지법")
        if is_lmo:
            applicable_laws.append("유전자변형생물체법")
        if is_veterinary:
            applicable_laws.append("가축전염병예방법")
        if is_aquatic:
            applicable_laws.append("수산생물질병관리법")

        if not applicable_laws:
            applicable_laws = ["감염병예방법", "화학생물무기금지법"]

        law_details = []
        for law_key in applicable_laws:
            knowledge = CORE_LAWS_KNOWLEDGE.get(law_key, {})
            full_name = knowledge.get("full_name", law_key)
            realtime_meta = self.fetch_latest_law_metadata(full_name)

            effective_date = realtime_meta.get("effective_date", "최신 시행") if realtime_meta else "최신 시행"
            official_url = realtime_meta.get("link", knowledge.get("law_url")) if realtime_meta else knowledge.get("law_url")
            mst = realtime_meta.get("mst", "") if realtime_meta else ""

            verbatim_articles = self.get_relevant_articles(mst=mst, query=query, law_key=law_key) if mst else []

            law_details.append({
                "law_key": law_key,
                "full_name": full_name,
                "short_name": knowledge.get("short_name", law_key),
                "ministry": knowledge.get("ministry", "관계 부처"),
                "effective_date": effective_date,
                "url": official_url,
                "mst": mst,
                "items": knowledge.get("items", {}),
                "practical_guide": knowledge.get("practical_guide", {}),
                "precautions": knowledge.get("precautions", {}),
                "articles": verbatim_articles,
            })

        return {
            "pathogen": pathogen_name,
            "applicable_laws": law_details,
            "checklist_items": [
                "분리신고",
                "이동(분양)신고",
                "보유·폐기신고",
                "제조신고 및 개발실험승인",
                "수입(반입)신고",
                "인수신고",
                "수출신고",
                "인도신고"
            ]
        }

    def format_law_context_for_prompt(self, law_data: Dict[str, Any]) -> str:
        """Formats legal search results into structured prompt context per law including verbatim articles."""
        lines = []
        lines.append("### [법제처(국가법령정보센터) 최신 법령 검색 및 법률별 규제·절차 데이터]")
        lines.append(f"대상 병원체: {law_data['pathogen']}")
        lines.append("")

        for idx, law in enumerate(law_data.get("applicable_laws", []), 1):
            lines.append(f"=== 적용 법률 {idx}: 《{law['full_name']}》 ({law['short_name']}) ===")
            lines.append(f"- 소관 부처: {law['ministry']}")
            lines.append(f"- 최신 시행일자: {law['effective_date']}")
            lines.append(f"- 법제처 공식 링크: {law['url']}")
            lines.append("- 해당 법률에서의 8대 필수 항목별 규제 요건:")
            for item, desc in law.get("items", {}).items():
                lines.append(f"  * [{item}]: {desc}")
            lines.append("- 해당 법률에 따른 실무 가이드 절차:")
            pg = law.get("practical_guide", {})
            if pg:
                lines.append(f"  * 접수/신고 포털: {pg.get('system_name')} ({pg.get('system_url')})")
                for step in pg.get("steps", []):
                    lines.append(f"  * {step}")
            lines.append("- 해당 법률에 따른 3. 주의사항 (형사처벌·벌칙, 양벌규정, 보관·운송 규정, 위험물질 취급 제한):")
            precautions = law.get("precautions", {})
            if precautions:
                for p_key, p_desc in precautions.items():
                    lines.append(f"  * [{p_key}]: {p_desc}")
            else:
                lines.append("  * 해당사항 없음")

            # Verbatim statutory provisions from law.go.kr
            articles = law.get("articles", [])
            if articles:
                lines.append("- [법제처(국가법령정보센터) 공식 법령 조문 원문 (최신 실시간 데이터)]:")
                for art in articles:
                    lines.append(f"  {art['text']}")
            lines.append("")

        return "\n".join(lines)

    def search_laws_for_general_query(self, query: str) -> Dict[str, Any]:
        """
        Identifies relevant statutory frameworks for general legal, institutional,
        and procedural queries (non-pathogen specific) and retrieves latest metadata and verbatim articles from law.go.kr.
        """
        q_lower = query.lower()
        selected_keys = []

        # Keyword mapping for relevant biological safety and biosecurity laws
        if any(k in q_lower for k in ["감염병", "고위험병원체", "생물테러", "질병관리청", "보건", "분리신고"]):
            selected_keys.append("감염병예방법")
        if any(k in q_lower for k in ["화학무기", "생물무기", "생물작용제", "독소", "생화학", "산업부", "산자부", "ktr", "제조신고", "보유신고", "25조", "제25조", "무기"]):
            selected_keys.append("화학생물무기금지법")
        if any(k in q_lower for k in ["유전자변형", "lmo", "연구시설", "ibc", "생물안전위원회", "개발실험", "과기부", "과기정통부", "시험연구"]):
            selected_keys.append("유전자변형생물체법")
        if any(k in q_lower for k in ["가축", "동물", "수의", "검역본부", "가축전염병"]):
            selected_keys.append("가축전염병예방법")
        if any(k in q_lower for k in ["수산", "어류", "수산생물", "수산질병", "국수품"]):
            selected_keys.append("수산생물질병관리법")

        # Fallback if no specific law keyword matched: provide the major regulatory frameworks
        if not selected_keys:
            selected_keys = ["감염병예방법", "화학생물무기금지법", "유전자변형생물체법"]

        law_details = []
        for law_key in selected_keys:
            knowledge = CORE_LAWS_KNOWLEDGE.get(law_key, {})
            full_name = knowledge.get("full_name", law_key)
            realtime_meta = self.fetch_latest_law_metadata(full_name)

            effective_date = realtime_meta.get("effective_date", "최신 시행") if realtime_meta else "최신 시행"
            official_url = realtime_meta.get("link", knowledge.get("law_url")) if realtime_meta else knowledge.get("law_url")
            mst = realtime_meta.get("mst", "") if realtime_meta else ""

            verbatim_articles = self.get_relevant_articles(mst=mst, query=query, law_key=law_key) if mst else []

            law_details.append({
                "law_key": law_key,
                "full_name": full_name,
                "short_name": knowledge.get("short_name", law_key),
                "ministry": knowledge.get("ministry", "관계 부처"),
                "effective_date": effective_date,
                "url": official_url,
                "mst": mst,
                "items": knowledge.get("items", {}),
                "practical_guide": knowledge.get("practical_guide", {}),
                "precautions": knowledge.get("precautions", {})
            })

        return {
            "query": query,
            "applicable_laws": law_details
        }

    def format_general_law_context_for_prompt(self, law_data: Dict[str, Any]) -> str:
        """
        Formats legal provisions for general legal/procedural queries in an objective statutory format
        with verbatim law articles directly from law.go.kr without imposing the 8-item template.
        """
        lines = []
        lines.append("### [법제처(국가법령정보센터) 최신 법령 정보 및 공식 조문 원문 데이터]")
        lines.append(f"질의 키워드/주제: {law_data.get('query', '')}")
        lines.append("")

        for idx, law in enumerate(law_data.get("applicable_laws", []), 1):
            lines.append(f"=== 관련 법률 {idx}: 《{law['full_name']}》 ({law['short_name']}) ===")
            lines.append(f"- 소관 부처: {law['ministry']}")
            lines.append(f"- 최신 시행일자: {law['effective_date']}")
            lines.append(f"- 법제처 공식 법령 링크: {law['url']}")
            lines.append("- 주요 법정 규정 및 조문별 요건:")
            for item, desc in law.get("items", {}).items():
                lines.append(f"  * {item}: {desc}")
            lines.append("- 실무 가이드 및 접수처 정보:")
            pg = law.get("practical_guide", {})
            if pg:
                lines.append(f"  * 신청/신고 시스템: {pg.get('system_name')} ({pg.get('system_url')})")
                for step in pg.get("steps", []):
                    lines.append(f"  * {step}")
            lines.append("- 벌칙 및 법적 주의사항 (형사처벌, 행정처분, 양벌규정 등):")
            precautions = law.get("precautions", {})
            if precautions:
                for p_key, p_desc in precautions.items():
                    lines.append(f"  * [{p_key}]: {p_desc}")
            else:
                lines.append("  * 규정된 주요 벌칙 확인 필요")

            # Verbatim statutory provisions directly from law.go.kr
            articles = law.get("articles", [])
            if articles:
                lines.append("- [법제처(국가법령정보센터) 공식 법령 조문 원문 (최신 실시간 데이터)]:")
                for art in articles:
                    lines.append(f"  {art['text']}")
            lines.append("")

        return "\n".join(lines)

