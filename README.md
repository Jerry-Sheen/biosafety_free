# 🧬 Biosafety Free: 병원체 규제 및 생물안전 법령 안내 오픈형 RAG 시스템 (완전 무료 에디션)

> **국내 병원체 및 생물작용제 관련 사내·연구 규정 문서**(PDF, HWP/HWPX, Word, Excel, CSV, JSON 등)와 **법제처 국가법령정보센터(law.go.kr)의 최신 법령 데이터**를 결합하여, 특정 병원균에 적용되는 국내외 법령 규제 및 **8대 법정 의무 신고·허가 절차**를 체계적으로 안내하는 AI 기반 듀얼 검색 RAG 시스템입니다.
> 
> **Biosafety Free**는 외부 제3자 사용자가 **유료 API 키 없이도 100% 무료로 즉시 이용**할 수 있도록 **ChromaDB + 로컬 오픈소스 LLM(CPU 최적화) + FastAPI 백엔드** 아키텍처를 도입하였으며, 고성능 상용 API(OpenAI/Gemini)를 원하는 사용자를 위한 **하이브리드 선택권(BYOK)**을 함께 제공합니다.

---

## 🌟 핵심 특징

### 1. 외부 사용자 100% 무료 질의응답 (비용 0원)
- **로컬 오픈소스 LLM 서빙**: 가벼운 CPU 환경(Intel i7, 16GB RAM)에서도 쾌적하게 동작하는 `Qwen2.5-1.5B-Instruct` (또는 3B) 모델을 FastAPI 백엔드에 1회 상주(Singleton)시켜 메모리 폭증 없이 다중 사용자 무료 답변을 실시간 스트리밍합니다.
- **고정 로컬 임베딩**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` 모델을 관리자(`admin.py`)와 사용자(`user.py`/`FastAPI`) 간 100% 동일하게 공유하여 검색 벡터 차원과 코사인 유사도 정합성을 완벽하게 보장합니다.

### 2. 하이브리드(무료/유료) 모드 선택권
- **[기본값] 🆓 완전 무료 모드**: API 키 입력 없이 즉시 질의응답 가능.
- **[선택] 💳 유료 고성능 모드**: 개인 OpenAI(`gpt-4o`, `gpt-4o-mini`) 또는 Google Gemini API 키를 직접 입력하여 초고속·초고성능 답변 선택 가능.

### 3. 병원체 규제 분류 및 법제처 듀얼 검색 (Dual Retrieval)
- **1단계: 영구 벡터 DB(ChromaDB) 기반 규제 분류 제시**
  - 감염병예방법(고위험/생물테러), 유전자변형생물체법(국가관리), 화학생물무기금지법(생물작용제), 가축전염병예방법(제1~3종), 수산생물질병관리법, 위험군 분류를 직관적으로 표기.
- **2단계: 국가법령정보센터(law.go.kr) 최신 법령 실시간 검색 및 8대 의무 항목 매핑**
  - 분리신고, 이동신고, 보유·폐기신고, 제조·실험승인, 수입신고, 인수신고, 수출신고, 인도신고 등 8대 법정 의무 자동 매핑.
- **3단계: 실무 가이드 및 온라인 전자민원 포털 안내**
  - 질병관리청, 산업통상자원부, 검역본부 등 소관 부처별 포털 URL 및 실무 절차(Step 1~Step 3) 안내.

### 4. 포트 충돌 방지 전용 포트 체계 (8600번대)
- 기존 서비스나 Streamlit 기본 포트와의 중복 충돌을 원천 차단하기 위해 **8600번대 전용 포트**를 배정하였습니다:
  - **FastAPI 백엔드 (`backend/main.py`)**: `http://localhost:8600`
  - **관리자 포털 (`admin.py`)**: `http://localhost:8601`
  - **사용자 포털 (`user.py`)**: `http://localhost:8602`

---

## 📂 프로젝트 구조

```
biosafety_free/
├── backend/
│   └── main.py               # 🧠 [Backend] FastAPI 비동기 RAG & 로컬 LLM 서빙 코어 (Port: 8600)
├── admin.py                  # 🛠️ [Admin] 관리자 문서 업로드/파싱/색인/DB 관리 독립 앱 (Port: 8601)
├── user.py                   # 💬 [User] 사용자 대화형 무료/유료 듀얼 모드 포털 (Port: 8602)
├── config.py                 # ⚙️ 중앙 애플리케이션 설정 (전용 포트, 모델, 경로)
├── .env.example              # 환경 변수 설정 템플릿
├── docker-compose.yml        # backend(8600) + admin(8601) + user(8602) 원클릭 오케스트레이션
├── Dockerfile                # 운영 환경용 멀티 서비스 도커 파일
├── requirements.txt          # 파이썬 의존성 패키지 목록
├── parsers/                  # 모듈형 문서 파서 패키지 (OCR, PDF, HWP, Markdown 등)
├── pipeline/                 # 문서 수집, 중복 제거, 품질 평가 및 청킹 파이프라인
├── vectorstore/              # 벡터 스토어 통합 인터페이스
├── src/
│   ├── config.py             # 설정 호환성 브릿지
│   ├── security/             # 보안 키 마스킹 및 로거
│   ├── parsers/              # IBM Docling, RapidOCR, HWP 구조화 파서
│   └── rag/
│       ├── embeddings.py     # SentenceTransformers 384차원 고정 임베딩 모듈
│       ├── local_llm.py      # CPU 최적화 오픈소스 LLM 스트리밍 서빙 엔진
│       ├── engine.py         # 하이브리드 듀얼 검색 RAG 스트리밍 엔진
│       ├── law_search.py     # 법제처 Open DRF API 연동 모듈
│       ├── vector_store.py   # ChromaDB 영구 저장소
│       ├── gemini_client.py  # Google Gemini 클라우드 API 클라이언트
│       └── openai_client.py  # OpenAI API 클라이언트
└── data/
    ├── chroma_db/            # 영구 벡터 데이터베이스 저장소
    └── uploads/              # 관리자 업로드 규정 문서 저장소
```

---

## 🚀 빠른 시작 가이드 (Quick Start)

### 방법 1. Docker Compose 원클릭 실행 (강력 추천)

Docker와 Docker Compose가 설치되어 있다면 아래 명령어로 모든 서비스가 일괄 구동됩니다:

```bash
# 1. 저장소 클론 및 환경 파일 생성
git clone https://github.com/Jerry-Sheen/biosafety_free.git
cd biosafety_free
cp .env.example .env

# 2. 전체 서비스 일괄 실행
docker compose up -d --build
```

- **관리자 포털 접속**: `http://localhost:8601` (기본 PIN: `admin1234`)
- **사용자 포털 접속**: `http://localhost:8602`
- **백엔드 API 문서**: `http://localhost:8600/docs`

---

### 방법 2. 로컬 파이썬 가상환경 직접 실행

```bash
# 1. 저장소 클론 및 가상환경 생성
git clone https://github.com/Jerry-Sheen/biosafety_free.git
cd biosafety_free
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. 의존성 패키지 설치
pip install -r requirements.txt

# 3. 환경 파일 복사
cp .env.example .env

# 4. [터미널 1] FastAPI 백엔드 실행 (Port: 8600)
python backend/main.py

# 5. [터미널 2] 관리자 포털 실행 (Port: 8601)
streamlit run admin.py --server.port 8601

# 6. [터미널 3] 사용자 포털 실행 (Port: 8602)
streamlit run user.py --server.port 8602
```

---

## 💡 하드웨어 요구 사양 (CPU 기반 로컬 추론)

- **CPU**: Intel Core i5 / i7 (4코어 8스레드 이상 권장)
- **RAM**: 16.0 GB 이상 (모델 메모리 약 2.5GB 점유)
- **GPU**: 외장 GPU 불필요 (순수 CPU 추론 최적화)
- **저장 공간**: 약 5GB 여유 공간 (ChromaDB 및 모델 가중치 캐시)

---

## 🔒 보안 및 라이선스
- API 키 및 관리자 PIN 번호는 `.env` 파일과 시스템 환경 변수를 통해서만 주입되며, `.gitignore`를 통해 Git 유출을 원천 방지합니다.
- 본 프로젝트는 MIT 라이선스에 따라 자유롭게 사용 및 수정이 가능합니다.
