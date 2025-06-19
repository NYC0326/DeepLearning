# RAG 기반 논문 분석 시스템

이 프로젝트는 **논문 PDF**를 자동으로 수집, 전처리, 벡터화하여 **질문-응답 기반 논문 리딩 시스템**을 구축하는 것이 목표입니다.  
사용자는 논문을 업로드하거나 질문을 입력하여 논문 내용을 요약하거나 해석된 정보를 받아볼 수 있습니다.


## 프로젝트 개요
### 기술 스택
- **Back-end**: FastAPI
- **Front-end**: Streamlit
- **Embedding**: Sentence-BERT 계열 모델
- **Vector Search**: FAISS
- **LLM**: Huggingface Transformers (Mistral, LLaMA 등)
- **논문 파싱**: GROBID
- **논문 수집**: arXiv API


## 전체 파이프라인

>1~4단계는 Kaggle 코드 참고해서 작성
### 1. ArXiv 논문 자동 수집
- 사용자 정의 키워드 기반으로 논문 검색
- PDF 파일로 저장
- 논문 메타데이터 (제목, 초록, ID 등)도 함께 저장

### 2. PDF → 구조화된 텍스트 변환
- **GROBID**를 사용하여 논문에서 구조화된 텍스트 추출
  - `title`, `abstract`, `body`, `references` 등

### 3. 텍스트 Chunking + Embedding
- 의미 기반 chunking 또는 고정 길이 chunking 수행
- `Sentence-BERT` 계열 모델로 임베딩 생성
- **실험**:
  - 의미 기반 vs 고정 길이 chunking
  - 임베딩 모델 비교: `multi-qa-MiniLM`, `all-MiniLM`, `instructor-xl`

### 4. FAISS 벡터 DB 구축
- 생성된 chunk 임베딩들을 FAISS index에 저장
- 각 chunk에 대한 메타데이터(JSON)도 별도 저장

### 5. 파일/폴더 구조 정리

```text
rag-agent/
├── retriever/                 # 벡터 검색 담당
│   ├── index_loader.py        # FAISS index + metadata 로드
│   └── search.py              # 질의 벡터 → top-k 검색
├── generator/                 # LLM 응답 생성
│   ├── prompt_template.py     # 프롬프트 생성
│   └── llm_runner.py          # LLM 호출 (HF/로컬)
├── evaluator/                 # 평가 도구
│   ├── precision.py
│   ├── cosine_similarity.py
│   └── log_writer.py          # 검색/응답 결과 기록
├── data/
│   ├── faiss_index.faiss      # 벡터 인덱스
│   └── metadata.json          # chunk ID → 본문, 논문 정보
└── run_query.py               # 최종 질의 처리 파이프라인
```

### 6. FAISS 검색 모듈
- 사용자 질의를 임베딩으로 변환
- FAISS index에서 top-k chunk 검색
- 결과는 chunk 메타 포함 형태로 반환

### 7. 프롬프트 생성기
- 검색된 chunk들을 하나의 context로 정리
- 사용자 질문과 함께 LLM 입력 프롬프트 생성

### 8. LLM 호출기
- 프롬프트를 기반으로 Huggingface pipeline을 통해 응답 생성
- **실험**:
  - 모델 비교: `mistral`, `llama`, 기타 lightweight LLM

### 9. 통합 질의 처리
- 질의 → 검색 → 프롬프트 생성 → 응답 생성 전체 프로세스 연결


## 실험 및 평가

### 10. 비교 실험 & 평가 모듈
- 검색 및 응답 로그 기록 (JSONL)
- 평가 지표:
  - Precision@k
  - Cosine Similarity (초록 vs chunk)
  - ROUGE (요약 질문)
  - NER Ratio
  - 주제 집중도 / 정보 밀도 등 정성 평가 지표


## 시스템 구성

### 11. FastAPI 백엔드 구성
- `/query`: 질문 입력 → 응답 반환
- `/upload`: PDF 업로드 → 구조화 및 인덱싱
- `/status`: 인덱싱 상태 등 확인

### 12. Streamlit 프론트엔드 구성
- 논문 업로드 인터페이스
- 질의 입력 → 응답 출력
- 검색된 chunk와 응답 내용 시각화
