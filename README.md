# 임대차 계약서 특약사항 검토 AI

부동산 임대차 계약서 PDF를 업로드하면, 조항을 정상/주의/위험으로 자동 분류하고
관련 법 조문을 근거로 위험 조항을 설명해주는 시스템입니다.

## 파이프라인 구조

```
PDF 업로드
  -> pdf_utils.py           : 텍스트 추출 + 조항 분리 (표준조항 "제n조" / 특약사항 번호매김 항목)
  -> contract_classifier/   : 직접 파인튜닝한 KLUE-BERT 분류기 (정상=0 / 주의=1 / 위험=2)
  -> rag/retriever.py       : 위험·주의 조항에 대해 관련 법 조문을 임베딩 유사도로 검색
  -> generation/explainer.py: 조항 + 분류결과 + 근거조문을 Gemini API에 넣어 설명 생성
```

법이 개정되면 `data/law_articles.json`만 갱신하면 되고, 분류 모델을 다시 학습할 필요는
없습니다 (RAG 구조의 핵심 이점).

## 폴더 구조

- `data/clauses.csv` : 분류기 학습 데이터 (정상/주의/위험 라벨링된 조항)
- `data/law_articles.json` : RAG용 법률 지식베이스 (주택임대차보호법·민법·형법 등)
- `contract_classifier/train.py` : 분류기 학습 스크립트
- `contract_classifier/predict.py` : 분류기 추론 함수
- `model/clause_classifier/` : 학습된 분류기 가중치
- `pdf_utils.py` : PDF 전처리 + 조항 분리
- `rag/retriever.py` : 법률 근거 검색
- `generation/explainer.py` : Gemini 기반 설명 생성 (+ 개인정보 마스킹)
- `pipeline.py` : 전체 파이프라인 오케스트레이터 (CLI 실행 가능)
- `app.py` : Streamlit 데모 앱

## 설치

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`.env.example`을 `.env`로 복사하고 Gemini API 키를 입력하세요.

```bash
cp .env.example .env
```

## 실행

분류기가 아직 학습되지 않았다면 먼저 학습:

```bash
cd contract_classifier
python train.py
```

CLI로 계약서 분석:

```bash
python pipeline.py 계약서.pdf
```

웹 데모 실행:

```bash
streamlit run app.py
```

## 주의사항

- 전자계약 등 텍스트 추출이 가능한 PDF를 가정합니다. 스캔본/이미지 PDF는 별도 OCR이 필요합니다.
- 본 결과는 참고용이며 법적 효력을 갖는 자문을 대체하지 않습니다.
- `data/law_articles.json`의 조문은 요지를 정리한 것으로, 실제 법률 자문 시에는 원문 확인이 필요합니다.
