# 임대차 계약서 특약사항 검토 AI

부동산 임대차 계약서 PDF를 업로드하면, 조항을 정상/주의/위험으로 자동 분류하고
관련 법 조문을 근거로 위험 조항을 설명해주는 시스템입니다.

## 파이프라인 구조

```
PDF 업로드
  -> pdf_utils.py           : 텍스트 추출 + 조항 분리 (표준조항 "제n조" / 특약사항 번호매김 항목)
  -> contract_classifier/   : 직접 파인튜닝한 KLUE-RoBERTa 분류기 (정상=0 / 주의=1 / 위험=2)
  -> rag/retriever.py       : 위험·주의 조항에 대해 관련 법 조문을 임베딩 유사도로 검색
  -> generation/explainer.py: 조항 + 분류결과 + 근거조문을 Gemini API에 넣어 설명 생성
```

법이 개정되면 `data/law_articles.json`만 갱신하면 되고, 분류 모델을 다시 학습할 필요는
없습니다 (RAG 구조의 핵심 이점).

### 분류기 성능

`klue/roberta-base`를 762건(정상 272 / 주의 247 / 위험 243, 중복 제거 및 라벨 균형화 완료)
데이터로 파인튜닝하여 **accuracy 0.980 / macro F1 0.980**을 달성했습니다. 자세한 데이터 증강·정제
과정과 성능 변화 추이는 `reports/project_summary_v5.pdf`(최종 리포트) 참고.

### RAG 임베딩 모델

기존 `jhgan/ko-sroberta-multitask`(STS/NLI 범용 모델)는 계약서 조항(구어체)과 법 조문(격식체)
간 문체 격차 때문에 유사도 점수가 0.5 안팎으로 낮게 나오는 문제가 있었습니다. 검색(query/passage
비대칭) 목적에 맞게 학습된 `nlpai-lab/KoE5`로 교체하여 이 문제를 개선했습니다 (관련 조문 유사도
0.68~0.70대로 상승). 인코딩 시 법 조문은 `passage: `, 검색 질의는 `query: ` 접두사를 붙이는 E5
컨벤션을 따릅니다.

모델 교체 후 baseline 유사도 자체가 높아져 무관한 조문까지 걸러지지 않는 문제가 생겨,
실제 샘플 계약서 테스트를 거쳐 근거 필터 임계값(`min_score`)을 `0.25 → 0.45`로 재조정했습니다.
자세한 튜닝 근거와 실계약서 검증 결과, 그리고 이 파이프라인(분류기+RAG+LLM 분리 구조)이 계약서를
통째로 LLM에 던지는 방식보다 왜 더 정확하고 저렴하고 검증 가능한지는 `reports/project_summary_v5.pdf` 참고.

### RAG 법률 지식베이스 확장 (38건 → 45건)

`min_score` 튜닝 과정에서, 분류기 학습 데이터의 소재 다양성 목록(보증금·갱신·원상복구·관리비·소음 등
14개 항목)과 기존 법 조문 38건을 대조해보니 **관리비, 층간소음, 설비 하자·수선비 전가, 임차인 사망 시
승계**처럼 실제로 관련 법이 존재하는데도 지식베이스에 빠져 있는 소재가 발견됐습니다. 이 공백만 메우기
위해 아래 7건을 같은 스타일(조문 요지 + 특약 상황 해석)로 추가했습니다.

- 민법 제626조(임차인의 상환청구권), 제627조(일부멸실 등과 감액청구권), 제634조(임차인의 통지의무),
  제640조(차임연체와 해지) — 원상복구·수선비 전가, 설비 고장 시 차임감액 금지, 연체 요건 완화 특약 대응
- 주택임대차보호법 제9조(주택 임차권의 승계) — 임차인 사망 시 계약 자동 종료·보증금 몰수 특약 대응
- 공동주택관리법 제23조(관리비 등의 납부 등), 제20조(층간소음의 방지 등) — 관리비 임의 부과, 소음 민원
  즉시 강제퇴거 특약 대응

**전체 법령을 통째로 넣지 않은 이유**: 이 지식베이스는 조문 원문을 그대로 쌓아두는 게 아니라, 각 조문에
"이게 어떤 특약 상황에 적용되는지"를 사람이 직접 해설을 붙여 큐레이션한 것입니다. 이 해설이 계약서
조항(구어체)과 법 조문(격식체) 사이의 문체 격차를 메워 검색 정확도를 높이는 핵심 요인이라, 원문을
기계적으로 대량 추가하면 관리 부담만 커지고 오히려 관련 없는 조문이 검색 결과에 섞여 정확도가
떨어질 수 있습니다. 그래서 실제 학습 데이터에 등장하는 소재 중 **법적 근거가 실재하는데 비어 있는
공백만 확인해서** 타겟팅으로 추가하는 방식을 택했습니다. 반대로 주차 관련 특약처럼 딱 맞는 특별법이
없는 소재는 억지로 조문을 끼워 맞추지 않고, 기존의 일반 조항(약관의 규제에 관한 법률 제6조 등 불공정
약관 일반원칙)으로 커버되도록 그대로 두었습니다 — 근거가 약할 때 특정 조문을 억지로 인용하기보다
"근거 불충분"이라고 솔직히 답하는 게 더 신뢰할 수 있는 결과라고 판단했기 때문입니다.

법 조문을 추가/수정한 뒤에는 임베딩 캐시를 새로 만들어야 합니다:

```bash
python3 -m rag.rebuild_embeddings
```

## 폴더 구조

- `data/clauses.csv` : 분류기 학습 데이터 (정상/주의/위험 라벨링된 조항, 762건)
- `data/law_articles.json` : RAG용 법률 지식베이스 (주택임대차보호법·민법·형법·공동주택관리법 등, 45건)
- `contract_classifier/train.py` : 분류기 학습 스크립트
- `contract_classifier/predict.py` : 분류기 추론 함수
- `model/clause_classifier/` : 학습된 분류기 가중치
- `pdf_utils.py` : PDF 전처리 + 조항 분리
- `rag/retriever.py` : 법률 근거 검색 (임베딩: nlpai-lab/KoE5)
- `rag/rebuild_embeddings.py` : 법 조문 임베딩 캐시 강제 재생성 스크립트
- `generation/explainer.py` : Gemini 기반 설명 생성 (+ 개인정보 마스킹)
- `pipeline.py` : 전체 파이프라인 오케스트레이터 (CLI 실행 가능)
- `app.py` : Streamlit 데모 앱
- `env/` : 환경변수 설정 파일 모음 (`.env`, `.env.example`)
- `reports/` : 버전별 프로젝트 요약 리포트 (`project_summary.pdf` ~ `project_summary_v5.pdf`,
  v5가 최종본)

## 설치

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`env/.env.example`을 `env/.env`로 복사하고 Gemini API 키를 입력하세요.

```bash
cp env/.env.example env/.env
```

`HF_TOKEN`은 선택 사항입니다 (모델이 전부 공개 모델이라 없어도 동작). 다만 Hugging Face의 익명 요청
속도 제한을 피하고 배포 환경에서 재배포 시 모델 재다운로드를 더 안정적으로 하려면 넣는 것을 권장합니다.
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)에서 Read 권한 토큰을 발급받아
`env/.env`에 넣으면 되고, Streamlit Cloud에 배포할 때는 GOOGLE_API_KEY와 동일하게 앱 Secrets에 `HF_TOKEN`을
추가하면 `app.py`가 자동으로 환경변수로 옮겨줍니다.

## 실행

`model/clause_classifier/`에 이미 학습된 분류기가 저장되어 있으므로, 아래 명령만으로 바로
실행할 수 있습니다. (데이터를 더 보강하거나 모델을 바꿔서 재학습하고 싶을 때만 학습 스크립트를
다시 실행하면 됩니다.)

CLI로 계약서 분석:

```bash
python3 pipeline.py sample_contract.pdf
```

웹 데모 실행 (venv 안에서, `streamlit` 명령이 다른 파이썬을 가리켜 모듈을 못 찾는 문제를 피하려면
`python3 -m` 형태 권장):

```bash
source .venv/bin/activate
python3 -m streamlit run app.py
```

분류기를 재학습하고 싶을 때 (예: `data/clauses.csv`를 더 보강한 뒤):

```bash
cd contract_classifier
python3 train.py
```

RAG 법 조문 임베딩을 강제로 재생성하고 싶을 때 (모델을 바꾸거나 `law_articles.json`을 수정한 뒤).
평소엔 `LawRetriever`가 모델명·조문 개수를 저장된 메타데이터와 비교해 자동으로 재생성하므로
꼭 실행할 필요는 없지만, 확실히 새로 만들어졌는지 확인하고 싶을 때 사용:

```bash
python3 -m rag.rebuild_embeddings
```

## 주의사항

- 전자계약 등 텍스트 추출이 가능한 PDF를 가정합니다. 스캔본/이미지 PDF는 별도 OCR이 필요합니다.
- 본 결과는 참고용이며 법적 효력을 갖는 자문을 대체하지 않습니다.
- `data/law_articles.json`의 조문은 요지를 정리한 것으로, 실제 법률 자문 시에는 원문 확인이 필요합니다.
