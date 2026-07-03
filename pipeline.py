"""
전체 파이프라인 오케스트레이터
------------------------------------------------
계약서 PDF -> 조항 분리 -> 분류 -> (위험/주의 조항만) RAG 검색 -> 설명 생성

이 모듈이 최종 요구사항 문서에서 설명한 3단계 에이전트 구조에 대응한다.
  1. 스크리닝: pdf_utils(전처리) + contract_classifier(분류 모델, 직접 파인튜닝)
  2. 법률 리서치: rag.retriever (임베딩 기반 검색)
  3. 설명: generation.explainer (Gemini API)
"""

from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from contract_classifier.predict import load_model, predict_clause
from pdf_utils import extract_text_from_pdf, split_clauses

RISK_ORDER = {"정상": 0, "주의": 1, "위험": 2}


def analyze_contract_text(
    text: str,
    generate_explanations: bool = True,
    retriever=None,
    explainer=None,
) -> dict:
    """계약서 텍스트(이미 추출된 문자열)를 분석한다."""
    tokenizer, model = load_model()
    clauses = split_clauses(text)

    summary = {"정상": 0, "주의": 0, "위험": 0}
    results = []

    for clause_info in clauses:
        clause_text = clause_info["text"]
        pred = predict_clause(clause_text, tokenizer, model)
        label = pred["label"]
        summary[label] += 1

        entry = {
            "clause": clause_text,
            "source": clause_info["source"],
            "label": label,
            "confidence": pred["confidence"],
            "references": [],
            "explanation": None,
        }

        # 정상 조항은 근거 검색/설명 생성을 건너뛴다 (비용·시간 절약)
        if label != "정상":
            if retriever is not None:
                entry["references"] = retriever.retrieve(clause_text, top_k=3)

            if generate_explanations and explainer is not None:
                try:
                    entry["explanation"] = explainer.explain(
                        clause_text, label, entry["references"]
                    )
                except Exception as e:  # API 실패 시 파이프라인 전체가 죽지 않도록 방어
                    entry["explanation"] = f"(설명 생성 실패: {e})"

        results.append(entry)

    # 위험 -> 주의 -> 정상 순으로 정렬해서 중요한 조항이 위로 오게 한다
    results.sort(key=lambda x: -RISK_ORDER[x["label"]])

    return {
        "clauses": results,
        "summary": summary,
        "total": len(clauses),
    }


def analyze_contract_pdf(
    pdf_path: str,
    generate_explanations: bool = True,
    retriever=None,
    explainer=None,
) -> dict:
    """계약서 PDF 파일 경로를 받아 분석 결과를 반환한다."""
    text = extract_text_from_pdf(pdf_path)
    return analyze_contract_text(
        text,
        generate_explanations=generate_explanations,
        retriever=retriever,
        explainer=explainer,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("사용법: python pipeline.py <계약서.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    # RAG/생성 단계는 무거운 의존성(sentence-transformers)/API 키가 필요하므로
    # 여기서 지연 초기화하고, 실패하면 분류 결과만이라도 보여준다.
    retriever = None
    explainer = None
    try:
        from rag.retriever import LawRetriever

        retriever = LawRetriever()
    except Exception as e:
        print(f"[경고] RAG 검색기 초기화 실패, 근거 조문 없이 진행합니다: {e}")

    try:
        from generation.explainer import GeminiExplainer

        explainer = GeminiExplainer()
    except Exception as e:
        print(f"[경고] Gemini 설명 생성기 초기화 실패, 설명 없이 진행합니다: {e}")

    result = analyze_contract_pdf(pdf_path, retriever=retriever, explainer=explainer)

    print(f"\n총 {result['total']}개 조항 분석 완료")
    print(f"정상 {result['summary']['정상']} / 주의 {result['summary']['주의']} / 위험 {result['summary']['위험']}\n")

    for entry in result["clauses"]:
        print(f"[{entry['label']}][{entry['source']}] {entry['clause'][:60]}")
        if entry["references"]:
            for ref in entry["references"]:
                print(f"   근거: {ref['title']} (유사도 {ref['score']})")
        if entry["explanation"]:
            print(f"   설명: {entry['explanation']}")
        print()
