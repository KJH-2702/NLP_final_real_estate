import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from pathlib import Path

LABELS = {0: "정상", 1: "주의", 2: "위험"}
# __file__ 기준 절대경로로 계산 -> 어느 위치에서 실행/임포트해도 안전
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = str(BASE_DIR / "model" / "clause_classifier")


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model


def split_clauses(text: str) -> list[str]:
    """계약서 텍스트를 조항 단위로 분리"""
    clauses = re.split(r'\n\s*(?=제?\s*\d+\s*조)', text)
    clauses = [c.strip() for c in clauses if len(c.strip()) > 10]
    return clauses


def predict_clause(clause: str, tokenizer, model) -> dict:
    inputs = tokenizer(
        clause,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).squeeze()
    label_id = torch.argmax(probs).item()
    return {
        "clause": clause,
        "label": LABELS[label_id],
        "label_id": label_id,
        "confidence": round(probs[label_id].item(), 4),
    }


def analyze_contract(text: str) -> list[dict]:
    """계약서 전체 텍스트를 받아 조항별 위험도 반환"""
    tokenizer, model = load_model()
    clauses = split_clauses(text)
    results = [predict_clause(c, tokenizer, model) for c in clauses]
    return results


if __name__ == "__main__":
    sample = """
    제1조 임대차 기간은 2024년 1월 1일부터 2025년 12월 31일까지로 한다.
    제2조 임차인은 임대인의 동의 없이 구조를 변경할 수 없다.
    제3조 임차인은 어떠한 사유로도 보증금 반환을 요구할 수 없다.
    """
    results = analyze_contract(sample)
    for r in results:
        print(f"[{r['label']}] ({r['confidence'] * 100:.1f}%) {r['clause'][:50]}...")
