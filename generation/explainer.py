"""
설명 생성 모듈 (Gemini API)
------------------------------------------------
분류기가 '주의'/'위험'으로 판정한 조항을, RAG로 검색된 법 조문을 근거로 삼아
일반인이 이해하기 쉬운 자연어 설명으로 풀어준다.

- 법 최신화: 이 모듈은 법 지식을 직접 암기하지 않는다. RAG가 넘겨준 조문
  텍스트 범위 안에서만 근거를 대도록 프롬프트로 강제한다. 법이 개정되면
  data/law_articles.json만 갱신하면 되고, 이 모듈은 손댈 필요가 없다.
- 개인정보 보호: 계약서에는 이름/연락처/주민번호 등이 포함될 수 있으므로,
  외부 API로 보내기 전에 정규식 기반으로 마스킹한다.
"""

import os
import re

SYSTEM_PROMPT = """당신은 대한민국 주택임대차 법률에 정통한 상담 전문가입니다.
사용자가 제공한 계약 조항, AI 분류 결과, 관련 법 조문을 바탕으로
왜 이 조항이 문제인지(또는 주의가 필요한지) 설명합니다.

반드시 지켜야 할 규칙:
1. 아래 제공된 '관련 법 조문' 범위 안에서만 근거를 제시하세요. 조문에 없는 내용을 지어내지 마세요.
2. 어떤 법률의 몇 조 때문에 문제가 되는지 명시하세요.
3. 전체 3~5문장, 임차인이 이해하기 쉬운 평이한 문장으로 설명하세요.
4. 마지막 문장은 임차인이 취할 수 있는 실질적인 행동을 한 가지 제안하세요.
5. 과장하지 말고, 근거가 부족하면 "단정하기 어렵다"고 솔직히 말하세요."""

# 이름/연락처/주민등록번호 등 개인정보 마스킹용 패턴
_PII_PATTERNS = [
    (re.compile(r"\d{2,3}-\d{3,4}-\d{4}"), "[연락처 비공개]"),
    (re.compile(r"\d{6}\s*-\s*\d{7}"), "[주민등록번호 비공개]"),
    (re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"), "[이메일 비공개]"),
]


def mask_pii(text: str) -> str:
    """조항 텍스트에서 연락처/주민번호/이메일 등을 마스킹한다."""
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _build_user_prompt(clause: str, label: str, references: list[dict]) -> str:
    if references:
        ref_lines = "\n".join(f"- {r['title']}: {r['text']}" for r in references)
    else:
        ref_lines = "(검색된 관련 조문 없음 - 일반적인 계약 공정성 관점에서만 신중하게 코멘트하세요)"

    return f"""[계약 조항]
{clause}

[AI 분류 결과]
{label}

[관련 법 조문]
{ref_lines}

위 정보를 바탕으로 이 조항에 대해 설명해 주세요."""


class GeminiExplainer:
    def __init__(self, api_key: str | None = None, model_name: str | None = None):
        from google import genai  # 지연 임포트

        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY 환경변수가 설정되어 있지 않습니다. "
                ".env 파일 또는 환경변수로 Gemini API 키를 설정하세요."
            )
        self.model_name = model_name or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        self.client = genai.Client(api_key=api_key)

    def explain(self, clause: str, label: str, references: list[dict]) -> str:
        masked_clause = mask_pii(clause)
        user_prompt = _build_user_prompt(masked_clause, label, references)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config={"system_instruction": SYSTEM_PROMPT, "temperature": 0.3},
        )
        return response.text.strip()


if __name__ == "__main__":
    sample_clause = "임차인은 어떠한 사유로도 보증금 반환을 요구할 수 없다."
    sample_refs = [
        {
            "law": "주택임대차보호법",
            "title": "주택임대차보호법 제10조 (강행규정)",
            "text": "이 법에 위반된 약정으로서 임차인에게 불리한 것은 그 효력이 없다.",
        }
    ]
    explainer = GeminiExplainer()
    print(explainer.explain(sample_clause, "위험", sample_refs))
