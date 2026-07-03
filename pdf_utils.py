"""
계약서 PDF 전처리 모듈
------------------------------------------------
1. PDF -> 텍스트 추출 (pymupdf)
2. 텍스트 -> 조항 단위 분리
   - 표준 조항: "제n조" 패턴 (임대차계약서 본문 기본 조항)
   - 특약사항: "특약사항" 섹션 헤더 이후 번호 매김 항목 (1. 2. 3. ...)
   두 종류를 모두 추출해서, 각 조항에 어느 섹션 출신인지 태깅한다.
   (표준양식이 아니라 특약사항 헤더가 없는 자유 양식 계약서의 경우
    표준 조항 분리 결과만 나올 수 있는데, 이 경우 분류기가 각 조항을
    직접 정상/주의/위험으로 걸러내는 역할을 겸한다.)
"""

import re

try:
    import fitz  # pymupdf
except ImportError:  # pragma: no cover
    fitz = None

# "제3조", "제 3 조" 등 앞에서 조항을 끊는다
STANDARD_CLAUSE_SPLIT = re.compile(r"\n\s*(?=제\s*\d+\s*조)")

# "특약사항", "특약 사항" 헤더 탐지
SPECIAL_SECTION_HEADER = re.compile(r"특\s*약\s*사\s*항")

# 특약사항 섹션 안에서 번호 매김 항목("1.", "1)", "가.", "- " 등)을 끊는다
NUMBERED_ITEM_SPLIT = re.compile(r"\n\s*(?=(?:\d{1,2}\s*[.\)]|[가-힣]\s*[.\)])\s*\S)")

# "임대인 (인)", "임차인:", "임대인 서명" 등 서명/날인 블록 시작점 -> 특약사항 섹션의 끝으로 간주
SIGNATURE_BLOCK_START = re.compile(r"\n\s*(?:임대인|임차인|계약자)\s*(?:\(인\)|서명|:)")

MIN_CLAUSE_LEN = 10


def extract_text_from_pdf(pdf_path: str) -> str:
    """PDF 파일에서 텍스트를 추출한다. (전자계약 PDF 기준, 스캔본/이미지 PDF는 미지원)"""
    if fitz is None:
        raise ImportError("pymupdf가 설치되어 있지 않습니다. `pip install pymupdf`로 설치하세요.")
    text_parts = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


_STARTS_WITH_JOHANG = re.compile(r"^제\s*\d+\s*조")


def split_standard_clauses(text: str) -> list[str]:
    """본문의 '제n조' 형태 표준 조항을 분리한다.

    특약사항 헤더 이후 텍스트는 표준 조항 분리 대상에서 제외하고(그렇지 않으면
    마지막 표준 조항이 특약사항 전체를 집어삼킨다), 조항 형태가 아닌
    제목/전문(前文) 조각도 걸러낸다.
    """
    special_match = SPECIAL_SECTION_HEADER.search(text)
    body = text[: special_match.start()] if special_match else text

    parts = STANDARD_CLAUSE_SPLIT.split(body)
    return [
        p.strip()
        for p in parts
        if len(p.strip()) > MIN_CLAUSE_LEN and _STARTS_WITH_JOHANG.match(p.strip())
    ]


def extract_special_terms_section(text: str) -> str | None:
    """'특약사항' 헤더 이후 텍스트를 반환한다. 헤더가 없으면 None."""
    match = SPECIAL_SECTION_HEADER.search(text)
    if not match:
        return None
    return text[match.end():]


def split_special_clauses(text: str) -> list[str]:
    """특약사항 섹션 안의 번호 매김 항목을 분리한다."""
    section = extract_special_terms_section(text)
    if not section:
        return []

    # 서명/날인 블록이 나오면 그 이전까지만 특약사항으로 취급한다
    sig_match = SIGNATURE_BLOCK_START.search(section)
    if sig_match:
        section = section[: sig_match.start()]

    items = NUMBERED_ITEM_SPLIT.split(section)
    return [i.strip() for i in items if len(i.strip()) > MIN_CLAUSE_LEN]


def split_clauses(text: str) -> list[dict]:
    """
    계약서 텍스트를 조항 단위로 분리한다.
    반환: [{"text": 조항 원문, "source": "표준조항" | "특약사항"}, ...]
    """
    results: list[dict] = []

    for clause in split_standard_clauses(text):
        results.append({"text": clause, "source": "표준조항"})

    for clause in split_special_clauses(text):
        results.append({"text": clause, "source": "특약사항"})

    # 표준조항도, 특약사항 헤더도 전혀 못 찾은 자유 양식 문서 대비 fallback:
    # 문장/줄바꿈 기준으로 최소한의 분리라도 시도한다.
    if not results:
        fallback = re.split(r"\n{1,}", text)
        results = [
            {"text": line.strip(), "source": "미분류"}
            for line in fallback
            if len(line.strip()) > MIN_CLAUSE_LEN
        ]

    return results


if __name__ == "__main__":
    sample = """
    제1조 임대차 기간은 2024년 1월 1일부터 2025년 12월 31일까지로 한다.
    제2조 임차인은 임대인의 동의 없이 구조를 변경할 수 없다.

    특약사항
    1. 임차인은 어떠한 사유로도 보증금 반환을 요구할 수 없다.
    2. 반려동물 사육 시 퇴거 시 특수청소비 100만원을 지급한다.
    """
    for c in split_clauses(sample):
        print(f"[{c['source']}] {c['text'][:40]}")
