"""
법률 조문 임베딩 캐시 재생성 스크립트
------------------------------------------------
임베딩 모델을 바꾸거나(현재: nlpai-lab/KoE5) data/law_articles.json 내용을 수정한 뒤
캐시(data/law_embeddings.npy)를 확실하게 새로 만들고 싶을 때 실행한다.

평소에는 LawRetriever가 모델명·조문 개수를 저장된 메타데이터와 비교해서 자동으로
캐시를 재생성하므로, 이 스크립트를 따로 돌릴 필요는 없다. 다만 "진짜 새로 만들어졌는지"를
확실히 확인하고 싶을 때, 또는 캐시 파일이 깨졌다고 의심될 때 강제로 지우고 재생성하는 용도.

실행 (프로젝트 루트에서):
    python3 -m rag.rebuild_embeddings
"""

from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / "env" / ".env")
except ImportError:
    pass

from rag.retriever import EMBED_CACHE_PATH, EMBED_META_PATH, LawRetriever


def main():
    for path in (EMBED_CACHE_PATH, EMBED_META_PATH):
        if path.exists():
            path.unlink()
            print(f"기존 캐시 삭제: {path}")

    retriever = LawRetriever()
    print(f"임베딩 재생성 완료: 조문 {len(retriever.articles)}개, 모델 = {retriever.model_name}")

    # 간단한 sanity check: 위험 조항 예시로 검색이 잘 되는지 확인
    samples = [
        "임차인은 어떠한 사유로도 보증금 반환을 요구할 수 없다.",
        "임대인은 임차인의 전입신고를 금지한다.",
        "임차인은 반려동물을 키울 수 없다.",
        "임차인이 관리비 명세를 요구해도 임대인은 항목을 공개하지 않는다.",
        "층간소음 민원이 한 번이라도 접수되면 임대인은 즉시 계약을 해지할 수 있다.",
        "보일러가 고장나도 수리 비용은 전액 임차인이 부담한다.",
        "임차인이 사망하면 계약은 자동으로 종료되고 보증금은 반환하지 않는다.",
        "차임을 1회만 연체해도 임대인은 즉시 계약을 해지하고 강제퇴거시킬 수 있다.",
    ]
    for query in samples:
        print(f"\n질의: {query}")
        results = retriever.retrieve(query)
        if not results:
            print("  (min_score 이상 매칭 없음)")
        for r in results:
            print(f"  [{r['score']}] {r['law']} {r['title']}")


if __name__ == "__main__":
    main()
