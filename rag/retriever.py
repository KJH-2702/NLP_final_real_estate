"""
법률 근거 검색(RAG) 모듈
------------------------------------------------
위험/주의로 분류된 조항에 대해, 관련 법 조문을 임베딩 유사도로 검색한다.
법이 개정되면 data/law_articles.json만 갱신하면 되고, 모델 재학습은 필요 없다.

임베딩 모델: nlpai-lab/KoE5 (한국어 검색 특화 모델)
  - 기존 jhgan/ko-sroberta-multitask는 STS/NLI(문장 간 유사도 판별)용으로 학습된 범용
    모델이라, 계약서 조항(구어체·짧은 문장)과 법 조문(격식체·긴 문장) 사이의 문체 격차를
    잘 메우지 못해 유사도 점수가 0.5 안팎으로 낮게 나오는 문제가 있었다.
  - KoE5는 query(질의)/passage(문서)를 비대칭으로 구분해 검색 목적에 맞게 학습된 모델이라
    이 문체 격차에 더 강건하다. E5 계열 컨벤션에 따라 인코딩 시 반드시
    "query: " / "passage: " 접두사를 붙여야 한다 (안 붙이면 성능이 크게 떨어진다).
"""

import json
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
LAW_JSON_PATH = BASE_DIR / "data" / "law_articles.json"
EMBED_CACHE_PATH = BASE_DIR / "data" / "law_embeddings.npy"
EMBED_META_PATH = BASE_DIR / "data" / "law_embeddings_meta.json"
DEFAULT_MODEL_NAME = "nlpai-lab/KoE5"


class LawRetriever:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, law_json_path: Path = LAW_JSON_PATH):
        from sentence_transformers import SentenceTransformer  # 지연 임포트 (무거운 의존성)

        with open(law_json_path, encoding="utf-8") as f:
            self.articles: list[dict] = json.load(f)

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embeddings = self._load_or_build_embeddings()

    def _article_corpus_texts(self) -> list[str]:
        # E5 계열 컨벤션: 검색 대상 문서(법 조문)에는 "passage: " 접두사
        return [f"passage: {a['title']} {a['text']}" for a in self.articles]

    def _cache_is_valid(self, expected_count: int) -> bool:
        """캐시가 지금 모델·지금 조문 개수로 만들어진 것인지 확인.
        (모델을 바꾸거나 law_articles.json이 변경되면 캐시를 자동으로 무효화한다)"""
        if not (EMBED_CACHE_PATH.exists() and EMBED_META_PATH.exists()):
            return False
        try:
            meta = json.loads(EMBED_META_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        return meta.get("model_name") == self.model_name and meta.get("article_count") == expected_count

    def _load_or_build_embeddings(self) -> np.ndarray:
        texts = self._article_corpus_texts()

        if self._cache_is_valid(len(texts)):
            return np.load(EMBED_CACHE_PATH)

        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        embeddings = np.asarray(embeddings)
        np.save(EMBED_CACHE_PATH, embeddings)
        EMBED_META_PATH.write_text(
            json.dumps({"model_name": self.model_name, "article_count": len(texts)}, ensure_ascii=False),
            encoding="utf-8",
        )
        return embeddings

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.45) -> list[dict]:
        """조항 텍스트를 받아 관련 법 조문 top_k개를 유사도 순으로 반환한다."""
        # E5 계열 컨벤션: 검색 질의(계약서 조항)에는 "query: " 접두사
        query_emb = self.model.encode(
            [f"query: {query}"], normalize_embeddings=True, show_progress_bar=False
        )[0]
        sims = self.embeddings @ query_emb
        top_idx = np.argsort(-sims)[:top_k]

        results = []
        for i in top_idx:
            score = float(sims[i])
            if score < min_score:
                continue
            article = self.articles[i]
            results.append(
                {
                    "law": article["law"],
                    "title": article["title"],
                    "text": article["text"],
                    "score": round(score, 4),
                }
            )
        return results


if __name__ == "__main__":
    retriever = LawRetriever()
    query = "임차인은 어떠한 사유로도 보증금 반환을 요구할 수 없다."
    for r in retriever.retrieve(query):
        print(f"[{r['score']}] {r['title']}")
