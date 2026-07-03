"""
법률 근거 검색(RAG) 모듈
------------------------------------------------
위험/주의로 분류된 조항에 대해, 관련 법 조문을 임베딩 유사도로 검색한다.
법이 개정되면 data/law_articles.json만 갱신하면 되고, 모델 재학습은 필요 없다.

임베딩 모델: jhgan/ko-sroberta-multitask (한국어 문장 임베딩, CPU에서도 가벼움)
"""

import json
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
LAW_JSON_PATH = BASE_DIR / "data" / "law_articles.json"
EMBED_CACHE_PATH = BASE_DIR / "data" / "law_embeddings.npy"
DEFAULT_MODEL_NAME = "jhgan/ko-sroberta-multitask"


class LawRetriever:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, law_json_path: Path = LAW_JSON_PATH):
        from sentence_transformers import SentenceTransformer  # 지연 임포트 (무거운 의존성)

        with open(law_json_path, encoding="utf-8") as f:
            self.articles: list[dict] = json.load(f)

        self.model = SentenceTransformer(model_name)
        self.embeddings = self._load_or_build_embeddings()

    def _article_corpus_texts(self) -> list[str]:
        return [f"{a['title']} {a['text']}" for a in self.articles]

    def _load_or_build_embeddings(self) -> np.ndarray:
        texts = self._article_corpus_texts()

        if EMBED_CACHE_PATH.exists():
            cached = np.load(EMBED_CACHE_PATH)
            if cached.shape[0] == len(texts):
                return cached

        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        embeddings = np.asarray(embeddings)
        np.save(EMBED_CACHE_PATH, embeddings)
        return embeddings

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.25) -> list[dict]:
        """조항 텍스트를 받아 관련 법 조문 top_k개를 유사도 순으로 반환한다."""
        query_emb = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
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
