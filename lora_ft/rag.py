"""Dense retriever over the passage corpus (sentence-transformers + cosine search)."""

import numpy as np
from sentence_transformers import SentenceTransformer

from lora_ft.config import DATA_DIR, EMBED_MODEL, device
from lora_ft.data import read_jsonl

INDEX_PATH = DATA_DIR / "passage_embeddings.npy"
# BGE models expect this prefix on queries (not on passages).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Retriever:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self.passages = read_jsonl(DATA_DIR / "passages.jsonl")
        self.encoder = SentenceTransformer(EMBED_MODEL, device=device())
        if INDEX_PATH.exists():
            self.embeddings = np.load(INDEX_PATH)
        else:
            self.embeddings = self.encoder.encode(
                [p["text"] for p in self.passages],
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            np.save(INDEX_PATH, self.embeddings)

    def search(self, questions: list[str]) -> list[list[dict]]:
        queries = self.encoder.encode(
            [QUERY_PREFIX + q for q in questions], normalize_embeddings=True
        )
        scores = queries @ self.embeddings.T
        top = np.argsort(-scores, axis=1)[:, : self.top_k]
        return [[self.passages[i] for i in row] for row in top]


def format_context(passages: list[dict]) -> str:
    return "\n\n".join(f"[{i + 1}] {p['text']}" for i, p in enumerate(passages))
