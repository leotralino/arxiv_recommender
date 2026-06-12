import numpy as np
from sentence_transformers import SentenceTransformer


class TextEncoder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: str | list[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts)

    def cosine_sim(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between two matrices.
        vec1 dim: (d1, D), vec2 dim: (d2, D) → returns (d1, d2)
        """
        norms1 = np.linalg.norm(vec1, axis=1, keepdims=True)
        norms2 = np.linalg.norm(vec2, axis=1, keepdims=True)
        norms1 = np.where(norms1 == 0, 1, norms1)
        norms2 = np.where(norms2 == 0, 1, norms2)
        return vec1 @ vec2.T / (norms1 * norms2.T)

    def get_top_k_similar(
        self,
        query_vec: np.ndarray,  # [1, D]
        content_vecs: np.ndarray,  # [N, D]
        k: int = 10,
    ) -> list[int]:
        """
        Get top-k most similar content vectors to the query vector.
        """
        sims = self.cosine_sim(query_vec, content_vecs)[0]
        top_k_indices = list(np.argsort(-sims)[:k])
        return top_k_indices

    def __repr__(self):
        return f"TextEncoder(model_name={self.model_name})"
