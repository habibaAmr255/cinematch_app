"""Adapter for your teammates' notebook output.

The notebook saves E_hybrid.npy (one L2-normalised row per movie, same order as the
cleaned table) plus fitted.joblib. This class only needs the matrix, so it works for
ANY model that produces one vector per movie (SBERT, SVD, hybrid, autoencoder...).
"""
import numpy as np
from .base import BaseRecommender


class EmbeddingRecommender(BaseRecommender):
    def __init__(self, movies, embeddings, name="Hybrid embeddings", text_encoder=None, text_fallback=None):
        super().__init__(movies)
        assert len(embeddings) == len(movies), (
            f"{len(embeddings)} embedding rows vs {len(movies)} movies: the row order/filters differ")
        self.E = np.asarray(embeddings, dtype="float32")
        self.name = name
        self.text_encoder = text_encoder      # optional callable: str -> vector
        self.text_fallback = text_fallback    # e.g. the TF-IDF baseline, used for free-text themes

    def similarity_to(self, row_ids):
        q = self.E[row_ids].mean(axis=0)
        q = q / (np.linalg.norm(q) + 1e-9)
        return self.E @ q

    def similarity_to_text(self, text):
        if self.text_encoder is None:
            return self.text_fallback.similarity_to_text(text) if self.text_fallback else np.zeros(len(self.E))
        v = np.asarray(self.text_encoder(text), dtype="float32")
        return self.E @ (v / (np.linalg.norm(v) + 1e-9))
