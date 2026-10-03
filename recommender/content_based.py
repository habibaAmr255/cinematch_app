"""Baseline: TF-IDF over overview + genres + keywords + directors + top cast."""
import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .base import BaseRecommender


def _doc(df):
    genres = df["genres"].str.replace(",", " ")
    kw = df["keywords"].str.replace(",", " ")
    cast5 = df["cast"].map(lambda s: " ".join(x.strip().replace(" ", "_") for x in s.split(",")[:5]))
    dirs = df["directors"].map(lambda s: " ".join(x.strip().replace(" ", "_") for x in s.split(",")))
    return (df["overview"] + " " + (genres + " ") * 3 + kw + " " + cast5 + " " + dirs + " " + dirs)


class ContentBasedRecommender(BaseRecommender):
    name = "Baseline (TF-IDF)"

    def __init__(self, movies, cache="models/tfidf_baseline.joblib"):
        super().__init__(movies)
        if cache and os.path.exists(cache):
            d = joblib.load(cache)
            if d["n"] == len(movies):
                self.vec, self.X = d["vec"], d["X"]
                return
        self.vec = TfidfVectorizer(stop_words="english", min_df=3, max_df=0.5,
                                   sublinear_tf=True, ngram_range=(1, 2), max_features=80000)
        self.X = self.vec.fit_transform(_doc(movies))
        if cache:
            try:
                joblib.dump({"vec": self.vec, "X": self.X, "n": len(movies)}, cache, compress=3)
            except OSError as e:                     # read-only filesystem on some hosts
                print(f"[content_based] could not write the baseline cache ({e}); "
                      "it will be rebuilt on the next start")

    def similarity_to(self, row_ids):
        q = self.X[row_ids].mean(axis=0)
        q = np.asarray(q)
        return np.asarray(self.X @ q.T).ravel()

    def similarity_to_text(self, text):
        q = self.vec.transform([text])
        if q.nnz == 0:
            return np.zeros(self.X.shape[0])
        return np.asarray((self.X @ q.T).todense()).ravel()
