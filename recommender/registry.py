"""Finds every available model for the sidebar dropdown.

To add a teammate model, drop a file in models/ and it appears automatically:
  models/<anything>.npy   -> one embedding row per movie (same order as movies_clean)
  models/<anything>.npz   -> arrays: embeddings (and optional ids to re-align rows)
Or register a custom class in CUSTOM below.

The notebook's full model set (hybrid embedding + XGBoost / stacked re-rankers) is loaded by
stacked_model.build_ml_models() when models/blocks.npz and models/ranker.joblib exist; build
them with `python scripts/build_ml_models.py --data <df_sampled.csv>`.
"""
import glob
import os
import numpy as np
from .content_based import ContentBasedRecommender
from .embedding_model import EmbeddingRecommender
from .stacked_model import build_ml_models

CUSTOM = {}   # e.g. CUSTOM["My model"] = lambda movies: MyModel(movies)

# Files written by scripts/build_ml_models.py. They are NOT one-vector-per-movie embedding
# files, so the folder scan below must not treat them as embedding models. E_hybrid.npy is the
# fused embedding, which is already offered as "Hybrid (cosine)" by the notebook bundle.
RESERVED = {"blocks.npz", "ok_mask.npy", "E_hybrid.npy"}


def _load_embedding(path, movies, fallback=None):
    name = os.path.splitext(os.path.basename(path))[0].replace("_", " ").strip().title()
    if path.endswith(".npz"):
        z = np.load(path)
        E = z["embeddings"] if "embeddings" in z else z[z.files[0]]
        if "ids" in z:                       # re-align by TMDB id
            pos = {int(i): k for k, i in enumerate(z["ids"])}
            full = np.zeros((len(movies), E.shape[1]), dtype="float32")   # movies they lack -> zero row
            for r, i in enumerate(movies["id"]):
                if int(i) in pos:
                    full[r] = E[pos[int(i)]]
            E = full
    else:
        E = np.load(path)
    return EmbeddingRecommender(movies, E, name=name, text_fallback=fallback)


def build_models(movies):
    """Returns {display_name: recommender}. A broken file never breaks the app."""
    baseline = ContentBasedRecommender(movies)
    models = {baseline.name: baseline}
    models.update(build_ml_models(movies, baseline))     # notebook models (if their files are present)
    for path in sorted(glob.glob("models/*.npy") + glob.glob("models/*.npz")):
        if os.path.basename(path) in RESERVED:
            continue
        try:
            r = _load_embedding(path, movies, baseline)
            models[r.name] = r
        except Exception as e:                          # noqa
            print(f"[registry] skipped {path}: {e}")
    for name, factory in CUSTOM.items():
        try:
            models[name] = factory(movies)
        except Exception as e:                          # noqa
            print(f"[registry] skipped {name}: {e}")
    return models
