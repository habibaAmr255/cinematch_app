"""The notebook's recommendation models, plugged into the app.

What ``movies_insights_V7.ipynb`` builds (sections "Saving everything for the app" and
"Recommendation models"), and what this file does with it:

  1. A hybrid embedding ``E``: the weighted fusion of 8 blocks (text, genres, keywords, cast,
     directors, writers, companies, meta).                        -> "Hybrid (cosine)"
  2. Re-rankers trained on (seed movie, candidate movie) pairs: XGBoost, and a stack of
     XGBoost + KNN + logistic regression.                        -> "XGBoost re-ranker",
                                                                    "Stacked model (XGB+KNN+LR)",
                                                                    "Hybrid + Stack (50/50)"
  3. ``recommend_v2``'s pipeline: filters -> profile -> hybrid search over a candidate pool ->
     model re-rank -> quality prior -> top-k.  The same steps run here behind the app's
     ``BaseRecommender`` methods, so the Similar page, My lists and the chatbot all work with
     these models without a single change in the UI layer.

Files in ``models/`` (all written by ``scripts/build_ml_models.py``, which re-runs the
notebook's cells):

    ==================  =========  =========================================================
    ``blocks.npz``      required   every block on its own
    ``ranker.joblib``   required   ``{"stack", "xgb", "features", "feat_blocks"}``
    ``fitted.joblib``   optional   fusion weights + how to embed free text (defaults below)
    ``ok_mask.npy``     optional   rows that may be recommended; recomputed when missing
    ``app_meta.pkl``    optional   only used to check the rows line up
    ==================  =========  =========================================================

Row ``i`` of every file must be row ``i`` of ``data/movies_clean.csv.gz``: the notebook's
``df`` after the duplicate-``tconst`` and ``adult`` filters, which is exactly how
``scripts/prepare_data.py`` builds that file.  ``app_meta.pkl`` is checked at load time.
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from .base import BaseRecommender

# Fallbacks if fitted.joblib is missing: the notebook's default weights and text model.
DEFAULT_W = dict(text=.37, genres=.15, keywords=.15, cast=.10, directors=.10,
                 writers=.03, companies=.05, meta=.05)
DEFAULT_TEXT_MODEL = "all-MiniLM-L6-v2"

# (display name, share of the trained model in the score, which trained model)
# gamma 0.0 = hybrid cosine only, 1.0 = the model only, 0.5 = half and half
# (the notebook's mode "hybrid" / "model" / "both").
SPECS = [("Hybrid (cosine)",            0.0, "stack"),
         ("Hybrid + Stack (50/50)",     0.5, "stack"),
         ("Stacked model (XGB+KNN+LR)", 1.0, "stack"),
         ("XGBoost re-ranker",          1.0, "xgb")]


def _minmax(x):
    x = np.asarray(x, float)
    return (x - x.min()) / (x.max() - x.min() + 1e-9)


class MLBundle:
    """Loads the notebook's files once; every ML recommender shares it.

    One copy of ``E`` (about 80 MB) stays in memory, no matter how many entries the sidebar
    offers, because ``recommender/registry.py`` builds them all from the same bundle.
    """

    def __init__(self, movies: pd.DataFrame, folder="models"):
        path = lambda f: os.path.join(folder, f)          # noqa: E731
        n = len(movies)

        z = np.load(path("blocks.npz"))
        self.blocks = {k: np.asarray(z[k], dtype="float32") for k in z.files}
        for k, v in self.blocks.items():
            if len(v) != n:
                raise ValueError(f"blocks.npz['{k}'] has {len(v)} rows but movies_clean has {n}: "
                                 "the row order / filters differ")

        saved = joblib.load(path("ranker.joblib"))
        self.rankers = {"stack": saved["stack"], "xgb": saved["xgb"]}
        self.feat_blocks = list(saved["feat_blocks"])
        missing = [b for b in self.feat_blocks if b not in self.blocks]
        if missing:
            raise ValueError(f"ranker needs blocks {missing} that are not in blocks.npz")

        # fused-block weights + how to embed free text (both optional)
        self.W, self.text_spec = dict(DEFAULT_W), DEFAULT_TEXT_MODEL
        if os.path.exists(path("fitted.joblib")):
            try:
                import recsys_utils                          # noqa: F401  (pickled vectorizers point at it)
                fitted = joblib.load(path("fitted.joblib"))
                self.W = dict(fitted.get("weights", self.W))
                self.text_spec = fitted.get("text", self.text_spec)
            except Exception as e:                           # noqa: BLE001
                print(f"[stacked_model] fitted.joblib not used ({e}); using the default weights")
        self.E = self.fuse(self.W)

        self._check_alignment(movies, path("app_meta.pkl"))
        self.ok_mask = self._load_ok_mask(movies, path("ok_mask.npy"))

        # arrays the pair features need (same definitions as the notebook)
        wr = movies["wr"].to_numpy(float)
        self.wr_n = (wr - wr.min()) / (wr.max() - wr.min() + 1e-9)
        self.votes = movies["vote_count"].fillna(0).to_numpy(float)
        self.yr = movies["release_year"].astype("float").to_numpy()
        self.rt = np.log1p(movies["runtime"].to_numpy(float))
        self.lang = movies["original_language"].to_numpy()

        self._sbert = None
        self._text_ok = True

    # ---------------------------------------------------------------- setup helpers
    def fuse(self, W):
        """Weighted fusion: sqrt(weight) makes the dot product a weighted sum of block cosines."""
        parts = [np.sqrt(W[k]) * v for k, v in self.blocks.items() if W.get(k, 0) > 0]
        return normalize(np.hstack(parts)).astype("float32")

    @staticmethod
    def _check_alignment(movies, meta_path):
        if not os.path.exists(meta_path):
            return
        meta = pd.read_pickle(meta_path)
        same = len(meta) == len(movies) and (meta["id"].astype("int64").to_numpy()
                                             == movies["id"].astype("int64").to_numpy()).all()
        if not same:
            raise ValueError("models/app_meta.pkl does not line up with data/movies_clean.csv.gz "
                             "(different rows or order). Re-run scripts/prepare_data.py on the same "
                             "df_sampled.csv the models were built from.")

    @staticmethod
    def _load_ok_mask(movies, mask_path):
        if os.path.exists(mask_path):
            m = np.load(mask_path).astype(bool)
            if len(m) == len(movies):
                return m
            print("[stacked_model] ok_mask.npy has the wrong length; recomputing it")
        # the notebook's rule: drop duplicate title+year (keep the most voted) and compilations
        key = (movies["title"].str.lower().str.strip() + "|"
               + movies["release_year"].astype("Int64").astype(str))
        dup = key.loc[movies["vote_count"].sort_values(ascending=False).index].duplicated().reindex(movies.index)
        compil = movies["title"].str.contains(
            r"trilogy|quadrilogy|box set|double feature|(?:complete|movie|film) (?:saga|series|collection)",
            case=False, regex=True)
        return ~(dup | compil).to_numpy()

    # ---------------------------------------------------------------- model inputs
    def pair_features(self, si, cj):
        """One row per (seed si, candidate cj): the notebook's 10 features, in the same order."""
        cols = [np.einsum("ij,ij->i", self.blocks[b][si], self.blocks[b][cj]) for b in self.feat_blocks]
        year_gap = np.nan_to_num(np.abs(self.yr[si] - self.yr[cj]), nan=15.0)
        run_gap = np.nan_to_num(np.abs(self.rt[si] - self.rt[cj]), nan=0.3)
        same_lang = (self.lang[si] == self.lang[cj]).astype(float)
        return np.column_stack(cols + [year_gap, run_gap, same_lang,
                                       self.wr_n[cj], np.log1p(self.votes[cj])])

    def embed_text(self, text):
        """Free text -> unit vector in the text-block space.

        Returns None when no encoder is available (sentence-transformers not installed), in which
        case the caller falls back to the baseline's TF-IDF scores.
        """
        if not self._text_ok:
            return None
        try:
            if isinstance(self.text_spec, str):              # SBERT, as in the notebook
                if self._sbert is None:
                    from sentence_transformers import SentenceTransformer
                    self._sbert = SentenceTransformer(self.text_spec)
                return self._sbert.encode([text], normalize_embeddings=True)[0].astype("float32")
            import recsys_utils as ru                         # TF-IDF + SVD fallback
            return ru.embed_svd(*self.text_spec, pd.Series([text]))[0]
        except Exception as e:                               # noqa: BLE001
            print(f"[stacked_model] free-text encoder unavailable ({e}); falling back to TF-IDF")
            self._text_ok = False
            return None


class StackedRecommender(BaseRecommender):
    """gamma = share of the trained model in the final score (0.0 hybrid only, 1.0 model only,
    0.5 half and half); ``ranker`` = which trained model, "stack" (XGB+KNN+LR) or "xgb"."""

    def __init__(self, movies, bundle: MLBundle, name, gamma, ranker="stack",
                 text_fallback=None, pool=100, min_text_sim=0.20):
        super().__init__(movies)
        self.b, self.name, self.gamma, self.ranker = bundle, name, gamma, ranker
        self.text_fallback = text_fallback   # the TF-IDF baseline, for free text without SBERT
        self.pool = pool                     # how many hybrid-nearest movies the model re-ranks
        self.min_text_sim = min_text_sim     # below this a theme text "matches nothing" (SBERT only)

    # ---- BaseRecommender contract ------------------------------------------------------------
    def similarity_to(self, row_ids):
        """Hybrid cosine to the mean of the given rows.  The trained model re-ranks inside
        ``recommend_similar``, because it needs the candidate pool, not one number per movie."""
        q = normalize(self.b.E[[int(r) for r in row_ids]].mean(axis=0, keepdims=True))
        return (self.b.E @ q.T).ravel()

    def _text_scores(self, text):
        """(cosine of every movie to the text, threshold below which nothing 'matches')."""
        v = self.b.embed_text(text)
        if v is not None:
            return self.b.blocks["text"] @ v, self.min_text_sim
        if self.text_fallback is not None:
            return self.text_fallback.similarity_to_text(text), 0.0
        return np.zeros(len(self.movies)), 0.0

    def similarity_to_text(self, text):
        return self._text_scores(text)[0]

    # ---- the pipeline from the notebook's recommend_v2 -----------------------------------------
    def _rank(self, profile, allowed, n, quality, txt_cos=None):
        """profile -> hybrid search among ``allowed`` -> model re-rank -> quality prior."""
        b = self.b
        n_allowed = int(allowed.sum())
        if n_allowed == 0:
            return self._finish(np.array([], dtype=int), np.zeros(len(self.movies)), 0)

        q = normalize(b.E[profile].mean(axis=0, keepdims=True))
        s = (b.E @ q.T).ravel()
        if txt_cos is not None:                              # a typed theme counts too
            s = 0.5 * s + 0.5 * txt_cos
        s = np.where(allowed, s, -np.inf)

        m = min(max(self.pool, 3 * n), n_allowed)            # candidate pool for the model
        rows = np.argpartition(-s, m - 1)[:m]
        rows = rows[np.argsort(-s[rows])]
        score = _minmax(s[rows])

        if self.gamma > 0:                                   # score every (profile, candidate) pair
            model = b.rankers[self.ranker]
            prob = np.mean([model.predict_proba(b.pair_features(np.full(len(rows), p), rows))[:, 1]
                            for p in profile], axis=0)
            score = (1 - self.gamma) * score + self.gamma * _minmax(prob)

        quality = quality or 0.0
        score = (1 - quality) * score + quality * _minmax(b.wr_n[rows])

        full = np.full(len(self.movies), -np.inf)
        full[rows] = score
        return self._finish(rows[np.argsort(-score)], full, n)

    def _allowed(self, **mask_args):
        return self._mask(**mask_args) & self.b.ok_mask      # never duplicates / box sets

    def why(self, seed_rows, row, top=2):
        """The embedding blocks that contributed most — the notebook's ``why_simple``.

        Instead of "these two share a genre", this can say "a similar story and the same studios",
        because it has the per-block similarities the fusion is built from.
        """
        b = self.b
        seeds = [int(s) for s in seed_rows]
        row = int(row)
        if not seeds:
            return ""
        try:
            contrib = {k: float(b.W.get(k, 0)) * float(np.mean([b.blocks[k][s] @ b.blocks[k][row]
                                                                for s in seeds]))
                       for k in b.blocks if b.W.get(k, 0) > 0}
        except Exception:                                     # noqa: BLE001 - shape mismatch, etc.
            return super().why(seed_rows, row, top)

        phrase = {"text": "a similar story", "genres": "the same genres", "keywords": "similar themes",
                  "cast": "a similar cast", "directors": "a similar director",
                  "writers": "similar writers", "companies": "the same studios",
                  "meta": "a similar era and length"}
        best = sorted(contrib, key=contrib.get, reverse=True)[:top]

        def bag(r, col, n=None):
            vals = [x.strip() for x in str(self.movies.at[r, col]).split(",") if x.strip()]
            return set(vals[:n] if n else vals)

        parts = []
        for name in best:
            shared = []
            if name == "genres":
                shared = sorted(g.title() for g in
                                {g.lower() for g in self.movies.at[row, "genres_l"]}
                                & set().union(*[{g.lower() for g in self.movies.at[s, "genres_l"]}
                                                for s in seeds]))
            elif name == "cast":
                shared = sorted(bag(row, "cast", 5) & set().union(*[bag(s, "cast", 5) for s in seeds]))
            elif name == "directors":
                shared = sorted(bag(row, "directors") & set().union(*[bag(s, "directors") for s in seeds]))
            parts.append(phrase.get(name, name) + (f" ({', '.join(shared[:3])})" if shared else ""))
        if not parts:
            return super().why(seed_rows, row, top)
        return "it is closest on " + " and ".join(parts)

    # ---- public methods used by the UI and the chatbot ------------------------------------------
    def recommend_similar(self, row_ids, n=10, beta=0.15, min_votes=50, **filters):
        """beta = weight of the weighted rating (the "Favor well-rated movies" slider)."""
        row_ids = [int(r) for r in row_ids]
        allowed = self._allowed(min_votes=min_votes, **filters)
        allowed[row_ids] = False
        return self._rank(row_ids, allowed, n, quality=beta)

    def recommend_by_filters(self, genres=None, year_range=None, language=None, max_runtime=None,
                             min_runtime=None, min_rating=None, director=None, actor=None,
                             text=None, n=10, min_votes=50, sort_by="wr"):
        kw = dict(genres=genres, year_range=year_range, language=language, max_runtime=max_runtime,
                  min_runtime=min_runtime, min_rating=min_rating, director=director, actor=actor)
        has_filter = any(kw.values())
        if not text and (sort_by != "wr" or not has_filter):
            # "most popular", "newest" or no input at all: nothing for a model to rank by similarity
            return super().recommend_by_filters(**kw, text=None, n=n, min_votes=min_votes, sort_by=sort_by)

        allowed = self._allowed(**kw, min_votes=min_votes)
        if not allowed.any():
            return self._rank([], allowed, n, 0.0)
        if text:                                             # profile = the 5 best text matches
            txt_cos, thr = self._text_scores(text)
            if txt_cos[allowed].max() <= thr:                # nothing matches -> empty (chatbot relaxes)
                return self._rank([], np.zeros(len(allowed), bool), n, 0.0)
            profile = list(np.argsort(-np.where(allowed, txt_cos, -np.inf))[:min(5, int(allowed.sum()))])
            return self._rank(profile, allowed, n, quality=0.3, txt_cos=txt_cos)
        # only filters: profile = the 10 best-rated movies inside them
        profile = list(np.argsort(-np.where(allowed, self.b.wr_n, -np.inf))[:min(10, int(allowed.sum()))])
        return self._rank(profile, allowed, n, quality=0.3)


def build_ml_models(movies, baseline=None, folder="models"):
    """{display name: recommender} for the notebook's models, or {} if its files are not there.
    A broken file never breaks the app."""
    need = [os.path.join(folder, f) for f in ("blocks.npz", "ranker.joblib")]
    if not all(os.path.exists(p) for p in need):
        print("[registry] the notebook's models are not in models/ yet - build them with:\n"
              "           python scripts/build_ml_models.py --data <path to df_sampled.csv>")
        return {}
    try:
        bundle = MLBundle(movies, folder)
    except Exception as e:                                   # noqa: BLE001
        print(f"[registry] notebook models skipped: {e}")
        return {}
    return {name: StackedRecommender(movies, bundle, name, g, r, text_fallback=baseline)
            for name, g, r in SPECS}
