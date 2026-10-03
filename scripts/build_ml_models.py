"""Build the app's model files by re-running the notebook's pipeline.

Source: ``movies_insights_V7.ipynb`` -> sections "Saving everything for the app",
"Recommendation models", "Training data for the models", "Models: XGBoost, KNN and a stack"
and "recommend_v2".  This script repeats those cells outside Jupyter and writes the result
into ``models/``, so the Streamlit app can offer them next to the TF-IDF baseline.

Output (``models/``)
--------------------
=================  ==========================================================
``blocks.npz``     the 8 embedding blocks, row i = movie i of the cleaned table
``fitted.joblib``  fitted vectorizers / SVDs + fusion weights + meta params
``ranker.joblib``  ``{"stack", "xgb", "features", "feat_blocks"}``
``ok_mask.npy``    rows that must never be recommended (dupes, box sets)
``app_meta.pkl``   the notebook's metadata table (used to check the row order)
``E_hybrid.npy``   the fused embedding, ``fuse(blocks, weights)``
``labels.csv``     ``"Title (Year)"`` labels, same row order
=================  ==========================================================

Row alignment
-------------
Row i of every file is row i of ``data/movies_clean.csv.gz``.  ``scripts/prepare_data.py``
builds that file as *read df_sampled.csv -> drop duplicate tconst -> drop adult -> reset
index*, which is exactly the two row filters the notebook applies, so the orders match.
The last step of this script re-reads the app's data file and asserts that the TMDB ids
line up, so a mismatch can never reach the app silently.

The text block
--------------
The notebook encodes ``title + tagline + overview + keywords`` with SBERT
(``all-MiniLM-L6-v2``, 384 dims).  Pass ``--sbert`` to do the same; that needs
``pip install sentence-transformers`` (which pulls in torch).  Without it, the notebook's own
fallback is used: TF-IDF + ``TruncatedSVD(128)``.  Everything still works, but free-text
themes (and the ``text`` feature of the re-rankers) are weaker.

Usage
-----
    python scripts/build_ml_models.py --data /path/to/df_sampled.csv
    python scripts/build_ml_models.py --data /path/to/df_sampled.csv --sbert
    python scripts/build_ml_models.py --data /path/to/df_sampled.csv --jobs 4 --no-eval
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)          # so `import recsys_utils` works from any directory
import recsys_utils as ru             # noqa: E402  (the pickled vectorizers point at ru.tok)

# --------------------------------------------------------------------------- notebook constants
# Cell 100: block weights (they sum to 1, so each reads as a share of the similarity).
W = dict(text=.37, genres=.15, keywords=.15, cast=.10, directors=.10,
         writers=.03, companies=.05, meta=.05)
# Cell 118: the re-rankers only use these blocks, so no cast / directors / writers.
FEAT_BLOCKS = ["text", "genres", "keywords", "companies", "meta"]
N_SEEDS, N_CAND = 4000, 30
MIN_VOTES = 100

# Cell 105: language codes -> display names (copied from the notebook).
LANG_NAMES = {
    "en": "English", "fr": "French", "es": "Spanish", "de": "German", "it": "Italian", "ja": "Japanese",
    "ko": "Korean", "zh": "Chinese", "cn": "Cantonese", "hi": "Hindi", "ru": "Russian", "pt": "Portuguese",
    "sv": "Swedish", "da": "Danish", "no": "Norwegian", "nb": "Norwegian", "fi": "Finnish", "nl": "Dutch",
    "pl": "Polish", "tr": "Turkish", "ar": "Arabic", "he": "Hebrew", "fa": "Persian", "th": "Thai",
    "id": "Indonesian", "ms": "Malay", "tl": "Tagalog", "vi": "Vietnamese", "cs": "Czech", "sk": "Slovak",
    "hu": "Hungarian", "ro": "Romanian", "bg": "Bulgarian", "el": "Greek", "uk": "Ukrainian", "sr": "Serbian",
    "hr": "Croatian", "sl": "Slovenian", "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "ml": "Malayalam",
    "kn": "Kannada", "mr": "Marathi", "pa": "Punjabi", "ur": "Urdu", "is": "Icelandic", "et": "Estonian",
    "lv": "Latvian", "lt": "Lithuanian", "ca": "Catalan", "eu": "Basque", "gl": "Galician", "la": "Latin",
    "sq": "Albanian", "ka": "Georgian", "hy": "Armenian", "mk": "Macedonian", "bs": "Bosnian",
    "af": "Afrikaans", "sw": "Swahili", "mn": "Mongolian", "ne": "Nepali", "kk": "Kazakh", "xx": "No language"}

_T0 = time.time()


def log(msg=""):
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- 1. load + clean
def load_and_clean(src: str):
    """Notebook cells 18, 81, 84, 85, 86, 87, 90 and 92, in that order.

    Returns the cleaned ``df`` (49,853 rows) plus the ``id / tconst / title / poster_path``
    table aligned with it.  The notebook gets that table from ``df_ins`` (cell 19 + 104);
    taking it from the same frame before those columns are dropped is equivalent, because
    ``df_ins`` is a copy of the same rows and is only filtered by ``adult``.
    """
    log(f"reading {src}")
    df = pd.read_csv(src)
    log(f"  raw shape {df.shape}")

    # cell 18 - duplicate movies (same IMDb id) go first, so every count uses one clean set
    df = df.drop_duplicates(subset="tconst", keep="first").reset_index(drop=True)
    log(f"  after tconst dedup {df.shape}")

    # cell 81 - keep the lookup columns aside, then drop what the models must not see
    meta_all = df[["id", "tconst", "title", "poster_path"]].copy()
    df["has_homepage"] = df.homepage.notna().astype(int)
    df = df.drop(columns=["id", "tconst", "backdrop_path", "poster_path", "homepage"])
    df = df.drop(columns="status")                       # constant after filtering

    # cell 84 - fill spoken_languages from original_language (the most common value)
    single = df[df.spoken_languages.notna() & ~df.spoken_languages.str.contains(",", na=False)]
    lang_map = single.groupby("original_language").spoken_languages.agg(lambda s: s.mode().iat[0])
    m = df.spoken_languages.isna()
    df.loc[m, "spoken_languages"] = df.loc[m, "original_language"].map(lang_map)

    # cell 85 - fill production_countries from the companies
    known = df.dropna(subset=["production_companies", "production_countries"])
    pairs = (known.assign(company=known.production_companies.str.split(", "))
                  .explode("company")[["company", "production_countries"]])
    comp2country = pairs.groupby("company").production_countries.agg(lambda s: s.mode().iat[0])

    def fill_country(companies):
        votes = pd.Series([comp2country.get(c) for c in companies.split(", ")]).dropna()
        return votes.mode().iat[0] if len(votes) else np.nan

    m = df.production_countries.isna() & df.production_companies.notna()
    df.loc[m, "production_countries"] = df.loc[m, "production_companies"].map(fill_country)

    # cell 86 - kNN keyword imputation (12,960 rows in the notebook's run)
    df = df.reset_index(drop=True)                       # X and df stay aligned
    Xk = TfidfVectorizer(stop_words="english", min_df=3, max_features=50000).fit_transform(
        df.overview.fillna("") + " " + df.genres.fillna(""))
    has = df.keywords.notna().to_numpy()
    miss = np.where(~has)[0]
    log(f"  keywords to impute: {len(miss)}")
    df["keywords_imputed"] = (~has).astype(int)
    if len(miss):
        nn = NearestNeighbors(n_neighbors=15, metric="cosine").fit(Xk[has])
        kw_known = df.keywords[has].str.split(", ").tolist()

        def impute_keywords(rows, top=5, batch=2000):
            out = []
            for s in range(0, len(rows), batch):
                dist, idx = nn.kneighbors(Xk[rows[s:s + batch]])
                for d, ix in zip(dist, idx):
                    c = Counter()
                    for dd, i in zip(d, ix):
                        for k in kw_known[i]:
                            c[k] += 1 - dd                    # similarity-weighted vote
                    out.append(", ".join(k for k, _ in c.most_common(top)))
                if s and s % 8000 == 0:
                    log(f"    imputed {s}/{len(rows)}")
            return out

        kw = df["keywords"].to_numpy(dtype=object)
        kw[miss] = impute_keywords(miss)
        df["keywords"] = kw

    # cell 87 - tagline flags and the raw text column
    df["has_tagline"] = df.tagline.notna().astype(int)
    df["tagline_len"] = df.tagline.fillna("").str.split().str.len()
    df["text"] = (df.overview.fillna("") + " " + df.tagline.fillna("") + " "
                  + df.keywords.fillna("").str.replace(",", " ")).str.strip()

    # cell 90 - release_date -> day / month / year, then drop the date itself
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_day"] = df["release_date"].dt.day
    df["release_month"] = df["release_date"].dt.month
    df["release_year"] = df["release_date"].dt.year
    df.drop(columns="release_date", inplace=True)

    # cell 92 - drop adult titles (the last row filter) and normalise the tag columns
    keep = ~df["adult"].to_numpy()
    meta = meta_all.loc[keep].reset_index(drop=True)
    df = df.loc[keep].reset_index(drop=True)
    log(f"  after adult filter {df.shape}")

    tag_cols = ["genres", "keywords", "production_companies", "production_countries",
                "spoken_languages", "cast", "writers", "directors"]
    for c in tag_cols:                                   # missing = empty string, never "unknown"
        df[c] = df[c].replace({"unknown": np.nan}).fillna("").astype(str)
    for c in ["title", "overview", "tagline"]:
        df[c] = df[c].fillna("").astype(str)
    for c in ["budget", "revenue", "runtime"]:           # 0 means "unknown" in this kind of data
        df[c] = df[c].where(df[c] > 0)
    return df, meta


def make_labels(df):
    """Cell 92: 'Title (Year)', with a '#index' suffix when title+year is ambiguous."""
    labels = df.title + " (" + df.release_year.astype("Int64").astype(str) + ")"
    return labels.where(~labels.duplicated(keep=False), labels + " #" + df.index.astype(str))


# --------------------------------------------------------------------------- 2. embedding blocks
def fit_tag(series, dims, min_df):
    """Cell 112's ``fit_tag``: TF-IDF -> TruncatedSVD, both kept so the app can reuse them."""
    vec = TfidfVectorizer(tokenizer=ru.tok, token_pattern=None, lowercase=False, min_df=min_df)
    X = vec.fit_transform(series)
    svd = TruncatedSVD(max(2, min(dims, X.shape[1] - 1)), random_state=42).fit(X)
    return vec, svd


def text_block(df, sbert: bool):
    """Cell 94: SBERT if asked for and available, else the notebook's TF-IDF + SVD fallback."""
    real_kw = df.keywords.where(df.keywords_imputed == 0, "")      # exclude guessed keywords
    sbert_text = (df.title + ". " + df.tagline + " " + df.overview
                  + " Keywords: " + real_kw.str.replace(",", ", ")).str.strip()

    if sbert:
        from sentence_transformers import SentenceTransformer        # noqa: PLC0415
        E = SentenceTransformer("all-MiniLM-L6-v2").encode(
            sbert_text.tolist(), batch_size=64, normalize_embeddings=True,
            show_progress_bar=True).astype("float32")
        spec = "all-MiniLM-L6-v2"
    else:
        log("  text block: TF-IDF + TruncatedSVD(128) fallback (no SBERT)")
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=5, max_df=0.5,
                              sublinear_tf=True, max_features=50000)
        X = vec.fit_transform(sbert_text)
        svd = TruncatedSVD(128, random_state=42).fit(X)
        E = normalize(svd.fit_transform(X)).astype("float32")
        spec = (vec, svd)

    weak = ((df.overview + " " + df.tagline).str.split().str.len() < 8).to_numpy()
    E[weak] *= 0.5                                       # very short text counts half
    log(f"  rows with weak text: {int(weak.sum())}")
    return E, spec, sbert_text


def build_blocks(df, sbert: bool):
    """Cells 94, 96, 98 and 112: every block, plus the objects needed to reproduce them."""
    blocks, fitted = {}, {}
    log("building the embedding blocks")

    blocks["text"], fitted["text"], text_series = text_block(df, sbert)

    # cell 96 - tag and people blocks
    cast5 = df.cast.map(lambda s: ", ".join(s.split(",")[:5]))     # top-5 billed
    specs = [("keywords", df.keywords, 64, 5), ("cast", cast5, 64, 3), ("directors", df.directors, 32, 2),
             ("writers", df.writers, 32, 2), ("companies", df.production_companies, 32, 20)]
    for name, series, dims, min_df in specs:
        vec, svd = fit_tag(series, dims, min_df)
        fitted[name] = (vec, svd)
        blocks[name] = normalize(svd.fit_transform(vec.transform(series))).astype("float32")
        log(f"  {name:<10} {blocks[name].shape}")
    blocks["keywords"][df.keywords_imputed.to_numpy() == 1] *= 0.5  # imputed keywords count half

    cv = CountVectorizer(tokenizer=ru.tok, token_pattern=None, lowercase=False, binary=True)
    fitted["genres"] = cv.fit(df.genres)
    blocks["genres"] = normalize(cv.transform(df.genres).toarray().astype("float32"))
    log(f"  {'genres':<10} {blocks['genres'].shape}")

    # cell 98 - meta block (year, runtime, budget, language one-hots), NaN -> 0 = neutral
    def z(x):
        x = np.asarray(x, float)
        return np.nan_to_num((x - np.nanmean(x)) / np.nanstd(x), nan=0.0)

    lang = df.original_language.fillna("other")
    lang = lang.where(lang.map(lang.value_counts()) >= 200, "other")
    x_meta = df[["release_year", "runtime", "budget"]].astype(float)
    blocks["meta"] = normalize(np.column_stack([z(df.release_year), z(np.log1p(df.runtime)),
                                                z(np.log1p(df.budget)),
                                                pd.get_dummies(lang).to_numpy(float)])).astype("float32")
    fitted["meta"] = {"year": (x_meta.release_year.mean(), x_meta.release_year.std(ddof=0)),
                      "runtime": (np.log1p(x_meta.runtime).mean(), np.log1p(x_meta.runtime).std(ddof=0)),
                      "budget": (np.log1p(x_meta.budget).mean(), np.log1p(x_meta.budget).std(ddof=0)),
                      "lang_cols": list(pd.get_dummies(lang).columns)}
    log(f"  {'meta':<10} {blocks['meta'].shape}")

    # cell 112 - the fitted objects must reproduce the saved blocks (agreement ~ 1)
    def agree(new, old):
        nz = np.linalg.norm(old, axis=1) > 0
        return float(((new[nz] * old[nz]).sum(1) /
                      (np.linalg.norm(new[nz], axis=1) * np.linalg.norm(old[nz], axis=1))).mean())

    for name, series, _, _ in specs:
        a = agree(ru.embed_svd(*fitted[name], series), blocks[name])
        log(f"  agreement {name:<10} {a:.4f}")
    log(f"  agreement {'genres':<10} {agree(ru.embed_genres(fitted['genres'], df.genres), blocks['genres']):.4f}")
    if isinstance(fitted["text"], tuple):
        log(f"  agreement {'text':<10} "
            f"{agree(ru.embed_svd(*fitted['text'], text_series), blocks['text']):.4f}")
    fitted["weights"] = dict(W)
    return blocks, fitted


def fuse(blocks, weights):
    """Cell 100's ``fuse``: sqrt(weight) makes the dot product a weighted sum of cosines."""
    parts = [np.sqrt(weights[k]) * v for k, v in blocks.items() if weights.get(k, 0) > 0]
    return normalize(np.hstack(parts)).astype("float32")


# --------------------------------------------------------------------------- 3. app metadata
def build_app_meta(df, meta, labels, wr, n_rows):
    """Cell 105: one metadata table for the app, same row order as the blocks."""
    year = df["release_year"]
    bad_year = year.isna() | ~year.between(1880, pd.Timestamp.today().year)
    log(f"  movies with a missing / impossible release_year: {int(bad_year.sum())}")
    year = year.where(~bad_year).astype("Int64")
    language = df.original_language.map(LANG_NAMES).fillna(df.original_language.str.upper())

    def to_list(s):            # "Tim Robbins, Morgan Freeman" -> ["tim robbins", "morgan freeman"]
        return [t.strip().lower() for t in s.split(",") if t.strip()]

    app = pd.DataFrame({
        "title": df.title, "label": labels, "year": year, "language": language, "overview": df.overview,
        "genres": df.genres, "cast": df.cast, "directors": df.directors,          # text, for display
        "genres_l": df.genres.map(to_list), "cast_l": df.cast.map(to_list),
        "directors_l": df.directors.map(to_list),
        "languages_l": df.spoken_languages.map(to_list), "countries_l": df.production_countries.map(to_list),
        "poster_path": meta.poster_path, "tconst": meta.tconst, "id": meta.id,
        "vote_average": df.vote_average, "vote_count": df.vote_count, "wr": wr})
    assert len(app) == n_rows, (len(app), n_rows)
    return app


# --------------------------------------------------------------------------- 4. training data
def build_training_data(df, app, blocks, ok_mask, E, rng, jobs, n_seeds=N_SEEDS, n_cand=N_CAND):
    """Cell 118: KNN candidates per seed movie, labelled by the notebook's weak rule."""
    # the same fuse() as before, with weights only for the blocks the re-rankers may see
    E_nt = fuse(blocks, {b: W[b] for b in FEAT_BLOCKS})
    votes = df.vote_count.fillna(0).to_numpy()
    wr_n = (app["wr"].to_numpy() - app["wr"].min()) / (app["wr"].max() - app["wr"].min() + 1e-9)
    pool_rows = np.where(ok_mask & (votes >= MIN_VOTES))[0]

    dir_set = [set(l) for l in app.directors_l]
    cast_set = [set(l[:3]) for l in app.cast_l]          # top-3 billed actors
    gen_set = [set(l) for l in app.genres_l]

    def same_team(i, j):
        return bool(dir_set[i] & dir_set[j]) or bool(cast_set[i] & cast_set[j])

    def good_pair(i, j):                                 # the weak label
        return same_team(i, j) and bool(gen_set[i] & gen_set[j])

    has_people = np.array([bool(dir_set[i]) and bool(cast_set[i]) for i in pool_rows])
    seeds_all = rng.choice(pool_rows[has_people], min(n_seeds, int(has_people.sum())), replace=False)

    nn_pool = NearestNeighbors(n_neighbors=n_cand + 1, metric="cosine", algorithm="brute",
                               n_jobs=jobs).fit(E_nt[pool_rows])
    _, nb = nn_pool.kneighbors(E_nt[seeds_all])
    cand = np.array([[j for j in pool_rows[r] if j != s][:n_cand] for s, r in zip(seeds_all, nb)])

    yr = df.release_year.to_numpy(float)
    rt = np.log1p(df.runtime.to_numpy(float))
    lang_arr = df.original_language.to_numpy()
    feats = FEAT_BLOCKS + ["year_gap", "runtime_gap", "same_lang", "cand_wr", "cand_logvotes"]

    def pair_features(si, cj, step=20000):
        cols = [np.concatenate([np.einsum("ij,ij->i", blocks[b][si[a:a + step]], blocks[b][cj[a:a + step]])
                                for a in range(0, len(si), step)]) for b in FEAT_BLOCKS]
        year_gap = np.nan_to_num(np.abs(yr[si] - yr[cj]), nan=15.0)     # missing -> typical gap
        run_gap = np.nan_to_num(np.abs(rt[si] - rt[cj]), nan=0.3)
        return np.column_stack(cols + [year_gap, run_gap, (lang_arr[si] == lang_arr[cj]).astype(float),
                                       wr_n[cj], np.log1p(votes[cj])])

    si, cj = np.repeat(seeds_all, n_cand), cand.ravel()
    X_pairs = pair_features(si, cj)
    y_pairs = np.array([good_pair(i, j) for i, j in zip(si, cj)]).astype(int)
    grp = np.repeat(np.arange(len(seeds_all)), n_cand)

    test_seed = rng.random(len(seeds_all)) < 0.2          # split by SEED movie, so no leakage
    tr, te = ~test_seed[grp], test_seed[grp]
    log(f"  pairs: {X_pairs.shape} | positive rate: {y_pairs.mean() * 100:.1f}% "
        f"| test seeds: {int(test_seed.sum())}")
    ctx = dict(df=df, app=app, blocks=blocks, E=E, votes=votes, wr_n=wr_n, seeds_all=seeds_all,
               cand=cand, pool_rows=pool_rows, gen_set=gen_set, good_pair=good_pair, n_cand=n_cand,
               pair_features=pair_features, feats=feats, X_pairs=X_pairs, y_pairs=y_pairs,
               grp=grp, tr=tr, te=te, test_seed=test_seed)
    return ctx


def train_rankers(ctx, jobs):
    """Cell 120: XGBoost alone plus the stack (XGBoost + KNN + logistic regression)."""
    from sklearn.base import clone
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    X_pairs, y_pairs, grp, tr = ctx["X_pairs"], ctx["y_pairs"], ctx["grp"], ctx["tr"]
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, eval_metric="logloss", random_state=42, n_jobs=jobs)
    knn = make_pipeline(StandardScaler(),
                        KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=jobs))
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    log("  training XGBoost ...")
    xgb_only = clone(xgb).fit(X_pairs[tr], y_pairs[tr])
    log("  training the stack (5-fold GroupKFold; this is the slow part) ...")
    folds = list(GroupKFold(n_splits=5).split(X_pairs[tr], y_pairs[tr], grp[tr]))
    stack = StackingClassifier([("xgb", xgb), ("knn", knn), ("lr", lr)],
                               final_estimator=LogisticRegression(), cv=folds,
                               stack_method="predict_proba").fit(X_pairs[tr], y_pairs[tr])
    try:
        imp = pd.Series(xgb_only.feature_importances_, index=ctx["feats"]).sort_values(ascending=False)
        log("  what XGBoost relies on:\n" + imp.round(3).to_string())
    except Exception:                                    # noqa: BLE001 - never fail on a report
        pass
    return {"stack": stack, "xgb": xgb_only}


def evaluate(ctx, rankers, rng):
    """Cell 122: hit-rate of every ranker on the held-out seed movies."""
    X_pairs, seeds_all, cand, pool_rows = ctx["X_pairs"], ctx["seeds_all"], ctx["cand"], ctx["pool_rows"]
    gen_set, test_seed, E = ctx["gen_set"], ctx["test_seed"], ctx["E"]
    good_pair, n_cand = ctx["good_pair"], ctx["n_cand"]
    test_ids = np.where(test_seed)[0]

    def minmax(x, axis=None):
        lo, hi = x.min(axis=axis, keepdims=True), x.max(axis=axis, keepdims=True)
        return (x - lo) / (hi - lo + 1e-9)

    def seed_scores(seed, js):
        team = np.mean([good_pair(seed, j) for j in js])
        gen = np.mean([len(gen_set[seed] & gen_set[j]) / max(1, len(gen_set[seed] | gen_set[j]))
                       for j in js])
        return team, gen

    def eval_pool(score):
        out = []
        for s in test_ids:
            order = np.argsort(-score[s * n_cand:(s + 1) * n_cand])[:10]
            out.append(seed_scores(seeds_all[s], cand[s][order]))
        return np.mean(out, axis=0)

    cos_rank = np.tile(np.linspace(1, 0, n_cand), (len(seeds_all), 1))   # candidates are cosine-sorted
    p_xgb = rankers["xgb"].predict_proba(X_pairs)[:, 1].reshape(-1, n_cand)
    p_stack = rankers["stack"].predict_proba(X_pairs)[:, 1].reshape(-1, n_cand)

    res = {"random order": eval_pool(rng.random(len(X_pairs))),
           "cosine (no people blocks)": eval_pool(cos_rank.ravel()),
           "XGBoost": eval_pool(p_xgb.ravel()),
           "Stack (XGB + KNN + LR)": eval_pool(p_stack.ravel()),
           "Blend 70% cosine + 30% stack": eval_pool((0.7 * cos_rank + 0.3 * minmax(p_stack, axis=1)).ravel()),
           "Blend 50% cosine + 50% stack": eval_pool((0.5 * cos_rank + 0.5 * minmax(p_stack, axis=1)).ravel())}

    te_seeds = seeds_all[test_seed]                       # reference: the full hybrid E
    sims = E[te_seeds] @ E[pool_rows].T
    sims[np.arange(len(te_seeds)), np.searchsorted(pool_rows, te_seeds)] = -np.inf
    top10 = pool_rows[np.argsort(-sims, axis=1)[:, :10]]
    res["hybrid E (reference)"] = np.mean([seed_scores(s, js) for s, js in zip(te_seeds, top10)], axis=0)

    table = pd.DataFrame(res, index=["team & genre hit @10", "genre overlap @10"]).T
    log("evaluation on the held-out movies:\n" + table.round(3).to_string())
    return table


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Rebuild the notebook's model files into models/.")
    ap.add_argument("--data", help="path to df_sampled.csv (the notebook's input)")
    ap.add_argument("--out", default=os.path.join(ROOT, "models"), help="output folder")
    ap.add_argument("--sbert", action="store_true",
                    help="use the notebook's SBERT text block (needs sentence-transformers)")
    ap.add_argument("--no-eval", action="store_true", help="skip the held-out evaluation table")
    ap.add_argument("--jobs", type=int, default=1,
                    help="threads for KNN/XGBoost (1 = safest; try 4 outside restricted shells)")
    ap.add_argument("--limit", type=int, default=0,
                    help="dry run: use only N random movies, to check the pipeline quickly")
    ap.add_argument("--seeds", type=int, default=N_SEEDS,
                    help=f"how many seed movies to train on (default {N_SEEDS})")
    args = ap.parse_args(argv)

    src = args.data or _find_data()
    if not src or not os.path.exists(src):
        ap.error("could not find df_sampled.csv - pass it with --data")
    os.makedirs(args.out, exist_ok=True)
    out = lambda name: os.path.join(args.out, name)      # noqa: E731

    # ---- 1. data -> blocks -----------------------------------------------------------------
    df, meta = load_and_clean(src)
    if args.limit and args.limit < len(df):
        pick = np.sort(np.random.default_rng(0).choice(len(df), args.limit, replace=False))
        df, meta = df.iloc[pick].reset_index(drop=True), meta.iloc[pick].reset_index(drop=True)
        log(f"DRY RUN: keeping {len(df)} random movies")
    labels = make_labels(df)
    blocks, fitted = build_blocks(df, args.sbert)
    E = fuse(blocks, fitted["weights"])
    dims = {k: int(v.shape[1]) for k, v in blocks.items()}
    log(f"E: {E.shape} ({E.nbytes / 1e6:.0f} MB) | blocks: {dims}")
    if not args.sbert and not args.limit:
        # the notebook's own numbers, for the blocks that do not depend on the text encoder
        assert dims["genres"] == 19, f"genres block has {dims['genres']} columns, expected 19"
        assert dims["meta"] == 24, f"meta block has {dims['meta']} columns, expected 24"

    # ---- 2. weighted rating (cell 102) and the metadata table (cell 105) --------------------
    v, R = df.vote_count.fillna(0), df.vote_average.fillna(0)
    C, m = R[v > 0].mean(), 100
    wr = ((v / (v + m)) * R + (m / (v + m)) * C).to_numpy()
    app = build_app_meta(df, meta, labels, wr, len(E))

    # ---- 3. rows that must never be recommended (cell 115) ---------------------------------
    key = df.title.str.lower().str.strip() + "|" + df.release_year.astype("Int64").astype(str)
    dup = key.loc[df.vote_count.sort_values(ascending=False).index].duplicated().reindex(df.index)
    compil = df.title.str.contains(
        r"trilogy|quadrilogy|box set|double feature|(?:complete|movie|film) (?:saga|series|collection)",
        case=False, regex=True)
    ok_mask = ~(dup | compil).to_numpy()
    log(f"duplicates: {int(dup.sum())} | compilations: {int(compil.sum())} | usable movies: {int(ok_mask.sum())}")

    # ---- 4. rankers (cells 118, 120, 127) --------------------------------------------------
    rng = np.random.default_rng(42)
    ctx = build_training_data(df, app, blocks, ok_mask, E, rng, args.jobs, n_seeds=args.seeds)
    rankers = train_rankers(ctx, args.jobs)
    if not args.no_eval:
        evaluate(ctx, rankers, rng)

    # ---- 5. save --------------------------------------------------------------------------
    log("saving ...")
    np.savez_compressed(out("blocks.npz"), **blocks)
    joblib.dump(fitted, out("fitted.joblib"), compress=3)
    joblib.dump({"stack": rankers["stack"], "xgb": rankers["xgb"],
                 "features": ctx["feats"], "feat_blocks": FEAT_BLOCKS}, out("ranker.joblib"), compress=3)
    np.save(out("ok_mask.npy"), ok_mask)
    app.to_pickle(out("app_meta.pkl"))
    np.save(out("E_hybrid.npy"), E)
    labels.to_csv(out("labels.csv"), index=False)

    # ---- 6. the app's data file must line up with these rows -------------------------------
    data_file = os.path.join(ROOT, "data", "movies_clean.csv.gz")
    if os.path.exists(data_file) and not args.limit:
        app_movies = pd.read_csv(data_file, usecols=["id", "title"])
        same = (len(app_movies) == len(app)
                and (app_movies["id"].astype("int64").to_numpy() == app["id"].astype("int64").to_numpy()).all())
        log(f"check against {data_file}: {'OK, rows line up' if same else 'MISMATCH'}")
        if not same:
            log("  !! the app's data/movies_clean.csv.gz was built from a different df_sampled.csv;")
            log("  !! re-run `python scripts/prepare_data.py <the same df_sampled.csv>`")
            return 1
    else:
        log(f"note: {data_file} not found, skipped the row-order check")

    for name in ["blocks.npz", "fitted.joblib", "ranker.joblib", "ok_mask.npy", "app_meta.pkl",
                 "E_hybrid.npy", "labels.csv"]:
        p = out(name)
        log(f"  {name:<15} {os.path.getsize(p) / 1e6:9.1f} MB")
    log("done - restart the app and pick a model in the sidebar")
    return 0


def _find_data():
    """Look for df_sampled.csv in the usual places around the project."""
    for cand in [os.path.join(os.getcwd(), "df_sampled.csv"),
                 os.path.join(ROOT, "df_sampled.csv"),
                 os.path.join(ROOT, "data", "df_sampled.csv"),
                 os.path.join(ROOT, "..", "Final Project-Movies", "Notebooks", "df_sampled.csv"),
                 os.path.join(ROOT, "..", "Notebooks", "df_sampled.csv")]:
        if os.path.exists(cand):
            return os.path.abspath(cand)
    return None


if __name__ == "__main__":
    sys.exit(main())
