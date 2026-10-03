# Helpers shared by the notebook and the app (the saved vectorizers point to tok in this file).
# Copied unchanged from the notebook cell `%%writefile recsys_utils.py`
# (movies_insights_V7.ipynb, section "Saving everything for the app").
# IMPORTANT: models/fitted.joblib stores TfidfVectorizer objects whose tokenizer is
# `recsys_utils.tok`, so this file must stay at the project root and keep working.
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

def tok(s):                       # same tokenizer as the notebook
    return [t.strip().lower().replace(" ", "_") for t in s.split(",") if t.strip()]

def embed_svd(vec, svd, series):  # TF-IDF -> SVD -> unit length (tags, people and the TF-IDF text fallback)
    return normalize(svd.transform(vec.transform(series))).astype("float32")

def embed_genres(cv, series):
    return normalize(cv.transform(series).toarray().astype("float32"))

def _z(x, mu, sd):
    return np.nan_to_num((np.asarray(x, float) - mu) / sd, nan=0.0)

def embed_meta(f, p):             # f needs release_year, runtime, budget, original_language
    lang = f["original_language"].fillna("other")
    lang = lang.where(lang.isin(p["lang_cols"]), "other")
    cols = [_z(f["release_year"], *p["year"]), _z(np.log1p(f["runtime"]), *p["runtime"]),
            _z(np.log1p(f["budget"]), *p["budget"])]
    cols += [(lang == l).to_numpy(float) for l in p["lang_cols"]]
    return normalize(np.column_stack(cols)).astype("float32")
