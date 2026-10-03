"""Load the cleaned movie table and add the helper columns the app needs."""
import re
import numpy as np
import pandas as pd

LANG_NAMES = {
    "en": "English", "fr": "French", "es": "Spanish", "de": "German", "it": "Italian",
    "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "cn": "Cantonese", "hi": "Hindi",
    "ru": "Russian", "pt": "Portuguese", "sv": "Swedish", "da": "Danish", "no": "Norwegian",
    "nb": "Norwegian", "fi": "Finnish", "nl": "Dutch", "pl": "Polish", "tr": "Turkish",
    "ar": "Arabic", "he": "Hebrew", "fa": "Persian", "th": "Thai", "id": "Indonesian",
    "cs": "Czech", "hu": "Hungarian", "ro": "Romanian", "el": "Greek", "uk": "Ukrainian",
    "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "ml": "Malayalam", "ur": "Urdu",
    "xx": "No language"}

def norm_title(s: str) -> str:
    """Lower-case, strip punctuation, drop a leading 'the' so 'The Matrix' == 'matrix'."""
    s = re.sub(r"[^a-z0-9 ]", " ", str(s).lower())
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"^(the|a|an) ", "", s)

def split_list(s: str):
    return [t.strip() for t in str(s).split(",") if t.strip()]

def load_movies(path="data/movies_clean.csv.gz", min_wr_votes=100) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["overview", "tagline", "genres", "keywords", "directors", "cast"]:
        df[c] = df[c].fillna("").astype(str)
    df["release_year"] = df["release_year"].astype("Int64")
    df["genres_l"] = df["genres"].map(split_list)
    df["language"] = df["original_language"].map(LANG_NAMES).fillna(df["original_language"].str.upper())
    df["label"] = df["title"] + " (" + df["release_year"].astype(str).replace("<NA>", "?") + ")"
    df["title_norm"] = df["title"].map(norm_title)
    # IMDb-style weighted rating: shrinks ratings that rest on few votes
    v, R = df["vote_count"].fillna(0), df["vote_average"].fillna(0)
    C = R[v > 0].mean()
    df["wr"] = (v / (v + min_wr_votes)) * R + (min_wr_votes / (v + min_wr_votes)) * C
    return df.reset_index(drop=True)
