"""Build data/movies_clean.csv.gz from the raw df_sampled.csv.

Row order follows your notebook (drop duplicate tconst -> drop adult -> reset index),
so row i here is row i of E_hybrid.npy / app_meta.pkl.

Usage:  python scripts/prepare_data.py path/to/df_sampled.csv
"""
import sys
import numpy as np
import pandas as pd

KEEP = ["id", "tconst", "title", "release_date", "release_year", "runtime", "vote_average",
        "vote_count", "averageRating", "numVotes", "popularity", "original_language",
        "overview", "tagline", "genres", "keywords", "directors", "cast",
        "production_countries", "poster_path", "budget", "revenue"]

def main(src, out="data/movies_clean.csv.gz"):
    df = pd.read_csv(src)
    df = df.drop_duplicates(subset="tconst", keep="first").reset_index(drop=True)
    df = df[~df["adult"]].reset_index(drop=True)          # same filter as the notebook
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year.astype("Int64")
    df["runtime"] = df["runtime"].where(df["runtime"] > 0)
    for c in ["budget", "revenue"]:
        df[c] = df[c].where(df[c] >= 1000)
    for c in ["overview", "tagline", "genres", "keywords", "directors", "cast",
              "production_countries"]:
        df[c] = df[c].fillna("").astype(str)
    df[KEEP].to_csv(out, index=False, compression="gzip")
    print("saved", out, df[KEEP].shape)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "df_sampled.csv")
