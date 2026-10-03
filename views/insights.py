"""Insights page: the main charts from the insights notebook, computed from the cleaned data."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from core import ui

sns.set_theme(style="whitegrid")
MIN_VOTES = 100                      # same reliability threshold as the notebook


@st.cache_data(show_spinner="Computing insights…")
def prep():
    df = ui.get_movies().copy()
    df["decade"] = (df["release_year"].astype("float") // 10 * 10)
    df["is_short"] = df["runtime"] < 40
    rel = df[df["vote_count"] >= MIN_VOTES].copy()
    ex = rel.explode("genres_l").rename(columns={"genres_l": "genre"})
    return df, rel, ex


df, rel, ex = prep()
st.title("📊 Insights")
a, b, c, d = st.columns(4)
a.metric("Movies", f"{len(df):,}")
b.metric("With 100+ votes", f"{len(rel):,}", f"{len(rel) / len(df):.0%} of all")
c.metric("Years covered", f"{int(df.release_year.min())}–{int(df.release_year.max())}")
d.metric("English share", f"{(df.original_language == 'en').mean():.0%}")
st.caption("Rating-based charts use only movies with at least 100 votes, because ratings on few votes are unreliable.")

t1, t2, t3, t4 = st.tabs(["Ratings & time", "Genres", "People & money", "Key findings"])

with t1:
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        df["decade"].dropna().astype(int).value_counts().sort_index().plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title("Movies per decade"); ax.set_xlabel("")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        rel.groupby("decade")["vote_average"].mean().plot(ax=ax, marker="o", color="darkorange")
        ax.set_title("Average rating per decade (100+ votes)"); ax.set_xlabel("")
        st.pyplot(fig)
    c3, c4 = st.columns(2)
    with c3:
        bins = pd.cut(df["vote_count"], [0, 20, 50, 100, 500, np.inf], right=False,
                      labels=["<20", "20-49", "50-99", "100-499", "500+"])
        t = df.groupby(bins, observed=True)["vote_average"].mean()
        fig, ax = plt.subplots(figsize=(6, 3.6))
        sns.barplot(x=t.index.astype(str), y=t.values, color="seagreen", ax=ax)
        ax.set_ylim(5, 7.5); ax.set_title("Average rating by number of votes"); ax.set_xlabel("vote count")
        st.pyplot(fig)
    with c4:
        r = rel.dropna(subset=["runtime"]).copy()
        r["runtime_bin"] = pd.cut(r["runtime"], [0, 80, 90, 100, 120, 140, 400],
                                  labels=["<80", "80-89", "90-99", "100-119", "120-139", "140+"])
        t = r.groupby("runtime_bin", observed=True)["vote_average"].mean()
        fig, ax = plt.subplots(figsize=(6, 3.6))
        sns.barplot(x=t.index.astype(str), y=t.values, color="mediumpurple", ax=ax)
        ax.set_ylim(5.5, 7.5); ax.set_title("Average rating by runtime (minutes)"); ax.set_xlabel("")
        st.pyplot(fig)

with t2:
    g = ex.groupby("genre").agg(n=("id", "count"), rating=("vote_average", "mean")).sort_values("rating")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 5))
        g["n"].sort_values().plot(kind="barh", ax=ax, color="steelblue")
        ax.set_title("Movies per genre (100+ votes)"); ax.set_ylabel("")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6, 5))
        g["rating"].plot(kind="barh", ax=ax, color="darkorange")
        ax.set_xlim(5.5, 7.5); ax.set_title("Average rating per genre"); ax.set_ylabel("")
        st.pyplot(fig)
    top_lang = df["language"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 3.4))
    sns.barplot(x=top_lang.index, y=top_lang.values, color="steelblue", ax=ax)
    ax.set_title("Top 10 original languages"); ax.set_xlabel("")
    st.pyplot(fig)

with t3:
    c1, c2 = st.columns(2)
    with c1:
        dd = rel.assign(director=rel["directors"].str.split(", ")).explode("director")
        t = dd.groupby("director").agg(n=("id", "count"), rating=("vote_average", "mean"))
        t = t[t["n"] >= 5].sort_values("rating", ascending=False).head(12).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6, 4.4))
        ax.barh(t.index, t["rating"], color="seagreen")
        ax.set_xlim(6.5, 9); ax.set_title("Best-rated directors (5+ movies with 100+ votes)")
        st.pyplot(fig)
    with c2:
        m = df.dropna(subset=["budget", "revenue"])
        fig, ax = plt.subplots(figsize=(6, 4.4))
        ax.scatter(m["budget"], m["revenue"], s=6, alpha=0.3, color="indianred")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("budget"); ax.set_ylabel("revenue")
        ax.set_title(f"Budget vs revenue ({len(m):,} movies with both recorded)")
        st.pyplot(fig)

with t4:
    st.markdown("""
**Ratings and quality**
- Documentary, Animation, History and War are the best-rated genres; Horror, Thriller and Sci-Fi the worst.
- Average ratings fall from about 7.1–7.3 in early decades to about 6.4 in the 2000s–2010s.
- Longer movies are rated higher (7.2 for 140+ minutes vs 6.2 for 90–99).

**Money**
- Budget does not predict rating, but it does predict revenue. Only about 20% of movies have budget and revenue recorded.

**People and content**
- Famous casts bring many more votes and more popularity, but not higher ratings.
- Anime, film noir and silent themes score highest; slasher, found footage and spoof themes score lowest.

**Caveats**
- Only 27% of movies have 100+ votes. Everything here is association, not causation.
""")
    st.caption("Full analysis, with more charts, is in the insights notebook.")
