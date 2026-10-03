import math
import streamlit as st
from core import ui

movies, rec = ui.get_movies(), ui.current_rec()
PAGE = 15

st.title("🔎 Browse movies")

all_genres = sorted({g for gl in movies["genres_l"] for g in gl})
top_langs = movies["language"].value_counts().index.tolist()
years = movies["release_year"].dropna().astype(int)

with st.expander("Search & filters", expanded=True):
    c1, c2 = st.columns([2, 3])
    q = c1.text_input("Title contains", placeholder="e.g. godfather")
    genres = c2.multiselect("Genres (movie must have all)", all_genres)
    c3, c4, c5, c6 = st.columns(4)
    yr = c3.slider("Release year", int(years.min()), int(years.max()), (1950, int(years.max())))
    lang = c4.selectbox("Original language", ["Any"] + top_langs[:25])
    max_rt = c5.slider("Max runtime (min, 0 = any)", 0, 240, 0, step=10)
    min_rating = c6.slider("Min rating (weighted)", 0.0, 9.0, 0.0, step=0.5,
                           help="Ratings on very few votes are pulled toward the average.")
    c7, c8 = st.columns(2)
    sort = c7.selectbox("Sort by", ["Best rated", "Most popular", "Newest"])
    min_votes = c8.slider("Min votes", 0, 1000, 50, step=10)

keep = rec.filter_mask(genres=genres or None, year_range=yr, language=None if lang == "Any" else lang,
                       max_runtime=max_rt or None, min_rating=min_rating or None, min_votes=min_votes)
if q:
    keep &= movies["title"].str.contains(q, case=False, regex=False).to_numpy()
sub = movies[keep]
col = {"Best rated": "wr", "Most popular": "popularity", "Newest": "release_year"}[sort]
sub = sub.sort_values(col, ascending=False)

pages = max(1, math.ceil(len(sub) / PAGE))
st.caption(f"{len(sub):,} movies match")
page = st.number_input("Page", 1, pages, 1) if pages > 1 else 1
view = sub.iloc[(page - 1) * PAGE: page * PAGE]
if view.empty:
    st.warning("No movies match these filters. Try loosening them.")
else:
    ui.movie_grid(view, key=f"browse{page}")
