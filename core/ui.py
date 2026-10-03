"""Streamlit helpers shared by all pages: cached loaders, sidebar, movie cards."""
import pandas as pd
import streamlit as st

from core import storage
from core.data import load_movies
from recommender.registry import build_models
from chatbot.engine import ChatEngine

POSTER = "https://image.tmdb.org/t/p/w342"


@st.cache_resource(show_spinner="Loading movies…")
def get_movies():
    return load_movies()


@st.cache_resource(show_spinner="Preparing the recommendation models (first run only)…")
def get_models():
    return build_models(get_movies())


@st.cache_resource
def get_engine():
    return ChatEngine(get_movies())


def current_rec():
    models = get_models()
    name = st.session_state.get("model_name")
    return models.get(name) or next(iter(models.values()))


def username():
    return st.session_state.get("username")


def user_lists():
    """(favorite ids, watched ids) of the logged-in user, as sets."""
    u = username()
    if not u:
        return set(), set()
    return set(storage.get_ids(u, "favorites")), set(storage.get_ids(u, "watched"))


# ------------------------------------------------------------------ sidebar
def sidebar():
    ss = st.session_state
    with st.sidebar:
        st.markdown("## 🎬 CineMatch")
        if ss.get("username"):
            st.success(f"Logged in as **{ss.username}**")
        else:
            st.info("Browsing as **guest**. Log in to save favorites and watched movies.")
        models = list(get_models())
        if "model_name" not in ss or ss.model_name not in models:
            ss.model_name = models[0]
        st.selectbox("Recommendation model", models, key="model_name",
                     help="New models from your teammates appear here automatically.")
        if ss.get("username"):
            if st.button("Log out"):
                for k in ("username", "mode", "chat"):
                    ss.pop(k, None)
                st.rerun()
        else:
            if st.button("Log in / Sign up"):
                ss.pop("mode", None)
                st.rerun()


# ------------------------------------------------------------------ cards
def _fmt_runtime(r):
    return f"{int(r)} min" if pd.notna(r) else "?"


def movie_card(m, key, rank=None):
    """One movie: poster, title, rating, details popover, and (if logged in) save buttons."""
    mid = int(m["id"])
    pp = m.get("poster_path")
    if isinstance(pp, str) and pp:
        st.image(POSTER + pp)
    else:
        st.markdown("<div style='height:180px;background:#2b2b3a;border-radius:8px;display:flex;"
                    "align-items:center;justify-content:center;font-size:40px'>🎬</div>",
                    unsafe_allow_html=True)
    head = f"**{rank}. {m['title']}**" if rank else f"**{m['title']}**"
    year = m["release_year"] if pd.notna(m["release_year"]) else "?"
    st.markdown(f"{head} ({year})")
    st.caption(f"⭐ {m['vote_average']:.1f} · {_fmt_runtime(m['runtime'])} · {m['genres']}")
    with st.popover("Details"):
        st.markdown(f"**{m['title']}** ({year}) · {m['language']}")
        if m["directors"]:
            st.markdown(f"**Director:** {m['directors']}")
        if m["cast"]:
            st.markdown("**Cast:** " + ", ".join(m["cast"].split(", ")[:6]))
        st.write(m["overview"] or "No overview available.")

    u = username()
    if u:
        favs, seen = user_lists()
        c1, c2 = st.columns(2)
        if c1.button("💔 Remove" if mid in favs else "❤️ Favorite", key=f"fav_{key}_{mid}"):
            storage.toggle(u, "favorites", mid)
            st.rerun()
        if c2.button("↩️ Unwatch" if mid in seen else "✔️ Watched", key=f"seen_{key}_{mid}"):
            storage.toggle(u, "watched", mid)
            st.rerun()
        tags = []
        if mid in favs:
            tags.append("❤️ favorite")
        if mid in seen:
            tags.append("✔️ watched")
        if tags:
            st.caption(" · ".join(tags))
    else:
        st.caption("🔒 Log in to save")


def movie_grid(df, key, cols=5, numbered=False):
    """Show a DataFrame of movies as rows of cards."""
    rows = list(df.reset_index(drop=True).iterrows())
    for start in range(0, len(rows), cols):
        columns = st.columns(cols)
        for col, (i, m) in zip(columns, rows[start:start + cols]):
            with col:
                movie_card(m, key, rank=i + 1 if numbered else None)
