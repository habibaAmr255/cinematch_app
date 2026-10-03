import streamlit as st
from core import storage, ui

st.title("❤️ My lists")
u = ui.username()
if not u:
    st.info("Lists are saved to your account. Log in or sign up to use them.")
    if st.button("Go to log in"):
        st.session_state.pop("mode", None)
        st.rerun()
    st.stop()

movies = ui.get_movies()
by_id = movies.set_index("id", drop=False)
fav_ids = storage.get_ids(u, "favorites")
seen_ids = storage.get_ids(u, "watched")
tab_f, tab_w, tab_r = st.tabs([f"❤️ Favorites ({len(fav_ids)})", f"✔️ Watched ({len(seen_ids)})", "✨ For you"])

with tab_f:
    sub = by_id.loc[[i for i in fav_ids if i in by_id.index]]
    if len(sub):
        ui.movie_grid(sub, "fav")
    else:
        st.write("No favorites yet. Add some from Browse or the chatbot.")
with tab_w:
    sub = by_id.loc[[i for i in seen_ids if i in by_id.index]]
    if len(sub):
        ui.movie_grid(sub, "seen")
    else:
        st.write("Nothing marked as watched yet.")
with tab_r:
    rows = [int(movies.index[movies["id"] == i][0]) for i in fav_ids[:8] if i in by_id.index]
    if not rows:
        st.write("Add a few favorites and we'll recommend more like them.")
    else:
        st.caption(f"Based on your latest favorites, using: {st.session_state.get('model_name')}")
        ui.movie_grid(ui.current_rec().recommend_similar(rows, n=10), "foryou")
