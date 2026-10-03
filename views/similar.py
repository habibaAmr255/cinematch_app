import streamlit as st
from core import ui

movies = ui.get_movies()
st.title("🎯 Because you liked…")

q = st.text_input("Type a movie title", placeholder="e.g. Sherlock Jr.")
if not q:
    st.info("Start typing a title, then pick the right one from the list.")
    st.stop()

hits = movies[movies["title"].str.contains(q, case=False, regex=False)]
hits = hits.sort_values("vote_count", ascending=False).head(25)
if hits.empty:
    st.warning("No movie with that title in the dataset.")
    st.stop()

label = st.selectbox("Which one?", hits["label"].tolist())
row = int(hits.index[hits["label"] == label][0])
c1, c2, c3 = st.columns(3)
n = c1.slider("How many?", 5, 20, 10, step=5)
boost = c2.slider("Favor well-rated movies", 0.0, 0.5, 0.15, step=0.05)
compare = c3.checkbox("Compare all models", value=False)

models = ui.get_models()
if compare and len(models) > 1:
    for tab, (name, rec) in zip(st.tabs(list(models)), models.items()):
        with tab:
            ui.movie_grid(rec.recommend_similar([row], n=n, beta=boost), key=f"sim_{name}")
else:
    ui.movie_grid(ui.current_rec().recommend_similar([row], n=n, beta=boost), key="sim")
