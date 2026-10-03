import streamlit as st
from core import ui

movies, engine, rec = ui.get_movies(), ui.get_engine(), ui.current_rec()
by_id = movies.set_index("id", drop=False)
ss = st.session_state

st.title("💬 Movie chatbot")
st.caption(f"Offline, rule-based, English only · answering from the dataset · model: "
           f"**{ss.get('model_name', 'Baseline (TF-IDF)')}**. Type **help** for what I understand.")

if "chat" not in ss:
    ss.chat = [{"role": "assistant", "ids": [],
                "content": "Hi! 👋 Ask me for a recommendation, or ask about a movie — for example "
                           "*something like Sherlock Jr. but newer*, *what is Inception about*, or "
                           "*is Se7en good*. I can also *compare* two films and explain *why* I picked one."}]
if "chat_ctx" not in ss:                 # conversation memory: last request, last results, seeds
    ss.chat_ctx = {}

for i, msg in enumerate(ss.chat):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        ids = [x for x in msg.get("ids", []) if x in by_id.index]
        if ids:
            ui.movie_grid(by_id.loc[ids], key=f"chat{i}", numbered=True)

if prompt := st.chat_input("What do you feel like watching?"):
    last_ids = next((m["ids"] for m in reversed(ss.chat) if m["role"] == "assistant" and m.get("ids")), [])
    reply = engine.handle(prompt, rec, ui.username(), last_ids, ctx=ss.chat_ctx)
    ss.chat.append({"role": "user", "content": prompt})
    ss.chat.append({"role": "assistant", "content": reply.text, "ids": reply.ids})
    st.rerun()

if len(ss.chat) > 1 and st.button("Clear chat"):
    del ss["chat"]
    ss["chat_ctx"] = {}
    st.rerun()
