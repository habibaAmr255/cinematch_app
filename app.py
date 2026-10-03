"""CineMatch: Streamlit entry point.   Run with:  streamlit run app.py"""
import streamlit as st

from core import storage, ui

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")
storage.init_db()

if not st.session_state.get("mode"):                        # not logged in and not a guest yet
    nav = st.navigation([st.Page("views/welcome.py", title="Welcome", icon="🎬")], position="hidden")
else:
    ui.sidebar()
    nav = st.navigation([
        st.Page("views/browse.py", title="Browse", icon="🔎", default=True),
        st.Page("views/similar.py", title="Similar movies", icon="🎯"),
        st.Page("views/chatbot.py", title="Chatbot", icon="💬"),
        st.Page("views/my_lists.py", title="My lists", icon="❤️"),
        st.Page("views/insights.py", title="Insights", icon="📊"),
    ])
nav.run()
