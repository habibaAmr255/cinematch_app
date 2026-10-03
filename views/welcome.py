import streamlit as st
from core import storage

st.title("🎬 CineMatch")
st.write("Find your next movie. Log in to keep a favorites list and a watched list, "
         "or just browse as a guest.")

tab_login, tab_signup, tab_guest = st.tabs(["Log in", "Sign up", "Continue as guest"])

with tab_login:
    with st.form("login_form"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.form_submit_button("Log in"):
            if storage.check_login(u, p):
                st.session_state.update(mode="user", username=u.strip())
                st.rerun()
            else:
                st.error("Wrong username or password.")

with tab_signup:
    with st.form("signup_form"):
        u2 = st.text_input("Choose a username", help="3-20 letters, numbers or underscore")
        p2 = st.text_input("Choose a password", type="password", help="At least 6 characters")
        p3 = st.text_input("Repeat the password", type="password")
        if st.form_submit_button("Create account"):
            if p2 != p3:
                st.error("The passwords don't match.")
            else:
                ok, msg = storage.create_user(u2, p2)
                if ok:
                    st.session_state.update(mode="user", username=u2.strip())
                    st.rerun()
                else:
                    st.error(msg)

with tab_guest:
    st.write("Browse, search and get recommendations. You just can't save favorites or watched movies.")
    if st.button("Continue as guest"):
        st.session_state.update(mode="guest", username=None)
        st.rerun()
