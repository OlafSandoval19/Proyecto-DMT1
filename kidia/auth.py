import streamlit as st

USERS = {
    "Olaf": {
        "password": "1999",
        "role": ""
    },
    "ITCG": {
        "password": "tecguzman123",
        "role": ""
    }
}

def login(username, password):
    if username in USERS and USERS[username]["password"] == password:
        st.session_state.authenticated = True
        st.session_state.user = username
        st.session_state.role = USERS[username]["role"]
        return True
    return False

def logout():
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.role = None
