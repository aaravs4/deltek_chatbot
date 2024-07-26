import streamlit as st
import requests

def login(username, password):
    url_login = "http://127.0.0.1:8000/login"
    data = {
        "username": username,
        "password": password
    }
    response = requests.post(url=url_login, data=data)
    return response.json()

def create(username, password):
    url_create = "http://127.0.0.1:8000/signup"
    data = {
        "email": username,
        "password": password,
        "id": None  
    }
    response = requests.post(url=url_create, json=data)
    return response.status_code

def protected():
    if "access_token" not in st.session_state:
        st.error("Please log in to continue.")
        return
    
    st.title('Deltek Chatbot')
    url = "http://127.0.0.1:8000/generate"
    query = st.chat_input("Ask me something about Deltek", key = "chat_input")
    if query:
        headers = {"Authorization": f"Bearer {st.session_state['access_token']}"}
        response = requests.post(url=url, json={"input": query}, headers=headers)
        answer = response.json()
        st.write(answer['answer'])


placeholder = st.empty()

if "access_token" not in st.session_state:
    form_mode = st.sidebar.radio("Choose form", ["Login", "Register"])

    if form_mode == "Login":
        with placeholder.form("Login"):
            username = st.text_input("Email")
            password = st.text_input("Password", type="password")
            formbutton = st.form_submit_button("Login")

        if formbutton:
            log = login(username, password)
            if "access_token" in log:
                st.session_state["access_token"] = log["access_token"]
                st.session_state["refresh_token"] = log.get("refresh_token")
                st.success("Login successful")
                placeholder.empty()
                
            else:
                st.error(log)
    elif form_mode == "Register":
        with placeholder.form("Register"):
            newuser = st.text_input("New Email")
            newpass = st.text_input("New Password", type="password")
            createbutton = st.form_submit_button("Register")

        if createbutton:
            st.write(f"Creating account for {newuser}...")
            code = create(newuser, newpass)
            st.write(f"API response status code: {code}")
            if code == 200:
                st.success("Account Created")
                placeholder.empty()  # Clear the registration form
            else:
                st.error(f"Error: {code}")

if "access_token" in st.session_state:
    protected()
