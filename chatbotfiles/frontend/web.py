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

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            
    url = "http://127.0.0.1:8000/generate"
    query = st.chat_input("Ask me something about Deltek", key = "chat_input")
    
    if query:
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        headers = {"Authorization": f"Bearer {st.session_state['access_token']}"}
        response = requests.post(url=url, json={"input": query}, headers=headers)
        answer = response.json()
        with st.chat_message("assistant"):
            st.markdown(answer['answer'])
        st.session_state.messages.append({"role": "assistant", "content": answer['answer']})
        # st.write(answer['answer'])


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
            code = create(newuser, newpass)
            if code == 200:
                st.success("Account created")
                placeholder.empty()  # Clear the registration form
            elif code == 400:
                st.error(f"Account already exists ")

if "access_token" in st.session_state:
    protected()
