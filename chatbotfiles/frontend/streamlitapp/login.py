import streamlit as st
import requests

st.markdown("# Login Page")
st.sidebar.markdown("# Login Page")

def login(username, password):
    url_login = "http://127.0.0.1:8000/login"
    data = {
        "username":username,
        "password":password
    }
    response = requests.post(url = url_login, data = data)
    return response.json()


# def protected():
#     if "access_token" not in st.session_state:
#         st.error("login")
#         return
    
#     st.title('Deltek Chatbot')
#     url = "http://127.0.0.1:8000/generate"
#     query = st.chat_input("Ask me a question about Deltek")
#     if(query):
#         response = requests.post(url = url, json = {"input": query})
#         answer = response.json()
#         # st.text_area(label = '', value = answer['answer'])
#         st.write(answer['answer'])



username = st.text_input("Email")
password  = st.text_input("Password")
if(st.button("Login")):
    log = login(username, password)
    if "access_token" in log:
        st.success("Login successful")
        st.session_state["access_token"] = log["access_token"]
        st.session_state["refresh_token"] = log.get("refresh_token")
        st.write("Successful login: navigate to chatbot page")
    else:
        
        st.write(log)

    if "access_token" in st.session_state:
        st.write("You are already logged in!")
    
    


