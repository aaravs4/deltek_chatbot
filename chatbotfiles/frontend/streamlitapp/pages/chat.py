import streamlit as st
import requests

st.markdown("# Deltek Chatbot")
st.sidebar.markdown("# Virtual Assistant")

if "access_token" not in st.session_state:
    st.error("go login")
    
else:   
    # st.title('Deltek Chatbot')
    url = "http://127.0.0.1:8000/generate"
    query = st.chat_input("Ask me a question about Deltek")
    if(query):
        headers = {
            "Authorization": f"Bearer {st.session_state['access_token']}"
        }
        response = requests.post(url = url, json = {"input": query},headers=headers)
        answer = response.json()
        # st.text_area(label = '', value = answer['answer'])
        # st.write(answer)
        st.write(answer['answer'])
