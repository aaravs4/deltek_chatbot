import streamlit as st
import requests

st.title('Deltek Chatbot')

st.text_input("Your question: ", key = "input")

query = st.session_state.input

button = st.button("Send")
url = "http://127.0.0.1:8000/generate"


if(button):
    response = requests.post(url = url, json = {"input": query})
    answer = response.json()
    st.text_area(label = 'Chatbot: ', value = answer['answer'])
