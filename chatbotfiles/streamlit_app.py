import streamlit as st
import requests

st.title('Deltek Chatbot')
url = "http://127.0.0.1:8000/generate"


# st.text_input("Your question: ", key = "input")
# query = st.session_state.input
# button = st.button("Send")
# if(button):
#     response = requests.post(url = url, json = {"input": query})
#     answer = response.json()
#     st.text_area(label = 'Answer: ', value = answer['answer'])

query = st.chat_input("Ask me a question about Deltek")
if(query):
    response = requests.post(url = url, json = {"input": query})
    answer = response.json()
    # st.text_area(label = '', value = answer['answer'])
    st.write(answer['answer'])


