import streamlit as st
import bcrypt
import sqlite3
import requests

# Authentication manager class
class AuthManager:
    def __init__(self, db_path='users.db'):
        self.db_path = db_path
        self.create_users_table()

    def create_users_table(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                access BOOLEAN NOT NULL DEFAULT 0
            )
        ''')
        conn.commit()
        conn.close()

    def register_user(self, username, password):
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            c.execute('INSERT INTO users (username, password, access) VALUES (?, ?, ?)', (username, hashed_password, 0))
            conn.commit()
            st.success("User registered successfully.")
        except sqlite3.IntegrityError:
            st.error("Username already exists.")
        conn.close()

    def check_credentials(self, username, password):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT password, access FROM users WHERE username = ?', (username,))
        user_data = c.fetchone()
        conn.close()
        if user_data and bcrypt.checkpw(password.encode('utf-8'), user_data[0]) and user_data[1] == 1:
            return True
        return False

# Login page class
class LoginPage:
    def __init__(self, auth_manager):
        self.auth_manager = auth_manager

    def display_login(self):
        st.title("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if self.auth_manager.check_credentials(username, password):
                st.session_state['logged_in'] = True
                st.success("Login successful")
                st.experimental_rerun()  # Rerun the app to reflect the new state
            else:
                st.error("Invalid username, password, or access denied.")

    def display(self):
        if 'logged_in' not in st.session_state:
            st.session_state['logged_in'] = False
        if st.session_state['logged_in']:
            st.write("Welcome to the AI chatbot!")
            display_chatbot()  # Call to display the chatbot
        else:
            self.display_login()

def display_chatbot():
    st.header("AI Chatbot")
    query = st.text_input("Ask a question:")
    if st.button("Enter"):
        response = get_chatbot_response(query)
        st.write("Response:", response)

def get_chatbot_response(query):
    url = "http://localhost:8000/api/response"
    payload = {"query": query}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json().get("response")
    else:
        return "Error: Unable to fetch response."

# Main entry point
if __name__ == '__main__':  
    auth_manager = AuthManager()
    login_page = LoginPage(auth_manager)
    login_page.display()
