The folder chatbot files contains a backend and frontend folder to use the Deltek chatbot with authentication. 
 - backend/appv2.py: creates endpoints to use deltek chatbot and authenticate users
 - frontend: contains scripts to create the chatbot page



 To run on your local server: 
 - see requirements.txt to download necessary packages with pip
 - create a new terminal and run uvicorn appv2:app --reload
 - make sure data.db is in the same directory as where you are running the app, so the app has access to the     database
 - create another terminal and run streamlit run web.py
 - login or create an account
