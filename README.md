The folder chatbotfiles contains two python scripts that can be used to try the Deltek chatbot. 
 - app.py: loads necessary models and creates an api endpoint which can generate an output given a query
 - streamlit_app.py: creats a simple web page that has an input text box and calls the api endpoint to display the output

 To run on your local server: 
 - see requirements.txt to download necessary packages with pip
 - create a new terminal and run uvicorn app:app --reload
 - create another terminal and run streamlit run streamlit_app.py
 - navigate to browser given by streamlit_app.py 
