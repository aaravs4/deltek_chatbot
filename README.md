The folder chatbot files contains a backend and frontend folder to use the Deltek chatbot with authentication. 
 - backend/appv2.py: creates endpoints to use deltek chatbot and authenticate users
 - frontend: contains scripts to create a login page and a chatbot page

To run, you need an Azure account and a configured SQL database. You should replace the variable in appv2.py called 'connection_string' with your database connection string. When configuring the database, you should allow your IP address access to the database under firewall settings.



 To run on your local server: 
 - see requirements.txt to download necessary packages with pip
 - replace connection string in appv2.py with your connection string from azure database
 - create a new terminal and run uvicorn appv2:app --reload
 - create another terminal and run streamlit run login.py
 - navigate to browser given by login.py 
