
import os
import pyodbc, struct
from azure import identity

from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel



connection_string = 'Driver={ODBC Driver 18 for SQL Server};Server=tcp:mysqlserverfordeltek.database.windows.net,1433;Database=auth_database;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30'


# class User(Base):
#     __tablename__ = "users"
#     id = Column(String, primary_key=True, index=True)
#     email = Column(String, unique=True, index=True)
#     password = Column(String)

class User(BaseModel):
    email: str
    password: str
    id: str | None
    
conn = None 
def get_conn():
    credential = identity.DefaultAzureCredential(exclude_interactive_browser_credential=False)
    token_bytes = credential.get_token("https://database.windows.net/.default").token.encode("UTF-16-LE")
    token_struct = struct.pack(f'<I{len(token_bytes)}s', len(token_bytes), token_bytes)
    SQL_COPT_SS_ACCESS_TOKEN = 1256  # This connection option is defined by microsoft in msodbcsql.h
    global conn
    conn = pyodbc.connect(connection_string, attrs_before={SQL_COPT_SS_ACCESS_TOKEN: token_struct})
    return conn
def get_db_conn():
    return conn
def close_conn():
    if conn:
        conn.close()

# Base.metadata.create_all(bind=engine)


