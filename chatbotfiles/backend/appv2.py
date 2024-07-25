from fastapi import FastAPI, HTTPException,status, Depends
from fastapi.responses import RedirectResponse
from uuid import uuid4
from pydantic import BaseModel
from transformers import pipeline
import torch
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils import (
    get_hashed_password,
    create_access_token,
    create_refresh_token,
    verify_password
)
from fastapi.security import OAuth2PasswordRequestForm
# from sqlalchemy.orm import Session
from models import User as DbUser, get_conn
from deps import get_current_user
conn = get_conn()



class UserOut(BaseModel):
    email: str
    password:str
    id:str | None


class UserAuth(BaseModel):
    email:str
    password:str
    id:str | None

class TokenSchema(BaseModel):
    access_token:str
    refresh_token:str


# Base.metadata.create_all(bind=engine)


class document_processer:
    def __init__(self, urls):
        self.model = SentenceTransformer('all-MiniLM-L6-v2') 
        self.bs4_strainer = bs4.SoupStrainer(class_=("main-content"))
        self.loader = WebBaseLoader(
            # web_paths=("https://deltek.com/en","https://www.deltek.com/en/about/contact-us", "https://www.deltek.com/en/small-business", "https://www.deltek.com/en/customers",
            #         "https://www.deltek.com/en/support", "https://www.deltek.com/en/partners"),
            web_paths = urls,
            bs_kwargs={"parse_only": self.bs4_strainer},
        )
        self.docs = self.loader.load()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        self.all_splits = self.text_splitter.split_documents(self.docs)
        self.all_splits_text = [split.page_content for split in self.all_splits]
    def getDocsfaster(self, query, k):
        doc_embeddings = self.model.encode(self.all_splits_text)
        query_embeddings = self.model.encode(query)
        results = cosine_similarity(doc_embeddings, query_embeddings.reshape(1,-1)).reshape((-1,))
        ixs = results.argsort()
        ixs = ixs[::-1]
        relevant_docs = []
        for i in ixs:
            relevant_docs.append(self.all_splits[i].page_content)
        relevant_docs = relevant_docs[:k]
        formatted_docs = "\n".join(doc for doc in relevant_docs)
        return formatted_docs
            
class query_processor:
    def __init__(self):
        self.question_answerer = pipeline("question-answering", model='distilbert-base-uncased-distilled-squad')
        self.pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")
    def getoutput(self, query, context):
        result = self.question_answerer(question= query, context=context)
        return result['answer']

class chatapp:
    def __init__(self):
        self.app = FastAPI()
        self.document_processor = document_processer(("https://deltek.com/en","https://www.deltek.com/en/about/contact-us", "https://www.deltek.com/en/small-business", "https://www.deltek.com/en/customers",
                    "https://www.deltek.com/en/support", "https://www.deltek.com/en/partners"))
        self.query_processor = query_processor()
    
    def getendpoints(self):
        @self.app.get("/")
        def hello():
            return "hello"

        @self.app.post("/generate", response_model=response)
        async def generate_something(query: userinput, user: DbUser = Depends(get_current_user)):
            query = query.input
            context = self.document_processor.getDocsfaster(query, k = 5)
            output = self.query_processor.getoutput(query, context)
            return(response(answer=output))
        @self.app.post('/signup', summary="Create new user", response_model=UserOut)
        def create_user(data: UserAuth):
            # querying database to check if user already exist
            # user = db.query(DbUser).filter(DbUser.email == data.email).first()
            # if user is not None:
            #         raise HTTPException(
            #         status_code=status.HTTP_400_BAD_REQUEST,
            #         detail="User with this email already exist"
            #     )
            # new_user = DbUser(
            #     email=data.email,
            #     password=get_hashed_password(data.password),
            #     id=str(uuid4())
            # )
            # db.add(new_user)
            # db.commit()
            # db.refresh(new_user)
            
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM Persons WHERE FirstName = ?", data.email)

            row = cursor.fetchone()
            if(row != None):
                raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exist"
                )
            else:
                new_user = DbUser(
                    email = data.email,
                    password = get_hashed_password(data.password),
                    id = str(uuid4())
                )
                cursor = conn.cursor()
                cursor.execute(f"INSERT INTO Persons (FirstName, LastName, UserID )VALUES (?, ?, ?)", new_user.email, new_user.password, new_user.id)
                conn.commit()

            return new_user
        @self.app.post('/login', summary="Create access and refresh tokens for user", response_model=TokenSchema)
        def login(form_data: OAuth2PasswordRequestForm = Depends()):
            # user = db.query(DbUser).filter(DbUser.email == form_data.username).first()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM Persons WHERE FirstName = ?", form_data.username)
            row = cursor.fetchone()
            
            if not row or not verify_password(form_data.password, row.LastName):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Incorrect email or password"
                )

            return {
                "access_token": create_access_token(row.FirstName),
                "refresh_token": create_refresh_token(row.LastName),
            }
        @app.get('/me', summary='Get details of currently logged in user', response_model=UserOut)
        def get_me(user: DbUser = Depends(get_current_user)):
            return user

class userinput(BaseModel):
    input: str
class response(BaseModel):
    answer: str

mainapp = chatapp()
app = mainapp.app
mainapp.getendpoints()










