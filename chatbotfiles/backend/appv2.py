from fastapi import FastAPI, HTTPException,status, Depends
from fastapi.responses import RedirectResponse
from uuid import uuid4
from pydantic import BaseModel
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM  
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
from deps import get_current_user
import sqlite3
import re





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
    def is_complete_sentence(self, sentence):
        # A complete sentence ends with ., ?, or !
        return re.match(r'.*[\.\?\!]$', sentence.strip()) is not None
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

        # sentences = []
        # for doc in relevant_docs:
        #     segments = doc.split('\n')
        #     for segment in segments:
        #         # Split each segment into sentences using regex
        #         segment_sentences = re.split(r'(?<=[.!?]) +', segment)
        #         # Filter out incomplete sentences

        #         complete_sentences = [sentence for sentence in segment_sentences if self.is_complete_sentence(sentence)]
        #         sentences.extend(complete_sentences)


        # formatted_docs = "\n\n".join(sentence for sentence in sentences)
        formatted_docs = "\n".join(doc for doc in relevant_docs)
        return formatted_docs
        
            
class query_processor:
    def __init__(self):
        self.question_answerer = pipeline("question-answering", model='distilbert-base-uncased-distilled-squad')
        self.pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained("llmware/bling-1.4b-0.1")  
        self.model =  AutoModelForCausalLM.from_pretrained("llmware/bling-1.4b-0.1")  
    # def getoutput(self, query, context):
    #     result = self.question_answerer(question= query, context=context)
    #     return result['answer']
    def getoutput(self,query, context):
        entries = {"context": context, 
          "query":query}
        new_prompt = "<human>: " + entries["context"] + "\n" + entries["query"] + "\n" + "<bot>:"
        inputs = self.tokenizer(new_prompt, return_tensors="pt")  
        start_of_output = len(inputs.input_ids[0])
        outputs = self.model.generate(
        inputs.input_ids.to("cpu"),
        attention_mask=inputs.attention_mask.to("cpu"),
        eos_token_id=self.tokenizer.eos_token_id,
        pad_token_id=self.tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.3,
        max_new_tokens=100,
        )
        output_text = self.tokenizer.decode(outputs[0][start_of_output:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return output_text


class chatapp:
    def __init__(self):
        self.app = FastAPI()
        self.document_processor = document_processer(("https://deltek.com/en","https://www.deltek.com/en/about/contact-us", "https://www.deltek.com/en/small-business", "https://www.deltek.com/en/customers",
                    "https://www.deltek.com/en/support", "https://www.deltek.com/en/partners", "https://www.deltek.com/en/products"))
        self.query_processor = query_processor()
    
    def getendpoints(self):
        @self.app.get("/")
        def hello():
            return "hello"

        @self.app.post("/generate", response_model=response)
        async def generate_something(query: userinput, user: UserOut = Depends(get_current_user)):
            query = query.input
            context = self.document_processor.getDocsfaster(query, k = 5)
            output = self.query_processor.getoutput(query, context)
            return(response(answer=output))
        @self.app.post('/signup', summary="Create new user", response_model=UserOut)
        def create_user(data: UserAuth):
            conn = sqlite3.connect("data.db")
            cursor = conn.cursor()
           
            cursor.execute("SELECT * FROM Users_new WHERE username = ?", (data.email,))

            row = cursor.fetchone()
            if(row != None):
                conn.close()
                raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exist"
                )
            else:
                new_user = UserOut(
                    email = data.email,
                    password = get_hashed_password(data.password),
                    id = str(uuid4())
                )
                
                cursor.execute(f"INSERT INTO Users_new (username, password, id )VALUES (?, ?, ?)", (new_user.email, new_user.password, new_user.id))
                conn.commit()
                conn.close()

            return new_user
        @self.app.post('/login', summary="Create access and refresh tokens for user", response_model=TokenSchema)
        def login(form_data: OAuth2PasswordRequestForm = Depends()):
            # user = db.query(DbUser).filter(DbUser.email == form_data.username).first()
            conn = sqlite3.connect("data.db")
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM Users_new WHERE username = ?", (form_data.username,))
            row = cursor.fetchone()
           
            if not row or not verify_password(form_data.password, row[1]):
                conn.close()
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Incorrect email or password"
                )
            conn.close()
            return {
                "access_token": create_access_token(row[0]),
                "refresh_token": create_refresh_token(row[0]),
            }
        
        @app.get('/me', summary='Get details of currently logged in user', response_model=UserOut)
        def get_me(user: UserOut = Depends(get_current_user)):
            return user

class userinput(BaseModel):
    input: str
class response(BaseModel):
    answer: str

mainapp = chatapp()
app = mainapp.app
mainapp.getendpoints()













