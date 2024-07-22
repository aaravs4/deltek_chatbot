from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import torch
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

##load models
question_answerer = pipeline("question-answering", model='distilbert-base-uncased-distilled-squad')
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")
model = SentenceTransformer('all-MiniLM-L6-v2') 

##process website
bs4_strainer = bs4.SoupStrainer(class_=("main-content"))
loader = WebBaseLoader(
    web_paths=("https://deltek.com/en","https://www.deltek.com/en/about/contact-us", "https://www.deltek.com/en/small-business", "https://www.deltek.com/en/customers",
            "https://www.deltek.com/en/support", "https://www.deltek.com/en/partners"),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
all_splits_text = [split.page_content for split in all_splits]

##make fastapi app
app = FastAPI()

class userinput(BaseModel):
    input: str
class response(BaseModel):
    answer: str


def getDocsfaster(query, all_splits_text, model, k):
    doc_embeddings = model.encode(all_splits_text)
    query_embeddings = model.encode(query)
    results = cosine_similarity(doc_embeddings, query_embeddings.reshape(1,-1)).reshape((-1,))
    ixs = results.argsort()
    ixs = ixs[::-1]
    relevant_docs = []

    for i in ixs:
        relevant_docs.append(all_splits[i].page_content)
    relevant_docs = relevant_docs[:k]
    formatted_docs = "\n".join(doc for doc in relevant_docs)
    return formatted_docs

def getoutput(query, context):
    result = question_answerer(question= query, context=context)
    return result['answer']

@app.get("/")
def hello():
    return "hello"

@app.post("/generate", response_model=response)
async def generate_something(query: userinput):
    query = query.input
    context = getDocsfaster(query, all_splits_text, model, k = 5)
    output = getoutput(query, context)
    return(response(answer=output))




