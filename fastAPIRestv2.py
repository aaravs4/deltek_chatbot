from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from transformers import pipeline
import torch
import requests
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# Initialize the question-answering model
question_answerer = pipeline("question-answering", model='distilbert-base-uncased-distilled-squad')

# Open-source embedding model
model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}  # Set True to compute cosine similarity

embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction='Generate a representation for this sentence that can be used to retrieve related sentences:'
)

def loadDocs():
    bs4_strainer = bs4.SoupStrainer(class_="main-content")
    loader = WebBaseLoader(
        web_paths=[
            "https://deltek.com/en",
            "https://www.deltek.com/en/about/contact-us",
            "https://www.deltek.com/en/small-business",
            "https://www.deltek.com/en/customers",
            "https://www.deltek.com/en/support",
            "https://www.deltek.com/en/partners"
        ],
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    return all_splits

def get_relevant_docs(query, all_splits, embeddings):
    db = FAISS.from_documents(all_splits, embeddings)
    relevant_docs = db.similarity_search(query, k=5)
    formatted_docs = '\n'.join(doc.page_content for doc in relevant_docs)
    return formatted_docs

splits = loadDocs()


app = FastAPI()

results = []
@app.get('/')
def root():
    return {"RESTful API": "Question Answering"}

@app.get('/api/response')
async def get_results():
    return results

class QueryRequest(BaseModel):
    question: str

@app.post('/api/response')
async def new_query(query_request: QueryRequest):
    question = query_request.question
    context = get_relevant_docs(question, splits, embeddings)
    result = question_answerer(question=question, context=context)
    formatted_result = {
        "question": {question},
        "answer": {result['answer']},
        "score": {round(result['score'], 4)},
        "start": {result['start']},
        "end": {result['end']}
    }
    results.append(formatted_result)
    return formatted_result

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='localhost', port=5000)
