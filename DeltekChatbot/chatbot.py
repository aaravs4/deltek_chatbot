from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# Ensure proper device handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Small open-source LLM
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device=device)

# Open-source embedding model
embeddings = HuggingFaceBgeEmbeddings()

def load_docs():
    bs4_strainer = bs4.SoupStrainer(class_=("main-content"))
    loader = WebBaseLoader(
        web_paths=(
            "https://deltek.com/en",  # Ensure this URL is valid
        ),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    return all_splits

def getDocs(query, all_splits, embeddings):
    db = FAISS.from_documents(all_splits, embeddings)
    relevant_docs = db.similarity_search(query, k=5)
    formatted_docs = '\n'.join(doc.page_content for doc in relevant_docs)
    return formatted_docs

def getOutput(query, context):
    messages = [
        {
            "role": "system",
            "content": f"You are a friendly chatbot who responds about queries related to Deltek from {context}. You will not make up answers."
        },
        {"role": "user", "content": query},
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    start_out = len(prompt)
    return outputs[0]["generated_text"][start_out:]

queries = []
splits = load_docs()

results = []
for query in queries:
    rel_docs = getDocs(query, splits, embeddings)
    output = getOutput(query, rel_docs)
    result = {"query": query, "response": output}
    results.append(result)

app = FastAPI()

@app.get('/')
def root():
    return {"RESTful API": "Chatbot"}

@app.get('/api/response')
async def get_results():
    return results

class QueryRequest(BaseModel):
    query: str

@app.post('/api/response')  # lets the user ask their own questions
async def new_query(query_request: QueryRequest):
    query = query_request.query
    rel_docs = getDocs(query, splits, embeddings)
    output = getOutput(query, rel_docs)
    result = {"query": query, "response": output}
    results.append(result)
    return result

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
