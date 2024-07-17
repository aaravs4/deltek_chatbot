
#app = FastAPI()

#@app.get("/")
#def root():
#    return{"Hello": "World"}
#import import_ipynb
#import tinyllama
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests

import torch
from transformers import pipeline
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

#small open source llm
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

##open source embedding model
model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction= 'Generate a representation for this sentence that can be used to retrieve related sentences:'
)

def load_docs():
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
    return all_splits
    # all_splits_text = [split.page_content for split in all_splits]



def getDocs(query, all_splits, embeddings):
    db = FAISS.from_documents(all_splits, embeddings)

    ##get relevant docs from vectorstore
    relevant_docs = db.similarity_search(query, k = 5)
    formatted_docs = '\n'.join(doc.page_content for doc in relevant_docs)

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

query1 = "Why is deltek trusted?"
query2 = "What does Deltek do?"
query3 = "How can I contact Deltek?"

queries = [query1, query2, query3]
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

@app.post('/api/response')#lets the user ask their own questions
async def new_query(query_request: QueryRequest):
    query = query_request.query
    rel_docs = getDocs(query, splits, embeddings)
    output = getOutput(query, rel_docs)
    result = {"query": query, "response": output}
    results.append(result)
    return result

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='localhost', port=5000)