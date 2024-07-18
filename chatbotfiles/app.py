from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import torch
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings


question_answerer = pipeline("question-answering", model='distilbert-base-uncased-distilled-squad')
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



app = FastAPI()

class userinput(BaseModel):
    input: str
class response(BaseModel):
    answer: str

def getDocs(query, all_splits, embeddings):
    db = FAISS.from_documents(all_splits, embeddings)

    ##get relevant docs from vectorstore
    relevant_docs = db.similarity_search(query, k = 5)
    formatted_docs = '\n'.join(doc.page_content for doc in relevant_docs)
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
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    context = getDocs(query, all_splits, embeddings)
    output = getoutput(query, context)
    return(response(answer=output))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

