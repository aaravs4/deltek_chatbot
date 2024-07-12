# import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM  
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
# from langchain.indexes import VectorstoreIndexCreator
# from sentence_transformers import SentenceTransformer
# from langchain.embeddings import OpenAIEmbeddings
# from sklearn.metrics.pairwise import cosine_similarity
from langchain.embeddings import HuggingFaceBgeEmbeddings


tokenizer = AutoTokenizer.from_pretrained("llmware/bling-1.4b-0.1")  
model = AutoModelForCausalLM.from_pretrained("llmware/bling-1.4b-0.1") 
model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    # query_instruction="What is Deltek?"
) 
def get_splits():
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
    ##all_splits_text = [split.page_content for split in all_splits]
    return all_splits

def similar_docs(query,all_splits, embeddings):
    # doc_embeddings = SentenceTransformer('all-MiniLM-L6-v2').encode(all_splits_text)
    # query_embeddings = SentenceTransformer('all-MiniLM-L6-v2').encode(query)
    # print(doc_embeddings.shape)
    # pairs = zip(all_splits_text, doc_embeddings)

    # vectorstore = FAISS.from_embeddings(pairs, SentenceTransformer('all-MiniLM-L6-v2'))
    # results = cosine_similarity(doc_embeddings, query_embeddings.reshape(1,-1)).reshape((-1,))
    # k = 4
    # ixs = results.argsort()
    # ixs = ixs[::-1]

    # relevant_docs = []
    # results2 = []
    # for i in ixs:
    #     relevant_docs.append(all_splits_text[i])
    #     results2.append(results[i])
    # relevant_docs = relevant_docs[:k]
    # formatted_docs = "\n\n".join(doc for doc in relevant_docs)
    # return formatted_docs
    db = FAISS.from_documents(all_splits, embeddings)
    relevant_docs = db.similarity_search(query)
    formatted_docs = '\n'.join(doc.page_content for doc in relevant_docs)
    return formatted_docs

def generate(context, query):
    entries = {"context": context, 
            "query":query}

    new_prompt = "<human>: " + entries["context"] + "\n" + entries["query"] + "\n" + "<bot>:"

    inputs = tokenizer(new_prompt, return_tensors="pt")  
    start_of_output = len(inputs.input_ids[0])

    outputs = model.generate(
            inputs.input_ids.to("cpu"),
            attention_mask=inputs.attention_mask.to("cpu"),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.3,
            max_new_tokens=100,
            )

    output_text = tokenizer.decode(outputs[0][start_of_output:], skip_special_tokens=True)
    return output_text




if(__name__ == "__main__"):
    splits = get_splits()
    while(True):
        query = input("Enter a question about Deltek(enter STOP to exit): ")
        if(query == "STOP"):
            break
        returned_docs = similar_docs(query,splits,embeddings)
        output_text = generate(context = returned_docs, query = query)
        print(output_text)
    





