import os
from langchain_community.document_loaders import WebBaseLoader

os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

url = "https://www.deltek.com/en"

loader = WebBaseLoader(url)
documents = loader.load()

documentData = ""
for document in documents:
    documentData += document.page_content

from huggingface_hub import InferenceClient 

userQuestion = input("Question: ")
client = InferenceClient(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    token="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",
)

for message in client.chat_completion(
	messages=[{"role": "user", "content": "Based on this context only: " + documentData + "     Question: "+ userQuestion}],
	max_tokens=500,
	stream=True,
):
    print(message.choices[0].delta.content, end="")