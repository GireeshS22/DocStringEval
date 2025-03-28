# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 13:49:56 2025

@author: ADMIN
"""

import time
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter


#%%

PINECONE_API_KEY = "pcsk_5YaVhG_5zm5bEQ6czBNYWHbAayaTEvU71fLpeJLSno2DxuMSuvzDVjDcYhDGXg6t5J5iJD"

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "rag-index-balaji"  # change if desired

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

#%%

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = PineconeVectorStore(index=index, embedding=embeddings)


#%%
# Load documents from URL
urls = [
    "https://peps.python.org/pep-0257/",
    "https://www.geeksforgeeks.org/python-docstrings/",
    "https://pandas.pydata.org/docs/development/contributing_docstring.html",
    "https://www.coding-guidelines.lftechnology.com/docs/python/docstrings/"
]
loader = WebBaseLoader(urls)
documents = loader.load()

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)



#%%
vector_store.add_documents(texts)

#%%
# Sample search

results = vector_store.similarity_search(query="What is pep standard number?",k=1)
for doc in results:
    print(f"* {doc.page_content} [{doc.metadata}]")
    
    
    
#%%
from langchain.llms import Ollama
from langchain.chains import RetrievalQA


llm = Ollama(model="qwen2.5:0.5b")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 1})
)

# Example query
query = "What is pep standard number?"
response = qa_chain.run(query)
print(response)
