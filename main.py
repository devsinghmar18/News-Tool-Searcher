import os
import streamlit as st
import pickle
import time
import requests
import faiss
from bs4 import BeautifulSoup
from transformers import pipeline
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
load_dotenv()

data = None

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_input_{i}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store.pkl"

main_placeholder = st.empty()

qa_pipeline = pipeline("question-answering", model="deepset/bert-base-cased-squad2")

def load_data(urls):
    temp_data = ""
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        for paragraph in paragraphs:
            temp_data += paragraph.get_text() + "\n\n"
    return temp_data

def split_documents(data, chunk_size=1000):
    chunks = []
    current_chunk = ""
    words = data.split()
    for word in words:
        if len(current_chunk) + len(word) < chunk_size:
            current_chunk += word + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = word + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def embed_documents(docs):
    model_name = 'google/flan-t5-base'
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(docs)
    return embeddings.astype('float32')  # Convert embeddings to float32

def build_faiss_index(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

if process_url_clicked:
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = load_data(urls)
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = split_documents(data)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    embeddings = embed_documents(docs)
    index = build_faiss_index(embeddings)
    with open(file_path, "wb") as f:
        pickle.dump(index, f)
    f.close()

query = main_placeholder.text_input("Question: ", key="query_input")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            index = pickle.load(f)
            result = qa_pipeline({"question": query, "context": load_data(urls)})
            st.header("Answer")
            st.write(result["answer"])
