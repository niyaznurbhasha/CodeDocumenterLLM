# vector_search.py
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

def build_vector_index(codes, model_name="Salesforce/codet5-base"):
    """
    Build a vector index for code snippets using FAISS.
    
    Keywords: **vector search**, **FAISS**, **retrieval-augmented generation (RAG)**
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    embeddings = []
    for code in codes:
        inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use the embedding of the [CLS] token (or first token) as representation
        embed = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.append(embed[0])
    embeddings = np.array(embeddings).astype("float32")
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index, embeddings

def search_vector_index(query, index, tokenizer, model, top_k=3):
    """
    Search the FAISS index with a query and return the top_k indices.
    
    Keywords: **vector search**, **FAISS**, **retrieval-augmented generation (RAG)**
    """
    inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    query_embedding = outputs.last_hidden_state[:, 0, :].numpy().astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    return indices, distances
