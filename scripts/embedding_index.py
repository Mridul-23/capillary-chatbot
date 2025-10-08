"""
embedding_index.py
------------------
Module to generate embeddings for text chunks and build FAISS index.
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

def build_faiss_index(chunks: List[str], model_name: str = "all-MiniLM-L6-v2"):
    """
    Generates embeddings for each chunk and builds a FAISS index.
    
    Args:
        chunks (List[str]): List of text chunks.
        model_name (str): Name of the SentenceTransformer model.
    
    Returns:
        index (faiss.IndexFlatIP): FAISS index with added embeddings.
        embeddings (np.ndarray): Array of normalized embeddings.
    """
    # Load embedding model
    model = SentenceTransformer(model_name)
    
    # Generate embeddings
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    
    # Normalize for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    return index, embeddings

if __name__ == "__main__":
    
    from chunking import chunk_text
    import json, os

    with open("../data/capillary_docs.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    all_text = " ".join([item["text"] for item in data])
    chunks = chunk_text(all_text)
    index, embeddings = build_faiss_index(chunks)
    
    print(f"FAISS index created with {index.ntotal} vectors.")
    
    os.makedirs("../faiss_index", exist_ok=True)
    faiss.write_index(index, "../faiss_index/capillary_chunks_index.faiss")
