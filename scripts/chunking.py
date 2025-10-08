"""
chunking.py
------------
Module to split large text into semantically meaningful chunks
with optional sentence overlap, suitable for RAG embedding.
"""

import re
from typing import List

def chunk_text(text: str, chunk_size: int = 50, overlap: int = 5) -> List[str]:
    """
    Splits text into overlapping chunks of sentences.
    
    Args:
        text (str): Input text to split.
        chunk_size (int): Number of sentences per chunk.
        overlap (int): Number of overlapping sentences between consecutive chunks.
    
    Returns:
        List[str]: List of text chunks.
    """
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    start = 0
    while start < len(sentences):
        end = start + chunk_size
        chunk_sentences = sentences[start:end]
        chunk_text = " ".join(chunk_sentences).strip()
        if chunk_text:
            chunks.append(chunk_text)
        start += chunk_size - overlap
    return chunks


if __name__ == "__main__":
    with open("../data/capillary_docs.json", "r", encoding="utf-8") as f:
        import json
        data = json.load(f)
    # Combine all texts into one string
    all_text = " ".join([item["text"] for item in data])
    chunks = chunk_text(all_text)
    print(f"Total chunks created: {len(chunks)}")
