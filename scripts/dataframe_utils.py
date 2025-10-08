"""
dataframe_utils.py
------------------
Helper module for creating Pandas DataFrame and saving ID mappings
for chunked text and FAISS integration.
"""

import pandas as pd
import json, os
from typing import List

def create_dataframe(chunks: List[str], save_csv_path: str, save_mapping_path: str):
    """
    Creates a Pandas DataFrame with chunk text, using index as unique ID.
    
    Args:
        chunks (List[str]): List of text chunks.
        save_csv_path (str): Path to save the DataFrame CSV.
        save_mapping_path (str): Path to save the ID mapping JSON.
    """
    df = pd.DataFrame({"text": chunks})
    df.to_csv(save_csv_path, index=True)  # index serves as chunk ID
    
    # Mapping: FAISS index position -> DataFrame index
    id_mapping = {i: i for i in range(len(chunks))}
    with open(save_mapping_path, "w", encoding="utf-8") as f:
        json.dump(id_mapping, f, indent=2)

if __name__ == "__main__":
    from chunking import chunk_text
    import json
    with open("../data/capillary_docs.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    all_text = " ".join([item["text"] for item in data])
    chunks = chunk_text(all_text)
    os.makedirs("../metadata", exist_ok=True)
    create_dataframe(
        chunks,
        save_csv_path="../metadata/capillary_chunks_df.csv",
        save_mapping_path="../metadata/capillary_chunks_id_mapping.json"
    )
    print("DataFrame and ID mapping saved successfully.")
