"""
app.py
------
Flask-based RAG chatbot using FAISS, Pandas, and Mistral via OpenRouter.
Retrieves relevant document chunks from CapillaryDocs and generates AI responses.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import faiss
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
import os

app = Flask(__name__)


# File paths
FAISS_INDEX_PATH = "faiss_index/capillary_chunks_index.faiss"
CSV_PATH = "metadata/capillary_chunks_df.csv"
ID_MAPPING_PATH = "metadata/capillary_chunks_id_mapping.json"


# Load FAISS index
index = faiss.read_index(FAISS_INDEX_PATH)


# Load DataFrame
df = pd.read_csv(CSV_PATH, index_col=0)


# Load ID mapping
with open(ID_MAPPING_PATH, "r", encoding="utf-8") as f:
    id_mapping = json.load(f)


# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# OpenRouter API settings
OPENROUTER_API_KEY = "YOUR_OPENROUTER_KEY"
MISTRAL_MODEL = "mistralai/mistral-small-3.2-24b-instruct:free"


# Helper functions
def query_mistral(prompt: str) -> str:
    """Send prompt to Mistral via OpenRouter and return the response."""
    url = "https://openrouter.ai/api/v1/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MISTRAL_MODEL, "prompt": prompt, "max_tokens": 1250, "temperature": 0.7}

    try:
        resp = requests.post(url, headers=headers, json=payload)
        data = resp.json()
        # Extract text safely
        choices = data.get("choices", [])
        if choices and "text" in choices[0]:
            return choices[0]["text"].strip()
        return "No response from model."
    except Exception as e:
        print("Error querying Mistral:", e)
        return "Error connecting to model."

def retrieve_docs(query: str, top_k: int = 5) -> str:
    """Retrieve top-k relevant document chunks from FAISS index."""
    query_vec = embed_model.encode([query])
    query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
    _, I = index.search(query_vec, top_k)
    results = [df.loc[int(id_mapping.get(str(idx)))]["text"] for idx in I[0] if str(idx) in id_mapping]
    return "\n\n".join(results)

def build_prompt(user_input: str, context: str) -> str:
    """Build a structured prompt for detailed, step-by-step answers."""
    if not context.strip():
        context = "No relevant information found in CapillaryDocs."
    return (
        "You are HelperBot, an AI assistant for Capillary Technologies documentation.\n"
        "Using the context below, provide a **detailed, step-by-step, structured answer** to the user's question.\n"
        "Include headings, numbered steps, API endpoints if relevant, and explain clearly so a user can follow instructions.\n"
        "If the answer is not found in the context, respond politely that you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"User Question:\n{user_input}\n\n"
        "Answer:"
    )


# Flask routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"answer": "Please enter a valid question."})
    
    context = retrieve_docs(user_input, top_k=5)
    prompt = build_prompt(user_input, context)
    answer = query_mistral(prompt)
    
    # Send raw Markdown/HTML for frontend to render
    return jsonify({"answer": answer})


# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
