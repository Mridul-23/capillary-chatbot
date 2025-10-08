# Capillary-Chatbot — CapillaryDocs RAG Chatbot

> A retrieval-augmented chatbot that indexes Capillary documentation (FAISS + SentenceTransformers), serves a Markdown-capable Discord/Jupyter-style frontend, and answers questions using Mistral via OpenRouter.

Clean, modular, and production-friendly. Designed for local development and easy deployment.

---

## Table of contents

* [Project structure](#project-structure)
* [Features](#features)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Quick start](#quick-start)
* [Pipeline scripts (one-shot build)](#pipeline-scripts-one-shot-build)
* [API & Flask app](#api--flask-app)
* [Frontend](#frontend)
* [Configuration & environment variables](#configuration--environment-variables)
* [Prompting & quality tuning](#prompting--quality-tuning)
* [Chunking strategy & tips](#chunking-strategy--tips)
* [Scaling & production notes](#scaling--production-notes)
* [Debugging & troubleshooting](#debugging--troubleshooting)
* [Security & privacy](#security--privacy)
* [Contributing](#contributing)
* [License](#license)

---

## Project structure

```
capillary-chatbot/
│
├─ data/
│   └─ capillary_docs.json          # raw scraped docs (input)
│
├─ scripts/
│   ├─ chunking.py                  # split large text into chunks
│   ├─ embedding_index.py           # create embeddings & FAISS index
│   └─ dataframe_utils.py           # create & save pandas dataframe + id mapping
│
├─ faiss_index/
│   └─ capillary_chunks_index.faiss
│
├─ metadata/
│   ├─ capillary_chunks_df.csv
│   └─ capillary_chunks_id_mapping.json
│
├─ templates/
│   └─ index.html                   # Discord/Jupyter-style frontend (Markdown)
│
├─ app.py                           # Flask RAG app (loads index, df, queries OpenRouter)
├─ build_rag_pipeline.py            # orchestrates chunk -> embed -> index -> metadata
├─ requirements.txt
└─ README.md
```

---

## Features

* Chunking by sentences with overlap to preserve context.
* Embeddings via `sentence-transformers` (default: `all-MiniLM-L6-v2`).
* FAISS index for fast semantic search.
* Pandas DataFrame mapping index positions → chunks (index used as ID).
* Flask backend that:

  * retrieves top-k chunks,
  * crafts a structured prompt for Mistral (via OpenRouter),
  * returns model responses (raw Markdown) to the frontend.
* Frontend: full-screen Discord/Jupyter-like chat, Markdown rendering using `marked.js`, loader UX.
* Minimal, modular scripts so pipeline steps can be reused or re-run.

---

## Prerequisites

* Python 3.10+ (3.11 recommended)
* pip
* (optional) A machine with enough RAM for embeddings. For ~1000 chunks, typical laptop is OK.
* OpenRouter API key with access to your chosen Mistral model.

---

## Installation

1. Clone repository

```bash
git clone https://github.com/Mridul-23/capillary-chatbot.git
cd project
```

2. Create virtual environment & activate

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

**Example `requirements.txt`**

```
flask
pandas
faiss-cpu       # or faiss-gpu if you have CUDA and want GPU speed
sentence-transformers
requests
python-dotenv   # optional but recommended
marked          # frontend lib is CDN-loaded; no need in requirements
```

*Note:* Use `faiss-cpu` on most dev machines. For large indexes or production, use `faiss-gpu` on a GPU server.

---

## Quick start

1. Place your scraped JSON into `data/capillary_docs.json`. Format:

```json
[
  {"url":"https://docs.capillarytech.com/docs/introduction", "text":"..."},
  ...
]
```

2. Build the RAG pipeline (chunks, embeddings, FAISS, metadata)

```bash
python build_rag_pipeline.py
```

This script (example included) will:

* read `data/capillary_docs.json`
* chunk text
* create DataFrame and save to `metadata/capillary_chunks_df.csv`
* generate embeddings and create `faiss_index/capillary_chunks_index.faiss`
* save `metadata/capillary_chunks_id_mapping.json`

3. Run Flask app

```bash
export OPENROUTER_API_KEY="sk-..."        # Linux / macOS
set OPENROUTER_API_KEY="sk-..."           # Windows Powershell
python app.py
```

Open: `http://127.0.0.1:5000`

---

## Pipeline scripts (one-shot build)

`build_rag_pipeline.py` should wire together:

```python
from chunking import chunk_text
from embedding_index import build_faiss_index
from dataframe_utils import create_dataframe
import faiss, json

# 1. Load raw
with open("data/capillary_docs.json") as f:
    data = json.load(f)
all_text = " ".join([d["text"] for d in data])

# 2. Chunk
chunks = chunk_text(all_text, chunk_size=50, overlap=5)

# 3. Save DF + mapping
create_dataframe(chunks, save_csv_path="metadata/capillary_chunks_df.csv", save_mapping_path="metadata/capillary_chunks_id_mapping.json")

# 4. Embed + index
index, embeddings = build_faiss_index(chunks)
faiss.write_index(index, "faiss_index/capillary_chunks_index.faiss")
```

Run once after changing source JSON. If you re-run, either overwrite metadata or use versioned filenames.

---

## API & Flask app

**Endpoints**

* `GET /` → Loads the chat UI (`templates/index.html`).
* `POST /chat` → Accepts `{ "message": "<user question>" }` and returns `{ "answer": "<Markdown response>" }`.

**Key behavior**

* The app:

  * loads the FAISS index and Pandas DF on startup,
  * encodes the incoming user query,
  * retrieves `top_k` most relevant chunks,
  * constructs a structured prompt instructing Mistral to produce multi-step, Markdown-styled answers,
  * sends prompt to OpenRouter, extracts response and returns raw Markdown to frontend for rendering.

**Example `curl`**

```bash
curl -X POST http://127.0.0.1:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Where to get user activity log"}'
```

---

## Frontend

* Built-in `templates/index.html` (Discord/Jupyter style).
* Uses `marked.js` (CDN) to render Markdown safely.
* Loader shown while waiting for backend response.
* No `\n -> <br>` replacement in backend — send Markdown and render on the frontend.

---

## Configuration & environment variables

* `OPENROUTER_API_KEY` — **required** (use .env or environment).
* `MISTRAL_MODEL` — default in `app.py` is set to the chosen model string. Update if you want another model.
* `TOP_K` — number of chunks to retrieve (default 3–5). Controlled in `retrieve_docs`.
* `MAX_TOKENS` — in `query_mistral` payload. Increase for longer answers but watch token usage.

**Recommended**: use `.env` and `python-dotenv`:

`.env`

```
OPENROUTER_API_KEY=sk-...
MISTRAL_MODEL=mistralai/mistral-small-3.2-24b-instruct:free
```

Load it in `app.py`:

```python
from dotenv import load_dotenv
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
```

**Do not commit** `.env` or keys to git.

---

## Prompting & quality tuning

* Use a structured prompt (see `build_prompt`) that explicitly asks for:

  * headings (`###`),
  * numbered steps,
  * bold for important terms,
  * examples and API endpoints if relevant,
  * concise summary at the end (optional).
* Increase `top_k` and `max_tokens` for richer answers.
* To make responses more verbose: ask the model to produce step-by-step instructions and to expand each bullet.

**Sample prompt fragment**

```
You are HelperBot, an expert in CapillaryDocs. Using the context below, produce a detailed, step-by-step answer with headings (###), numbered steps, and code/API examples if available. Use Markdown formatting.
```

---

## Chunking strategy & tips

* Sentence-based splitting + overlap preserves semantics:

  * `chunk_size = 50` sentences, `overlap = 5` is a solid default for long docs.
* Alternative: token-based chunking (measure tokens with a tokenizer) for strict token control.
* Save chunk-to-original mapping (URL, section title) in the DataFrame for traceability.
* Keep chunk count reasonable (hundreds to low thousands) to fit memory & embedding time.

---

## Scaling & production notes

* Production deployment:

  * Use Gunicorn + systemd or a container (Docker) + nginx reverse proxy.
  * Example: `gunicorn -w 4 app:app`
* For larger indexes:

  * Use FAISS IVF (IndexIVFFlat) with training for faster search & smaller memory footprint.
  * Persist index on SSD and load into RAM on startup. Consider memory-mapped indexes for very large datasets.
* GPU:

  * Use `faiss-gpu` and model acceleration (embedding on GPU) if you have a GPU server.
* Caching:

  * Cache recent query embeddings or model responses to reduce cost and latency.

---

## Debugging & troubleshooting

**Common issues**

* `No response from model.`

  * Print raw response JSON from OpenRouter. Different models return text under different keys:

    * `choices[0]["text"]` OR `choices[0]["message"]["content"]` — adapt extractor accordingly.
  * Check HTTP status codes: 401 (invalid key), 429 (rate limit), 500 (server).
* Empty `context` (FAISS returned results but DataFrame lookup failed)

  * Verify `id_mapping` keys match DataFrame indices (string vs int mismatch).
  * Print `I` (indices) from `index.search()` and confirm mapping: `id_mapping[str(idx)]`.
* FAISS write error (`No such file or directory`)

  * Create directories first: `os.makedirs("faiss_index", exist_ok=True)`.
* `25 batches` during embedding

  * That's normal: it’s internal batching based on model batch size and number of chunks.
* Frontend shows raw `<br>` or markdown literal

  * Ensure frontend uses `marked.parse()` (or similar) to render Markdown and that backend sends raw Markdown (not HTML-escaped).

**Useful debug prints**

* Print `context`, `prompt[:1000]`, OpenRouter status and `data` (raw JSON) during development.

---

## Security & privacy

* **Do not store API keys in VCS.** Use environment variables or secrets manager.
* If you log user queries & responses, ensure privacy: redact PII if required, secure logs.
* Consider rate-limiting and authentication for your Flask endpoint in production.

---

## Contributing

* PRs welcome. Keep modules small and single-responsibility.
* Suggested additions:

  * UI improvements (threads, source tracing).
  * Per-user conversation history and RAG state.
  * Dockerfile + Compose for easy deployment.
  * Integration tests for pipeline.

---

## Example usage

1. Build pipeline:

```bash
python build_rag_pipeline.py
```

2. Run app:

```bash
export OPENROUTER_API_KEY="sk-..."
python app.py
```

3. Ask in UI:

```
Where to get user activity log
```

Result: HelperBot returns a multi-step, Markdown-formatted guide with API endpoints and steps.

---

## License

MIT License — feel free to use, modify, and extend. Give me a shout (or a star) if you like it.
