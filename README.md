# Local Knowledge Oracle

A local RAG (Retrieval-Augmented Generation) system that lets you query any collection of PDFs and Markdown files using natural language — with follow-up questions and zero hallucinations.

---

## What it does

- Ingests a folder of PDFs and `.md` files (books, manuals, notes — anything)
- Embeds and indexes them locally using ChromaDB (no data leaves your machine except for the OpenAI API calls)
- Answers questions grounded strictly in your documents
- Says **"I don't know"** if the answer isn't in the docs
- Remembers chat history so you can ask follow-up questions

---

## Project structure

```
project/
├── config.py       # All settings (paths, chunk size, model names, system prompt)
├── oracle.py       # Core RAG logic (load → chunk → embed → retrieve → answer)
├── main.py         # Entry point and interactive REPL
├── requirements.txt
├── docs/           # ← Put your PDFs and .md files here
└── chroma_db/      # Auto-created on first run (your local vector index)
```

---

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Add your documents**

```bash
mkdir docs
# Copy any PDFs or Markdown files into docs/
# Examples: textbooks, technical guides, story PDFs, research papers
```

**3. Set your OpenAI API key**

```bash
export OPENAI_API_KEY="sk-..."
```

Get your key at: https://platform.openai.com/api-keys

---

## Run

```bash
python main.py
```

On the **first run** it builds the vector index (takes a minute depending on how many docs you have). On every run after that it loads instantly from disk.

```
═══════════════════════════════════════════════════════════
  🔮  Local Knowledge Oracle  —  type 'exit' to quit
═══════════════════════════════════════════════════════════

You: What is the main argument in chapter 3?
Oracle: ...

You: Can you expand on the second point?
Oracle: ...
```

To force a full re-index (e.g. after adding new documents):

```bash
python main.py --rebuild
```

---

## Configuration

All settings are in `config.py`. Key options:

| Setting | Default | What it controls |
|---|---|---|
| `DOCS_DIR` | `./docs` | Where your documents live |
| `CHROMA_DIR` | `./chroma_db` | Where the vector index is saved |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks (prevents context loss at boundaries) |
| `RETRIEVER_K` | `4` | Chunks retrieved per question |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI model used for answers |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model used to create vectors |

---

## File roles

**`config.py`** — Single source of truth for all constants. Change chunk sizes, swap models, or edit the system prompt here without touching any logic.

**`oracle.py`** — The entire RAG pipeline as clean functions: `load_documents()`, `split_documents()`, `get_vectorstore()`, `build_retriever()`, `build_qa_chain()`. Import and reuse these in your own code.

**`main.py`** — Wires everything together and runs the interactive terminal session. Replace this file if you want a web UI or API instead.

---

## Supported file types

| Format | Notes |
|---|---|
| `.pdf` | Text-based PDFs (scanned image PDFs are not supported without OCR) |
| `.md` | Markdown files including headers, code blocks, and tables |

---

## Requirements

- Python 3.9+
- An OpenAI API key (used for embeddings and chat completions)
- Internet connection for API calls (the vector index itself is local)