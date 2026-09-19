# Document Q&A with Citations

**Business question this automates:** "Read N documents and answer a specific
comparative question" — the manual step in competitor benchmarking / ICP
research. This tool retrieves the relevant sections and generates a sourced
answer instead of you reading every document end to end.

## What's here

- `app.py` — Streamlit app. Loads `.txt` files from `sample_docs/`, chunks
  them, retrieves the most relevant chunks for a question using TF-IDF +
  cosine similarity, then asks Claude to answer using only those chunks,
  citing the source file for each claim.
- `sample_docs/` — two short fictional company overviews so the app runs
  out of the box.
- `requirements.txt`

## To run it

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
streamlit run app.py
```

Try asking: *"Which company has higher customer concentration risk?"* or
*"Compare their approaches to growth strategy."*

## Using your own documents

Drop `.txt` files into `sample_docs/` (or point `DOCS_DIR` in `app.py` at a
different folder). For PDFs (e.g. real 10-Ks), extract text first — the
`pdf` handling in most PDF libraries (e.g. `pypdf`) will do this in a few
lines — then save the extracted text as `.txt` before running the app.

## Why TF-IDF instead of embeddings

This keeps the project fully self-contained: no vector database, no
embedding API calls, nothing to download beyond three pip packages. For a
handful of documents, keyword-based retrieval works fine. If you scale this
up (hundreds of documents, or questions that need semantic matching beyond
shared vocabulary), swap `build_index`/`retrieve` for an embedding model
(e.g. `sentence-transformers`) plus a vector store (e.g. `chromadb`) — the
rest of the app (prompting, citation, UI) doesn't need to change.

## What this demonstrates

- Retrieval grounded in real source documents, not the model's memory
- Citations tied to a specific chunk, not just "trust me"
- A workflow shaped around an actual business research task, not a generic chatbot

## What I'd add with more time

- Swap TF-IDF for embeddings once the document set grows
- Multi-document structured extraction (e.g. auto-pull revenue, risk factors into a comparison table)
- PDF ingestion built directly into the app instead of a manual pre-step
- 
