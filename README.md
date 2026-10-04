# Document Q&A with Citations

**Document Q&A with Citations** is a research tool that automates a common business research task:

> "Read multiple documents and answer a specific comparative question."

Instead of manually reading every document from beginning to end, the application retrieves the most relevant sections and generates a sourced answer based only on those sections.

This can be useful for tasks such as **competitor benchmarking, market research, and ICP research**.

## What's Included

- `app.py` - Streamlit application that loads `.txt` files from `sample_docs/`, splits them into chunks, retrieves the most relevant chunks using TF-IDF and cosine similarity, and asks Claude to generate an answer using only those retrieved sections.
- `sample_docs/` - Two fictional company overviews that allow the application to run out of the box.
- `requirements.txt` - Required Python packages.

## Running the Application

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=your_key_here
```

Then start the application:

```bash
streamlit run app.py
```

Try questions such as:

- "Which company has higher customer concentration risk?"
- "Compare their approaches to growth strategy."

## Using Your Own Documents

Add your own `.txt` files to the `sample_docs/` folder.

You can also point `DOCS_DIR` in `app.py` to a different folder containing your documents.

For PDFs, such as company 10-K filings, extract the text first and save it as `.txt` before running the application. Libraries such as `pypdf` can be used to extract PDF text programmatically.

## Why TF-IDF Instead of Embeddings?

The project uses **TF-IDF and cosine similarity** to keep the application lightweight and self-contained.

There is:

- No vector database
- No embedding API
- No additional model downloads
- Minimal setup beyond the required Python packages

For a small collection of documents, keyword-based retrieval can work well.

For larger document collections or questions that require deeper semantic matching, the retrieval system could be upgraded to use an embedding model such as `sentence-transformers` alongside a vector database such as `ChromaDB`.

The rest of the application, including the prompting, citation system, and user interface, could remain largely unchanged.

## What This Demonstrates

- **Retrieval-augmented generation:** Answers are grounded in retrieved source documents rather than relying solely on the model's knowledge.
- **Source citations:** Each claim is tied back to a specific source chunk.
- **Business-focused AI:** The application is designed around a practical research workflow rather than functioning as a generic chatbot.
- **Information retrieval:** Uses TF-IDF and cosine similarity to identify relevant sections across multiple documents.

## What I'd Add With More Time

- Replace TF-IDF with embeddings as the document collection grows.
- Add structured multi-document extraction, such as automatically pulling revenue and risk factors into a comparison table.
- Add direct PDF ingestion so users can upload documents without manually extracting the text first.
