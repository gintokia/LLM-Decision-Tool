"""
A small RAG (retrieval-augmented generation) tool: ask a question across a
folder of documents (competitor overviews, market reports, etc.) and get a
sourced answer, with citations back to the specific document and chunk.

This mirrors the manual competitor-benchmarking / ICP research work — the
idea is to automate the "read N documents, answer a specific question"
step, not to replace the judgment step.

Retrieval here uses TF-IDF + cosine similarity, which keeps this fully
self-contained (no vector DB, no embedding API calls, runs offline except
for the final answer-generation call to Claude). That's a reasonable
starting point for a small, static document set. If you outgrow it
(hundreds+ of documents, need semantic matching beyond keyword overlap),
swap the retrieval section for a proper embedding model + vector store
(e.g. sentence-transformers + Chroma) — the rest of the app stays the same.
"""

import os
import glob
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import anthropic

DOCS_DIR = "sample_docs"
CHUNK_SIZE = 500  # characters per chunk, roughly a paragraph


def load_and_chunk_documents(docs_dir):
    """Reads all .txt files in docs_dir and splits each into paragraph-ish chunks."""
    chunks = []
    for path in sorted(glob.glob(os.path.join(docs_dir, "*.txt"))):
        filename = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for i, para in enumerate(paragraphs):
            chunks.append({
                "source": filename,
                "chunk_id": i,
                "text": para,
            })
    return chunks


@st.cache_resource
def build_index(docs_dir):
    chunks = load_and_chunk_documents(docs_dir)
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return chunks, vectorizer, matrix


def retrieve(question, chunks, vectorizer, matrix, top_k=4):
    q_vec = vectorizer.transform([question])
    scores = cosine_similarity(q_vec, matrix).flatten()
    top_indices = scores.argsort()[::-1][:top_k]
    return [(chunks[i], scores[i]) for i in top_indices if scores[i] > 0]


def build_prompt(question, retrieved):
    context_blocks = []
    for chunk, score in retrieved:
        context_blocks.append(
            f"[Source: {chunk['source']}, chunk {chunk['chunk_id']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_blocks)

    prompt = f"""You are answering a business research question using only the context below.
Cite the source filename for every claim you make, in the form (source: filename).
If the context doesn't contain the answer, say so directly rather than guessing.

Context:
{context}

Question: {question}

Answer:"""
    return prompt


def get_answer(client, prompt):
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def main():
    st.set_page_config(page_title="Document Q&A", layout="centered")
    st.title("Document Q&A with Citations")
    st.caption(
        "Ask a question across the documents in sample_docs/. "
        "Answers are grounded in retrieved chunks and cite their source."
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("Set the ANTHROPIC_API_KEY environment variable before running this app.")
        st.stop()

    client = anthropic.Anthropic(api_key=api_key)
    chunks, vectorizer, matrix = build_index(DOCS_DIR)

    st.write(f"Indexed {len(chunks)} chunks from {len(set(c['source'] for c in chunks))} documents.")

    question = st.text_input("Your question", placeholder="e.g. Which company has higher customer concentration risk?")

    if st.button("Ask") and question:
        with st.spinner("Retrieving relevant sections..."):
            retrieved = retrieve(question, chunks, vectorizer, matrix)

        if not retrieved:
            st.warning("No relevant content found in the documents for this question.")
            return

        with st.spinner("Generating answer..."):
            prompt = build_prompt(question, retrieved)
            answer = get_answer(client, prompt)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Retrieved sources"):
            for chunk, score in retrieved:
                st.markdown(f"**{chunk['source']}** (chunk {chunk['chunk_id']}, relevance {score:.2f})")
                st.text(chunk["text"])


if __name__ == "__main__":
    main()
