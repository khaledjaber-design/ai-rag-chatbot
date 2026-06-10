"""RAG Chatbot — ask questions about your own document.

Pipeline: chunk the document -> retrieve the most relevant chunks for a
question (TF-IDF similarity) -> ask Claude to answer using only those chunks.
Retrieval runs locally; only answer generation calls the LLM.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"


def read_document(path: str) -> str:
    """Read a .txt or .pdf file into plain text."""
    p = Path(path)
    if not p.exists():
        sys.exit(f"File not found: {path}")
    if p.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(p))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return p.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    """Split text into overlapping word-windows."""
    words = text.split()
    chunks: list[str] = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + size]))
        i += size - overlap
    return [c for c in chunks if c.strip()]


class Retriever:
    """TF-IDF retriever: returns the chunks most similar to a query."""

    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(chunks)

    def top_k(self, query: str, k: int = 4) -> list[str]:
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix)[0]
        ranked = sims.argsort()[::-1][:k]
        return [self.chunks[i] for i in ranked if sims[i] > 0]


def answer(question: str, context: str) -> str:
    """Ask Claude to answer the question using only the retrieved context."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask questions about a document (RAG).")
    ap.add_argument("document", help="Path to a .txt or .pdf file")
    args = ap.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first: copy .env.example to .env and add your key.")

    chunks = chunk_text(read_document(args.document))
    if not chunks:
        sys.exit("The document appears to be empty.")
    retriever = Retriever(chunks)

    print(f"\nLoaded {len(chunks)} chunks from '{args.document}'.")
    print("Ask a question about it (Ctrl+C to quit).\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not question:
            continue
        context = "\n\n".join(retriever.top_k(question))
        if not context:
            print("Bot: I couldn't find anything relevant in the document.\n")
            continue
        print(f"Bot: {answer(question, context)}\n")


if __name__ == "__main__":
    main()
