# 🤖 RAG Chatbot — Ask Questions About Your Documents

A simple, working **Retrieval-Augmented Generation (RAG)** chatbot. Point it at a
document (`.txt` or `.pdf`), then ask questions in plain English. It finds the most
relevant passages and uses an LLM (Anthropic's Claude) to answer **grounded in your
document** — not from the model's general knowledge.

## How it works (RAG in 3 steps)
1. **Chunk** — the document is split into overlapping passages
2. **Retrieve** — for each question, the most relevant chunks are found (TF-IDF similarity)
3. **Generate** — the question + retrieved passages are sent to Claude, which answers using only that context

This is the core pattern behind most "chat with your docs" AI products.

## Tech
- **Python**
- **Anthropic Claude** — answer generation
- **scikit-learn** — TF-IDF retrieval
- **pypdf** — PDF reading

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env        # then open .env and add your Anthropic API key
```

## Run
```bash
python rag_chatbot.py sample_docs/about_rag.txt
```
Then ask questions, for example: *"What does retrieval-augmented generation mean?"*

Use your own document:
```bash
python rag_chatbot.py path/to/your-file.pdf
```

## Example
```
You: What problem does RAG solve?
Bot: RAG grounds the model's answers in your own documents, reducing
     hallucination by retrieving relevant passages before generating a reply.
```

## Notes
Built as a portfolio demo of the RAG pattern. Lightweight by design — retrieval runs
locally (TF-IDF), and only answer generation calls the LLM.
