# 📚 Document Q&A System

An educational Retrieval Augmented Generation (RAG) application
built using Python, Hugging Face Transformers, Sentence Transformers,
FAISS and Streamlit.

The goal of this project is to understand how a basic RAG pipeline
works internally without relying on LangChain, LlamaIndex or paid APIs.

---

## Project Overview

The application allows a user to upload a PDF and ask questions
about its contents.

The system:

1. Extracts text from the PDF.
2. Cleans the extracted text.
3. Splits the text into overlapping chunks.
4. Converts each chunk into an embedding.
5. Stores embeddings in a FAISS vector index.
6. Converts the user's question into an embedding.
7. Searches for the most relevant chunks.
8. Filters chunks using a similarity threshold.
9. Uses an extractive Transformer QA model to find an answer.
10. Displays the answer and source information.

---

## Architecture

```text
PDF
 ↓
Text Extraction
 ↓
Text Cleaning
 ↓
Chunking
 ↓
Sentence Transformer
 ↓
Embeddings
 ↓
FAISS
 ↓
Question
 ↓
Question Embedding
 ↓
Similarity Search
 ↓
Top-K Retrieval
 ↓
Similarity Threshold
 ↓
Retrieved Chunks
 ↓
Extractive QA Transformer
 ↓
Answer
 ↓
Streamlit
```

---

## Project Structure

```text
document_qa_system/
├── app.py                 Streamlit entry point
├── models/
│   └── model_loader.py    Embedding + QA model loading (cached)
├── nlp/
│   ├── preprocessing.py   Text cleaning and chunking
│   ├── inference.py       Embeddings, FAISS, extractive QA
│   └── scoring.py         Filtering and ranking
├── utils/
│   ├── file_utils.py      PDF text extraction
│   └── visualization.py   Streamlit display helpers
└── data/                  Local documents (not tracked by git)
```

---

## Installation

```bash
pip install -r requirements.txt
```

The models are downloaded from Hugging Face the first time the app
runs, so the first launch needs an internet connection.

---

## Usage

```bash
streamlit run app.py
```

Then upload a PDF from the sidebar and ask a question.

---

## Configuration

The sidebar exposes the parameters that shape the RAG pipeline:

| Setting | Meaning |
| --- | --- |
| Chunk size | Words per chunk before embedding. |
| Chunk overlap | Words shared between neighbouring chunks. |
| Top-K | How many chunks the FAISS search returns. |
| Similarity threshold | Minimum retrieval score a chunk must reach. |

---

## Two Scores, Two Meanings

The application reports two different numbers, and they answer
different questions:

- **Retrieval similarity** — how close the chunk is to the question
  in embedding space. It comes from FAISS.
- **QA score** — how confident the Transformer is about the answer
  span it selected inside that chunk.

A chunk can be highly relevant while the QA model still fails to
extract a good span, so neither score alone is a confidence value.
Candidates are ranked on a weighted combination of the two.

---

## Limitations

- Scanned PDFs without a text layer produce no text (no OCR).
- The QA model is extractive: it copies a span from the document
  and cannot summarise or reason across passages.
- The context budget is counted in words, not Transformer tokens.
- The FAISS index lives in memory and is rebuilt per upload.