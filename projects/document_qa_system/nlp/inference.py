"""
Inference and retrieval functions for the Document Q&A System.

This module implements the main RAG pipeline:

Document chunks
        ↓
Embeddings
        ↓
FAISS index
        ↓
Question embedding
        ↓
Similarity search
        ↓
Top-K chunks
        ↓
Similarity filtering
        ↓
Context construction
        ↓
Transformer Question Answering
        ↓
Answer
"""

from typing import Dict, List

import faiss
import numpy as np
import torch


# ============================================================
# EMBEDDINGS
# ============================================================

def generate_embeddings(
    texts: List[str],
    embedding_model
) -> np.ndarray:
    """
    Convert text chunks into dense numerical vectors.

    Sentence Transformers converts each piece of text into
    a vector representation.

    Example:

        "What is overfitting?"

    becomes something conceptually similar to:

        [0.12, -0.34, 0.87, ...]

    normalize_embeddings=True makes every vector have
    approximately unit length.

    This allows FAISS inner-product search to behave like
    cosine similarity.
    """

    if not texts:
        return np.empty((0, 0), dtype="float32")

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings.astype("float32")


# ============================================================
# FAISS INDEX
# ============================================================

def build_faiss_index(
    embeddings: np.ndarray
) -> faiss.IndexFlatIP:
    """
    Build a FAISS vector index using inner product.

    IndexFlatIP performs exact inner-product similarity search.

    Because our embeddings are normalized:

        Inner Product ≈ Cosine Similarity

    This is a simple and understandable FAISS index
    for a student project.
    """

    if embeddings.size == 0:
        raise ValueError(
            "Cannot build a FAISS index from empty embeddings."
        )

    # Number of dimensions in each embedding vector.
    dimension = embeddings.shape[1]

    # IndexFlatIP = exact inner-product search.
    index = faiss.IndexFlatIP(dimension)

    # Add all document vectors to FAISS.
    index.add(embeddings)

    return index


# ============================================================
# VECTOR SEARCH
# ============================================================

def retrieve_relevant_chunks(
    question: str,
    embedding_model,
    faiss_index,
    chunks: List[Dict],
    top_k: int = 3,
) -> List[Dict]:
    """
    Search FAISS for the chunks most similar to the question.

    Steps:

        Question
            ↓
        Question embedding
            ↓
        FAISS search
            ↓
        Top-K chunks

    Each returned chunk receives a "similarity_score" field.
    """

    if not question.strip():
        return []

    if faiss_index.ntotal == 0:
        return []

    # Convert the question into an embedding.
    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    # FAISS expects the number of results to retrieve.
    top_k = min(top_k, faiss_index.ntotal)

    # Search the vector index.
    scores, indices = faiss_index.search(
        question_embedding,
        top_k,
    )

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0],
    ):
        # -1 means FAISS did not return a valid result.
        if index_position == -1:
            continue

        chunk = chunks[index_position].copy()

        # Store retrieval similarity with the chunk.
        #
        # The key name must match the one used by nlp.scoring and
        # utils.visualization.
        chunk["similarity_score"] = float(score)

        results.append(chunk)

    return results


# ============================================================
# CONTEXT BUILDING
# ============================================================

def build_context(
    chunks: List[Dict],
    max_words: int = 450,
) -> str:
    """
    Combine retrieved chunks into a limited context.

    We limit the context because Transformer models have
    maximum input lengths.

    We use words here for educational simplicity.

    Important:

        450 words != 450 Transformer tokens.

    A production system should preferably perform
    token-aware context management.
    """

    if not chunks:
        return ""

    context_parts = []
    word_count = 0

    for chunk in chunks:

        text = chunk.get("text", "").strip()

        if not text:
            continue

        words = text.split()

        remaining_words = max_words - word_count

        if remaining_words <= 0:
            break

        selected_words = words[:remaining_words]

        context_parts.append(
            " ".join(selected_words)
        )

        word_count += len(selected_words)

    return "\n\n".join(context_parts)


# ============================================================
# EXTRACTIVE QUESTION ANSWERING
# ============================================================

def run_question_answering(
    question: str,
    context: str,
    tokenizer,
    qa_model,
) -> Dict:
    """
    Run the Hugging Face extractive Question Answering model.

    The model receives:

        Question
        +
        Context

    It predicts:

        start position
        end position

    Those positions identify an answer span inside
    the supplied context.

    The model does NOT freely generate an answer.
    """

    if not question.strip():
        return {
            "answer": "",
            "qa_score": 0.0,
        }

    if not context.strip():
        return {
            "answer": "",
            "qa_score": 0.0,
        }

    # Convert question and context into Transformer tokens.
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation="only_second",
        max_length=512,
    )

    # Disable gradient calculation because this is inference,
    # not model training.
    with torch.no_grad():

        outputs = qa_model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    # The tokenized input holds the question AND the context.
    #
    # An extractive answer may only come from the context, so we
    # mask out every token that is not part of it.
    #
    # sequence_ids() marks question tokens with 0, context tokens
    # with 1, and special tokens with None.
    sequence_ids = inputs.sequence_ids(0)

    context_mask = torch.tensor(
        [
            sequence_id == 1
            for sequence_id in sequence_ids
        ]
    )

    if not bool(context_mask.any()):

        return {
            "answer": "",
            "qa_score": 0.0,
        }

    start_logits = start_logits.masked_fill(
        ~context_mask,
        float("-inf"),
    )

    end_logits = end_logits.masked_fill(
        ~context_mask,
        float("-inf"),
    )

    # Find the most likely starting token inside the context.
    start_index = int(
        torch.argmax(start_logits)
    )

    # The answer must end at or after it starts, so we only search
    # the tokens from start_index onwards.
    end_index = start_index + int(
        torch.argmax(end_logits[start_index:])
    )

    # Extract the predicted answer tokens.
    answer_tokens = inputs.input_ids[
        0,
        start_index:end_index + 1,
    ]

    # Convert tokens back into readable text.
    answer = tokenizer.decode(
        answer_tokens,
        skip_special_tokens=True,
    ).strip()

    if not answer:

        return {
            "answer": "",
            "qa_score": 0.0,
        }

    # Convert start/end logits into probabilities.
    #
    # We use the masked logits so the probabilities are spread over
    # the context tokens only.
    start_probabilities = torch.softmax(
        start_logits,
        dim=-1,
    )

    end_probabilities = torch.softmax(
        end_logits,
        dim=-1,
    )

    start_probability = start_probabilities[start_index]

    end_probability = end_probabilities[end_index]

    # This is a simple QA confidence approximation.
    #
    # It is NOT the same thing as FAISS retrieval similarity.
    qa_score = float(
        (
            start_probability * end_probability
        ).item()
    )

    return {
        "answer": answer,
        "qa_score": qa_score,
    }


# ============================================================
# COMPLETE RAG ANSWER FUNCTION
# ============================================================

def answer_from_chunks(
    question: str,
    retrieved_chunks: List[Dict],
    tokenizer,
    qa_model,
    max_context_words: int = 450,
) -> List[Dict]:
    """
    Run extractive Question Answering once per retrieved chunk.

    Retrieval and similarity filtering happen before this function
    is called, so the chunks arriving here are already the ones
    considered relevant.

    Running the QA model per chunk (instead of on one merged
    context) means every candidate answer keeps its own page
    number and retrieval similarity, which is what the UI shows.

    Returns a list of candidate answers:

        [
            {
                "answer": "...",
                "qa_score": 0.87,
                "similarity_score": 0.74,
                "page": 3,
                "chunk_id": 12,
                "context": "...",
            },
            ...
        ]

    Chunks the model could not answer from are left out, so an
    empty list means "no answer was extracted".
    """

    candidates = []

    for chunk in retrieved_chunks:

        # Limit each chunk to the context budget of the model.
        context = build_context(
            [chunk],
            max_words=max_context_words,
        )

        if not context:
            continue

        qa_result = run_question_answering(
            question=question,
            context=context,
            tokenizer=tokenizer,
            qa_model=qa_model,
        )

        # Skip chunks that produced no answer span.
        if not qa_result["answer"]:
            continue

        candidates.append(
            {
                "answer": qa_result["answer"],
                "qa_score": qa_result["qa_score"],
                "similarity_score": chunk.get(
                    "similarity_score",
                    0.0,
                ),
                "page": chunk.get("page", "Unknown"),
                "chunk_id": chunk.get("chunk_id"),
                "context": context,
            }
        )

    return candidates