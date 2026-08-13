"""
Model loading utilities.

This module is responsible for loading and caching:
1. Sentence Transformer embedding model
2. Hugging Face extractive Question Answering model
"""

import streamlit as st

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


# Lightweight model used to convert text into vectors.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# DistilBERT fine-tuned on the SQuAD question-answering dataset.
QA_MODEL_NAME = "distilbert-base-cased-distilled-squad"


@st.cache_resource
def load_embedding_model():
    """
    Load the Sentence Transformer embedding model.

    st.cache_resource() prevents Streamlit from loading the
    model again every time the application reruns.
    """

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return model


@st.cache_resource
def load_qa_model():
    """
    Load the tokenizer and extractive QA model.

    We intentionally load the tokenizer and model separately
    instead of using:

        pipeline("question-answering", ...)

    This makes the internal Transformer QA process easier
    for students to understand.
    """

    tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME)

    model = AutoModelForQuestionAnswering.from_pretrained(
        QA_MODEL_NAME
    )

    # Put the model into evaluation mode because we are doing
    # inference, not training.
    model.eval()

    return tokenizer, model