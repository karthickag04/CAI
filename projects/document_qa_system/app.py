"""
Streamlit entry point for the Document Q&A System.

Run with:

    streamlit run app.py
"""

import streamlit as st

from models.model_loader import (
    load_embedding_model,
    load_qa_model,
)
from nlp.inference import (
    answer_from_chunks,
    build_faiss_index,
    generate_embeddings,
    retrieve_relevant_chunks,
)
from nlp.preprocessing import (
    chunk_text,
    clean_pages,
)
from nlp.scoring import (
    filter_by_similarity,
    get_best_answer,
    interpret_relevance,
    rank_by_similarity,
    rank_answers,
)
from utils.file_utils import (
    extract_text_from_pdf,
    get_pdf_statistics,
)
from utils.visualization import (
    display_answer_candidates,
    display_answer_details,
    display_document_statistics,
    display_retrieved_chunks,
)


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Document Q&A System",
    page_icon="📚",
    layout="wide",
)


# ---------------------------------------------------------
# Application title
# ---------------------------------------------------------

st.title("📚 Document Q&A System")

st.write(
    "Ask questions about your PDF using "
    "Transformers, embeddings, FAISS and RAG."
)

st.caption(
    "Educational project: PDF → Retrieval → "
    "Extractive Question Answering"
)


# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------

st.sidebar.header("⚙️ RAG Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF",
    type=["pdf"],
)

chunk_size = st.sidebar.slider(
    "Chunk size (words)",
    min_value=100,
    max_value=1000,
    value=500,
    step=50,
)

chunk_overlap = st.sidebar.slider(
    "Chunk overlap (words)",
    min_value=0,
    max_value=300,
    value=100,
    step=25,
)

top_k = st.sidebar.slider(
    "Top-K retrieved chunks",
    min_value=1,
    max_value=10,
    value=3,
)

similarity_threshold = st.sidebar.slider(
    "Similarity threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.05,
)


# ---------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "pages" not in st.session_state:
    st.session_state.pages = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None

if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None

if "qa_model" not in st.session_state:
    st.session_state.qa_model = None

if "document_ready" not in st.session_state:
    st.session_state.document_ready = False


# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------

try:
    with st.spinner("Loading embedding model..."):
        embedding_model = load_embedding_model()

    with st.spinner("Loading Question Answering model..."):
        # load_qa_model() returns the tokenizer and the model.
        tokenizer, qa_model = load_qa_model()

    st.session_state.embedding_model = embedding_model
    st.session_state.tokenizer = tokenizer
    st.session_state.qa_model = qa_model

except Exception as exc:
    st.error(
        "The Transformer models could not be loaded."
    )

    st.exception(exc)

    st.stop()


# ---------------------------------------------------------
# PDF processing
# ---------------------------------------------------------

if uploaded_file is not None:

    # Reprocess the document when a different PDF is uploaded.
    if (
        st.session_state.document_name
        != uploaded_file.name
    ):
        try:
            with st.spinner("Extracting PDF text..."):
                raw_pages = extract_text_from_pdf(
                    uploaded_file
                )

            if not raw_pages:
                st.error(
                    "No pages were found in this PDF."
                )
                st.stop()

            with st.spinner("Cleaning document text..."):
                pages = clean_pages(raw_pages)

            if not pages:
                st.error(
                    "No extractable text was found. "
                    "This may be a scanned PDF."
                )

                st.info(
                    "OCR support can be added as a future "
                    "extension."
                )

                st.stop()

            if chunk_overlap >= chunk_size:
                st.error(
                    "Chunk overlap must be smaller "
                    "than chunk size."
                )
                st.stop()

            with st.spinner("Creating document chunks..."):
                chunks = chunk_text(
                    pages,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

            if not chunks:
                st.error(
                    "The document could not be divided "
                    "into chunks."
                )
                st.stop()

            with st.spinner(
                "Generating document embeddings..."
            ):
                texts = [
                    chunk["text"]
                    for chunk in chunks
                ]

                embeddings = generate_embeddings(
                    texts,
                    embedding_model,
                )

            with st.spinner(
                "Building FAISS vector index..."
            ):
                index = build_faiss_index(
                    embeddings
                )

            # Store processed document information.
            st.session_state.document_name = (
                uploaded_file.name
            )

            st.session_state.pages = pages
            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.document_ready = True

            st.success(
                "Document processed successfully."
            )

        except ValueError as exc:
            st.error(str(exc))

        except Exception as exc:
            st.error(
                "An unexpected error occurred while "
                "processing the PDF."
            )

            st.exception(exc)


# ---------------------------------------------------------
# Document information
# ---------------------------------------------------------

if st.session_state.document_ready:

    st.subheader("📄 Document Information")

    statistics = get_pdf_statistics(
        st.session_state.pages,
        st.session_state.chunks,
    )

    display_document_statistics(
        statistics
    )

    st.write(
        f"**Current document:** "
        f"`{st.session_state.document_name}`"
    )

    st.divider()

    # -----------------------------------------------------
    # Question input
    # -----------------------------------------------------

    st.subheader("❓ Ask a Question")

    question = st.text_input(
        "Enter a question about the document",
        placeholder=(
            "Example: What is overfitting?"
        ),
    )

    ask_button = st.button(
        "🔍 Find Answer",
        type="primary",
        use_container_width=True,
    )

    if ask_button:

        if not question.strip():
            st.warning(
                "Please enter a question."
            )
            st.stop()

        try:
            # -------------------------------------------------
            # Retrieval
            # -------------------------------------------------

            with st.spinner(
                "Searching the document..."
            ):
                retrieved_chunks = (
                    retrieve_relevant_chunks(
                        question=question,
                        embedding_model=embedding_model,
                        faiss_index=st.session_state.index,
                        chunks=st.session_state.chunks,
                        top_k=top_k,
                    )
                )

            if not retrieved_chunks:
                st.warning(
                    "No document chunks were retrieved."
                )
                st.stop()

            # Rank the retrieved chunks by similarity.
            retrieved_chunks = rank_by_similarity(
                retrieved_chunks
            )

            # -------------------------------------------------
            # Similarity filtering
            # -------------------------------------------------

            filtered_chunks = filter_by_similarity(
                retrieved_chunks,
                threshold=similarity_threshold,
            )

            if not filtered_chunks:
                st.warning(
                    "The document did not contain a "
                    "sufficiently relevant passage."
                )

                best_similarity = retrieved_chunks[0].get(
                    "similarity_score",
                    0.0,
                )

                st.info(
                    f"Best retrieval similarity was "
                    f"{best_similarity:.3f}, while your "
                    f"threshold is "
                    f"{similarity_threshold:.2f}."
                )

                st.write(
                    "Try lowering the similarity threshold "
                    "or asking a question closer to the "
                    "document's content."
                )

                display_retrieved_chunks(
                    retrieved_chunks
                )

                st.stop()

            # -------------------------------------------------
            # Display retrieval results
            # -------------------------------------------------

            st.subheader("🎯 Retrieval Result")

            best_retrieval_score = (
                filtered_chunks[0]
                .get("similarity_score", 0.0)
            )

            relevance = interpret_relevance(
                best_retrieval_score
            )

            st.write(
                f"**Best retrieval:** "
                f"{relevance}"
            )

            display_retrieved_chunks(
                filtered_chunks
            )

            # -------------------------------------------------
            # Question Answering
            # -------------------------------------------------

            with st.spinner(
                "Finding the answer in the "
                "retrieved text..."
            ):
                answer_candidates = answer_from_chunks(
                    question=question,
                    retrieved_chunks=filtered_chunks,
                    tokenizer=tokenizer,
                    qa_model=qa_model,
                )

            if not answer_candidates:
                st.warning(
                    "Relevant text was found, but the "
                    "Question Answering model could not "
                    "extract a reliable answer."
                )

                st.info(
                    "Try asking the question using "
                    "different wording."
                )

                st.stop()

            # Rank using retrieval + QA scores.
            answer_candidates = rank_answers(
                answer_candidates
            )

            best_answer = get_best_answer(
                answer_candidates
            )

            if best_answer is None:
                st.warning(
                    "No answer could be selected."
                )
                st.stop()

            # -------------------------------------------------
            # Display final answer
            # -------------------------------------------------

            display_answer_details(
                best_answer
            )

            # Show alternative QA candidates.
            display_answer_candidates(
                answer_candidates
            )

        except Exception as exc:
            st.error(
                "An error occurred while searching "
                "the document."
            )

            st.exception(exc)

else:
    # ---------------------------------------------------------
    # Initial empty state
    # ---------------------------------------------------------

    st.info(
        "👈 Upload a PDF from the sidebar to begin."
    )

    st.markdown(
        """
        ### How this application works

        1. Upload a PDF.
        2. Extract its text.
        3. Split the text into overlapping chunks.
        4. Convert chunks into embeddings.
        5. Store embeddings in FAISS.
        6. Enter a question.
        7. Convert the question into an embedding.
        8. Retrieve the most relevant chunks.
        9. Use a Transformer QA model to find the answer.
        10. Display the answer and source page.
        """
    )