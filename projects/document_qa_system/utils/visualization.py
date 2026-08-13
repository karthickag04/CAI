"""
Streamlit display and visualization helpers.
"""

from typing import Dict, List

import streamlit as st


def display_document_statistics(
    statistics: Dict,
) -> None:
    """
    Display document statistics using Streamlit metrics.
    """
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Pages",
        statistics["pages"],
    )

    col2.metric(
        "Text Pages",
        statistics["non_empty_pages"],
    )

    col3.metric(
        "Characters",
        f"{statistics['characters']:,}",
    )

    col4.metric(
        "Chunks",
        statistics["chunks"],
    )


def display_retrieved_chunks(
    chunks: List[Dict],
) -> None:
    """
    Display retrieved chunks inside expandable sections.
    """
    st.subheader("🔎 Retrieved Sources")

    if not chunks:
        st.info("No relevant chunks were retrieved.")
        return

    for position, chunk in enumerate(chunks, start=1):
        page = chunk.get("page", "Unknown")
        similarity = chunk.get(
            "similarity_score",
            0.0,
        )

        title = (
            f"Source {position} | "
            f"Page {page} | "
            f"Similarity {similarity:.3f}"
        )

        with st.expander(title):
            st.write(chunk["text"])


def display_answer_details(
    answer: Dict,
) -> None:
    """
    Display the selected answer and its source information.
    """
    st.subheader("📌 Answer")

    st.success(answer["answer"])

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Page",
        answer.get("page", "Unknown"),
    )

    col2.metric(
        "Retrieval Similarity",
        f"{answer.get('similarity_score', 0.0):.3f}",
    )

    col3.metric(
        "QA Score",
        f"{answer.get('qa_score', 0.0):.3f}",
    )

    combined = answer.get(
        "combined_score",
        0.0,
    )

    st.caption(
        f"Combined ranking score: {combined:.3f}"
    )


def display_answer_candidates(
    answers: List[Dict],
) -> None:
    """
    Display alternative candidate answers for learning/debugging.
    """
    if not answers:
        return

    with st.expander("🧪 View QA Candidates"):
        rows = []

        for answer in answers:
            rows.append(
                {
                    "Answer": answer.get("answer", ""),
                    "Page": answer.get("page", ""),
                    "Similarity": round(
                        answer.get(
                            "similarity_score",
                            0.0,
                        ),
                        3,
                    ),
                    "QA Score": round(
                        answer.get(
                            "qa_score",
                            0.0,
                        ),
                        3,
                    ),
                    "Combined": round(
                        answer.get(
                            "combined_score",
                            0.0,
                        ),
                        3,
                    ),
                }
            )

        st.dataframe(
            rows,
            use_container_width=True,
        )