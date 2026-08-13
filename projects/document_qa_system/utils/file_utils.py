"""
PDF file handling utilities.
"""

from typing import Dict, List

from pypdf import PdfReader


def extract_text_from_pdf(uploaded_file) -> List[Dict]:
    """
    Extract text from a Streamlit uploaded PDF.

    Returns:
        [
            {
                "page": 1,
                "text": "..."
            },
            ...
        ]

    Page numbering is human-friendly and starts at 1.
    """
    try:
        reader = PdfReader(uploaded_file)
    except Exception as exc:
        raise ValueError(
            "The uploaded file could not be read as a PDF."
        ) from exc

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        pages.append(
            {
                "page": page_number,
                "text": text,
            }
        )

    return pages


def get_pdf_statistics(
    pages: List[Dict],
    chunks: List[Dict],
) -> Dict:
    """
    Generate simple document statistics for the UI.
    """
    total_characters = sum(
        len(page["text"])
        for page in pages
    )

    non_empty_pages = sum(
        bool(page["text"].strip())
        for page in pages
    )

    return {
        "pages": len(pages),
        "non_empty_pages": non_empty_pages,
        "characters": total_characters,
        "chunks": len(chunks),
    }