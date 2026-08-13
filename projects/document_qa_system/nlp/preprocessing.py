"""
Text preprocessing and chunking utilities.
"""

import re
from typing import Dict, List


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text.

    PDF extraction can introduce:
    - excessive whitespace
    - repeated newlines
    - strange spacing

    This function performs simple cleaning while preserving
    the actual meaning of the text.
    """
    if not text:
        return ""

    # Replace multiple whitespace characters with a single space.
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing whitespace.
    text = text.strip()

    return text


def clean_pages(pages: List[Dict]) -> List[Dict]:
    """
    Clean text for every page and remove pages that contain
    no meaningful text.
    """
    cleaned_pages = []

    for page in pages:
        cleaned = clean_text(page["text"])

        if cleaned:
            cleaned_pages.append(
                {
                    "page": page["page"],
                    "text": cleaned,
                }
            )

    return cleaned_pages


def chunk_text(
    pages: List[Dict],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Dict]:
    """
    Split page text into overlapping word-based chunks.

    Parameters
    ----------
    pages:
        List containing page number and page text.

    chunk_size:
        Approximate number of words in each chunk.

    chunk_overlap:
        Number of words shared between neighboring chunks.

    Returns
    -------
    List of dictionaries containing:
        - chunk_id
        - page
        - text
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    chunks = []
    chunk_id = 0

    step = chunk_size - chunk_overlap

    for page in pages:
        words = page["text"].split()

        if not words:
            continue

        # Move through the page using overlapping windows.
        for start in range(0, len(words), step):
            end = start + chunk_size

            chunk_words = words[start:end]

            if not chunk_words:
                continue

            chunk_text_value = " ".join(chunk_words)

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "page": page["page"],
                    "text": chunk_text_value,
                }
            )

            chunk_id += 1

            # Stop once the end of this page has been reached.
            if end >= len(words):
                break

    return chunks