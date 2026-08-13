"""
Scoring, filtering, and ranking utilities.
"""

from typing import Dict, List


def filter_by_similarity(
    chunks: List[Dict],
    threshold: float,
) -> List[Dict]:
    """
    Keep only chunks whose retrieval similarity is greater than
    or equal to the configured threshold.
    """
    return [
        chunk
        for chunk in chunks
        if chunk.get("similarity_score", 0.0) >= threshold
    ]


def rank_by_similarity(
    chunks: List[Dict],
) -> List[Dict]:
    """
    Sort chunks from most relevant to least relevant.
    """
    return sorted(
        chunks,
        key=lambda item: item.get("similarity_score", 0.0),
        reverse=True,
    )


def rank_answers(
    answers: List[Dict],
) -> List[Dict]:
    """
    Rank QA results by a combined score.

    Retrieval similarity tells us:
        "How relevant is this chunk to the question?"

    QA score tells us:
        "How confident is the QA model about the answer span?"

    We combine both rather than treating either score as a
    universal confidence value.
    """
    for answer in answers:
        retrieval_score = float(
            answer.get("similarity_score", 0.0)
        )

        qa_score = float(
            answer.get("qa_score", 0.0)
        )

        # Weighted score used only for ranking candidate answers.
        combined_score = (
            0.6 * retrieval_score
            + 0.4 * qa_score
        )

        answer["combined_score"] = combined_score

    return sorted(
        answers,
        key=lambda item: item.get("combined_score", 0.0),
        reverse=True,
    )


def get_best_answer(
    answers: List[Dict],
) -> Dict | None:
    """
    Return the highest-ranked answer or None.
    """
    if not answers:
        return None

    ranked = rank_answers(answers)

    return ranked[0]


def interpret_relevance(score: float) -> str:
    """
    Convert a similarity score into a simple educational label.

    These ranges are heuristic and are NOT universal probabilities.
    """
    if score >= 0.75:
        return "Very relevant"
    if score >= 0.60:
        return "Relevant"
    if score >= 0.50:
        return "Possibly relevant"

    return "Low relevance"