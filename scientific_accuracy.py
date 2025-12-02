# scientific_accuracy.py

from dataclasses import dataclass
from typing import List
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from linguistic_clarity import tokenize_words  # reuse tokenizer

DEBUG = True

if DEBUG:
    print("[scientific_accuracy] Module imported")

# TODO: replace with your real patterns
QUASI_DEF_PATTERNS: List[str] = [
    r"\bis defined as\b",
    r"\bcan be defined as\b",
    r"\brefers to\b",
    r"\bmeans that\b",
]


@dataclass
class ScientificAccuracyScores:
    quasi_definitions_per_answer: int
    bias_markers_per_answer: int
    cosine_similarity_q_a: float


def _count_quasi_definitions(answer: str) -> int:
    if DEBUG:
        print("[scientific_accuracy] _count_quasi_definitions called")
    count = 0
    lower = answer.lower()
    for pattern in QUASI_DEF_PATTERNS:
        matches = re.findall(pattern, lower)
        count += len(matches)
    if DEBUG:
        print(f"[scientific_accuracy] quasi_definitions={count}")
    return count


def _count_bias_markers(question: str, answer: str) -> int:
    if DEBUG:
        print("[scientific_accuracy] _count_bias_markers called")

    loaded_terms = [
        "obviously",
        "clearly",
        "of course",
        "everyone knows",
        "always",
        "never",
        "terrible",
        "disaster",
        "perfect",
    ]

    q_words = set(w.lower() for w in tokenize_words(question))
    a_words = [w.lower() for w in tokenize_words(answer)]
    loaded_terms = {t.lower() for t in loaded_terms}

    count = 0
    for w in a_words:
        if w in loaded_terms and w in q_words:
            count += 1

    if DEBUG:
        print(f"[scientific_accuracy] bias_markers={count}")
    return count


def _cosine_similarity_q_a(question: str, answer: str) -> float:
    if DEBUG:
        print("[scientific_accuracy] _cosine_similarity_q_a called")
    texts = [question, answer]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(texts)
    sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0, 0]
    if DEBUG:
        print(f"[scientific_accuracy] cosine_similarity={sim:.4f}")
    return float(sim)


def evaluate_scientific_accuracy(
    question: str,
    answer: str,
) -> ScientificAccuracyScores:
    if DEBUG:
        print("[scientific_accuracy] evaluate_scientific_accuracy START")

    quasi_defs = _count_quasi_definitions(answer)
    bias_markers = _count_bias_markers(question, answer)
    cos_sim = _cosine_similarity_q_a(question, answer)

    if DEBUG:
        print(
            "[scientific_accuracy] DONE | "
            f"quasi_defs={quasi_defs}, "
            f"bias_markers={bias_markers}, "
            f"cos_sim={cos_sim:.4f}"
        )

    return ScientificAccuracyScores(
        quasi_definitions_per_answer=quasi_defs,
        bias_markers_per_answer=bias_markers,
        cosine_similarity_q_a=cos_sim,
    )