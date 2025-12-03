# scientific_accuracy.py

from dataclasses import dataclass
from typing import List, Tuple
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from linguistic_clarity import tokenize_words  # reuse tokenizer

DEBUG = True

if DEBUG:
    print("[scientific_accuracy] Module imported")


# ---------------------------------------------------------------------
# Data structure for returning scores
# ---------------------------------------------------------------------


@dataclass
class ScientificAccuracyScores:
    """
    Container for scientific-accuracy related signals.

    - quasi_definitions_per_answer: how many definition-like cues the answer uses
    - bias_markers_per_answer: how many loaded / biased phrases are repeated
      from the question into the answer
    - cosine_similarity_q_a: TF–IDF cosine similarity between question and answer
    """

    quasi_definitions_per_answer: int
    bias_markers_per_answer: int
    cosine_similarity_q_a: float


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def _normalise_spaces(text: str) -> str:
    """Lowercase and collapse whitespace so multi-word phrases match reliably."""
    return re.sub(r"\s+", " ", text.lower()).strip()


# ---------------------------------------------------------------------
# Quasi-definition detection (improved version)
# ---------------------------------------------------------------------
# We normalise whitespace and lowercase the text, then look for
# definition-like phrases using regex. Multi-word phrases and
# variants like "is commonly understood as" are supported.

QUASI_DEF_PATTERNS: List[Tuple[str, float]] = [
    # pattern                            weight
    (r"\bis defined as\b",               1.5),
    (r"\bcan be defined as\b",           1.5),
    (r"\bis formally defined as\b",      1.8),
    (r"\bis typically defined as\b",     1.5),

    # Meaning / explanation patterns
    (r"\brefers to\b",                   1.0),
    (r"\bmeans that\b",                  1.0),
    (r"\bmeans\b",                       0.8),
    (r"\bis understood as\b",            1.2),
    (r"\bis commonly understood as\b",   1.4),
    (r"\bis generally understood as\b",  1.4),

    # Equivalence / alias / naming
    (r"\bis known as\b",                 1.0),
    (r"\bis also known as\b",            1.2),
    (r"\bis called\b",                   1.0),
    (r"\bcan be described as\b",         1.0),
    (r"\bis described as\b",             1.0),

    # Categorisation patterns
    (r"\bis a type of\b",                1.2),
    (r"\bis a form of\b",                1.2),
    (r"\bis a kind of\b",                1.2),
    (r"\bis an example of\b",            1.0),

    # Explanation / reformulation
    (r"\bin other words\b",              0.8),
    (r"\bthat is to say\b",              1.0),
    (r"\bthat is\b",                     0.7),
    (r"\bi\.e\.\b",                      0.7),

    # “The term ...” style definitions
    (r"\bthe term\b.*\bis defined as\b", 1.8),
    (r"\bthe term\b.*\brefers to\b",     1.5),
    (r"\bby definition\b",               1.5),

    # Characterisation
    (r"\bcan be characterised as\b",     1.2),
    (r"\bis characterised by\b",         1.0),
]


def _count_quasi_definitions(answer: str) -> int:
    """
    Count quasi-definitional cues in the answer.

    We:
    - normalise whitespace and lowercase,
    - run each regex pattern over the text (multi-word safe),
    - accumulate a weighted count,
    - return an integer (rounded) to keep the external API simple.
    """
    if DEBUG:
        print("[scientific_accuracy] _count_quasi_definitions called")

    norm = _normalise_spaces(answer)
    weighted_sum = 0.0

    for pattern, weight in QUASI_DEF_PATTERNS:
        matches = re.findall(pattern, norm)
        if matches:
            if DEBUG:
                print(
                    f"[scientific_accuracy] quasi pattern={pattern!r}, "
                    f"matches={len(matches)}, weight={weight}"
                )
            weighted_sum += len(matches) * weight

    quasi_count = int(round(weighted_sum))

    if DEBUG:
        print(
            "[scientific_accuracy] quasi_definitions(weighted)="
            f"{weighted_sum}, rounded={quasi_count}"
        )

    return quasi_count


# ---------------------------------------------------------------------
# Bias / loaded-language marker detection (improved, multi-word safe)
# ---------------------------------------------------------------------


LOADED_BIAS_TERMS: List[str] = [
    # Over-certainty
    "obviously",
    "clearly",
    "of course",
    "everyone knows",
    "undoubtedly",
    "certainly",

    # Absolutes
    "always",
    "never",

    # Strongly emotional / evaluative
    "terrible",
    "disaster",
    "disastrous",
    "perfect",
    "flawless",
    "ridiculous",
    "nonsense",
]


def _count_phrase_occurrences(text: str, phrase: str) -> int:
    """
    Count non-overlapping occurrences of 'phrase' in 'text', after
    normalising case and whitespace. Works for multi-word phrases.
    """
    norm_text = _normalise_spaces(text)
    norm_phrase = _normalise_spaces(phrase)

    if not norm_phrase:
        return 0

    return norm_text.count(norm_phrase)


def _count_bias_markers(question: str, answer: str) -> int:
    """
    Count how many *loaded / biased* phrases are repeated in both
    question and answer.

    We:
    - normalise whitespace and case,
    - for each loaded term, check if it appears in BOTH Q and A,
    - if yes, count how many times it appears in the answer.

    This fixes the old issue where multi-word phrases like
    "of course" or "everyone knows" would never match when using
    token-level comparison only.
    """
    if DEBUG:
        print("[scientific_accuracy] _count_bias_markers called")

    q_norm = _normalise_spaces(question)
    a_norm = _normalise_spaces(answer)

    count = 0

    for term in LOADED_BIAS_TERMS:
        t_norm = _normalise_spaces(term)
        if not t_norm:
            continue

        # Only consider the term if it appears in both Q and A
        if t_norm in q_norm and t_norm in a_norm:
            occurrences_in_answer = _count_phrase_occurrences(a_norm, t_norm)
            if DEBUG and occurrences_in_answer:
                print(
                    f"[scientific_accuracy] bias term={term!r}, "
                    f"occurrences_in_answer={occurrences_in_answer}"
                )
            count += occurrences_in_answer

    if DEBUG:
        print(f"[scientific_accuracy] bias_markers={count}")

    return count


# ---------------------------------------------------------------------
# Cosine similarity between question and answer (TF–IDF)
# ---------------------------------------------------------------------


def _cosine_similarity_q_a(question: str, answer: str) -> float:
    """
    Compute TF–IDF cosine similarity between question and answer.

    Returns a float in [0, 1]. If one of the texts is empty or the
    vectoriser cannot form a vocabulary, returns 0.0.
    """
    if DEBUG:
        print("[scientific_accuracy] _cosine_similarity_q_a called")

    q = (question or "").strip()
    a = (answer or "").strip()

    if not q or not a:
        if DEBUG:
            print("[scientific_accuracy] Empty question or answer; cos_sim=0.0")
        return 0.0

    try:
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([q, a])
        sim_matrix = cosine_similarity(tfidf[0:1], tfidf[1:2])
        cos_sim = float(sim_matrix[0, 0])

    except ValueError as e:
        # Typically raised if there's no valid vocabulary
        if DEBUG:
            print(
                "[scientific_accuracy] ValueError in TF-IDF / cosine: "
                f"{e}; returning 0.0"
            )
        return 0.0

    if DEBUG:
        print(f"[scientific_accuracy] cosine_similarity_q_a={cos_sim:.4f}")

    return cos_sim


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------


def evaluate_scientific_accuracy(
    question: str,
    answer: str,
) -> ScientificAccuracyScores:
    """
    Main entry point for the scientific-accuracy rubric.

    Inputs
    ------
    question : str
        The original question posed to the model.
    answer : str
        The model's answer to be evaluated.

    Returns
    -------
    ScientificAccuracyScores
        - quasi_definitions_per_answer: int
        - bias_markers_per_answer: int
        - cosine_similarity_q_a: float
    """
    if DEBUG:
        print("[scientific_accuracy] evaluate_scientific_accuracy called")

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