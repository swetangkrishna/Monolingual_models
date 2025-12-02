# linguistic_clarity.py

from dataclasses import dataclass
from typing import List, Optional
import re

DEBUG = True

if DEBUG:
    print("[linguistic_clarity] Module imported")

SENTENCE_SPLIT_REGEX = re.compile(r"[.!?]+")
WORD_REGEX = re.compile(r"\b[\w'-]+\b", re.UNICODE)
VOWELS = "aeiouy"

# TODO: replace with your full connector list
CONNECTORS = [
    "however",
    "therefore",
    "indeed",
    "for this reason",
    "in contrast",
    "on the other hand",
]


def split_sentences(text: str) -> List[str]:
    if DEBUG:
        print("[linguistic_clarity] split_sentences called")
    return [s.strip() for s in SENTENCE_SPLIT_REGEX.split(text) if s.strip()]


def tokenize_words(text: str) -> List[str]:
    if DEBUG:
        print("[linguistic_clarity] tokenize_words called")
    return WORD_REGEX.findall(text)


def count_syllables_in_word(word: str) -> int:
    word = word.lower()
    word = re.sub(r"[^a-z]", "", word)
    if not word:
        return 0

    syllables = 0
    prev_is_vowel = False
    for ch in word:
        is_vowel = ch in VOWELS
        if is_vowel and not prev_is_vowel:
            syllables += 1
        prev_is_vowel = is_vowel

    if word.endswith("e") and syllables > 1:
        syllables -= 1

    return max(syllables, 1)


@dataclass
class LinguisticClarityScores:
    words_per_sentence: float
    jargon_per_sentence: float
    flesch_reading_ease: float
    connectors_per_sentence: float


def evaluate_linguistic_clarity(
    answer: str,
    jargon_list: Optional[List[str]] = None,
    connectors: Optional[List[str]] = None,
) -> LinguisticClarityScores:
    if DEBUG:
        print("[linguistic_clarity] evaluate_linguistic_clarity START")

    sentences = split_sentences(answer)
    num_sentences = max(len(sentences), 1)

    words = tokenize_words(answer)
    num_words = len(words)

    if DEBUG:
        print(f"[linguistic_clarity] num_sentences={num_sentences}, num_words={num_words}")

    words_per_sentence = num_words / num_sentences if num_sentences > 0 else 0.0

    jargon_per_sentence = 0.0
    if jargon_list:
        if DEBUG:
            print(f"[linguistic_clarity] jargon_list size={len(jargon_list)}")
        jargon_set = {j.lower() for j in jargon_list}
        jargon_count = sum(1 for w in words if w.lower() in jargon_set)
        jargon_per_sentence = jargon_count / num_sentences
        if DEBUG:
            print(f"[linguistic_clarity] jargon_count={jargon_count}")

    syllables = sum(count_syllables_in_word(w) for w in words) or 1
    words_per_sentence_for_flesch = num_words / num_sentences
    syllables_per_word = syllables / max(num_words, 1)

    flesch = 206.835 - 1.015 * words_per_sentence_for_flesch - 84.6 * syllables_per_word

    connectors_to_use = [c.lower() for c in (connectors or CONNECTORS)]
    lower_answer = answer.lower()
    connectors_count = 0
    for c in connectors_to_use:
        connectors_count += len(
            re.findall(r"\b" + re.escape(c) + r"\b", lower_answer)
        )
    connectors_per_sentence = connectors_count / num_sentences

    if DEBUG:
        print(
            "[linguistic_clarity] DONE | "
            f"words_per_sentence={words_per_sentence:.2f}, "
            f"jargon_per_sentence={jargon_per_sentence:.2f}, "
            f"flesch={flesch:.2f}, "
            f"connectors_per_sentence={connectors_per_sentence:.2f}"
        )

    return LinguisticClarityScores(
        words_per_sentence=words_per_sentence,
        jargon_per_sentence=jargon_per_sentence,
        flesch_reading_ease=flesch,
        connectors_per_sentence=connectors_per_sentence,
    )