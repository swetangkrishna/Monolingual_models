# references_check.py

from dataclasses import dataclass
from typing import List, Optional, Set
import re

DEBUG = True

if DEBUG:
    print("[references_check] Module imported")


@dataclass
class ReferenceMetadata:
    id: str
    title: str
    source_type: str     # "journal", "conference", "arxiv", "other"
    year: Optional[int]


@dataclass
class ReferenceScores:
    hallucinated_citations_per_answer: int
    source_quality_score: float
    recency_score: float


def _extract_citation_ids(answer: str) -> List[str]:
    if DEBUG:
        print("[references_check] _extract_citation_ids called")
    ids = re.findall(r"\[([^\]]+)\]", answer)
    if DEBUG:
        print(f"[references_check] found citation ids={ids}")
    return ids


def _compute_source_quality_score(
    used_citation_ids: List[str],
    reference_db: List[ReferenceMetadata],
) -> float:
    if DEBUG:
        print("[references_check] _compute_source_quality_score called")

    if not used_citation_ids:
        if DEBUG:
            print("[references_check] no citations used")
        return 0.0

    db_map = {ref.id: ref for ref in reference_db}
    type_weight = {
        "journal": 1.0,
        "conference": 0.8,
        "arxiv": 0.6,
        "other": 0.4,
    }

    scores = []
    for cid in used_citation_ids:
        ref = db_map.get(cid)
        if ref is None:
            continue
        scores.append(type_weight.get(ref.source_type.lower(), 0.4))

    if not scores:
        if DEBUG:
            print("[references_check] no matching refs in DB")
        return 0.0

    avg = sum(scores) / len(scores)
    if DEBUG:
        print(f"[references_check] source_quality_score={avg:.3f}")
    return avg


def _compute_recency_score(
    used_citation_ids: List[str],
    reference_db: List[ReferenceMetadata],
    current_year: int = 2025,
) -> float:
    if DEBUG:
        print("[references_check] _compute_recency_score called")

    if not used_citation_ids:
        if DEBUG:
            print("[references_check] no citations used")
        return 0.0

    db_map = {ref.id: ref for ref in reference_db}
    ages = []
    for cid in used_citation_ids:
        ref = db_map.get(cid)
        if ref is None or ref.year is None:
            continue
        ages.append(max(0, current_year - ref.year))

    if not ages:
        if DEBUG:
            print("[references_check] no ages found")
        return 0.0

    avg_age = sum(ages) / len(ages)
    score = max(0.0, 1.0 - avg_age / 20.0)
    if DEBUG:
        print(f"[references_check] avg_age={avg_age:.2f}, recency_score={score:.3f}")
    return score


def evaluate_references(
    answer: str,
    reference_db: List[ReferenceMetadata],
) -> ReferenceScores:
    if DEBUG:
        print("[references_check] evaluate_references START")

    used_cids = _extract_citation_ids(answer)
    reference_ids: Set[str] = {ref.id for ref in reference_db}

    hallucinated = sum(1 for cid in used_cids if cid not in reference_ids)
    if DEBUG:
        print(f"[references_check] hallucinated_citations={hallucinated}")

    source_quality = _compute_source_quality_score(used_cids, reference_db)
    recency_score = _compute_recency_score(used_cids, reference_db)

    if DEBUG:
        print(
            "[references_check] DONE | "
            f"hallucinated={hallucinated}, "
            f"source_quality={source_quality:.3f}, "
            f"recency={recency_score:.3f}"
        )

    return ReferenceScores(
        hallucinated_citations_per_answer=hallucinated,
        source_quality_score=source_quality,
        recency_score=recency_score,
    )