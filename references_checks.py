"""
references_checks.py

Usage overview
--------------

You give it the *full text* of an AI-generated answer (like your
`gpt-5.1(RQ1).txt`, `deepseek-R1(RQ2).txt`, etc.), and it will:

1. Find the final "References" section (case-insensitive).
2. Parse each reference line/paragraph into structured metadata.
   - Handles formats like:
       References
           1. Shi, F. et al. (2022). ...
           2. ...
     and
       References
       Kang, D., ... (2025). ...
       Zhou, R., ... (2025). ...
3. Call free web APIs (Crossref + arXiv) to:
   - Check if each reference is real (title/DOI-based match).
   - Infer source type: "journal", "conference", "arxiv", or "other".
   - Extract publication year.
4. Compute:
   - hallucinated_citations_per_answer
   - source_quality_score (journal > conference > arxiv > other > unknown)
   - recency_score (newer = higher)

Requirements:
    pip install requests
    (and internet access)

IMPORTANT:
    Please set USER_AGENT to include a real contact email to be polite
    to the Crossref API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict
import re
import time
from datetime import datetime
from urllib.parse import quote_plus

import requests
import difflib


# Toggle for debug prints
DEBUG = False


# ---------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------

@dataclass
class ReferenceMetadata:
    """
    Parsed from the References section of an answer.

    id:     local ID assigned per answer, e.g. "R1", "R2", ...
    title:  paper title, if we can parse it
    authors: raw author string (e.g. "Kang, D., Hwang, S., ...")
    venue:  journal/conference/arxiv/etc as free text if visible
    year:   publication year if parseable, else None
    url:    URL if any (publisher, arxiv, DOI link)
    doi:    DOI string if any, e.g. "10.48550/arXiv.2510.27269"
    raw:    raw reference block string (for debugging)
    """
    id: str
    title: Optional[str]
    authors: Optional[str]
    venue: Optional[str]
    year: Optional[int]
    url: Optional[str]
    doi: Optional[str]
    raw: str


@dataclass
class VerifiedReference:
    """
    Result of trying to verify a reference via online APIs.
    """
    id: str
    found: bool
    match_score: float
    source_type: str  # "journal", "conference", "arxiv", "other", "unknown"
    year: Optional[int]
    raw_api_source: str  # "crossref", "arxiv", "none"


@dataclass
class ReferenceScores:
    hallucinated_citations_per_answer: int
    source_quality_score: float
    recency_score: float


# ---------------------------------------------------------------------
# Utility: HTTP + similarity
# ---------------------------------------------------------------------

CROSSREF_BASE = "https://api.crossref.org/works"
ARXIV_BASE = "http://export.arxiv.org/api/query"

# PLEASE customise this to your own email per Crossref rules
USER_AGENT = "reference-checker/1.0 (mailto:your-email@example.com)"


def _http_get(url: str, params: Dict | None = None, timeout: int = 10) -> Optional[requests.Response]:
    """
    Simple GET with retry + backoff for politeness and robustness.
    """
    for attempt in range(3):
        try:
            resp = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers={"User-Agent": USER_AGENT},
            )
            if resp.status_code == 200:
                return resp

            # Retry on rate limiting / transient
            if resp.status_code in (429, 503):
                if DEBUG:
                    print(f"[http] status={resp.status_code}, retry {attempt + 1}")
                time.sleep(1 + attempt)
            else:
                if DEBUG:
                    print(f"[http] status={resp.status_code}, giving up")
                return None
        except requests.RequestException as e:
            if DEBUG:
                print(f"[http] exception={e}, retry {attempt + 1}")
            time.sleep(0.5 * (attempt + 1))
    return None


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _title_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _normalize_text(a), _normalize_text(b)).ratio()


# ---------------------------------------------------------------------
# Step 1: Extract "References" chunk from full answer text
# ---------------------------------------------------------------------

def _extract_references_chunk(text: str) -> str:
    """
    Find the LAST occurrence of a line that looks like a References heading
    (e.g. "References" or "5 References") and return the text after it.

    We stop when we hit a heading like "Final Summary" or "Summary" if present.
    """
    matches = list(re.finditer(r'(?im)^\s*(?:\d+\s*)?References\b.*$', text))
    if not matches:
        return ""

    m = matches[-1]
    refs_text = text[m.end():]

    # Stop at "Final Summary" / "Summary" style headings if present
    stop_match = re.search(r'(?im)^\s*(Final Summary(?: and Recommendation)?|Summary)\b.*$', refs_text)
    if stop_match:
        refs_text = refs_text[:stop_match.start()]

    return refs_text


def _group_reference_blocks(refs_text: str) -> List[str]:
    """
    Turn the raw References text into a list of reference blocks.
    Handles:
      - Numbered lists:
            1. Shi, F. ...
            2. Ahia, O. ...
      - Plain paragraphs separated by blank lines:
            Kang, D., ... (2025). ...
            <blank>
            Zhou, R., ... (2025). ...
    """
    lines = [ln.rstrip() for ln in refs_text.splitlines()]
    blocks: List[str] = []
    current: List[str] = []

    def is_new_ref_start(line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        # Numbered like "1." or "1)" at start of line
        if re.match(r'^\d+[\.\)]\s+', s):
            return True
        # Author-style: "Lastname, X. ..." with a year somewhere
        if re.match(r'^[A-Z][^,]+,\s', s) and re.search(r'\(\d{4}\)', s):
            return True
        return False

    for ln in lines:
        if not ln.strip():
            # Blank line: end of current block
            if current:
                blocks.append(" ".join(current).strip())
                current = []
            continue

        if is_new_ref_start(ln) and current:
            # Start of a new reference, flush previous
            blocks.append(" ".join(current).strip())
            current = [ln]
        else:
            current.append(ln)

    if current:
        blocks.append(" ".join(current).strip())

    # Remove empty blocks
    return [b for b in blocks if b]


def _parse_block_to_metadata(block: str, idx: int) -> ReferenceMetadata:
    """
    Parse a single reference block into ReferenceMetadata.
    Very heuristic, but tuned to your AI-generated formats.
    """
    raw_block = block

    # Strip "[citation:x]" markers if present
    block = re.sub(r"\[citation:\d+\]", "", block)

    # Remove leading "1." or "1)"
    block = re.sub(r'^\s*\d+[\.\)]\s*', '', block)

    # DOI
    doi_match = re.search(r'\b10\.\d{4,9}/\S+\b', block)
    doi = doi_match.group(0) if doi_match else None

    # URL (if any)
    url_match = re.search(r'(https?://\S+)', block)
    url = url_match.group(1) if url_match else None

    # Year
    year = None
    year_match = re.search(r'\((\d{4})\)', block)
    if year_match:
        try:
            year = int(year_match.group(1))
        except ValueError:
            year = None

    # Authors + rest
    authors = None
    rest = block
    if year_match:
        authors = block[:year_match.start()].strip().rstrip(".")
        rest = block[year_match.end():].strip()
    else:
        # Fallback: authors = up to the first period
        m = re.match(r'^(.*?\.)\s*(.*)$', block)
        if m:
            authors = m.group(1).strip().rstrip(".")
            rest = m.group(2).strip()

    # Clean leading ". " from rest
    rest = rest.lstrip(". ").strip()

    # Title & venue: title = first sentence after year
    title = None
    venue = None
    if rest:
        parts = rest.split(". ")
        title = parts[0].strip().rstrip(".")
        if len(parts) > 1:
            venue = ". ".join(parts[1:]).strip()

    return ReferenceMetadata(
        id=f"R{idx}",
        title=title or None,
        authors=authors or None,
        venue=venue,
        year=year,
        url=url,
        doi=doi,
        raw=raw_block,
    )


def parse_references_from_text(answer_text: str) -> List[ReferenceMetadata]:
    """
    High-level parser:
      1. Grab the References chunk.
      2. Group into blocks.
      3. Parse each block into metadata.
      4. Filter to plausible references (must have a year and title).
    """
    refs_chunk = _extract_references_chunk(answer_text)
    if not refs_chunk:
        return []

    blocks = _group_reference_blocks(refs_chunk)
    metas: List[ReferenceMetadata] = [
        _parse_block_to_metadata(b, i + 1) for i, b in enumerate(blocks)
    ]

    # Keep only blocks that look like real references
    filtered: List[ReferenceMetadata] = [
        m for m in metas
        if m.year is not None and m.title is not None and m.title.strip()
    ]

    # Renumber ids R1, R2, ... after filtering
    for i, m in enumerate(filtered, start=1):
        m.id = f"R{i}"

    if DEBUG:
        print("[parse] extracted references:")
        for m in filtered:
            print(" ", m.id, "|", m.authors, "|", m.year, "|", m.title)

    return filtered


# ---------------------------------------------------------------------
# Step 2: Online verification (Crossref + arXiv)
# ---------------------------------------------------------------------

def _lookup_crossref(ref: ReferenceMetadata) -> Optional[VerifiedReference]:
    """
    Try to verify a reference via Crossref.

    - If DOI present: query directly.
    - Else: search by title (using 'query.bibliographic') and pick
      the best title match.
    """
    if not ref.title:
        return None

    if DEBUG:
        print(f"[crossref] lookup id={ref.id} title={ref.title!r}")

    # 1. Direct DOI lookup
    if ref.doi:
        url = f"{CROSSREF_BASE}/{quote_plus(ref.doi)}"
        resp = _http_get(url)
        if resp is None:
            return None
        try:
            data = resp.json()["message"]
        except Exception:
            return None
    else:
        # 2. Search by title
        params = {
            "query.bibliographic": ref.title,
            "rows": 5,
        }
        resp = _http_get(CROSSREF_BASE, params=params)
        if resp is None:
            return None

        try:
            items = resp.json()["message"]["items"]
        except Exception:
            return None

        if not items:
            return None

        best_item = None
        best_score = 0.0
        for item in items:
            titles = item.get("title") or []
            if not titles:
                continue
            api_title = titles[0]
            score = _title_similarity(ref.title, api_title)
            if score > best_score:
                best_score = score
                best_item = item

        if best_item is None:
            return None

        data = best_item

    titles = data.get("title") or [""]
    api_title = titles[0]
    match_score = _title_similarity(ref.title, api_title)

    # Year
    year = None
    for key in ("published-print", "published-online", "created", "issued"):
        if key in data and "date-parts" in data[key]:
            try:
                year = data[key]["date-parts"][0][0]
                break
            except Exception:
                continue

    # Source type
    ctype = data.get("type", "")  # "journal-article", "proceedings-article", etc.
    container_titles = data.get("container-title") or []
    container = container_titles[0].lower() if container_titles else ""

    if "arxiv" in container or (ref.venue and "arxiv" in ref.venue.lower()) or (ref.url and "arxiv.org" in ref.url.lower()):
        source_type = "arxiv"
    elif ctype == "journal-article":
        source_type = "journal"
    elif ctype in ("proceedings-article", "proceedings"):
        source_type = "conference"
    else:
        source_type = "other"

    return VerifiedReference(
        id=ref.id,
        found=True,
        match_score=match_score,
        source_type=source_type,
        year=year,
        raw_api_source="crossref",
    )


def _lookup_arxiv(ref: ReferenceMetadata) -> Optional[VerifiedReference]:
    """
    Fallback arXiv lookup using the Atom feed when we suspect an arXiv paper.
    """
    if not ref.title:
        return None

    if DEBUG:
        print(f"[arxiv] lookup id={ref.id} title={ref.title!r}")

    query = f'ti:"{ref.title}"'
    params = {"search_query": query, "start": 0, "max_results": 3}
    resp = _http_get(ARXIV_BASE, params=params)
    if resp is None:
        return None

    text = resp.text
    entries = re.split(r"<entry>", text)[1:]
    if not entries:
        return None

    best_score = 0.0
    best_year = None

    for entry in entries:
        m_title = re.search(r"<title>(.*?)</title>", entry, flags=re.DOTALL)
        if not m_title:
            continue
        api_title = m_title.group(1)
        score = _title_similarity(ref.title, api_title)
        if score > best_score:
            best_score = score
            m_pub = re.search(r"<published>(\d{4})-\d{2}-\d{2}</published>", entry)
            if m_pub:
                try:
                    best_year = int(m_pub.group(1))
                except ValueError:
                    best_year = None

    if best_score == 0.0:
        return None

    return VerifiedReference(
        id=ref.id,
        found=True,
        match_score=best_score,
        source_type="arxiv",
        year=best_year,
        raw_api_source="arxiv",
    )


def _verify_reference(ref: ReferenceMetadata, min_match: float = 0.7) -> VerifiedReference:
    """
    High-level verification:
      * Try Crossref.
      * If venue/url looks like arXiv, also try arXiv.
      * If neither yields a good match, mark as not found.
    """
    # 1. Crossref
    cr = _lookup_crossref(ref)
    if cr is not None and cr.match_score >= min_match:
        return cr

    # 2. arXiv if suspected
    looks_arxiv = (
        (ref.venue and "arxiv" in ref.venue.lower()) or
        (ref.url and "arxiv.org" in ref.url.lower())
    )
    if looks_arxiv:
        ar = _lookup_arxiv(ref)
        if ar is not None and ar.match_score >= min_match:
            return ar

    # 3. Not verified, fall back to whatever we guessed locally
    guessed_source_type = "arxiv" if looks_arxiv else "unknown"

    return VerifiedReference(
        id=ref.id,
        found=False,
        match_score=0.0,
        source_type=guessed_source_type,
        year=ref.year,
        raw_api_source="none",
    )


# ---------------------------------------------------------------------
# Step 3: Scoring
# ---------------------------------------------------------------------

def _compute_source_quality_score(verified_refs: List[VerifiedReference]) -> float:
    """
    Weighted average based on source_type:
        journal  = 1.0
        conference = 0.8
        arxiv    = 0.6
        other    = 0.4
        unknown  = 0.2
    """
    if not verified_refs:
        return 0.0

    type_weight = {
        "journal": 1.0,
        "conference": 0.8,
        "arxiv": 0.6,
        "other": 0.4,
        "unknown": 0.2,
    }

    scores = [type_weight.get(vr.source_type.lower(), 0.2) for vr in verified_refs]
    return sum(scores) / len(scores)


def _compute_recency_score(
    verified_refs: List[VerifiedReference],
    current_year: Optional[int] = None,
) -> float:
    """
    Simple linear recency:
        0 years old  -> 1.0
        10 years old -> 0.5
        20+ years    -> 0.0
    """
    if current_year is None:
        current_year = datetime.utcnow().year

    ages = []
    for vr in verified_refs:
        if vr.year is None:
            continue
        ages.append(max(0, current_year - vr.year))

    if not ages:
        return 0.0

    avg_age = sum(ages) / len(ages)
    return max(0.0, 1.0 - avg_age / 20.0)


# ---------------------------------------------------------------------
# Step 4: Public entrypoint for your pipeline
# ---------------------------------------------------------------------

def evaluate_answer_text(
    answer_text: str,
    current_year: Optional[int] = None,
    min_match_score_for_real: float = 0.7,
) -> ReferenceScores:
    """
    Main function to call for a single AI-generated answer.

    It:
      * Parses the References section.
      * Verifies each reference online.
      * Counts hallucinated references and computes quality/recency scores.

    A reference is considered "hallucinated" if:
      - We cannot verify it via Crossref/arXiv with title similarity >= threshold.
    """
    # 1. Parse references from the raw text
    refs = parse_references_from_text(answer_text)

    if not refs:
        return ReferenceScores(
            hallucinated_citations_per_answer=0,
            source_quality_score=0.0,
            recency_score=0.0,
        )

    if current_year is None:
        current_year = datetime.utcnow().year

    verified_refs: List[VerifiedReference] = []
    hallucinated_count = 0

    for ref in refs:
        vr = _verify_reference(ref, min_match=min_match_score_for_real)
        verified_refs.append(vr)
        if (not vr.found) or (vr.match_score < min_match_score_for_real):
            hallucinated_count += 1

    source_quality = _compute_source_quality_score(verified_refs)
    recency = _compute_recency_score(verified_refs, current_year=current_year)

    return ReferenceScores(
        hallucinated_citations_per_answer=hallucinated_count,
        source_quality_score=source_quality,
        recency_score=recency,
    )


# ---------------------------------------------------------------------
# Small CLI example
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python references_checks.py path/to/answer.txt")
        raise SystemExit(1)

    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()

    scores = evaluate_answer_text(txt)
    print("Hallucinated citations:", scores.hallucinated_citations_per_answer)
    print("Source quality score:  ", scores.source_quality_score)
    print("Recency score:         ", scores.recency_score)