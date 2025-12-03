# main_rubric.py

from dataclasses import dataclass, asdict
from typing import Dict, List
from pathlib import Path
import json
import traceback

from linguistic_clarity import (
    LinguisticClarityScores,
    evaluate_linguistic_clarity,
)
from scientific_accuracy import (
    ScientificAccuracyScores,
    evaluate_scientific_accuracy,
)
from references_checks import (
    ReferenceMetadata,
    ReferenceScores,
    evaluate_answer_text,
)

DEBUG = True

if DEBUG:
    print("[main_rubric] Module imported")

RQ_TEXT: Dict[str, str] = {
    "RQ1": "How does lexical gap between languages impact the reasoning abilities of models?",
    "RQ2": (
        "Do monolingual models perform differently on reasoning tasks depending on "
        "the language in which they are trained? And does it affect their efficiency?"
    ),
}

# 👉 IMPORTANT: fill this with your actual files, e.g.

ANSWER_FILES = {
     "RQ1": {
         "gpt-5.1": "gpt-5.1(RQ1).txt",
         "deepseek-R1": "deepseek-R1(RQ1).txt",
     },
     "RQ2": {
         "gpt-5.1": "gpt-5.1(RQ2).txt",
         "deepseek-R1": "deepseek-R1(RQ2).txt",
     },
}
# ANSWER_FILES: Dict[str, Dict[str, str]] = {
    # currently empty → results will be empty
#}

BASE_DIR = Path.cwd()

# TODO: your jargon/connectors/reference DB
JARGON_LIST: List[str] = []
CONNECTORS: List[str] = []
REFERENCE_DB: List[ReferenceMetadata] = []


@dataclass
class RubricResult:
    linguistic_clarity: LinguisticClarityScores
    scientific_accuracy: ScientificAccuracyScores
    references: ReferenceScores

    def as_dict(self):
        return {
            "linguistic_clarity": asdict(self.linguistic_clarity),
            "scientific_accuracy": asdict(self.scientific_accuracy),
            "references": asdict(self.references),
        }


class RubricEvaluator:
    def __init__(
        self,
        jargon_list: List[str],
        connectors: List[str],
        reference_db: List[ReferenceMetadata],
    ):
        if DEBUG:
            print("[main_rubric] RubricEvaluator __init__ called")
        self.jargon_list = jargon_list
        self.connectors = connectors
        self.reference_db = reference_db

    def evaluate(self, question: str, answer: str) -> RubricResult:
        if DEBUG:
            print("[main_rubric] RubricEvaluator.evaluate START")

        if DEBUG:
            print("[main_rubric] -> linguistic_clarity")
        ling = evaluate_linguistic_clarity(
            answer,
            jargon_list=self.jargon_list,
            connectors=self.connectors,
        )

        if DEBUG:
            print("[main_rubric] -> scientific_accuracy")
        sci = evaluate_scientific_accuracy(question, answer)

        if DEBUG:
            print("[main_rubric] -> references")
        refs = evaluate_answer_text(answer)

        if DEBUG:
            print("[main_rubric] RubricEvaluator.evaluate DONE")

        return RubricResult(
            linguistic_clarity=ling,
            scientific_accuracy=sci,
            references=refs,
        )


evaluator = RubricEvaluator(
    jargon_list=JARGON_LIST,
    connectors=CONNECTORS,
    reference_db=REFERENCE_DB,
)


def read_answer_file(relative_path: str) -> str:
    full_path = BASE_DIR / relative_path
    if DEBUG:
        print(f"[main_rubric] Reading file: {full_path}")
    if not full_path.exists():
        print(f"[main_rubric][ERROR] File does not exist: {full_path}")
        raise FileNotFoundError(f"Answer file not found: {full_path}")
    with open(full_path, "r", encoding="utf-8") as f:
        text = f.read()
    if DEBUG:
        print(f"[main_rubric] Read {len(text)} characters from {full_path}")
    return text


def _print_scores_pretty(rq_id: str, model_name: str, rr: RubricResult):
    """Print all metric scores and subcategories clearly for debugging."""
    print("\n" + "=" * 80)
    print(f"[SCORES] {rq_id} / {model_name}")
    print("-" * 80)

    # Linguistic clarity
    lc = rr.linguistic_clarity
    print("[Linguistic clarity]")
    print(f"  - words_per_sentence      : {lc.words_per_sentence:.3f}")
    print(f"  - jargon_per_sentence     : {lc.jargon_per_sentence:.3f}")
    print(f"  - flesch_reading_ease     : {lc.flesch_reading_ease:.3f}")
    print(f"  - connectors_per_sentence : {lc.connectors_per_sentence:.3f}")

    # Scientific accuracy
    sa = rr.scientific_accuracy
    print("[Scientific accuracy]")
    print(f"  - quasi_definitions_per_answer : {sa.quasi_definitions_per_answer}")
    print(f"  - bias_markers_per_answer      : {sa.bias_markers_per_answer}")
    print(f"  - cosine_similarity_q_a        : {sa.cosine_similarity_q_a:.4f}")

    # References
    rf = rr.references
    print("[References]")
    print(f"  - hallucinated_citations_per_answer : {rf.hallucinated_citations_per_answer}")
    print(f"  - source_quality_score              : {rf.source_quality_score:.3f}")
    print(f"  - recency_score                     : {rf.recency_score:.3f}")
    print("=" * 80 + "\n")


def evaluate_all_answers() -> Dict[str, Dict[str, dict]]:
    if DEBUG:
        print("[main_rubric] evaluate_all_answers START")

    if not ANSWER_FILES:
        print("[main_rubric][WARNING] ANSWER_FILES is empty – no answers to evaluate.")
        print("  -> The output JSON will be empty {} until you populate ANSWER_FILES.")
        return {}

    results: Dict[str, Dict[str, dict]] = {}

    for rq_id, models in ANSWER_FILES.items():
        question_text = RQ_TEXT[rq_id]
        results[rq_id] = {}
        if DEBUG:
            print(f"[main_rubric] Processing {rq_id} | question='{question_text[:60]}...'")

        for model_name, rel_path in models.items():
            if DEBUG:
                print(f"[main_rubric] --- Model: {model_name}, file={rel_path}")

            try:
                answer_text = read_answer_file(rel_path)

                if DEBUG:
                    print(f"[main_rubric] Evaluating {rq_id} / {model_name}")
                rubric_result = evaluator.evaluate(question_text, answer_text)

                # Store as dict
                results[rq_id][model_name] = rubric_result.as_dict()

                # 🔍 Print metric + sub-metric scores clearly
                _print_scores_pretty(rq_id, model_name, rubric_result)

            except Exception as e:
                print(
                    f"[main_rubric][ERROR] Failed for {rq_id} / {model_name} "
                    f"with file {rel_path}"
                )
                print(f"[main_rubric][ERROR] {type(e).__name__}: {e}")
                traceback.print_exc()

    if DEBUG:
        print("[main_rubric] evaluate_all_answers DONE")

    return results


if __name__ == "__main__":
    if DEBUG:
        print("[main_rubric] __main__ entry point")
    all_results = evaluate_all_answers()
    out_path = BASE_DIR / "rubric_results_RQ1_RQ2.json"
    if DEBUG:
        print(f"[main_rubric] Writing results to {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("Saved all rubric results to", out_path)