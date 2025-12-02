# Rubric-Based Evaluation System for Research Question Answers

## 📘 Overview

This project provides an automated **rubric-based evaluation framework** for assessing model-generated answers to two predefined research questions (RQ1 and RQ2).  
The system evaluates each answer along three major dimensions:

1. **Linguistic Clarity** – How readable, clear, and coherent the answer is.
2. **Scientific Accuracy** – Whether the answer is relevant, rigorous, unbiased, and well structured.
3. **Reference Quality** – What citations are used and how credible they are (optional, based on metadata).

The evaluation results are printed to console and also saved into a structured JSON file for further analysis.

---

## 🧠 Research Questions

The evaluation focuses on analyzing model responses to:

### **RQ1**  
**How does lexical gap between languages impact the reasoning abilities of models?**

### **RQ2**  
**Do monolingual models perform differently on reasoning tasks across multiple languages?  
Does this performance difference affect their efficiency?**

Each model produces a text answer for each question. These answers are evaluated independently through the rubric.

---

## 📂 Input Structure

The evaluation script expects answer files in the following format:

```
project/
│
├── gpt-5.1(RQ1).txt
├── deepseek-R1(RQ1).txt
├── gpt-5.1(RQ2).txt
└── deepseek-R1(RQ2).txt
```

Each file contains the full textual answer from one model.

---

## ⚙️ Evaluation Pipeline (Step-by-Step)

The entire evaluation is orchestrated through the `RubricEvaluator` class, which contains logic for clarity scoring, scientific scoring, and citation scoring.

### ### 1️⃣ Input Processing

1. The system loads the answer file (e.g., `gpt-5.1(RQ1).txt`).
2. The corresponding research question text is fetched.
3. The answer is passed into the evaluation pipeline.

---

## ✏️ 2️⃣ Linguistic Clarity Scoring

The evaluator checks:

### **✓ Sentence Length**
- Splits the answer into sentences.
- Computes average words per sentence.

### **✓ Jargon Usage**
- Matches words against a predefined list of domain-specific jargon terms.
- Computes jargon density per sentence.

### **✓ Readability (Flesch Reading Ease Score)**
- Uses a custom syllable counter.
- Computes readability using the Flesch formula.

### **✓ Cohesion via Connectors**
- Counts usages of logical connectors such as *however*, *therefore*, *moreover*, etc.
- Measures the logical flow of the response.

All these metrics are wrapped into a `LinguisticClarityScores` dataclass.

---

## 🔬 3️⃣ Scientific Accuracy Scoring

This section evaluates whether the answer is **scientifically grounded**, **relevant**, and **unbiased**.

### **✓ Quasi-Definitions**
Checks for phrases like:
- *“is defined as”*
- *“refers to”*
- *“is understood as”*

More definitions → stronger conceptual clarity.

### **✓ Bias Marker Detection**
Detects subjective or emotionally loaded words like:
- *clearly*, *obviously*, *always*, *never*, *perfect*, etc.

If such terms appear in both the question and answer → potential bias reinforcement.

### **✓ Cosine Similarity**
Uses TF–IDF vectorization to measure:
**How closely the answer content aligns with the question.**

Produces a score between **0.0** and **1.0**.

All results are stored in `ScientificAccuracyScores`.

---

## 📚 4️⃣ Reference Evaluation (Optional)

If citations appear inside the answer, the system extracts them using:

- **Numeric citation pattern:** `[1]`, `[2]`, etc.
- **Author-year pattern:** `(Smith, 2020)`

If a **reference metadata database** is supplied, the evaluator also computes:

- **Citation coverage**
- **Citation quality (journal > conference > preprint)**
- **Citation recency**

Since no metadata is currently enabled, these scores default to **0**.

---

## 🏁 5️⃣ Final Output

After evaluating all models for both research questions, the system saves results to:

```
rubric_results_RQ1_RQ2.json
```

This JSON file contains:

```json
{
  "RQ1": {
    "gpt-5.1": { ... linguistic, scientific, and reference scores ... },
    "deepseek-R1": { ... }
  },
  "RQ2": {
    "gpt-5.1": { ... },
    "deepseek-R1": { ... }
  }
}
```

---

## 🔧 How to Run

Run the Jupyter notebook or Python script containing:

```python
if __name__ == "__main__":
    all_results = evaluate_all_answers()
```

This will:

- Load all answer files
- Evaluate each answer systematically
- Print results
- Save JSON output

---

## 📦 Project Structure

```
project/
│
├── prac.ipynb
├── gpt-5.1(RQ1).txt
├── deepseek-R1(RQ1).txt
├── gpt-5.1(RQ2).txt
├── deepseek-R1(RQ2).txt
└── rubric_results_RQ1_RQ2.json  ← auto-generated output
```

---

## 🧩 Extensibility

You can easily extend:

- **The jargon list** for domain tuning  
- **Connector list** for style analysis  
- **Reference metadata** for real citation scoring  
- **Bias marker lists** for deeper fairness analysis  
- **Cosine similarity** to include embeddings (e.g., sentence transformers)

---

## 📝 Summary

This project provides a fully automated, reproducible, and extensible evaluation pipeline for comparing LLM responses across:

- **Readability**
- **Scientific depth**
- **Relevance**
- **Citation behavior**

It is especially useful for research comparing LLM model families or multilingual reasoning behavior.

---

If you'd like, I can also generate:

- A version with diagrams  
- A GitHub-friendly short README  
- A PDF documentation file  

Just ask!
