# Garuda - Intelligent Resume Ranking System

## Overview

Garuda is a production-oriented resume ranking system developed for the **Redrob Resume Ranking Hackathon**.

The system ranks candidate resumes against a target Job Description using a deterministic, explainable, CPU-only scoring pipeline designed for large-scale recruitment workflows.

Unlike approaches that rely on hosted Large Language Models or GPU inference during ranking, Garuda performs candidate evaluation entirely offline using engineered features extracted from structured candidate profiles. This design enables scalable, reproducible, and efficient ranking suitable for production environments.

The final output is a ranked list of the **Top 100 candidates**, accompanied by concise, human-readable reasoning for every recommendation.

---

# Key Features

* Deterministic ranking pipeline
* CPU-only execution
* No external API calls
* No hosted LLM dependency during inference
* Explainable candidate ranking
* Multi-factor candidate evaluation
* Structured reasoning generation
* Edge-case testing utilities
* Submission validation tools
* Fully reproducible pipeline

---

# Repository Structure

```text
Garuda/
│
├── .gitignore
├── README.md
├── LICENSE
├── DESIGN.md
├── rank.py
├── requirements.txt
├── submission_metadata.yaml
│
├── sandbox/
│   ├── README.md
│   └── Garuda_Resume_Ranking_Sandbox.ipynb
│
└── tests/
    ├── README.md
    ├── Garuda_Testing_and_Validation.ipynb
    ├── generate_edge_cases.py
    └── validate_submission.py
```

---

# System Architecture

```text
Candidate Dataset
        │
        ▼
JSON Loader
        │
        ▼
Feature Extraction
        │
        ├── Experience Analysis
        ├── Skill Matching
        ├── Career Progression
        ├── Education
        ├── Recruiter Signals
        ├── Availability
        ├── Profile Completeness
        └── Consistency Checks
        │
        ▼
Weighted Scoring Engine
        │
        ▼
Candidate Ranking
        │
        ▼
Reasoning Generator
        │
        ▼
submission.csv
```

---

# Ranking Methodology

Garuda evaluates each candidate using multiple complementary signals rather than relying solely on keyword matching.

The scoring pipeline incorporates:

* Relevant years of experience
* Current role alignment
* Career progression
* Technical skill relevance
* AI/ML experience
* Project and work-history relevance
* Educational background
* Recruiter engagement signals
* Profile completeness
* Notice period
* Work preferences
* Candidate consistency signals

Each feature contributes to a normalized weighted score.

The final ranking is deterministic and reproducible.

---

# Explainability

Each ranked candidate is accompanied by automatically generated reasoning based entirely on factual information contained within the candidate profile.

Reasoning may reference:

* Relevant experience
* Technical skills
* Current role
* Career strengths
* Recruiter engagement signals
* Potential hiring considerations

No unsupported or fabricated information is introduced during reasoning generation.

---

# Performance

Measured using the official competition dataset.

| Metric        | Value                     |
| ------------- | ------------------------- |
| Dataset Size  | 100,000 Candidates        |
| Loading Time  | ~7.9 seconds                |
| Ranking Time  | ~60.5 seconds               |
| Total Runtime | ~68.8 seconds               |
| Compute       | CPU Only                  |
| GPU           | Not Required              |
| Network       | Disabled                  |
| Output        | Top 100 Ranked Candidates |

---

# Installation

Clone the repository:

```bash
git clone https://github.com/Re-veil-eb/Garuda.git
cd Garuda
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Ranker

Place the official `candidates.jsonl` dataset in the repository root.

Execute:

```bash
python rank.py \
    --candidates ./candidates.jsonl \
    --out ./submission.csv
```

The generated CSV follows the competition specification:

```text
candidate_id,rank,score,reasoning
```

---

# Submission Validation

Validate the generated submission:

```bash
python tests/validate_submission.py submission.csv
```

Expected output:

```text
Submission is valid.
```

---

# Edge Case Testing

Generate synthetic edge-case profiles:

```bash
python tests/generate_edge_cases.py
```

Evaluate the generated dataset:

```bash
python rank.py \
    --candidates tests/edge_case_candidates.jsonl \
    --out edge_case_submission.csv
```

Validate the generated submission:

```bash
python tests/validate_submission.py edge_case_submission.csv
```

---

# Sandbox

The repository includes a Google Colab notebook demonstrating end-to-end execution on a lightweight sample dataset.

The sandbox demonstrates:

* Environment setup
* Dependency installation
* Ranking execution
* Submission generation
* Output validation

---

# Testing

The repository includes automated testing utilities for local verification.

Included tools:

* Submission validator
* Edge-case dataset generator
* End-to-end testing notebook

These utilities help verify correctness before submission.

---

# Reproducibility

Garuda is fully deterministic.

Given identical input data, the generated ranking will always be identical.

The ranking process:

* Uses no randomness
* Makes no external network requests
* Produces deterministic scores
* Generates reproducible output

---

# Competition Constraint Compliance

| Constraint              | Status |
| ----------------------- | ------ |
| CPU Only                | ✅      |
| No GPU                  | ✅      |
| No External APIs        | ✅      |
| Runtime < 5 Minutes     | ✅      |
| Memory < 16 GB          | ✅      |
| Deterministic Execution | ✅      |
| Reproducible Pipeline   | ✅      |

---

# Documentation

Additional project documentation is provided in:

* **DESIGN.md** – Methodology, feature engineering, scoring strategy, and design decisions.
* **tests/** – Validation scripts and testing utilities.
* **sandbox/** – Google Colab notebook for reproducible execution.

---

# AI Usage

AI-assisted development tools were used during design discussions, documentation, code review, and iterative refinement.

The ranking pipeline itself performs **no external AI inference** during execution and fully complies with the competition's compute constraints.

---

# Future Improvements

Potential enhancements include:

* Learning-to-Rank models
* Adaptive feature weighting
* Semantic skill normalization
* Graph-based career trajectory modeling
* Recruiter feedback integration
* Fairness-aware ranking
* Score calibration using hiring outcomes

---

# License

This project is released under the MIT License.

---

# Acknowledgements

Developed as part of the **Redrob Resume Ranking Hackathon**.

The repository contains the complete source code required to reproduce the ranking pipeline. All feature engineering, ranking logic, testing, validation, and documentation are included to support transparent and reproducible evaluation.
