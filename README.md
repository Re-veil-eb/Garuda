# Garuda - Intelligent Resume Ranking System

## Overview

Garuda is a production-oriented resume ranking system developed for the **Redrob Resume Ranking Hackathon**. The system ranks candidates against a given Job Description using an interpretable, deterministic, CPU-only scoring pipeline designed to satisfy real-world recruitment constraints.

Unlike approaches that rely on hosted Large Language Models or GPU inference, Garuda performs ranking entirely offline using engineered features extracted from candidate profiles. This makes the system scalable, reproducible, explainable, and suitable for large candidate pools.

The final output is a ranked list of the **Top 100 candidates** together with human-readable reasoning for every recommendation.

---

# Key Features

* Deterministic ranking
* CPU-only execution
* No external API calls
* No hosted LLM dependency during inference
* Runtime under competition limits
* Explainable ranking
* Multi-factor candidate evaluation
* Edge-case testing
* Submission validation
* Fully reproducible pipeline

---

# Repository Structure

```
Garuda/
│
├── README.md
├── LICENSE
├── .gitignore
│
├── rank.py
├── DESIGN.md
├── requirements.txt
├── submission_metadata.yaml
├── submission.csv
│
├── sample_candidates.json
├── sample_candidates.jsonl
│
├── sandbox/
│   └── ResumeRankingDemo.ipynb
│
├── tests/
│   ├── generate_edge_cases.py
│   ├── validate_submission.py
│   ├── edge_case_candidates.jsonl
│   └── README.md
│
└── docs/
```

---

# System Architecture

```
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
Ranking
        │
        ▼
Reasoning Generator
        │
        ▼
submission.csv
```

---

# Ranking Methodology

Garuda evaluates every candidate using multiple independent signals instead of relying solely on keyword overlap.

The scoring engine combines:

* Relevant years of experience
* Current role alignment
* Career progression
* Technical skills
* AI / ML expertise
* Project relevance
* Education
* Recruiter engagement signals
* Profile completeness
* Notice period
* Work preferences
* Hiring signals
* Consistency validation

Each feature contributes to a normalized weighted score.

The final score is deterministic and reproducible.

---

# Anti-Gaming Measures

The ranking pipeline includes safeguards against common manipulation techniques including:

* Keyword stuffing
* Inflated skill lists
* Suspicious experience timelines
* Inconsistent profiles
* Unrealistic expertise claims
* Low-quality profile signals

These checks help naturally reduce the ranking of synthetic or honeypot candidates.

---

# Explainability

Every ranked candidate receives an automatically generated explanation based on factual information extracted from the profile.

The reasoning includes:

* Years of experience
* Relevant technologies
* Domain alignment
* Career strengths
* Recruiter signals
* Potential concerns (when applicable)

No hallucinated information is introduced.

---

# Performance

Measured on the official dataset.

| Metric     |                     Value |
| ---------- | ------------------------: |
| Candidates |                   100,000 |
| Runtime    |               ~40 seconds |
| Compute    |                       CPU |
| GPU        |              Not Required |
| Network    |                  Disabled |
| Output     | Top 100 Ranked Candidates |

---

# Installation

Clone the repository.

```bash
git clone https://github.com/Re-veil-eb/Garuda.git
cd Garuda
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

# Running the Ranker

Place the official `candidates.jsonl` file in the repository root.

Execute:

```bash
python rank.py \
    --candidates ./candidates.jsonl \
    --out ./submission.csv
```

The generated CSV follows the competition specification:

```
candidate_id,rank,score,reasoning
```

---

# Validation

Run the official validator.

```bash
python tests/validate_submission.py submission.csv
```

Expected output:

```
Submission is valid.
```

---

# Edge Case Testing

Generate adversarial candidate profiles.

```bash
python tests/generate_edge_cases.py
```

Evaluate them using:

```bash
python rank.py \
    --candidates tests/edge_case_candidates.jsonl \
    --out edge_case_submission.csv
```

---

# Sample Execution

For quick testing without the full dataset:

```bash
python rank.py \
    --candidates sample_candidates.jsonl \
    --out sample_submission.csv
```

---

# Reproducibility

The ranking process is completely deterministic.

Given identical input data, the generated submission will always be identical.

No randomness is used during inference.

No external services are contacted.

---

# Competition Constraints

| Constraint      | Status |
| --------------- | ------ |
| CPU Only        | ✅      |
| No GPU          | ✅      |
| No Network      | ✅      |
| Runtime < 5 min | ✅      |
| Memory < 16 GB  | ✅      |
| Reproducible    | ✅      |

---

# AI Usage

AI-assisted development tools were used during implementation for design discussions, code review, documentation, and iterative refinement.

The ranking pipeline itself performs no external AI inference during execution and complies fully with the competition's compute constraints.

---

# Future Improvements

* Learning-to-Rank models
* Adaptive feature weighting
* Semantic skill normalization
* Graph-based career trajectory modeling
* Recruiter feedback integration
* Fairness-aware ranking
* Calibration using hiring outcomes

---

# License

This project is released under the MIT License.

---

# Author

Developed as part of the **Redrob Resume Ranking Hackathon**.

All ranking logic, feature engineering, testing, and evaluation pipeline are fully reproducible from the source code contained in this repository.
