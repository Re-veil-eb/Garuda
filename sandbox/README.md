# Google Colab Sandbox

## Overview

This notebook provides a lightweight demonstration of the Garuda Resume Ranking System.

The purpose of this sandbox is to allow reviewers to verify that the ranking pipeline executes successfully in a clean, CPU-only environment without requiring the full competition dataset.

To keep execution lightweight, the notebook uses a small sample dataset (≤100 candidates). The complete 100,000-candidate evaluation is reproduced from the repository during Stage 3 using the official dataset.

---

# Environment

Platform:

* Google Colab
* Python 3.12
* CPU Runtime
* No GPU Required
* No Internet/API Calls during ranking

---

# Repository

Clone the repository:

```bash
git clone https://github.com/Re-veil-eb/Garuda.git
cd Garuda
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

Dependencies include:

* pandas
* numpy
* scikit-learn

---

# Sample Dataset

The notebook includes a sample dataset:

```
sample_candidates.jsonl
```

containing approximately 100 candidate profiles.

This dataset exists only for demonstration purposes.

---

# Running the Ranker

Execute:

```bash
python rank.py \
    --candidates sample_candidates.jsonl \
    --out sample_submission.csv
```

Expected console output:

```
Loaded XX candidates
Scored XX candidates
Wrote sample_submission.csv
```

---

# Output

The generated CSV follows the official competition format:

```
candidate_id,rank,score,reasoning
```

The notebook displays:

* Top ranked candidates
* Generated reasoning
* Final ranking score

---

# Validation

Run the validator:

```bash
python tests/validate_submission.py sample_submission.csv
```

This verifies:

* Correct CSV format
* Required columns
* Valid ranking
* Score ordering
* Duplicate detection

---

# Competition Constraints

This notebook demonstrates execution under the same constraints expected during evaluation.

| Constraint          | Status |
| ------------------- | ------ |
| CPU Only            | ✅      |
| GPU Required        | ❌      |
| Network Calls       | ❌      |
| Deterministic       | ✅      |
| Runtime < 5 Minutes | ✅      |

---

# Reproducing the Official Submission

The full submission can be reproduced using the official dataset supplied by the organizers.

```bash
python rank.py \
    --candidates candidates.jsonl \
    --out submission.csv
```

During local testing:

* Dataset Size: 100,000 candidates
* Runtime: ~40 seconds
* CPU Only
* No external API calls

---

# Notes

The official competition dataset (`candidates.jsonl`) is intentionally excluded from the repository because it exceeds GitHub's file size limit and is distributed separately by the organizers.

Reviewers can replace the sample dataset with the official dataset using the same command shown above.

---

# Expected Files

```
sample_candidates.jsonl
rank.py
requirements.txt
tests/
submission_metadata.yaml
```

---

# Contact

This notebook accompanies the Garuda repository submitted for the Redrob Resume Ranking Hackathon and serves as a reproducible sandbox for verifying the ranking pipeline.
