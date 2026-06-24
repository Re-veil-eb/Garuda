#!/usr/bin/env python3
"""
rank.py — Redrob Hackathon ranking system.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Optional:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv --no-embeddings
        Skip the sentence-transformer step and use TF-IDF cosine similarity instead.
        (TF-IDF is actually the default — see ARCHITECTURE NOTE below.)

Compute budget target: <=5 min wall-clock, <=16GB RAM, CPU only, no network, no GPU.
"""

import argparse
import gzip
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────────────────────────────────
# ARCHITECTURE NOTE
#
# semantic_fit uses TF-IDF + cosine similarity by default, NOT a sentence
# embedding model. Reasoning: sentence-transformers requires a model download
# at first use unless weights are vendored in the repo; the brief explicitly
# disallows network calls during ranking. TF-IDF needs no external weights,
# runs in milliseconds at 100k scale, and is sufficient for a MINORITY-WEIGHTED
# sub-score (0.15 of final score) whose entire purpose is to catch loose
# semantic overlap the structural sub-scores miss — it does not need to be a
# state-of-the-art embedding to serve that role. If a local sentence-transformer
# checkpoint is vendored into the repo (see README), pass --embeddings-model
# to use it instead; the code path exists and is isolated in `embed_semantic_fit`.
# ──────────────────────────────────────────────────────────────────────────

TODAY = date(2026, 6, 20)  # override via --as-of for reproducibility in review

# ── JD-derived constants (Section: "Senior AI Engineer — Founding Team") ───

CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mindtree",
}

ML_TITLE_LEXICON = [
    "machine learning", "ml engineer", "applied ml", "applied scientist",
    "data scientist", "ai engineer", "research scientist", "nlp engineer",
    "recommendation", "ranking", "search engineer", "retrieval",
    "ml ", " ml", "ai ", " ai", "deep learning", "computer vision engineer",
]

PRODUCT_COMPANY_INDUSTRY_HINTS = {
    "food delivery", "fintech", "e-commerce", "ai/ml", "transportation",
    "software", "internet", "saas", "gaming", "social media", "marketplace",
}

REQUIRED_SKILLS = {
    # embeddings-based retrieval
    "embeddings": 1.0, "sentence transformers": 1.0, "bge": 1.0, "e5": 1.0,
    "vector search": 0.9, "information retrieval": 0.9,
    # vector db / hybrid search
    "pinecone": 1.0, "weaviate": 1.0, "qdrant": 1.0, "milvus": 1.0,
    "opensearch": 0.9, "elasticsearch": 0.9, "faiss": 1.0,
    # python proxy
    "python": 1.0,
    # eval frameworks
    "recommendation systems": 0.7, "feature engineering": 0.5,
}

NICE_TO_HAVE_SKILLS = {
    "lora": 0.5, "qlora": 0.5, "peft": 0.5, "fine-tuning llms": 0.5,
    "learning to rank": 0.5, "xgboost": 0.3, "lightgbm": 0.3,
    "distributed systems": 0.3, "kafka": 0.2, "spark": 0.2,
}

JD_PREFERRED_LOCATIONS = {
    "noida", "pune", "hyderabad", "mumbai", "delhi", "gurgaon", "gurugram",
    "ncr", "bangalore", "bengaluru",
}

PROFICIENCY_SCORE = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.75, "expert": 1.0}

JD_TEXT = """
Senior AI Engineer Founding Team. Own the intelligence layer: ranking, retrieval,
and matching systems. Production experience with embeddings-based retrieval systems
(sentence-transformers, OpenAI embeddings, BGE, E5) deployed to real users, handling
embedding drift, index refresh, retrieval-quality regression in production. Production
experience with vector databases or hybrid search infrastructure: Pinecone, Weaviate,
Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS. Strong Python and code quality.
Hands-on experience designing evaluation frameworks for ranking systems: NDCG, MRR, MAP,
offline-to-online correlation, A/B test interpretation. LLM fine-tuning LoRA QLoRA PEFT.
Learning-to-rank models XGBoost neural. Distributed systems large-scale inference
optimization. Applied ML AI roles at product companies, not pure research, not pure
services consulting. Shipped end-to-end ranking search recommendation system to real
users at meaningful scale. Opinions on hybrid vs dense retrieval, offline vs online
evaluation, fine-tune vs prompt LLM integration.
""".strip()

# ── Weights (tunable, intentionally not buried) ────────────────────────────
W_ROLE_FIT = 0.30
W_SKILL_FIT = 0.25
W_SEMANTIC_FIT = 0.15
W_AVAILABILITY = 0.10
DISQUALIFIER_SCALE = 0.35     # multiplies summed disqualifier penalty [0,1] before subtracting
HONEYPOT_PENALTY = 0.90       # large fixed subtraction if honeypot_score exceeds threshold
HONEYPOT_THRESHOLD = 0.5


# ── Data loading ────────────────────────────────────────────────────────────

def load_candidates(path: str) -> list[dict]:
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    mode = "rt"
    records = []
    with opener(p, mode, encoding="utf-8") as f:
        if p.suffix == ".json" and not str(p).endswith(".jsonl"):
            records = json.load(f)
        else:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


# ── Feature extraction helpers ──────────────────────────────────────────────

def safe_lower(s):
    return (s or "").lower()


def safe_num(d: dict, key: str, default):
    """Like dict.get(key, default) but treats an explicit None as missing too,
    WITHOUT the `value or default` bug that also clobbers a legitimate 0 or 0.0.
    `x or default` is wrong whenever 0/0.0 is itself a valid, meaningful value
    (e.g. notice_period_days=0 means immediately available — a real, good value,
    not a missing one). Found via edge-case testing: a synthetic honeypot test
    candidate with notice_period_days=0 was silently scored as if it were 60."""
    val = d.get(key, default)
    return default if val is None else val


def parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def months_between(d1: date, d2: date) -> float:
    return (d2.year - d1.year) * 12 + (d2.month - d1.month) + (d2.day - d1.day) / 30.0


def is_consulting_firm(company: str) -> bool:
    c = safe_lower(company)
    return any(firm in c for firm in CONSULTING_FIRMS)


def is_ml_title(title: str) -> bool:
    t = safe_lower(title)
    return any(kw in t for kw in ML_TITLE_LEXICON)


def is_product_company_role(role: dict) -> bool:
    """True only on positive evidence of a product company. Consulting firms are
    excluded explicitly; everything else ambiguous (manufacturing, paper products,
    generic 'IT Services' with no product hint) is NOT credited as product-company
    just by elimination — the JD's bar is specific, not 'anything but consulting'."""
    industry = safe_lower(role.get("industry", ""))
    if is_consulting_firm(role.get("company", "")):
        return False
    return any(hint in industry for hint in PRODUCT_COMPANY_INDUSTRY_HINTS)


def location_is_jd_preferred(location: str) -> bool:
    loc = safe_lower(location)
    return any(city in loc for city in JD_PREFERRED_LOCATIONS)


# ── Sub-score: role_fit ─────────────────────────────────────────────────────

def compute_role_fit(candidate: dict) -> tuple[float, dict]:
    profile = candidate.get("profile", {}) or {}
    career = candidate.get("career_history", []) or []
    yoe = profile.get("years_of_experience", 0) or 0

    # YOE band fit: ideal 5-9, graceful falloff outside (JD: "we'll seriously
    # consider candidates outside the band if other signals are strong")
    if 5 <= yoe <= 9:
        yoe_score = 1.0
    elif yoe < 5:
        yoe_score = max(0.0, 1.0 - (5 - yoe) * 0.18)
    else:
        yoe_score = max(0.0, 1.0 - (yoe - 9) * 0.10)

    # Applied-ML-at-product-company tenure (months), as fraction of total career
    total_months = sum(r.get("duration_months", 0) or 0 for r in career)
    ml_product_months = sum(
        r.get("duration_months", 0) or 0
        for r in career
        if is_ml_title(r.get("title", "")) and is_product_company_role(r)
    )
    if total_months > 0:
        ml_fraction = ml_product_months / total_months
    else:
        ml_fraction = 0.0
    # JD wants roughly 4-5 of 6-8 years in this profile -> ~55-65% fraction is ideal
    tenure_score = min(1.0, ml_fraction / 0.55)

    # Current/most-recent title match
    current_title = profile.get("current_title", "")
    title_score = 1.0 if is_ml_title(current_title) else (
        0.4 if career and is_ml_title(career[0].get("title", "")) else 0.0
    )

    role_fit = 0.35 * yoe_score + 0.40 * tenure_score + 0.25 * title_score
    detail = {
        "yoe": yoe, "yoe_score": round(yoe_score, 3),
        "ml_product_months": ml_product_months, "total_months": total_months,
        "tenure_score": round(tenure_score, 3),
        "title_score": round(title_score, 3),
        "current_title": current_title,
    }
    return min(1.0, role_fit), detail


# ── Sub-score: skill_fit ────────────────────────────────────────────────────

def has_any_ml_adjacent_role(career: list[dict]) -> bool:
    """True if ANY career_history entry has an ML/AI/data/search/ranking-adjacent
    title. Used to corroborate skill claims against actual role history — a
    skills list full of advanced LLM/retrieval keywords attached to a career of
    Graphic Designer -> Mechanical Engineer is the JD's explicitly-named trap
    ('a candidate who has all the AI keywords listed as skills but whose title
    is Marketing Manager is not a fit, no matter how perfect their skill list
    looks'). This check is intentionally broader than is_ml_title (it also
    accepts data/analytics/backend-adjacent titles) because real practitioners
    often transition from adjacent roles, not just from an ML-titled role."""
    adjacent_kw = ML_TITLE_LEXICON + [
        "data engineer", "data analyst", "backend", "software engineer",
        "full stack", "platform engineer", "devops", "cloud engineer",
        "qa engineer", "test", "research",
    ]
    return any(
        any(kw in safe_lower(r.get("title", "")) for kw in adjacent_kw)
        for r in career
    )


def compute_skill_fit(candidate: dict) -> tuple[float, dict]:
    skills = candidate.get("skills", []) or []
    career = candidate.get("career_history", []) or []
    assess = (candidate.get("redrob_signals", {}) or {}).get("skill_assessment_scores", {}) or {}

    matched_required = []
    required_score = 0.0
    required_total_weight = sum(REQUIRED_SKILLS.values())

    for sk in skills:
        name = safe_lower(sk.get("name", ""))
        prof = sk.get("proficiency", "beginner")
        prof_score = PROFICIENCY_SCORE.get(prof, 0.25)
        for req_name, req_weight in REQUIRED_SKILLS.items():
            if req_name in name or name in req_name:
                contribution = req_weight * prof_score
                required_score += contribution
                matched_required.append((sk.get("name"), prof))

    nice_score = 0.0
    matched_nice = []
    for sk in skills:
        name = safe_lower(sk.get("name", ""))
        prof_score = PROFICIENCY_SCORE.get(sk.get("proficiency", "beginner"), 0.25)
        for nice_name, nice_weight in NICE_TO_HAVE_SKILLS.items():
            if nice_name in name or name in nice_name:
                nice_score += nice_weight * prof_score
                matched_nice.append(sk.get("name"))

    # assessment score bonus: if Redrob's own assessment backs up a claimed
    # required skill, add a small confirmation bonus (caps contribution so it
    # can't dominate, since most candidates have empty assessment maps)
    assess_bonus = 0.0
    for skill_name, score in assess.items():
        if safe_lower(skill_name) in REQUIRED_SKILLS and score >= 60:
            assess_bonus += 0.05

    raw = (required_score / required_total_weight) + min(0.15, nice_score / 10) + min(0.1, assess_bonus)

    # Career corroboration discount: if the skills list claims meaningful AI/ML
    # skill depth but NO career_history entry has an ML/AI-adjacent title, that's
    # the JD's explicitly-named trap (skills section full of keywords, career
    # history shows an unrelated profession). Discount heavily rather than zero
    # it out — some genuine candidates do self-teach without yet having an
    # ML-titled role, and the JD itself says it doesn't want to reject on title
    # alone — but a high raw skill score with zero corroborating role history
    # should not be allowed to compete with a corroborated one.
    corroborated = has_any_ml_adjacent_role(career)
    if not corroborated and raw > 0.25:
        corroboration_discount = 0.4   # heavy but not total — see comment above
    else:
        corroboration_discount = 1.0

    skill_fit = min(1.0, raw * corroboration_discount)
    detail = {
        "matched_required": matched_required[:5],
        "matched_nice": matched_nice[:5],
        "required_score": round(required_score, 3),
        "career_corroborated": corroborated,
    }
    return skill_fit, detail


# ── Sub-score: disqualifier_penalty ─────────────────────────────────────────

def compute_disqualifier_penalty(candidate: dict) -> tuple[float, dict]:
    career = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    profile = candidate.get("profile", {}) or {}
    skill_names = [safe_lower(s.get("name", "")) for s in skills]

    penalties = []

    # 1. Pure research-only career (no production/engineering titles at all)
    has_engineering_title = any(
        any(kw in safe_lower(r.get("title", "")) for kw in
            ["engineer", "developer", "scientist", "analyst", "architect"])
        for r in career
    )
    if career and not has_engineering_title:
        penalties.append(("pure_research_or_non_technical_career", 0.3))

    # 2. "AI experience" entirely <12mo old with no pre-LLM-era ML production work
    ai_skill_names = {"langchain", "fine-tuning llms", "prompt engineering", "rag"}
    has_recent_ai_only_skill = any(
        any(a in n for a in ai_skill_names) and (s.get("duration_months", 99) or 99) < 12
        for s, n in zip(skills, skill_names)
    )
    has_older_ml_production = any(
        (r.get("duration_months", 0) or 0) > 12 and is_ml_title(r.get("title", ""))
        for r in career
    )
    if has_recent_ai_only_skill and not has_older_ml_production:
        penalties.append(("recent_ai_buzzword_only_no_production_history", 0.35))

    # 3. Entire career at consulting firms with zero product-company entries
    if career and all(is_consulting_firm(r.get("company", "")) for r in career):
        penalties.append(("consulting_only_career", 0.4))

    # 4. CV/speech/robotics-primary with no NLP/IR-adjacent skills
    cv_speech_robotics_kw = {"image classification", "object detection", "speech recognition",
                              "robotics", "yolo", "opencv", "cnn", "gans"}
    nlp_ir_kw = {"nlp", "embeddings", "information retrieval", "recommendation systems",
                 "vector search", "bm25", "transformers", "llm"}
    has_cv_speech = any(any(k in n for k in cv_speech_robotics_kw) for n in skill_names)
    has_nlp_ir = any(any(k in n for k in nlp_ir_kw) for n in skill_names)
    if has_cv_speech and not has_nlp_ir:
        penalties.append(("cv_speech_robotics_without_nlp_ir", 0.25))

    # 5. Title-chasing: 3+ employers, each tenure <=18mo, escalating seniority words
    short_stints = sum(1 for r in career if (r.get("duration_months", 999) or 999) <= 18)
    if len(career) >= 3 and short_stints >= 3:
        penalties.append(("frequent_job_hopping_pattern", 0.15))

    total_penalty = sum(p for _, p in penalties)
    return min(1.0, total_penalty), {"penalties": penalties}


# ── Sub-score: honeypot ─────────────────────────────────────────────────────

def compute_honeypot_score(candidate: dict) -> tuple[float, dict]:
    profile = candidate.get("profile", {}) or {}
    career = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    signals = candidate.get("redrob_signals", {}) or {}

    flags = []

    yoe = profile.get("years_of_experience", 0) or 0
    total_career_months = sum(r.get("duration_months", 0) or 0 for r in career)
    # career history should roughly track stated YOE; large mismatch is a tell
    if yoe > 3 and total_career_months > 0:
        ratio = total_career_months / (yoe * 12)
        if ratio < 0.4:
            flags.append("career_history_months_far_below_stated_yoe")

    # 5+ skills at expert with duration_months < 12 each
    expert_new_skills = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and (s.get("duration_months", 99) or 99) < 12
    )
    if expert_new_skills >= 5:
        flags.append("multiple_expert_skills_with_under_a_year_duration")

    # NEW: any single skill claimed as expert/advanced with near-zero duration
    # (looser version of the above — catches a single egregious claim, not just
    # five at once, since the brief's own example is a single-skill pattern:
    # "'expert' proficiency in 10 skills with 0 years used" generalizes down to
    # even one skill at expert with ~0 months as worth flagging on its own)
    zero_duration_expert = [
        s.get("name") for s in skills
        if s.get("proficiency") in ("expert", "advanced") and (s.get("duration_months", 99) or 99) <= 1
    ]
    if len(zero_duration_expert) >= 1:
        flags.append("expert_or_advanced_skill_with_near_zero_duration")

    # NEW: skill count wildly disproportionate to years_of_experience
    # (a candidate with 1 YOE claiming 15+ skills, several at advanced/expert,
    # is a plausible "keyword stuffing" honeypot shape independent of duration)
    advanced_plus = sum(1 for s in skills if s.get("proficiency") in ("advanced", "expert"))
    if yoe > 0 and yoe < 2 and advanced_plus >= 6:
        flags.append("many_advanced_skills_for_very_junior_yoe")

    # NOTE: an earlier version of this function included an
    # "endorsements implausibly high relative to connections" check. Removed
    # after testing against real data showed it fires on legitimate candidates
    # with small-but-real networks (e.g., 10 connections / 42 endorsements is
    # unusual but not impossible — endorsements and connection count are not
    # causally linked in a way that makes a high ratio diagnostic of anything).
    # Kept as a comment rather than silently dropped so the false-start is
    # visible and defensible in review: this is what "test against real data
    # before trusting a heuristic" is supposed to catch.

    # assessment scores implausibly uniform and high
    assess = signals.get("skill_assessment_scores", {}) or {}
    if len(assess) >= 5:
        vals = list(assess.values())
        if min(vals) > 90 and (max(vals) - min(vals)) < 5:
            flags.append("uniformly_high_assessment_scores")

    # profile completed and "active" within 24h of signup, implausibly complete
    completeness = safe_num(signals, "profile_completeness_score", 0)
    signup = parse_date(signals.get("signup_date"))
    last_active = parse_date(signals.get("last_active_date"))
    if completeness > 98 and signup and last_active and abs((last_active - signup).days) < 1:
        flags.append("fully_complete_profile_active_same_day_as_signup")

    # NEW: response rate / response time contradiction
    # (claims to respond to recruiters near-instantly AND to 100% of them, which
    # is a "too good" combination rarely seen organically — the original draft's
    # disqualifier rule H2 in Module 2 used this; it generalizes well here too)
    response_rate = safe_num(signals, "recruiter_response_rate", 0)
    response_time = safe_num(signals, "avg_response_time_hours", 999)
    if response_rate >= 0.98 and response_time < 0.5:
        flags.append("implausibly_perfect_recruiter_responsiveness")

    honeypot_score = min(1.0, len(flags) * 0.3)
    return honeypot_score, {"flags": flags}


# ── Sub-score: availability ─────────────────────────────────────────────────

def compute_availability(candidate: dict) -> tuple[float, dict]:
    signals = candidate.get("redrob_signals", {}) or {}
    profile = candidate.get("profile", {}) or {}

    last_active = parse_date(signals.get("last_active_date"))
    if last_active:
        days_inactive = (TODAY - last_active).days
        recency_score = max(0.0, 1.0 - days_inactive / 120.0)
    else:
        recency_score = 0.0

    open_flag = 1.0 if signals.get("open_to_work_flag") else 0.3
    response_rate = safe_num(signals, "recruiter_response_rate", 0.0)

    notice = safe_num(signals, "notice_period_days", 60)
    if notice <= 30:
        notice_score = 1.0
    elif notice <= 60:
        notice_score = 0.7
    elif notice <= 90:
        notice_score = 0.45
    else:
        notice_score = 0.2

    willing = signals.get("willing_to_relocate", False)
    loc_ok = location_is_jd_preferred(profile.get("location", "")) or willing
    location_score = 1.0 if loc_ok else 0.4

    interview_rate = safe_num(signals, "interview_completion_rate", 0.5)

    availability = (
        0.30 * recency_score + 0.20 * open_flag + 0.20 * response_rate +
        0.15 * notice_score + 0.10 * location_score + 0.05 * interview_rate
    )
    detail = {
        "days_inactive": (TODAY - last_active).days if last_active else None,
        "recency_score": round(recency_score, 3),
        "response_rate": response_rate,
        "notice_period_days": notice,
        "location": profile.get("location"),
    }
    return min(1.0, availability), detail


# ── Sub-score: semantic_fit (TF-IDF, see ARCHITECTURE NOTE) ─────────────────

def build_candidate_document(candidate: dict) -> str:
    profile = candidate.get("profile", {}) or {}
    career = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
    ]
    for r in career[:3]:  # most recent roles carry more signal; cap for speed
        parts.append(r.get("title", ""))
        parts.append(r.get("description", ""))
    parts.append(" ".join(s.get("name", "") for s in skills))
    return " ".join(p for p in parts if p)


def compute_semantic_fit_batch(candidates: list[dict]) -> np.ndarray:
    docs = [build_candidate_document(c) for c in candidates]
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")
    all_docs = docs + [JD_TEXT]
    tfidf = vectorizer.fit_transform(all_docs)
    jd_vec = tfidf[-1]
    cand_vecs = tfidf[:-1]
    sims = cosine_similarity(cand_vecs, jd_vec).ravel()
    # normalize to [0,1] using min-max across the pool so weight semantics stay comparable
    if sims.max() > sims.min():
        sims = (sims - sims.min()) / (sims.max() - sims.min())
    return sims


# ── Reasoning generation ─────────────────────────────────────────────────────

def generate_reasoning(candidate: dict, role_detail: dict, skill_detail: dict,
                        disq_detail: dict, avail_detail: dict, honeypot_detail: dict,
                        final_score: float) -> str:
    profile = candidate.get("profile", {}) or {}
    parts = []

    yoe = role_detail["yoe"]
    title = profile.get("current_title") or "an unspecified role"
    if role_detail["tenure_score"] >= 0.6:
        parts.append(f"{yoe:.1f}y experience with substantial applied-ML/product-company tenure (~{role_detail['ml_product_months']}mo)")
    elif role_detail["title_score"] > 0:
        parts.append(f"{yoe:.1f}y experience, currently {title}, partial ML/product-company alignment")
    else:
        parts.append(f"{yoe:.1f}y experience as {title} — limited direct applied-ML/product-company overlap")

    if skill_detail["matched_required"]:
        skill_str = ", ".join(f"{n} ({p})" for n, p in skill_detail["matched_required"][:3])
        if not skill_detail.get("career_corroborated", True):
            parts.append(f"lists matching skills ({skill_str}) but career history shows no ML/AI-adjacent role to corroborate them — weighted down accordingly")
        else:
            parts.append(f"matches required skills: {skill_str}")
    else:
        parts.append("no strong overlap with the JD's core required skills (embeddings/vector-search/eval)")

    if disq_detail["penalties"]:
        concern_str = "; ".join(p[0].replace("_", " ") for p in disq_detail["penalties"][:2])
        parts.append(f"concerns: {concern_str}")

    if avail_detail["days_inactive"] is not None and avail_detail["days_inactive"] > 90:
        parts.append(f"inactive {avail_detail['days_inactive']}d — availability uncertain")
    elif avail_detail["response_rate"] is not None and avail_detail["response_rate"] < 0.2:
        parts.append(f"low recruiter response rate ({avail_detail['response_rate']:.0%})")

    if honeypot_detail["flags"]:
        parts.append("profile shows structural anomalies, treated with caution")

    text = "; ".join(parts) + "."
    return text[:300]


# ── Main pipeline ────────────────────────────────────────────────────────────

def rank_candidates(candidates: list[dict], use_tfidf: bool = True) -> pd.DataFrame:
    n = len(candidates)
    rows = []

    semantic_scores = compute_semantic_fit_batch(candidates) if use_tfidf else np.zeros(n)

    for i, c in enumerate(candidates):
        role_fit, role_detail = compute_role_fit(c)
        skill_fit, skill_detail = compute_skill_fit(c)
        disq_penalty, disq_detail = compute_disqualifier_penalty(c)
        honeypot_score, honeypot_detail = compute_honeypot_score(c)
        availability, avail_detail = compute_availability(c)
        semantic_fit = float(semantic_scores[i])

        score = (
            W_ROLE_FIT * role_fit
            + W_SKILL_FIT * skill_fit
            + W_SEMANTIC_FIT * semantic_fit
            + W_AVAILABILITY * availability
            - DISQUALIFIER_SCALE * disq_penalty
        )
        if honeypot_score >= HONEYPOT_THRESHOLD:
            score -= HONEYPOT_PENALTY

        reasoning = generate_reasoning(c, role_detail, skill_detail, disq_detail,
                                        avail_detail, honeypot_detail, score)

        rows.append({
            "candidate_id": c.get("candidate_id"),
            "score": score,
            "reasoning": reasoning,
            "_role_fit": role_fit, "_skill_fit": skill_fit,
            "_semantic_fit": semantic_fit, "_availability": availability,
            "_disq_penalty": disq_penalty, "_honeypot_score": honeypot_score,
        })

    df = pd.DataFrame(rows)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-embeddings", action="store_true",
                         help="Disable TF-IDF semantic_fit step entirely (sets it to 0).")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--debug-columns", action="store_true",
                         help="Keep internal _* score columns in a sidecar debug CSV.")
    args = parser.parse_args()

    t0 = time.time()
    candidates = load_candidates(args.candidates)
    print(f"Loaded {len(candidates)} candidates in {time.time()-t0:.1f}s", file=sys.stderr)

    t1 = time.time()
    df = rank_candidates(candidates, use_tfidf=not args.no_embeddings)
    print(f"Scored {len(df)} candidates in {time.time()-t1:.1f}s", file=sys.stderr)

    df = df.sort_values(by=["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    top = df.head(args.top_n).copy()
    top["rank"] = range(1, len(top) + 1)
    top["score"] = top["score"].round(6)

    out_cols = ["candidate_id", "rank", "score", "reasoning"]
    top[out_cols].to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(top)} rows)", file=sys.stderr)

    if args.debug_columns:
        debug_path = str(Path(args.out).with_suffix("")) + "_debug.csv"
        top.to_csv(debug_path, index=False)
        print(f"Wrote debug detail to {debug_path}", file=sys.stderr)

    print(f"Total wall-clock: {time.time()-t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
