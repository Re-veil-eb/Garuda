#!/usr/bin/env python3
"""
Generate adversarial/edge-case synthetic candidates to stress-test rank.py.

Unlike the earlier "synthetic_100k" load test (which just replicated real records
for timing purposes), this generates structurally distinct edge cases designed to
break specific assumptions in the scoring code: missing/null fields, empty arrays,
extreme values, malformed dates, duplicate IDs, and unicode/encoding edge cases.

The goal is to find crashes or silently wrong scores BEFORE they happen on the real
100k pool, where at least a few of these patterns are statistically likely to occur
even if they're absent from the 1000-record sample.
"""
import json

cases = []

# 1. Completely empty optional arrays (career_history minItems=1 in schema, but
#    test what happens if a real record violates that, or has empty skills/education)
cases.append({
    "candidate_id": "CAND_9000001",
    "profile": {
        "anonymized_name": "Empty Arrays Test", "headline": "", "summary": "",
        "location": "", "country": "", "years_of_experience": 0,
        "current_title": "", "current_company": "", "current_company_size": "1-10",
        "current_industry": ""
    },
    "career_history": [],
    "education": [],
    "skills": [],
    "certifications": [],
    "languages": [],
    "redrob_signals": {
        "profile_completeness_score": 0, "signup_date": "2026-06-01",
        "last_active_date": "2026-06-01", "open_to_work_flag": False,
        "profile_views_received_30d": 0, "applications_submitted_30d": 0,
        "recruiter_response_rate": 0.0, "avg_response_time_hours": 0,
        "skill_assessment_scores": {}, "connection_count": 0,
        "endorsements_received": 0, "notice_period_days": 0,
        "expected_salary_range_inr_lpa": {"min": 0, "max": 0},
        "preferred_work_mode": "remote", "willing_to_relocate": False,
        "github_activity_score": -1, "search_appearance_30d": 0,
        "saved_by_recruiters_30d": 0, "interview_completion_rate": 0.0,
        "offer_acceptance_rate": -1, "verified_email": False,
        "verified_phone": False, "linkedin_connected": False
    }
})

# 2. Null values where the schema technically allows them (end_date null is normal,
#    but test null grade, null in unexpected places)
cases.append({
    "candidate_id": "CAND_9000002",
    "profile": {
        "anonymized_name": "Null Fields Test", "headline": "AI Engineer",
        "summary": "Test candidate with nulls", "location": "Pune", "country": "India",
        "years_of_experience": 7.0, "current_title": "Senior ML Engineer",
        "current_company": "TestCo", "current_company_size": "201-500",
        "current_industry": "AI/ML"
    },
    "career_history": [
        {"company": "TestCo", "title": "Senior ML Engineer", "start_date": "2022-01-01",
         "end_date": None, "duration_months": 53, "is_current": True,
         "industry": "AI/ML", "company_size": "201-500", "description": None}
    ],
    "education": [
        {"institution": "Test University", "degree": "B.Tech", "field_of_study": "CS",
         "start_year": 2015, "end_year": 2019, "grade": None, "tier": "unknown"}
    ],
    "skills": [
        {"name": "Python", "proficiency": "expert", "endorsements": 0, "duration_months": None}
    ],
    "certifications": [], "languages": [],
    "redrob_signals": {
        "profile_completeness_score": 70, "signup_date": "2024-01-01",
        "last_active_date": "2026-06-15", "open_to_work_flag": True,
        "profile_views_received_30d": 10, "applications_submitted_30d": 1,
        "recruiter_response_rate": 0.5, "avg_response_time_hours": 24,
        "skill_assessment_scores": {}, "connection_count": 100,
        "endorsements_received": 10, "notice_period_days": 30,
        "expected_salary_range_inr_lpa": {"min": 20, "max": 30},
        "preferred_work_mode": "hybrid", "willing_to_relocate": True,
        "github_activity_score": 50, "search_appearance_30d": 50,
        "saved_by_recruiters_30d": 2, "interview_completion_rate": 0.5,
        "offer_acceptance_rate": 0.5, "verified_email": True,
        "verified_phone": True, "linkedin_connected": True
    }
})

# 3. Extreme values: very high YOE, huge career_history list, huge skills list
cases.append({
    "candidate_id": "CAND_9000003",
    "profile": {
        "anonymized_name": "Extreme Values Test", "headline": "Veteran Engineer",
        "summary": "x" * 5000,  # very long summary text
        "location": "Noida, Uttar Pradesh", "country": "India",
        "years_of_experience": 45.0, "current_title": "Principal AI Architect",
        "current_company": "BigCo", "current_company_size": "10001+",
        "current_industry": "AI/ML"
    },
    "career_history": [
        {"company": f"Company{i}", "title": "ML Engineer", "start_date": "2000-01-01",
         "end_date": "2001-01-01", "duration_months": 12, "is_current": False,
         "industry": "AI/ML", "company_size": "10001+", "description": "desc"}
        for i in range(10)  # schema says maxItems 10, test the boundary
    ],
    "education": [
        {"institution": f"Univ{i}", "degree": "PhD", "field_of_study": "CS",
         "start_year": 1990 + i, "end_year": 1994 + i, "grade": "10 CGPA", "tier": "tier_1"}
        for i in range(5)  # schema says maxItems 5
    ],
    "skills": [
        {"name": f"Skill{i}", "proficiency": "expert", "endorsements": 999, "duration_months": 500}
        for i in range(50)  # no maxItems on skills in schema -- test a large list
    ],
    "certifications": [], "languages": [],
    "redrob_signals": {
        "profile_completeness_score": 100, "signup_date": "2010-01-01",
        "last_active_date": "2026-06-20", "open_to_work_flag": True,
        "profile_views_received_30d": 99999, "applications_submitted_30d": 0,
        "recruiter_response_rate": 1.0, "avg_response_time_hours": 0.01,
        "skill_assessment_scores": {f"Skill{i}": 100 for i in range(50)},
        "connection_count": 50000, "endorsements_received": 50000,
        "notice_period_days": 180, "expected_salary_range_inr_lpa": {"min": 999, "max": 9999},
        "preferred_work_mode": "flexible", "willing_to_relocate": True,
        "github_activity_score": 100, "search_appearance_30d": 99999,
        "saved_by_recruiters_30d": 9999, "interview_completion_rate": 1.0,
        "offer_acceptance_rate": 1.0, "verified_email": True,
        "verified_phone": True, "linkedin_connected": True
    }
})

# 4. Malformed/inconsistent salary (min > max, already known to occur in real data per CAND_0000009)
cases.append({
    "candidate_id": "CAND_9000004",
    "profile": {
        "anonymized_name": "Salary Inverted Test", "headline": "ML Engineer",
        "summary": "Test", "location": "Hyderabad, Telangana", "country": "India",
        "years_of_experience": 6.0, "current_title": "ML Engineer",
        "current_company": "MidCo", "current_company_size": "501-1000",
        "current_industry": "Fintech"
    },
    "career_history": [
        {"company": "MidCo", "title": "ML Engineer", "start_date": "2021-01-01",
         "end_date": None, "duration_months": 65, "is_current": True,
         "industry": "Fintech", "company_size": "501-1000", "description": "ML work"}
    ],
    "education": [
        {"institution": "Test U", "degree": "M.Tech", "field_of_study": "ML",
         "start_year": 2017, "end_year": 2019, "grade": "8.0", "tier": "tier_2"}
    ],
    "skills": [
        {"name": "Embeddings", "proficiency": "advanced", "endorsements": 10, "duration_months": 40},
        {"name": "FAISS", "proficiency": "advanced", "endorsements": 5, "duration_months": 30}
    ],
    "certifications": [], "languages": [],
    "redrob_signals": {
        "profile_completeness_score": 80, "signup_date": "2024-01-01",
        "last_active_date": "2026-06-10", "open_to_work_flag": True,
        "profile_views_received_30d": 20, "applications_submitted_30d": 2,
        "recruiter_response_rate": 0.6, "avg_response_time_hours": 30,
        "skill_assessment_scores": {}, "connection_count": 200,
        "endorsements_received": 15, "notice_period_days": 30,
        "expected_salary_range_inr_lpa": {"min": 35.0, "max": 12.0},  # inverted!
        "preferred_work_mode": "hybrid", "willing_to_relocate": True,
        "github_activity_score": 40, "search_appearance_30d": 80,
        "saved_by_recruiters_30d": 5, "interview_completion_rate": 0.7,
        "offer_acceptance_rate": 0.4, "verified_email": True,
        "verified_phone": True, "linkedin_connected": True
    }
})

# 5. Honeypot-style: explicit JD example -- "8 years experience at a company founded 3 years ago"
#    Simulated via implausible single-role tenure vs YOE, at a clearly-named young company
cases.append({
    "candidate_id": "CAND_9000005",
    "profile": {
        "anonymized_name": "Impossible Tenure Honeypot", "headline": "AI Engineer | Expert",
        "summary": "8 years of deep AI expertise.", "location": "Pune, Maharashtra",
        "country": "India", "years_of_experience": 8.0, "current_title": "AI Engineer",
        "current_company": "Rephrase.ai", "current_company_size": "51-200",
        "current_industry": "AI/ML"
    },
    "career_history": [
        {"company": "Rephrase.ai", "title": "AI Engineer", "start_date": "2018-01-01",
         "end_date": None, "duration_months": 96, "is_current": True,
         "industry": "AI/ML", "company_size": "51-200",
         "description": "8 years at this AI company."}
    ],  # Rephrase.ai is referenced elsewhere in the real data with founding-era tenure starting ~2020
    "education": [
        {"institution": "IIT Bombay", "degree": "B.Tech", "field_of_study": "CS",
         "start_year": 2010, "end_year": 2014, "grade": "9.0", "tier": "tier_1"}
    ],
    "skills": [
        {"name": "Embeddings", "proficiency": "expert", "endorsements": 100, "duration_months": 96},
        {"name": "FAISS", "proficiency": "expert", "endorsements": 100, "duration_months": 96},
        {"name": "Pinecone", "proficiency": "expert", "endorsements": 100, "duration_months": 96},
    ],
    "certifications": [], "languages": [],
    "redrob_signals": {
        "profile_completeness_score": 100, "signup_date": "2026-06-19",
        "last_active_date": "2026-06-19", "open_to_work_flag": True,
        "profile_views_received_30d": 50, "applications_submitted_30d": 1,
        "recruiter_response_rate": 1.0, "avg_response_time_hours": 0.05,
        "skill_assessment_scores": {"Embeddings": 99, "FAISS": 99, "Pinecone": 99},
        "connection_count": 500, "endorsements_received": 300,
        "notice_period_days": 0, "expected_salary_range_inr_lpa": {"min": 40, "max": 60},
        "preferred_work_mode": "onsite", "willing_to_relocate": True,
        "github_activity_score": 95, "search_appearance_30d": 500,
        "saved_by_recruiters_30d": 50, "interview_completion_rate": 1.0,
        "offer_acceptance_rate": 1.0, "verified_email": True,
        "verified_phone": True, "linkedin_connected": True
    }
})

# 6. Keyword-stuffed non-technical title (second instance of the trap pattern, different role)
cases.append({
    "candidate_id": "CAND_9000006",
    "profile": {
        "anonymized_name": "Keyword Stuffer Test", "headline": "Sales Executive | AI Enthusiast | RAG Pinecone FAISS Expert",
        "summary": "Sales professional exploring AI tools.", "location": "Mumbai, Maharashtra",
        "country": "India", "years_of_experience": 5.0, "current_title": "Sales Executive",
        "current_company": "RetailCo", "current_company_size": "201-500",
        "current_industry": "Retail"
    },
    "career_history": [
        {"company": "RetailCo", "title": "Sales Executive", "start_date": "2021-06-01",
         "end_date": None, "duration_months": 60, "is_current": True,
         "industry": "Retail", "company_size": "201-500",
         "description": "Enterprise sales, quota carrying."}
    ],
    "education": [
        {"institution": "Generic University", "degree": "MBA", "field_of_study": "Marketing",
         "start_year": 2015, "end_year": 2017, "grade": "70%", "tier": "tier_3"}
    ],
    "skills": [
        {"name": "Pinecone", "proficiency": "advanced", "endorsements": 30, "duration_months": 20},
        {"name": "FAISS", "proficiency": "advanced", "endorsements": 25, "duration_months": 18},
        {"name": "RAG", "proficiency": "advanced", "endorsements": 40, "duration_months": 22},
        {"name": "Embeddings", "proficiency": "advanced", "endorsements": 35, "duration_months": 19},
        {"name": "LangChain", "proficiency": "advanced", "endorsements": 28, "duration_months": 17},
        {"name": "Vector Search", "proficiency": "advanced", "endorsements": 30, "duration_months": 21},
    ],
    "certifications": [], "languages": [],
    "redrob_signals": {
        "profile_completeness_score": 85, "signup_date": "2025-01-01",
        "last_active_date": "2026-06-15", "open_to_work_flag": True,
        "profile_views_received_30d": 30, "applications_submitted_30d": 5,
        "recruiter_response_rate": 0.4, "avg_response_time_hours": 50,
        "skill_assessment_scores": {}, "connection_count": 300,
        "endorsements_received": 50, "notice_period_days": 30,
        "expected_salary_range_inr_lpa": {"min": 15, "max": 25},
        "preferred_work_mode": "hybrid", "willing_to_relocate": True,
        "github_activity_score": -1, "search_appearance_30d": 100,
        "saved_by_recruiters_30d": 8, "interview_completion_rate": 0.6,
        "offer_acceptance_rate": 0.3, "verified_email": True,
        "verified_phone": True, "linkedin_connected": True
    }
})

# 7. Unicode / special characters stress test
cases.append({
    "candidate_id": "CAND_9000007",
    "profile": {
        "anonymized_name": "Ünïcödé Tëst Çandidate", "headline": "AI工程师 | Senior Engineer™",
        "summary": "Test with emoji 🚀 and quotes \"like this\" and 'this' and commas, in text.",
        "location": "Bengaluru, Karnataka", "country": "India",
        "years_of_experience": 6.5, "current_title": "AI Engineer",
        "current_company": "TestCo™", "current_company_size": "201-500",
        "current_industry": "AI/ML"
    },
    "career_history": [
        {"company": "TestCo™", "title": "AI Engineer", "start_date": "2020-01-01",
         "end_date": None, "duration_months": 78, "is_current": True,
         "industry": "AI/ML", "company_size": "201-500",
         "description": "Built embeddings, \"retrieval\" systems, and RAG pipelines."}
    ],
    "education": [
        {"institution": "Test University", "degree": "M.Tech", "field_of_study": "CS",
         "start_year": 2016, "end_year": 2018, "grade": "9.0", "tier": "tier_2"}
    ],
    "skills": [
        {"name": "Embeddings", "proficiency": "advanced", "endorsements": 20, "duration_months": 50},
        {"name": "FAISS", "proficiency": "advanced", "endorsements": 15, "duration_months": 45}
    ],
    "certifications": [], "languages": [],
    "redrob_signals": {
        "profile_completeness_score": 90, "signup_date": "2023-06-01",
        "last_active_date": "2026-06-18", "open_to_work_flag": True,
        "profile_views_received_30d": 40, "applications_submitted_30d": 2,
        "recruiter_response_rate": 0.7, "avg_response_time_hours": 15,
        "skill_assessment_scores": {}, "connection_count": 400,
        "endorsements_received": 30, "notice_period_days": 30,
        "expected_salary_range_inr_lpa": {"min": 22, "max": 35},
        "preferred_work_mode": "hybrid", "willing_to_relocate": True,
        "github_activity_score": 60, "search_appearance_30d": 150,
        "saved_by_recruiters_30d": 10, "interview_completion_rate": 0.8,
        "offer_acceptance_rate": 0.5, "verified_email": True,
        "verified_phone": True, "linkedin_connected": True
    }
})

# 8. Missing redrob_signals fields entirely (schema says required, but test resilience anyway)
cases.append({
    "candidate_id": "CAND_9000008",
    "profile": {
        "anonymized_name": "Partial Signals Test", "headline": "Backend Engineer",
        "summary": "Test", "location": "Chennai, Tamil Nadu", "country": "India",
        "years_of_experience": 4.0, "current_title": "Backend Engineer",
        "current_company": "SmallCo", "current_company_size": "11-50",
        "current_industry": "Software"
    },
    "career_history": [
        {"company": "SmallCo", "title": "Backend Engineer", "start_date": "2022-06-01",
         "end_date": None, "duration_months": 48, "is_current": True,
         "industry": "Software", "company_size": "11-50", "description": "Backend work."}
    ],
    "education": [],
    "skills": [{"name": "Python", "proficiency": "intermediate", "endorsements": 5, "duration_months": 30}],
    "certifications": [], "languages": [],
    "redrob_signals": {
        "profile_completeness_score": 50,
        # signup_date intentionally omitted
        "last_active_date": "2026-05-01", "open_to_work_flag": True,
        "profile_views_received_30d": 5, "applications_submitted_30d": 1,
        "recruiter_response_rate": 0.3,
        # avg_response_time_hours intentionally omitted
        "skill_assessment_scores": {}, "connection_count": 50,
        "endorsements_received": 5, "notice_period_days": 60,
        "expected_salary_range_inr_lpa": {"min": 10, "max": 15},
        "preferred_work_mode": "remote", "willing_to_relocate": False,
        "github_activity_score": -1, "search_appearance_30d": 20,
        "saved_by_recruiters_30d": 1, "interview_completion_rate": 0.4,
        "offer_acceptance_rate": -1, "verified_email": True,
        "verified_phone": False, "linkedin_connected": False
    }
})

with open("./tests/edge_case_candidates.jsonl", "w") as f:
    for c in cases:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

print(f"Wrote {len(cases)} edge-case candidates")
