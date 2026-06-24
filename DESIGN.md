# Ranking System Design — Redrob Hackathon

## Scoring philosophy

The JD is unusually explicit about what it wants and doesn't want. That's a gift: most of
the JD's signal is checkable against **structured fields** (years_of_experience, company
names/sizes, industry, career_history durations, current_title), not against free-text
semantic similarity. Free-text similarity (embeddings) is necessary but must be a minority
voice in the final score, because the JD itself says keyword/semantic stuffing is the trap.

Final score per candidate = weighted sum of independently-computed, inspectable sub-scores:

    score = w1 * role_fit          (structural: title/company/industry/seniority match)
          + w2 * skill_fit          (structural: required skills present + proficiency + recency)
          + w3 * disqualifier_penalty (structural: hard JD disqualifiers, large negative)
          + w4 * semantic_fit       (embedding similarity: JD vs profile+career text)
          + w5 * availability       (behavioral: redrob_signals-derived availability/engagement)
          - honeypot_penalty        (large negative if honeypot heuristics fire)

Each sub-score is [0,1] (except disqualifier_penalty and honeypot_penalty, which are
negative-only modifiers). Weights are tunable constants at the top of the script, not
buried in logic, so they can be defended in the Stage 5 interview.

Why this shape:
- Inspectable: every sub-score can be printed per-candidate, which is exactly what the
  `reasoning` column needs (Stage 4 graders explicitly check for specific facts).
- Defensible: in a 30-minute interview, "role_fit=0.8 because they're 6.2 YOE at a product
  company in applied ML" is something a human said on purpose. A single opaque embedding
  cosine similarity is not.
- Resistant to the stated trap: a keyword-stuffed Marketing Manager profile scores low on
  role_fit (title/career_history don't match) even if semantic_fit is inflated by skill
  keyword overlap.

## Sub-score design

### 1. role_fit (weight 0.30)
Structural match against the JD's explicit profile:
- years_of_experience in [5,9] ideal, with graceful falloff outside (not a hard cutoff —
  JD says "we'll seriously consider candidates outside the band").
- Applied-ML-at-product-company tenure: sum duration_months across career_history entries
  where (a) industry/company suggests a product company, not a pure-services firm, and
  (b) title suggests ML/AI/data/search/ranking work. This directly operationalizes "4-5
  years in applied ML/AI roles at product companies."
- current_title / most-recent career_history title matched against an ML/AI-adjacent
  title lexicon (ranking, search, recommendation, ML, AI, NLP, applied scientist, etc.)
- Company-size / industry sanity: candidate's recent employers should look like product
  companies (industries like "Food Delivery", "Fintech", "E-commerce", "AI/ML",
  "Transportation" rather than "IT Services" repeatedly, which is the consulting-firm
  signal flagged as a partial disqualifier).

### 2. skill_fit (weight 0.25)
JD's "absolutely need" skills, checked against the candidate's skills[] (name +
proficiency + duration_months) and skill_assessment_scores where present:
- embeddings-based retrieval (sentence-transformers, OpenAI/BGE/E5 embeddings, "Embeddings")
- vector DB / hybrid search (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS)
- Python (proxy: presence of Python, or strongly Python-coded skills like Django/FastAPI/
  PyTorch/pandas-adjacent, since "Python" itself is sometimes implicit)
- evaluation framework experience (NDCG, MRR, MAP explicitly named, or "Feature Engineering"
  + "Information Retrieval" + "Recommendation Systems" as proxies)
Each matched skill contributes proficiency-weighted score (expert=1.0, advanced=0.75,
intermediate=0.5, beginner=0.25) with a recency discount based on duration_months (a skill
used 80 months ago but never recently is weighted down relative to one actively in use,
approximated here by duration_months itself since the schema has no "last used" field per
skill — duration_months is the closest proxy and is treated as "how long they've used it,"
which for this dataset's generation logic also serves as a tenure/depth signal).
"Nice to have" skills (LoRA/QLoRA/PEFT, learning-to-rank, HR-tech exposure, distributed
systems, open-source) add a smaller bonus, never a requirement.

### 3. disqualifier_penalty (weight applied directly, large negative)
Hard, JD-explicit disqualifiers, each independently checked and summed as negative log-odds
style penalties (so multiple disqualifiers compound but one alone doesn't necessarily zero
the candidate, matching the JD's own hedged tone — "we will probably not move forward"):
- Pure-research-only career (all career_history titles read as research/academic, zero
  production/engineering titles) → large penalty.
- "AI experience" entirely within the last 12 months AND pre-LLM-era ML/production
  experience absent (no career_history entries >12 months old containing engineering/
  ML/data titles) → penalty (the "LangChain wrapper" disqualifier).
- Career entirely at the named consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant,
  Capgemini, HCL, Tech Mahindra, Mindtree) with zero product-company entries → penalty
  (explicitly softened if ANY entry is at a non-consulting company).
- Senior-sounding title (Lead/Principal/Architect/Manager-of-Engineers, etc.) but current
  role's career_history description and skills suggest no hands-on coding in the most
  recent role, AND current role tenure > 18 months → small penalty (weakest signal in the
  set, least confidently detectable from this schema, weighted lowest of the disqualifiers).
- CV/speech/robotics-primary with no NLP/IR-adjacent skills at all → penalty.
- Title-chasing pattern: 3+ employers in career_history within a span where each tenure is
  <= 18 months AND titles show seniority escalation → penalty.

### 4. semantic_fit (weight 0.15 — deliberately minority weight)
Cosine similarity between a JD embedding and a candidate document embedding built from
headline + summary + current_title + concatenated career_history descriptions + skill
names. Using a small local sentence-transformer model (CPU-friendly, no network call,
satisfies the compute constraint). This catches genuine semantic fits that don't hit
exact skill-lexicon keywords (the JD's own "Tier 5 candidate without the words RAG or
Pinecone" example) — exactly why it can't be zero-weighted, but also exactly why it
can't dominate, since it's the same channel keyword-stuffers exploit.

### 5. availability (weight 0.10)
Derived from redrob_signals, reflecting the JD's own explicit call-out: "a perfect-on-paper
candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is, for
hiring purposes, not actually available."
- last_active_date recency (decay function, steep beyond ~60 days)
- open_to_work_flag (boolean bump)
- recruiter_response_rate (direct)
- notice_period_days (JD explicitly prefers <30 days, tolerates more with a higher bar)
- willing_to_relocate OR already in Noida/Pune/NCR/Hyderabad/Mumbai/Delhi (location fit
  per JD's explicit city list)
- interview_completion_rate (mild positive — shows follow-through)
This is intentionally the lowest-weighted *positive* sub-score (engagement matters, but
the JD's primary ask is technical/role fit) while still being large enough to clearly
separate an engaged ideal-fit candidate from a disengaged one, per the JD's own framing.

### 6. honeypot_penalty (independent, large negative, computed before scoring)
Structural-impossibility checks, independent of skill/role matching so honeypots can't
"buy their way out" with a good keyword profile (that's the point of a honeypot):
- years_of_experience implies a career start year; if any career_history start_date
  predates a plausible founding date for a named company that is well-known to be young
  (heuristic: a small denylist of companies + an implied-too-long-tenure check is fragile;
  instead use the more robust, schema-derivable signal below).
- Internal date consistency: sum of career_history duration_months should roughly not
  exceed years_of_experience * 12 by a large margin, and should not be near-zero while
  years_of_experience is large (career history that doesn't account for stated experience).
- Implausible skill profile: 5+ skills at "expert" proficiency with duration_months < 12
  each (expert-level claimed with no time to have gotten there).
- Implausible assessment uniformity: skill_assessment_scores with many entries all
  clustered implausibly high (>90) with no variance.
- profile_completeness_score implausibly high (>98) combined with signup_date and
  last_active_date being the same day or within 24 hours (full profile completed and
  "active" the moment of signup — a synthetic-generation tell, not a real user pattern).
These are summed as a 0-1 honeypot_score; candidates above a threshold receive a large
fixed penalty subtracted from final score (effectively pushing them out of top-100 without
needing it to be a hard filter — keeps the system from being "filter-then-rank," which the
JD frames as the wrong mental model anyway, and matches the brief's "rank tier 0").

## Reasoning generation
Templated but per-candidate, built from the actual sub-score breakdown and the actual
fields that drove it — not a fill-in-the-blank template, since Stage 4 explicitly penalizes
templated reasoning and rewards specificity. The reasoning generator pulls the top 1-2
contributing facts from whichever sub-scores were highest, and explicitly states the
weakest aspect of the fit if disqualifier_penalty or low availability pulled the score
down — "honest concerns" is an explicit grading criterion.

## Tie-breaking
Per spec: ties broken by candidate_id ascending. This is implemented as the final sort key
after score, so it's never ambiguous and matches the validator's literal expectation.

## Validation log (against real data)

Tested against a real 1000-candidate slice of candidates.jsonl (not synthetic). Findings
that changed the implementation, kept here so the reasoning is auditable and defensible
in the Stage 5 interview:

1. **Original honeypot heuristics fired on 0/1000 real candidates.** The duration/expert,
   assessment-uniformity, and same-day-signup checks were reasoned from first principles
   with no real honeypot example to calibrate against, and the real data turned out to be
   internally consistent on exactly those dimensions (years_of_experience matches earliest
   career_history start date to within ~1 year for 971/1000 candidates). Broadened the
   heuristic set rather than abandoning it; one new check (advanced+ skill count
   disproportionate to YOE) found a genuine, defensible catch: an Operations Manager with
   1.9 YOE holding 7 "advanced"-rated AI/retrieval skills, several with duration_months
   that individually exceed the candidate's total career length when converted to years.

2. **A proposed "endorsements implausibly high relative to connections" honeypot check was
   added, tested, and removed.** On real data it fired on candidates with small-but-real
   networks (10 connections / 42 endorsements) — a benign, unremarkable pattern, not a
   synthetic-generation tell. Endorsements and connection count are not causally linked in
   a way that makes their ratio diagnostic of anything. This is kept as a comment in the
   code rather than silently deleted, specifically to demonstrate the "test before trusting
   a heuristic" discipline if asked about it at Stage 5.

3. **The single most important fix came from manually inspecting rank #8 of a real run:**
   a "Graphic Designer | AI enthusiast | Building with LLMs" candidate, with a career
   history of Graphic Designer -> Mechanical Engineer (zero ML/AI/data roles ever), but a
   skills list dense with advanced-rated retrieval/LLM keywords at durations long enough
   (8-18 months) to dodge the duration-based honeypot checks. This is the JD's own
   explicitly-named trap pattern, present in real data, and the original skill_fit function
   had no defense against it — skill claims were scored independent of career corroboration.
   Fixed by adding `has_any_ml_adjacent_role()`: skill_fit is discounted to 40% of its raw
   value when the candidate's career_history contains no ML/AI/data-adjacent title at all.
   This single fix moved that candidate from rank 8 to rank 11 in the same 1000-candidate
   test set — not eliminated (the JD doesn't want pure title-based rejection), but no longer
   competitive with corroborated candidates.

4. **Targeted adversarial edge-case generation found a real arithmetic bug.** Eight
   synthetic candidates were constructed to probe specific failure modes (empty arrays,
   null fields, extreme values, malformed salary, unicode/emoji text, a deliberate
   impossible-tenure honeypot, a second keyword-stuffer instance, missing redrob_signals
   fields). None crashed the pipeline, which is good, but inspecting sub-scores surfaced a
   real bug: the common `signals.get(key, default) or default` pattern used throughout the
   availability and honeypot functions silently overwrites a genuine `0`/`0.0` value with
   the fallback default, because `0 or default` evaluates to `default` in Python regardless
   of whether the key was present. This matters because several redrob_signals fields are
   legitimately and meaningfully zero: notice_period_days=0 (immediately available — a
   strong positive signal) and avg_response_time_hours=0 (instant response — relevant to
   both availability scoring and the honeypot responsiveness check) were both being
   silently replaced by their fallback defaults (60 and 999 respectively), understating
   genuine availability and weakening the honeypot signal it was meant to support. Fixed
   by adding a `safe_num()` helper that only falls back on an explicit `None` or missing
   key, not on a legitimate zero, and replacing all six vulnerable call sites. Re-verified
   against the real 1000-candidate set afterward: rank #1 unchanged, the two previously
   confirmed fixes (Graphic Designer keyword-stuffer discount, Operations Manager honeypot
   exclusion) both still hold, and the overall score distribution shifted only slightly —
   consistent with a precision fix rather than a logic change.

## Compute budget (measured, not estimated)
100k candidates, 5 minutes, 16GB RAM, CPU-only, no network, no GPU.
- Load-tested on a 100k synthetic-but-structurally-similar pool: **~22 seconds wall-clock,
  ~1.5GB peak RSS** — both with large margin against the 5-minute / 16GB limits.
- semantic_fit uses TF-IDF + cosine similarity, not a sentence-transformer model (see
  ARCHITECTURE NOTE at the top of rank.py). This avoids any network dependency for model
  weights and is fast enough at 100k scale that it isn't the bottleneck.
- Action item before final submission: re-run this timing/memory check against the actual
  full 100k candidates.jsonl (not a synthetic stand-in) on the machine intended to match
  Stage 3's sandboxed reproduction environment, since real career_history/skills array
  lengths may differ from the synthetic test set and could shift the constant factor.

These four findings are the direct payoff of testing against both real data and deliberately
adversarial synthetic data before relying on heuristics designed from the brief's prose
alone. The honeypot detector should still be treated as a best-effort layer, not a
guarantee — with zero labeled honeypot examples available pre-submission, false negatives
on whatever the true honeypot generation pattern actually is remain possible. The
career-corroboration discount on skill_fit is the higher-confidence defense of the two,
since it's structural and doesn't depend on guessing the generator's specific recipe.
