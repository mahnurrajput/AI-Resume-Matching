"""
test_core.py
============
Comprehensive Core Implementation Tests for AI Resume Matching System

Tests every aspect of the matching engine — correctness, edge cases,
consistency, speed, metadata integrity, and result quality.

Run from project root:
    python test_core.py
    python test_core.py --quick        # skip slow batch/real-data tests
    python test_core.py --section 3    # run only section 3

Sections:
  1  Engine Initialization & Loading
  2  Embedding & Normalization
  3  Single Resume Matching (Correctness)
  4  Result Structure & Data Integrity
  5  Domain Specificity (does Java resume get Java jobs?)
  6  Score Sanity (ordering, range, self-consistency)
  7  Edge Cases (empty, short, special chars, very long)
  8  Batch Matching (correctness + speed vs single)
  9  Speed Benchmarks
  10 Real Resume from Dataset
  11 Semantic Understanding (synonyms, paraphrases)
  12 Metadata Integrity (nan handling, missing fields)
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import faiss

# ── allow import from models/ regardless of where script is run from ──────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.matching_engine import MatchingEngine, MatchResult


# ══════════════════════════════════════════════════════════════════════════════
# TEST RUNNER HELPERS
# ══════════════════════════════════════════════════════════════════════════════

PASS  = "  ✓ PASS"
FAIL  = "  ✗ FAIL"
INFO  = "  ─ INFO"
WARN  = "  ⚠ WARN"

_results = {"passed": 0, "failed": 0, "warned": 0}


def check(condition: bool, label: str, detail: str = ""):
    if condition:
        print(f"{PASS}  {label}")
        _results["passed"] += 1
    else:
        print(f"{FAIL}  {label}  {'→ ' + detail if detail else ''}")
        _results["failed"] += 1


def warn(label: str, detail: str = ""):
    print(f"{WARN}  {label}  {'→ ' + detail if detail else ''}")
    _results["warned"] += 1


def info(label: str):
    print(f"{INFO}  {label}")


def section(n: int, title: str):
    print(f"\n{'═'*65}")
    print(f"  SECTION {n} — {title}")
    print(f"{'═'*65}")


def summary():
    print(f"\n{'═'*65}")
    print(f"  TEST SUMMARY")
    print(f"{'═'*65}")
    print(f"  Passed  : {_results['passed']}")
    print(f"  Failed  : {_results['failed']}")
    print(f"  Warnings: {_results['warned']}")
    total = _results["passed"] + _results["failed"]
    pct   = (_results["passed"] / total * 100) if total > 0 else 0
    print(f"  Score   : {_results['passed']}/{total}  ({pct:.1f}%)")
    print(f"{'═'*65}")
    if _results["failed"] == 0:
        print("  All tests passed ✓")
    else:
        print(f"  {_results['failed']} test(s) failed — review output above")
    print(f"{'═'*65}\n")


# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE RESUME TEXTS
# ══════════════════════════════════════════════════════════════════════════════

RESUME_JAVA = """
experienced java developer with 6 years building enterprise web applications
spring boot spring mvc hibernate jpa restful web services microservices
angular javascript html css aws cloud oracle database postgresql
junit testing maven jenkins ci cd git agile scrum
design patterns factory singleton dao mvc
docker kubernetes deployed on ibm websphere tomcat
"""

RESUME_DATA_SCIENCE = """
data scientist with 4 years experience in machine learning deep learning
python pandas numpy scikit-learn tensorflow pytorch keras
natural language processing nlp computer vision neural networks
sql postgresql mongodb data wrangling feature engineering
model deployment flask fastapi docker aws sagemaker
statistical analysis regression classification clustering
tableau power bi data visualization jupyter notebooks
"""

RESUME_HR = """
human resources professional with 8 years experience
talent acquisition recruitment onboarding employee relations
performance management compensation benefits payroll
hris workday bamboohr applicant tracking systems
labor law compliance policy development training development
organizational development culture building employee engagement
conflict resolution workforce planning succession planning
"""

RESUME_FINANCE = """
financial analyst with 5 years in investment banking
financial modeling valuation dcf analysis excel vba
bloomberg terminal reuters equity research
mergers acquisitions due diligence financial statements
cfa level 2 candidate portfolio management risk analysis
python for financial modeling pandas numpy matplotlib
forecasting budgeting variance analysis reporting
"""

RESUME_CHEF = """
executive chef with 12 years culinary experience
french cuisine italian cuisine asian fusion menu development
kitchen management food costing inventory control
team leadership staff training health safety regulations
catering events banquet management fine dining
food presentation plating techniques molecular gastronomy
michelin star restaurant experience pastry baking
"""

RESUME_NETWORK_SECURITY = """
network security engineer with 7 years cybersecurity experience
firewall configuration cisco checkpoint palo alto
intrusion detection prevention systems ids ips
penetration testing vulnerability assessment ethical hacking
siem tools splunk ibm qradar security operations center soc
incident response threat hunting malware analysis
cissp ceh certified network security protocols tcp ip vpn
"""

RESUME_VERY_SHORT = "python developer"

RESUME_ONE_SENTENCE = "experienced software engineer with skills in java and python looking for backend developer role"

RESUME_VERY_LONG = (
    "senior software architect with 15 years experience in enterprise systems "
    "java spring boot microservices kubernetes docker aws azure gcp "
    "system design distributed systems high availability fault tolerance "
    "database design sql nosql postgresql mysql mongodb redis elasticsearch "
    "api design restful graphql grpc message queues kafka rabbitmq "
    "ci cd jenkins github actions terraform infrastructure as code "
    "team leadership technical mentoring agile scrum kanban "
    "performance optimization profiling load testing jmeter "
    "security best practices oauth jwt encryption ssl tls "
) * 15  # ~2250 words — stress test long text truncation

RESUME_SPECIAL_CHARS = """
résumé — software engineer (5+ years) @ tech companies
skills: c++, c#, .net, asp.net, node.js, react.js, vue.js
databases: mysql, postgresql, mongodb, redis
cloud: aws (ec2, s3, rds, lambda), azure, gcp
"""

RESUME_ALL_CAPS = """
SENIOR DATA ENGINEER WITH 6 YEARS EXPERIENCE IN BIG DATA
APACHE SPARK HADOOP HIVE KAFKA AIRFLOW
AWS GLUE EMR REDSHIFT S3
PYTHON SCALA SQL
ETL PIPELINE DEVELOPMENT DATA WAREHOUSING
"""

RESUME_NUMBERS_ONLY = "1234567890 00000 99999 12345"

RESUME_GIBBERISH = "xkzqwp mfnvbrt lsdhqw pvxmkz nrtbcv qwxpzl"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — ENGINE INITIALIZATION & LOADING
# ══════════════════════════════════════════════════════════════════════════════

def test_section_1(engine):
    section(1, "Engine Initialization & Loading")

    check(engine is not None,
          "Engine object created successfully")

    check(engine.index is not None,
          "FAISS index loaded")

    check(engine.job_meta is not None,
          "Job metadata loaded")

    check(engine.model is not None,
          "Sentence-BERT model loaded")

    check(engine.index.ntotal > 0,
          f"FAISS index has vectors",
          f"ntotal = {engine.index.ntotal:,}")

    check(engine.index.ntotal > 100_000,
          f"FAISS index has expected scale (>100k jobs)",
          f"actual = {engine.index.ntotal:,}")

    check(len(engine.job_meta) == engine.index.ntotal,
          f"Metadata row count matches index vector count",
          f"metadata={len(engine.job_meta):,}  index={engine.index.ntotal:,}")

    info(f"Index size    : {engine.index.ntotal:,} vectors")
    info(f"Metadata rows : {len(engine.job_meta):,}")
    info(f"Model         : {engine.model.get_sentence_embedding_dimension()}-dim embeddings")

    # Check required metadata columns exist
    required_cols = ["title", "company", "location", "experience_level",
                     "work_type", "remote_allowed", "salary_min", "salary_max"]
    for col in required_cols:
        check(col in engine.job_meta.columns,
              f"Metadata has column: '{col}'")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — EMBEDDING & NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def test_section_2(engine):
    section(2, "Embedding & Normalization")

    vec = engine._embed(RESUME_JAVA)

    check(vec is not None,
          "Embedding returned (not None)")

    check(vec.shape == (1, 384),
          f"Embedding shape is (1, 384)",
          f"actual shape = {vec.shape}")

    check(vec.dtype == np.float32,
          f"Embedding dtype is float32",
          f"actual dtype = {vec.dtype}")

    norm = float(np.linalg.norm(vec[0]))
    check(abs(norm - 1.0) < 1e-5,
          f"Embedding is L2-normalized (norm ≈ 1.0)",
          f"actual norm = {norm:.8f}")

    # Two different texts should produce different embeddings
    vec2 = engine._embed(RESUME_DATA_SCIENCE)
    cosine_sim = float(np.dot(vec[0], vec2[0]))
    check(cosine_sim < 0.99,
          "Different texts produce different embeddings",
          f"similarity = {cosine_sim:.4f}")

    info(f"Embedding norm     : {norm:.8f}  (should be 1.0)")
    info(f"Java vs DS sim     : {cosine_sim:.4f}  (should be < 0.99)")

    # Same text twice should produce identical embeddings (deterministic)
    vec3 = engine._embed(RESUME_JAVA)
    identical = np.allclose(vec[0], vec3[0], atol=1e-6)
    check(identical,
          "Same text produces identical embedding (deterministic)")

    # Embedding dimension matches FAISS index dimension
    index_dim = engine.index.d
    check(vec.shape[1] == index_dim,
          f"Embedding dim matches FAISS index dim",
          f"embed={vec.shape[1]}  index={index_dim}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SINGLE RESUME MATCHING (CORRECTNESS)
# ══════════════════════════════════════════════════════════════════════════════

def test_section_3(engine):
    section(3, "Single Resume Matching — Correctness")

    results = engine.match(RESUME_JAVA, top_k=10)

    check(results is not None,
          "match() returns a result")

    check(isinstance(results, list),
          "match() returns a list")

    check(len(results) == 10,
          f"match() returns exactly top_k results",
          f"requested=10  received={len(results)}")

    check(all(isinstance(r, MatchResult) for r in results),
          "All results are MatchResult objects")

    # Scores are in descending order
    scores = [r.score for r in results]
    check(scores == sorted(scores, reverse=True),
          "Results are sorted by score descending",
          f"scores = {[round(s, 4) for s in scores]}")

    # All scores between 0 and 1
    check(all(0.0 <= s <= 1.0 for s in scores),
          "All scores in valid range [0, 1]",
          f"min={min(scores):.4f}  max={max(scores):.4f}")

    # Ranks are sequential 1..10
    ranks = [r.rank for r in results]
    check(ranks == list(range(1, 11)),
          "Ranks are sequential 1..10",
          f"actual = {ranks}")

    # Top-1 score should be reasonably high for a real resume
    check(results[0].score > 0.5,
          f"Top-1 score is meaningful (>0.5)",
          f"actual = {results[0].score:.4f}")

    info(f"Top-1 match  : '{results[0].title}' at '{results[0].company}'  score={results[0].score:.4f}")
    info(f"Top-5 scores : {[round(r.score, 4) for r in results[:5]]}")

    # Test different top_k values
    for k in [1, 5, 20, 50]:
        r = engine.match(RESUME_JAVA, top_k=k)
        check(len(r) == k,
              f"top_k={k} returns exactly {k} results",
              f"actual = {len(r)}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — RESULT STRUCTURE & DATA INTEGRITY
# ══════════════════════════════════════════════════════════════════════════════

def test_section_4(engine):
    section(4, "Result Structure & Data Integrity")

    results = engine.match(RESUME_JAVA, top_k=10)
    r = results[0]

    # Check all fields exist and are not missing
    check(hasattr(r, "rank"),             "MatchResult has 'rank' field")
    check(hasattr(r, "score"),            "MatchResult has 'score' field")
    check(hasattr(r, "job_idx"),          "MatchResult has 'job_idx' field")
    check(hasattr(r, "job_id"),           "MatchResult has 'job_id' field")
    check(hasattr(r, "title"),            "MatchResult has 'title' field")
    check(hasattr(r, "company"),          "MatchResult has 'company' field")
    check(hasattr(r, "location"),         "MatchResult has 'location' field")
    check(hasattr(r, "experience_level"), "MatchResult has 'experience_level' field")
    check(hasattr(r, "work_type"),        "MatchResult has 'work_type' field")
    check(hasattr(r, "remote_allowed"),   "MatchResult has 'remote_allowed' field")
    check(hasattr(r, "salary_min"),       "MatchResult has 'salary_min' field")
    check(hasattr(r, "salary_max"),       "MatchResult has 'salary_max' field")

    # to_dict() works and has all expected keys
    d = r.to_dict()
    check(isinstance(d, dict),            "to_dict() returns a dict")

    expected_keys = ["rank", "score", "job_id", "title", "company",
                     "location", "experience_level", "work_type",
                     "remote_allowed", "salary_min", "salary_max"]
    for key in expected_keys:
        check(key in d, f"to_dict() contains key: '{key}'")

    # Type checks
    check(isinstance(r.rank,  int),   "rank is int")
    check(isinstance(r.score, float), "score is float")
    check(isinstance(r.title, str),   "title is str")

    # job_idx must be a valid index into the job metadata
    check(0 <= r.job_idx < engine.index.ntotal,
          f"job_idx is valid FAISS index",
          f"job_idx={r.job_idx}  ntotal={engine.index.ntotal}")

    # Score in to_dict() should be rounded to 4 decimal places
    score_str = str(d["score"])
    decimal_places = len(score_str.split(".")[-1]) if "." in score_str else 0
    check(decimal_places <= 4,
          f"Score in to_dict() rounded to ≤4 decimal places",
          f"actual = {d['score']}")

    # No result should have an empty title (metadata integrity)
    titles_empty = sum(1 for res in results if not str(res.title).strip() or str(res.title) == "nan")
    if titles_empty > 0:
        warn(f"{titles_empty}/10 results have empty/nan title — check job metadata quality")
    else:
        check(True, "All top-10 results have non-empty titles")

    info(f"Sample result dict: {d}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — DOMAIN SPECIFICITY
# Does the right type of resume get the right type of job?
# ══════════════════════════════════════════════════════════════════════════════

def test_section_5(engine):
    section(5, "Domain Specificity — Right Resume Gets Right Jobs")

    test_cases = [
        {
            "label"    : "Java Developer",
            "resume"   : RESUME_JAVA,
            "keywords" : ["java", "developer", "software", "engineer", "backend", "fullstack", "full stack"],
            "anti_kw"  : ["chef", "cook", "culinary", "nurse", "doctor", "teacher"],
        },
        {
            "label"    : "Data Scientist",
            "resume"   : RESUME_DATA_SCIENCE,
            "keywords" : ["data", "scientist", "analyst", "machine learning", "engineer", "ml", "ai"],
            "anti_kw"  : ["chef", "cook", "nurse", "attorney", "lawyer"],
        },
        {
            "label"    : "HR Professional",
            "resume"   : RESUME_HR,
            "keywords" : ["hr", "human resources", "recruiter", "talent", "people", "coordinator"],
            "anti_kw"  : ["java", "python", "chef", "cook", "surgeon"],
        },
        {
            "label"    : "Chef",
            "resume"   : RESUME_CHEF,
            "keywords" : ["chef", "cook", "culinary", "kitchen", "food", "restaurant", "pastry"],
            "anti_kw"  : ["software", "developer", "java", "python", "accountant"],
        },
        {
            "label"    : "Finance Analyst",
            "resume"   : RESUME_FINANCE,
            "keywords" : ["finance", "financial", "analyst", "investment", "banking", "accountant", "cfa"],
            "anti_kw"  : ["chef", "cook", "java developer", "nurse"],
        },
        {
            "label"    : "Network Security Engineer",
            "resume"   : RESUME_NETWORK_SECURITY,
            "keywords" : ["security", "network", "cyber", "firewall", "engineer", "analyst", "soc"],
            "anti_kw"  : ["chef", "cook", "hr", "recruiter", "teacher"],
        },
    ]

    for tc in test_cases:
        results    = engine.match(tc["resume"], top_k=10)
        top5_titles = [r.title.lower() for r in results[:5]]
        all_titles  = " ".join(top5_titles)

        # Check that at least 3/5 top results contain relevant keywords
        relevant_hits = sum(
            1 for title in top5_titles
            if any(kw in title for kw in tc["keywords"])
        )

        # Check that anti-keywords don't dominate
        irrelevant_hits = sum(
            1 for title in top5_titles
            if any(kw in title for kw in tc["anti_kw"])
        )

        check(relevant_hits >= 3,
              f"{tc['label']}: ≥3/5 top results are domain-relevant",
              f"relevant={relevant_hits}/5  top titles: {top5_titles}")

        check(irrelevant_hits == 0,
              f"{tc['label']}: no irrelevant domain jobs in top-5",
              f"irrelevant={irrelevant_hits}  titles: {top5_titles}")

        info(f"  {tc['label']:28s} → top match: '{results[0].title}'  score={results[0].score:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — SCORE SANITY
# ══════════════════════════════════════════════════════════════════════════════

def test_section_6(engine):
    section(6, "Score Sanity — Ordering, Range, Consistency")

    # --- 6A: Scores decrease monotonically ---
    results = engine.match(RESUME_JAVA, top_k=20)
    scores  = [r.score for r in results]
    is_monotone = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    check(is_monotone,
          "Scores are monotonically non-increasing (rank 1 ≥ rank 2 ≥ ... ≥ rank N)")

    # --- 6B: Score gap between top-1 and rank-10 is meaningful ---
    gap = scores[0] - scores[-1]
    check(gap > 0.01,
          f"Score gap between rank-1 and rank-20 is meaningful (>0.01)",
          f"gap = {gap:.4f}")

    info(f"Score range (top-20): {scores[0]:.4f} → {scores[-1]:.4f}  gap={gap:.4f}")

    # --- 6C: Two very different resumes should get different scores ---
    r_java = engine.match(RESUME_JAVA,  top_k=1)[0].score
    r_chef = engine.match(RESUME_CHEF,  top_k=1)[0].score
    # Both get their own relevant top-1, scores should both be > 0.4
    check(r_java > 0.4,
          f"Java resume top-1 score is healthy (>0.4)",
          f"score = {r_java:.4f}")
    check(r_chef > 0.4,
          f"Chef resume top-1 score is healthy (>0.4)",
          f"score = {r_chef:.4f}")

    # --- 6D: Cosine similarity between same text twice should be exactly 1.0 ---
    v1 = engine._embed(RESUME_JAVA)
    v2 = engine._embed(RESUME_JAVA)
    self_sim = float(np.dot(v1[0], v2[0]))
    check(abs(self_sim - 1.0) < 1e-5,
          f"Same text self-similarity = 1.0",
          f"actual = {self_sim:.8f}")

    # --- 6E: Very different resumes should have lower mutual similarity ---
    v_java = engine._embed(RESUME_JAVA)
    v_chef = engine._embed(RESUME_CHEF)
    v_ds   = engine._embed(RESUME_DATA_SCIENCE)

    java_chef = float(np.dot(v_java[0], v_chef[0]))
    java_ds   = float(np.dot(v_java[0], v_ds[0]))

    # Java and DS are closer (both tech) than Java and Chef
    check(java_ds > java_chef,
          f"Java↔DataScience similarity > Java↔Chef similarity",
          f"Java↔DS={java_ds:.4f}  Java↔Chef={java_chef:.4f}")

    info(f"Java ↔ DataScience similarity : {java_ds:.4f}")
    info(f"Java ↔ Chef similarity        : {java_chef:.4f}")

    # --- 6F: Top-K results should not have duplicate job indices ---
    results_20 = engine.match(RESUME_JAVA, top_k=20)
    indices    = [r.job_idx for r in results_20]
    check(len(indices) == len(set(indices)),
          "No duplicate job indices in top-K results")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — EDGE CASES
# ══════════════════════════════════════════════════════════════════════════════

def test_section_7(engine):
    section(7, "Edge Cases — Unusual Inputs")

    # --- 7A: Empty string should raise ValueError ---
    try:
        engine.match("", top_k=5)
        check(False, "Empty string raises ValueError")
    except ValueError:
        check(True, "Empty string raises ValueError")
    except Exception as e:
        check(False, "Empty string raises ValueError", f"got {type(e).__name__}: {e}")

    # --- 7B: Whitespace only should raise ValueError ---
    try:
        engine.match("   \n\t  ", top_k=5)
        check(False, "Whitespace-only raises ValueError")
    except ValueError:
        check(True, "Whitespace-only raises ValueError")
    except Exception as e:
        check(False, "Whitespace-only raises ValueError", f"got {type(e).__name__}: {e}")

    # --- 7C: Very short text (2 words) should still work ---
    try:
        r = engine.match(RESUME_VERY_SHORT, top_k=5)
        check(len(r) == 5, "Very short text (2 words) still returns results")
        check(r[0].score > 0.0, "Very short text produces non-zero score",
              f"score = {r[0].score:.4f}")
        info(f"Very short text top match: '{r[0].title}'  score={r[0].score:.4f}")
    except Exception as e:
        check(False, "Very short text does not crash", f"{type(e).__name__}: {e}")

    # --- 7D: One sentence ---
    try:
        r = engine.match(RESUME_ONE_SENTENCE, top_k=5)
        check(len(r) == 5, "One-sentence resume returns results")
        info(f"One-sentence top match  : '{r[0].title}'  score={r[0].score:.4f}")
    except Exception as e:
        check(False, "One-sentence resume does not crash", f"{type(e).__name__}: {e}")

    # --- 7E: Very long text (truncation test) ---
    try:
        r = engine.match(RESUME_VERY_LONG, top_k=5)
        check(len(r) == 5, "Very long text (>2000 words) returns results without crash")
        check(r[0].score > 0.0, "Very long text produces non-zero score",
              f"score = {r[0].score:.4f}")
        info(f"Very long text top match: '{r[0].title}'  score={r[0].score:.4f}")
    except Exception as e:
        check(False, "Very long text does not crash", f"{type(e).__name__}: {e}")

    # --- 7F: Special characters ---
    try:
        r = engine.match(RESUME_SPECIAL_CHARS, top_k=5)
        check(len(r) == 5, "Special characters (accents, symbols) handled")
        info(f"Special chars top match : '{r[0].title}'  score={r[0].score:.4f}")
    except Exception as e:
        check(False, "Special characters do not crash", f"{type(e).__name__}: {e}")

    # --- 7G: ALL CAPS text ---
    try:
        r = engine.match(RESUME_ALL_CAPS, top_k=5)
        check(len(r) == 5, "ALL CAPS text handled")
        info(f"ALL CAPS top match      : '{r[0].title}'  score={r[0].score:.4f}")
    except Exception as e:
        check(False, "ALL CAPS text does not crash", f"{type(e).__name__}: {e}")

    # --- 7H: Numbers-only text ---
    try:
        r = engine.match(RESUME_NUMBERS_ONLY, top_k=5)
        check(len(r) == 5, "Numbers-only text returns results (no crash)")
        if r[0].score < 0.3:
            warn("Numbers-only text has very low top-1 score (expected)",
                 f"score = {r[0].score:.4f}")
        info(f"Numbers-only top match  : '{r[0].title}'  score={r[0].score:.4f}")
    except Exception as e:
        check(False, "Numbers-only text does not crash", f"{type(e).__name__}: {e}")

    # --- 7I: Gibberish text ---
    try:
        r = engine.match(RESUME_GIBBERISH, top_k=5)
        check(len(r) == 5, "Gibberish text returns results (no crash)")
        if r[0].score < 0.3:
            warn("Gibberish text has very low top-1 score (expected)",
                 f"score = {r[0].score:.4f}")
        info(f"Gibberish top match     : '{r[0].title}'  score={r[0].score:.4f}")
    except Exception as e:
        check(False, "Gibberish text does not crash", f"{type(e).__name__}: {e}")

    # --- 7J: top_k = 1 ---
    try:
        r = engine.match(RESUME_JAVA, top_k=1)
        check(len(r) == 1, "top_k=1 returns exactly 1 result")
    except Exception as e:
        check(False, "top_k=1 does not crash", f"{type(e).__name__}: {e}")

    # --- 7K: Unicode / non-ASCII ---
    unicode_resume = "développeur logiciel avec expérience en Python et données"
    try:
        r = engine.match(unicode_resume, top_k=5)
        check(len(r) == 5, "Unicode/non-ASCII text handled")
        info(f"Unicode top match       : '{r[0].title}'  score={r[0].score:.4f}")
    except Exception as e:
        check(False, "Unicode text does not crash", f"{type(e).__name__}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — BATCH MATCHING
# ══════════════════════════════════════════════════════════════════════════════

def test_section_8(engine):
    section(8, "Batch Matching — Correctness & Consistency with Single")

    batch_resumes = [
        RESUME_JAVA,
        RESUME_DATA_SCIENCE,
        RESUME_HR,
        RESUME_FINANCE,
        RESUME_CHEF,
    ]

    # --- 8A: Basic batch call ---
    batch_results = engine.match_batch(batch_resumes, top_k=10, show_progress=False)

    check(isinstance(batch_results, list),
          "match_batch() returns a list")

    check(len(batch_results) == len(batch_resumes),
          f"match_batch() returns one result list per resume",
          f"expected={len(batch_resumes)}  actual={len(batch_results)}")

    for i, results in enumerate(batch_results):
        check(len(results) == 10,
              f"Batch resume {i+1}: returns 10 results",
              f"actual = {len(results)}")

    # --- 8B: Batch results match single results (consistency) ---
    for i, (batch_res, resume_text) in enumerate(zip(batch_results, batch_resumes)):
        single_res = engine.match(resume_text, top_k=10)

        # Top-1 job index should be identical
        batch_top1  = batch_res[0].job_idx
        single_top1 = single_res[0].job_idx
        check(batch_top1 == single_top1,
              f"Batch resume {i+1}: top-1 match same as single match",
              f"batch={batch_top1}  single={single_top1}")

        # Top-1 score should be very close (floating point tolerance)
        batch_score  = batch_res[0].score
        single_score = single_res[0].score
        check(abs(batch_score - single_score) < 0.001,
              f"Batch resume {i+1}: score matches single (±0.001)",
              f"batch={batch_score:.4f}  single={single_score:.4f}")

    # --- 8C: Empty batch ---
    empty_result = engine.match_batch([], top_k=10, show_progress=False)
    check(empty_result == [],
          "match_batch([]) returns empty list")

    # --- 8D: Single-item batch ---
    single_batch = engine.match_batch([RESUME_JAVA], top_k=5, show_progress=False)
    check(len(single_batch) == 1,
          "match_batch with 1 item returns list of 1")
    check(len(single_batch[0]) == 5,
          "Single-item batch returns correct top_k")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — SPEED BENCHMARKS
# ══════════════════════════════════════════════════════════════════════════════

def test_section_9(engine):
    section(9, "Speed Benchmarks")

    # --- 9A: Single query latency ---
    # Warm up
    engine.match(RESUME_JAVA, top_k=10)

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        engine.match(RESUME_JAVA, top_k=10)
        times.append(time.perf_counter() - t0)

    avg_single = np.mean(times) * 1000
    min_single = np.min(times)  * 1000
    max_single = np.max(times)  * 1000

    check(avg_single < 1000,
          f"Single query latency < 1000ms",
          f"avg = {avg_single:.1f}ms")

    if avg_single < 100:
        check(True, f"Single query latency < 100ms (excellent)",
              f"avg = {avg_single:.1f}ms")
    elif avg_single < 300:
        warn(f"Single query latency {avg_single:.1f}ms (acceptable on CPU)")
    else:
        warn(f"Single query latency {avg_single:.1f}ms (slow — consider GPU)")

    info(f"Single query: avg={avg_single:.1f}ms  min={min_single:.1f}ms  max={max_single:.1f}ms")

    # --- 9B: Batch of 10 vs 10 single queries ---
    batch_texts = [RESUME_JAVA, RESUME_DATA_SCIENCE, RESUME_HR,
                   RESUME_FINANCE, RESUME_CHEF, RESUME_NETWORK_SECURITY,
                   RESUME_ONE_SENTENCE, RESUME_VERY_SHORT,
                   RESUME_ALL_CAPS, RESUME_SPECIAL_CHARS]

    t0 = time.perf_counter()
    engine.match_batch(batch_texts, top_k=10, show_progress=False)
    batch_time = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    for text in batch_texts:
        engine.match(text, top_k=10)
    sequential_time = (time.perf_counter() - t0) * 1000

    info(f"Batch (10 resumes)  : {batch_time:.1f}ms")
    info(f"Sequential (10×1)   : {sequential_time:.1f}ms")
    info(f"Batch speedup       : {sequential_time/batch_time:.2f}x")

    # Batch should not be drastically slower than sequential
    check(batch_time < sequential_time * 2.0,
          "Batch not drastically slower than sequential",
          f"batch={batch_time:.1f}ms  sequential={sequential_time:.1f}ms")

    # --- 9C: FAISS search isolation (just the search, not encoding) ---
    vec = engine._embed(RESUME_JAVA)
    times_faiss = []
    for _ in range(10):
        t0 = time.perf_counter()
        engine.index.search(vec, 10)
        times_faiss.append(time.perf_counter() - t0)

    avg_faiss = np.mean(times_faiss) * 1000
    check(avg_faiss < 100,
          f"FAISS search alone < 100ms",
          f"avg = {avg_faiss:.2f}ms")

    info(f"FAISS search alone  : {avg_faiss:.2f}ms  (over 108k vectors)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — REAL RESUME FROM DATASET
# ══════════════════════════════════════════════════════════════════════════════

def test_section_10(engine):
    section(10, "Real Resume from Dataset")

    resume_file = "data_processed/resumes_cleaned.csv"
    if not os.path.exists(resume_file):
        warn("resumes_cleaned.csv not found — skipping real-data tests")
        return

    df = pd.read_csv(resume_file).dropna(subset=["cleaned_text"])
    df = df[df["cleaned_text"].str.strip() != ""].reset_index(drop=True)

    info(f"Dataset loaded: {len(df):,} resumes")

    # --- 10A: Test first 5 resumes from each source ---
    for source in df["source"].unique():
        source_df = df[df["source"] == source].head(3)
        info(f"Testing source: {source} ({len(source_df)} samples)")

        for _, row in source_df.iterrows():
            text = row["cleaned_text"]
            try:
                results = engine.match(text, top_k=5)
                check(len(results) == 5,
                      f"[{source}] Real resume returns 5 results")
                check(results[0].score > 0.3,
                      f"[{source}] Real resume top-1 score > 0.3",
                      f"score = {results[0].score:.4f}")
            except Exception as e:
                check(False,
                      f"[{source}] Real resume does not crash",
                      f"{type(e).__name__}: {e}")

    # --- 10B: Word count extremes ---
    shortest = df.nsmallest(3, "word_count")
    longest  = df.nlargest(3, "word_count")

    for _, row in shortest.iterrows():
        try:
            r = engine.match(row["cleaned_text"], top_k=5)
            check(len(r) == 5,
                  f"Shortest resume ({row['word_count']} words) returns results")
        except Exception as e:
            check(False,
                  f"Shortest resume does not crash",
                  f"{type(e).__name__}: {e}")

    for _, row in longest.iterrows():
        try:
            r = engine.match(row["cleaned_text"], top_k=5)
            check(len(r) == 5,
                  f"Longest resume ({row['word_count']} words) returns results")
        except Exception as e:
            check(False,
                  f"Longest resume does not crash",
                  f"{type(e).__name__}: {e}")

    # --- 10C: Labeled resumes get category-appropriate matches ---
    labeled = df[df["category"].notna() & (df["category"] != "NaN")]
    if len(labeled) > 0:
        info(f"Testing labeled resumes: {len(labeled):,} available")

        category_kw = {
            "INFORMATION-TECHNOLOGY": ["software", "developer", "engineer", "java", "python", "data", "it"],
            "FINANCE"               : ["finance", "financial", "analyst", "accountant", "banking"],
            "HR"                    : ["hr", "human resources", "recruiter", "talent"],
            "CHEF"                  : ["chef", "cook", "culinary", "kitchen", "food"],
            "SALES"                 : ["sales", "account", "business development"],
            "ADVOCATE"              : ["lawyer", "attorney", "legal", "counsel"],
            "FITNESS"               : ["fitness", "trainer", "wellness", "health", "coach"],
        }

        correct_domain = 0
        tested         = 0

        for cat, keywords in category_kw.items():
            cat_df = labeled[labeled["category"] == cat].head(5)
            if cat_df.empty:
                continue

            for _, row in cat_df.iterrows():
                results   = engine.match(row["cleaned_text"], top_k=5)
                top5_titles = " ".join([r.title.lower() for r in results[:5]])
                if any(kw in top5_titles for kw in keywords):
                    correct_domain += 1
                tested += 1

        if tested > 0:
            pct = correct_domain / tested * 100
            check(pct >= 60,
                  f"Labeled resumes: ≥60% get domain-relevant top-5 matches",
                  f"correct={correct_domain}/{tested}  ({pct:.1f}%)")
            info(f"Domain match rate: {correct_domain}/{tested}  ({pct:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — SEMANTIC UNDERSTANDING
# Tests SBERT's core strength — synonyms and paraphrases
# ══════════════════════════════════════════════════════════════════════════════

def test_section_11(engine):
    section(11, "Semantic Understanding — Synonyms & Paraphrases")

    # --- 11A: Synonym test ---
    # "software engineer" and "programmer" and "developer" should get similar jobs
    r_engineer   = engine.match("experienced software engineer java spring", top_k=10)
    r_developer  = engine.match("experienced software developer java spring", top_k=10)
    r_programmer = engine.match("experienced programmer java spring",        top_k=10)

    titles_engineer   = set(r.title.lower() for r in r_engineer[:5])
    titles_developer  = set(r.title.lower() for r in r_developer[:5])
    titles_programmer = set(r.title.lower() for r in r_programmer[:5])

    overlap_eng_dev = len(titles_engineer & titles_developer)
    overlap_eng_prg = len(titles_engineer & titles_programmer)

    check(overlap_eng_dev >= 2,
          f"'engineer' and 'developer' share ≥2 top-5 matches",
          f"overlap = {overlap_eng_dev}  engineer={titles_engineer}  developer={titles_developer}")

    check(overlap_eng_prg >= 1,
          f"'engineer' and 'programmer' share ≥1 top-5 matches",
          f"overlap = {overlap_eng_prg}")

    info(f"engineer↔developer overlap  : {overlap_eng_dev}/5")
    info(f"engineer↔programmer overlap : {overlap_eng_prg}/5")

    # --- 11B: Paraphrase test ---
    # Same experience described differently should produce close embedding vectors
    text_a = "i have 5 years experience building web applications using python"
    text_b = "5 years of web development experience with python programming"

    va = engine._embed(text_a)
    vb = engine._embed(text_b)
    sim_ab = float(np.dot(va[0], vb[0]))

    check(sim_ab > 0.80,
          f"Paraphrased sentences have high embedding similarity (>0.80)",
          f"similarity = {sim_ab:.4f}")

    info(f"Paraphrase similarity : {sim_ab:.4f}")

    # --- 11C: Unrelated fields should have low similarity ---
    v_java  = engine._embed("java spring boot microservices backend developer")
    v_chef2 = engine._embed("head chef french cuisine kitchen management pastry")
    sim_unrelated = float(np.dot(v_java[0], v_chef2[0]))

    check(sim_unrelated < 0.6,
          f"Unrelated fields have lower similarity (<0.6)",
          f"java↔chef similarity = {sim_unrelated:.4f}")

    info(f"Java↔Chef similarity  : {sim_unrelated:.4f}")

    # --- 11D: Adding more context should not destroy relevance ---
    short_text = "python machine learning"
    long_text  = "python machine learning data science scikit-learn tensorflow model training deployment"

    rs = engine.match(short_text, top_k=5)
    rl = engine.match(long_text,  top_k=5)

    titles_short = set(r.title.lower() for r in rs)
    titles_long  = set(r.title.lower() for r in rl)
    overlap_sl   = len(titles_short & titles_long)

    check(overlap_sl >= 1,
          f"Short and expanded text share ≥1 top-5 matches",
          f"overlap = {overlap_sl}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — METADATA INTEGRITY & NaN HANDLING
# ══════════════════════════════════════════════════════════════════════════════

def test_section_12(engine):
    section(12, "Metadata Integrity & NaN Handling")

    results = engine.match(RESUME_JAVA, top_k=50)

    # Count nan occurrences in each field
    nan_counts = {
        "title"            : 0,
        "company"          : 0,
        "location"         : 0,
        "experience_level" : 0,
        "work_type"        : 0,
    }

    for r in results:
        for field in nan_counts:
            val = str(getattr(r, field))
            if val.lower() in ("nan", "none", "", "null"):
                nan_counts[field] += 1

    total = len(results)
    for field, count in nan_counts.items():
        pct = count / total * 100
        if pct == 0:
            check(True, f"Field '{field}': no NaN values in top-50")
        elif pct < 20:
            warn(f"Field '{field}': {count}/{total} ({pct:.0f}%) NaN — data quality issue")
        else:
            warn(f"Field '{field}': {count}/{total} ({pct:.0f}%) NaN — significant missing data")
        info(f"  {field:20s}: {total-count}/{total} populated  ({100-pct:.0f}%)")

    # Salary nan is expected — report but don't fail
    salary_nan = sum(
        1 for r in results
        if r.salary_min is None or str(r.salary_min).lower() in ("nan", "none")
    )
    salary_pct = salary_nan / total * 100
    info(f"  {'salary_min':20s}: {total-salary_nan}/{total} populated  ({100-salary_pct:.0f}%)")
    if salary_pct > 50:
        warn(f"Salary data missing for {salary_pct:.0f}% of results — expected for LinkedIn dataset")

    # to_dict() should never raise an exception even for NaN fields
    all_dicts_ok = True
    for r in results:
        try:
            d = r.to_dict()
            if not isinstance(d, dict):
                all_dicts_ok = False
        except Exception as e:
            all_dicts_ok = False
            info(f"  to_dict() failed for result: {e}")

    check(all_dicts_ok,
          "to_dict() succeeds for all top-50 results (including NaN fields)")

    # Check that job_idx values are within valid range
    invalid_idx = [r.job_idx for r in results if not (0 <= r.job_idx < engine.index.ntotal)]
    check(len(invalid_idx) == 0,
          f"All job_idx values are valid FAISS indices",
          f"invalid indices: {invalid_idx}")

    # Check that all job_ids are non-empty
    empty_job_ids = sum(1 for r in results if not str(r.job_id).strip() or str(r.job_id) == "nan")
    if empty_job_ids > 0:
        warn(f"{empty_job_ids}/50 results have empty job_id")
    else:
        check(True, "All top-50 results have non-empty job_id")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Core implementation test suite")
    parser.add_argument("--quick",   action="store_true",
                        help="Skip slow tests (sections 10)")
    parser.add_argument("--section", type=int, default=0,
                        help="Run only this section number (0 = all)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "═"*65)
    print("  AI RESUME MATCHING — CORE IMPLEMENTATION TEST SUITE")
    print("  12 Sections | Correctness · Edge Cases · Speed · Quality")
    print("═"*65)

    # Initialize engine once — shared across all tests
    print("\nInitializing engine (loading FAISS index + SBERT model)...")
    t0     = time.perf_counter()
    engine = MatchingEngine()
    init_t = (time.perf_counter() - t0) * 1000
    info(f"Engine initialized in {init_t:.0f}ms")

    sections = {
        1 : ("Engine Initialization & Loading",        test_section_1),
        2 : ("Embedding & Normalization",              test_section_2),
        3 : ("Single Resume Matching — Correctness",   test_section_3),
        4 : ("Result Structure & Data Integrity",      test_section_4),
        5 : ("Domain Specificity",                     test_section_5),
        6 : ("Score Sanity",                           test_section_6),
        7 : ("Edge Cases",                             test_section_7),
        8 : ("Batch Matching",                         test_section_8),
        9 : ("Speed Benchmarks",                       test_section_9),
        10: ("Real Resume from Dataset",               test_section_10),
        11: ("Semantic Understanding",                 test_section_11),
        12: ("Metadata Integrity & NaN Handling",      test_section_12),
    }

    skip = set()
    if args.quick:
        skip.add(10)
        info("--quick flag: skipping section 10 (real dataset)")

    for num, (title, fn) in sections.items():
        if args.section and num != args.section:
            continue
        if num in skip:
            print(f"\n  [Skipped] Section {num} — {title}")
            continue
        try:
            fn(engine)
        except Exception as e:
            print(f"\n  [CRASH] Section {num} crashed: {type(e).__name__}: {e}")
            _results["failed"] += 1

    summary()


if __name__ == "__main__":
    main()
