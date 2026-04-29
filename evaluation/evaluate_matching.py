"""
evaluate_matching.py
====================
Phase 4 — Matching System Evaluation

Three evaluation strategies that together give a complete, honest picture
of how well the SBERT + FAISS matching system performs.

This system uses Sentence-BERT for semantic retrieval — not a supervised
classifier — so there are no train/test splits or gradient-based metrics.
The strategies below are the correct evaluation approach for retrieval systems.

─────────────────────────────────────────────────────────────────────────────
DATA SCOPE
─────────────────────────────────────────────────────────────────────────────

  Total resumes      : 11,654  (across 3 datasets)
  Labeled resumes    : 2,481   (Dataset 2 only — the only source with category labels)
  Unlabeled resumes  : 9,173   (Dataset 1 + Dataset 3 — no category labels)

    Strategies 1 and 3 use the 2,481 labeled resumes (category labels required).
  Strategy 2 uses all 11,654 resumes (no labels needed).

─────────────────────────────────────────────────────────────────────────────
STRATEGY 1 — Precision@K + MRR  (labeled resumes, K = 1, 3, 5, 10)
─────────────────────────────────────────────────────────────────────────────
The primary retrieval metric. For each labeled resume, retrieve top-K jobs
and check how many belong to the correct category (determined by job title
keyword matching against the resume's category label).

  Precision@K : what fraction of the top-K retrieved jobs are relevant?
  MRR         : Mean Reciprocal Rank — 1/position of the first relevant hit.
                MRR = 1.0 means the top-1 result is always correct.
                MRR = 0.5 means the first correct result is at rank 2 on average.

Recall@K is deliberately excluded. With 108,702 jobs and potentially
thousands of relevant ones per category, a top-10 list covering 5 of 3,000
relevant jobs gives R@10 = 0.0017 — correct mathematically but misleading
to report without a lengthy explanation. Precision@K tells the real story.

─────────────────────────────────────────────────────────────────────────────
STRATEGY 2 — Score Separation  (all 11,654 resumes, no labels needed)
─────────────────────────────────────────────────────────────────────────────
For each resume, compare the mean cosine score of its top-10 matched jobs
against the mean cosine score of 10 randomly sampled jobs from the index.
Reports the mean separation gap and its distribution.

This answers: does the system actually discriminate, or does everything
score similarly? A healthy system shows a consistent, wide gap. No labels
are required, so this covers the full 11,654-resume dataset.

─────────────────────────────────────────────────────────────────────────────
STRATEGY 3 — Word Count Sensitivity  (labeled resumes)
─────────────────────────────────────────────────────────────────────────────
Groups resumes into word-count buckets and reports P@10 per bucket.
Answers: do short resumes (thin content) get worse results than long ones?
Directly relevant to the data quality observations in the project report.

─────────────────────────────────────────────────────────────────────────────
OUTPUTS
─────────────────────────────────────────────────────────────────────────────

  CSV files (outputs/evaluation/):
    metrics_summary.csv          — headline P@K and MRR numbers
    word_count_metrics.csv       — Strategy 3 breakdown by word-count bucket

  Plots (outputs/evaluation/):
    01_precision_at_k.png        — P@K curve across K = 1, 3, 5, 10
    02_score_separation.png      — top-10 vs random score distributions
    03_word_count_sensitivity.png — P@10 by resume word-count bucket

─────────────────────────────────────────────────────────────────────────────
RUNTIME ESTIMATE (CPU)
─────────────────────────────────────────────────────────────────────────────
    Strategy 1 + 3 : ~8–12 min   (encode + search 2,481 resumes once)
  Strategy 2     : ~3–5 min    (sample 500 from 11,654, random pairs)
  Total          : ~12–18 min

    The labeled resumes are encoded once and reused across Strategies 1 and 3.
"""

import os
import time
import numpy as np
import pandas as pd
import faiss
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

RESUME_FILE       = "data_processed/resumes_cleaned.csv"
FAISS_INDEX_FILE  = "models/faiss_index.bin"
JOB_METADATA_FILE = "models/job_metadata.csv"
OUTPUT_DIR        = "outputs/evaluation"

MODEL_NAME        = "all-MiniLM-L6-v2"
MAX_SEQ_LENGTH    = 384

# K values for Precision@K — keep small and meaningful
K_VALUES          = [1, 3, 5, 10]

# How many results to retrieve per query (must be >= max(K_VALUES))
RETRIEVAL_TOP_K   = 10

# Strategy 2: number of resumes to sample for score separation analysis
# 500 is enough for a stable distribution estimate; use fewer for speed
SCORE_SEP_SAMPLE  = 500

# Random pairs per resume for score separation comparison
RANDOM_PAIRS_N    = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120


# ──────────────────────────────────────────────────────────────────────────────
# CATEGORY → JOB TITLE KEYWORD MAPPING
# ──────────────────────────────────────────────────────────────────────────────
#
# A job is "relevant" for a resume if its lowercase title contains any of the
# listed keywords. Keys must exactly match the category values in the CSV.
# The categories here are the 25 categories from Dataset 2 (resumes_cleaned.csv).

CATEGORY_KEYWORDS = {
    "INFORMATION-TECHNOLOGY"  : ["information technology", "it support", "systems administrator", "it specialist", "technical support"],
    "DATA-SCIENCE"             : ["data scientist", "data science", "machine learning", "ml engineer", "ai engineer"],
    "JAVA-DEVELOPER"           : ["java", "j2ee", "spring", "backend developer", "software engineer"],
    "PYTHON-DEVELOPER"         : ["python", "django", "flask", "backend developer", "software engineer"],
    "WEB-DESIGNING"            : ["web designer", "ui designer", "ux designer", "frontend", "web developer"],
    "DEVOPS-ENGINEER"          : ["devops", "site reliability", "cloud engineer", "infrastructure", "sre"],
    "DATABASE"                 : ["database", "dba", "sql developer", "data engineer"],
    "HR"                       : ["hr", "human resources", "recruiter", "talent acquisition", "people operations"],
    "ADVOCATE"                 : ["lawyer", "attorney", "legal", "counsel", "paralegal"],
    "ARTS"                     : ["artist", "creative", "designer", "art director", "illustrator"],
    "MECHANICAL-ENGINEER"      : ["mechanical", "manufacturing", "process engineer", "product engineer"],
    "SALES"                    : ["sales", "account executive", "business development", "account manager"],
    "HEALTH-AND-FITNESS"       : ["fitness", "personal trainer", "health coach", "wellness", "nutritionist"],
    "CIVIL-ENGINEER"           : ["civil", "structural", "construction", "site engineer"],
    "FINANCE"                  : ["finance", "financial analyst", "accountant", "cpa", "controller"],
    "HADOOP"                   : ["hadoop", "big data", "spark", "data engineer", "etl"],
    "BLOCKCHAIN"               : ["blockchain", "smart contract", "solidity", "web3", "crypto"],
    "ETL-DEVELOPER"            : ["etl", "data pipeline", "data engineer", "informatica", "talend"],
    "OPERATIONS-MANAGER"       : ["operations", "operations manager", "supply chain", "logistics"],
    "PMO"                      : ["project manager", "pmo", "scrum master", "agile coach", "program manager"],
    "BUSINESS-ANALYST"         : ["business analyst", "product analyst", "requirements", "systems analyst"],
    "DOTNET-DEVELOPER"         : [".net", "dotnet", "c#", "asp.net", "microsoft developer"],
    "AUTOMATION-TESTING"       : ["qa", "quality assurance", "test engineer", "automation", "selenium"],
    "ELECTRICAL-ENGINEERING"   : ["electrical", "electronics", "power systems", "embedded"],
    "NETWORK-SECURITY-ENGINEER": ["network", "security", "cybersecurity", "infosec", "firewall"],
}

# Also support the alternate format used in Dataset 2 CSV (plain text, no hyphens).
# We build a unified lookup by adding both forms.
_ALSO_SUPPORT = {
    "Data Science"             : CATEGORY_KEYWORDS["DATA-SCIENCE"],
    "Java Developer"           : CATEGORY_KEYWORDS["JAVA-DEVELOPER"],
    "Python Developer"         : CATEGORY_KEYWORDS["PYTHON-DEVELOPER"],
    "Web Designing"            : CATEGORY_KEYWORDS["WEB-DESIGNING"],
    "DevOps Engineer"          : CATEGORY_KEYWORDS["DEVOPS-ENGINEER"],
    "Database"                 : CATEGORY_KEYWORDS["DATABASE"],
    "HR"                       : CATEGORY_KEYWORDS["HR"],
    "Advocate"                 : CATEGORY_KEYWORDS["ADVOCATE"],
    "Arts"                     : CATEGORY_KEYWORDS["ARTS"],
    "Mechanical Engineer"      : CATEGORY_KEYWORDS["MECHANICAL-ENGINEER"],
    "Sales"                    : CATEGORY_KEYWORDS["SALES"],
    "Health and Fitness"       : CATEGORY_KEYWORDS["HEALTH-AND-FITNESS"],
    "Civil Engineer"           : CATEGORY_KEYWORDS["CIVIL-ENGINEER"],
    "Finance"                  : CATEGORY_KEYWORDS["FINANCE"],
    "Hadoop"                   : CATEGORY_KEYWORDS["HADOOP"],
    "Blockchain"               : CATEGORY_KEYWORDS["BLOCKCHAIN"],
    "ETL Developer"            : CATEGORY_KEYWORDS["ETL-DEVELOPER"],
    "Operations Manager"       : CATEGORY_KEYWORDS["OPERATIONS-MANAGER"],
    "PMO"                      : CATEGORY_KEYWORDS["PMO"],
    "Business Analyst"         : CATEGORY_KEYWORDS["BUSINESS-ANALYST"],
    "DotNet Developer"         : CATEGORY_KEYWORDS["DOTNET-DEVELOPER"],
    "Automation Testing"       : CATEGORY_KEYWORDS["AUTOMATION-TESTING"],
    "Electrical Engineering"   : CATEGORY_KEYWORDS["ELECTRICAL-ENGINEERING"],
    "Network Security Engineer": CATEGORY_KEYWORDS["NETWORK-SECURITY-ENGINEER"],
    "Information Technology"   : CATEGORY_KEYWORDS["INFORMATION-TECHNOLOGY"],
}
CATEGORY_KEYWORDS.update(_ALSO_SUPPORT)


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_data():
    """
    Load all required artefacts.

    Returns:
        labeled    : DataFrame — 2,481 resumes from Dataset 2 with category labels
        all_resumes: DataFrame — all 11,654 resumes (for Strategy 2)
        index      : faiss.Index — 108,702-vector job index
        job_meta   : DataFrame — job metadata (title, company, location, etc.)
        model      : SentenceTransformer — all-MiniLM-L6-v2

    Column contract (from resumes_cleaned.csv produced by resume_pipeline.py):
        cleaned_text, category, source, word_count

    Column contract (from job_metadata.csv produced by embedding_generator.py):
        embed_idx (index col), job_id, title, company, location,
        experience_level, work_type, remote_allowed, salary_min, salary_max
    """

    print("Loading data and models...")

    # ── Resumes ───────────────────────────────────────────────────────────────
    all_resumes = pd.read_csv(RESUME_FILE)

    # Sanity check: required columns must exist
    required_resume_cols = ["cleaned_text", "category", "source", "word_count"]
    missing = [c for c in required_resume_cols if c not in all_resumes.columns]
    if missing:
        raise ValueError(
            f"resumes_cleaned.csv is missing expected columns: {missing}. "
            f"Re-run resume_pipeline.py to regenerate."
        )

    # Filter to labeled resumes that have a known category mapping
    labeled = all_resumes[
        all_resumes["category"].notna()
        & (all_resumes["category"].str.strip() != "")
        & (all_resumes["category"] != "NaN")
        & all_resumes["category"].isin(CATEGORY_KEYWORDS)
    ].copy().reset_index(drop=True)

    print(f"  All resumes            : {len(all_resumes):,}")
    print(f"  Labeled + mapped       : {len(labeled):,}")
    print(f"  Unique categories      : {labeled['category'].nunique()}")

    # ── FAISS index ───────────────────────────────────────────────────────────
    if not os.path.exists(FAISS_INDEX_FILE):
        raise FileNotFoundError(
            f"FAISS index not found at '{FAISS_INDEX_FILE}'. "
            f"Run faiss_index_builder.py first."
        )
    index = faiss.read_index(FAISS_INDEX_FILE)
    print(f"  FAISS index            : {index.ntotal:,} vectors")

    # ── Job metadata ──────────────────────────────────────────────────────────
    if not os.path.exists(JOB_METADATA_FILE):
        raise FileNotFoundError(
            f"Job metadata not found at '{JOB_METADATA_FILE}'. "
            f"Run embedding_generator.py first."
        )
    job_meta = pd.read_csv(JOB_METADATA_FILE, index_col="embed_idx")
    job_meta["title_lower"] = job_meta["title"].fillna("").str.lower()
    print(f"  Job metadata           : {len(job_meta):,} rows")

    # ── SBERT model ───────────────────────────────────────────────────────────
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = MAX_SEQ_LENGTH
    print(f"  SBERT model            : {MODEL_NAME} ({model.get_sentence_embedding_dimension()}-dim)")

    return labeled, all_resumes, index, job_meta, model


# ──────────────────────────────────────────────────────────────────────────────
# RELEVANCE HELPER
# ──────────────────────────────────────────────────────────────────────────────

# Cache: category → frozenset of relevant embed_idx values
_relevance_cache: dict = {}


def get_relevant_indices(category: str, job_meta: pd.DataFrame) -> set:
    """
    Return the set of embed_idx values for jobs relevant to a category.
    A job is relevant if its lowercase title contains any keyword for that category.
    Results are cached after the first call for each category.
    """
    if category in _relevance_cache:
        return _relevance_cache[category]

    keywords = CATEGORY_KEYWORDS.get(category, [])
    if not keywords:
        _relevance_cache[category] = set()
        return set()

    mask = job_meta["title_lower"].apply(
        lambda t: any(kw in t for kw in keywords)
    )
    result = set(job_meta[mask].index.tolist())
    _relevance_cache[category] = result
    return result


# ──────────────────────────────────────────────────────────────────────────────
# METRIC FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def precision_at_k(retrieved: list, relevant: set, k: int) -> float:
    """Fraction of top-K retrieved items that are in the relevant set."""
    if k <= 0:
        return 0.0
    hits = sum(1 for idx in retrieved[:k] if idx in relevant)
    return hits / k


def mrr(retrieved: list, relevant: set) -> float:
    """
    Mean Reciprocal Rank: 1 / (rank of first relevant result).
    Returns 0.0 if no relevant result is found in the retrieved list.
    """
    for rank, idx in enumerate(retrieved, start=1):
        if idx in relevant:
            return 1.0 / rank
    return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# STRATEGY 1 + 3 CORE: ENCODE + EVALUATE ALL LABELED RESUMES
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_labeled_resumes(
    labeled  : pd.DataFrame,
    model    : SentenceTransformer,
    index    : faiss.Index,
    job_meta : pd.DataFrame,
) -> pd.DataFrame:
    """
    Encode all labeled resumes and compute Precision@K and MRR for each one.
    This single pass is shared by Strategies 1 and 3 to avoid encoding
    the same resumes multiple times.

    Returns a DataFrame with one row per resume and columns:
        category, word_count, source,
        P@1, P@3, P@5, P@10,
        MRR, top1_score, n_relevant
    """

    print(f"\nEvaluating {len(labeled):,} labeled resumes (Strategies 1, 3)...")
    print(f"  Encoding in batches of 64...")

    texts      = labeled["cleaned_text"].fillna("").tolist()
    categories = labeled["category"].tolist()

    # Encode all resumes in one batched call — much faster than one-by-one
    t0         = time.time()
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # we normalize below before FAISS search
    ).astype("float32")
    faiss.normalize_L2(embeddings)
    print(f"  Encoding done in {time.time() - t0:.1f}s")

    results = []

    print("  Searching FAISS index and computing metrics...")
    for i in tqdm(range(len(labeled)), desc="  Evaluating"):
        category = categories[i]
        vec      = embeddings[i : i + 1]   # shape (1, 384)

        scores, indices = index.search(vec, RETRIEVAL_TOP_K)
        retrieved       = indices[0].tolist()
        relevant_set    = get_relevant_indices(category, job_meta)

        if not relevant_set:
            # Category has no matching jobs in the index — skip this resume
            continue

        row = {
            "category"  : category,
            "word_count": labeled.iloc[i]["word_count"],
            "source"    : labeled.iloc[i]["source"],
            "MRR"       : mrr(retrieved, relevant_set),
            "top1_score": float(scores[0][0]) if len(scores[0]) > 0 else 0.0,
            "n_relevant": len(relevant_set),
        }
        for k in K_VALUES:
            row[f"P@{k}"] = precision_at_k(retrieved, relevant_set, k)

        results.append(row)

    df = pd.DataFrame(results)
    print(f"  Evaluated : {len(df):,} resumes  ({len(labeled) - len(df)} skipped — no relevant jobs found)")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# STRATEGY 1 — OVERALL Precision@K + MRR
# ──────────────────────────────────────────────────────────────────────────────

def strategy1_overall_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and print the overall headline metrics from the per-resume results.
    Returns a summary DataFrame.
    """

    print("\n" + "=" * 60)
    print("  STRATEGY 1 — Overall Precision@K + MRR")
    print("=" * 60)

    metric_cols  = [f"P@{k}" for k in K_VALUES] + ["MRR"]
    summary_rows = []

    for col in metric_cols:
        if col not in results_df.columns:
            continue
        mean_val = results_df[col].mean()
        std_val  = results_df[col].std()
        print(f"  {col:<8} : {mean_val:.4f}  ±  {std_val:.4f}")
        summary_rows.append({
            "metric": col,
            "mean"  : round(mean_val, 4),
            "std"   : round(std_val,  4),
            "n"     : len(results_df),
        })

    print(f"\n  Evaluated on : {len(results_df):,} labeled resumes")
    print(f"  Categories   : {results_df['category'].nunique()}")

    summary_df = pd.DataFrame(summary_rows)
    path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
    summary_df.to_csv(path, index=False)
    print(f"  Saved: {path}")

    return summary_df


# ──────────────────────────────────────────────────────────────────────────────
# STRATEGY 2 — SCORE SEPARATION (all resumes, no labels)
# ──────────────────────────────────────────────────────────────────────────────

def strategy2_score_separation(
    all_resumes : pd.DataFrame,
    model       : SentenceTransformer,
    index       : faiss.Index,
    n_sample    : int = SCORE_SEP_SAMPLE,
) -> pd.DataFrame:
    """
    Sample resumes from the full dataset (including unlabeled), retrieve their
    top-10 matched jobs, sample 10 random jobs for comparison, and record the
    mean cosine scores for each group.

    Returns a DataFrame with columns: top10_mean, random_mean, gap
    (one row per sampled resume).
    """

    print("\n" + "=" * 60)
    print("  STRATEGY 2 — Score Separation (all resumes, no labels needed)")
    print("=" * 60)

    # Drop rows with empty text, then sample
    valid   = all_resumes[all_resumes["cleaned_text"].notna()
                          & (all_resumes["cleaned_text"].str.strip() != "")].copy()
    sample  = valid.sample(min(n_sample, len(valid)), random_state=42).reset_index(drop=True)

    print(f"  Sampling {len(sample):,} resumes from {len(valid):,} valid resumes")

    n_jobs  = index.ntotal
    rng     = np.random.default_rng(42)
    rows    = []

    texts      = sample["cleaned_text"].fillna("").tolist()
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype("float32")
    faiss.normalize_L2(embeddings)

    print("  Computing score separation...")
    for i in tqdm(range(len(sample)), desc="  Score separation"):
        vec = embeddings[i : i + 1]

        # Top-10 matched scores
        scores, _ = index.search(vec, 10)
        top10_mean = float(np.mean(scores[0]))

        # Random job scores — reconstruct stored vectors and compute dot product
        rand_idxs = rng.integers(0, n_jobs, size=RANDOM_PAIRS_N).tolist()
        rand_scores = []
        for r_idx in rand_idxs:
            try:
                job_vec = np.zeros((index.d,), dtype="float32")
                index.reconstruct(int(r_idx), job_vec)
                sim = float(np.dot(vec[0], job_vec))
            except Exception:
                sim = 0.0
            rand_scores.append(sim)

        random_mean = float(np.mean(rand_scores))
        rows.append({
            "top10_mean" : top10_mean,
            "random_mean": random_mean,
            "gap"        : top10_mean - random_mean,
        })

    sep_df     = pd.DataFrame(rows)
    mean_gap   = sep_df["gap"].mean()
    pct_positive = (sep_df["gap"] > 0).mean() * 100

    print(f"\n  Mean top-10 score    : {sep_df['top10_mean'].mean():.4f}")
    print(f"  Mean random score    : {sep_df['random_mean'].mean():.4f}")
    print(f"  Mean separation gap  : {mean_gap:.4f}")
    print(f"  Resumes with gap > 0 : {pct_positive:.1f}%")

    if mean_gap >= 0.15:
        print("  Interpretation: STRONG discrimination — system clearly prefers relevant jobs.")
    elif mean_gap >= 0.07:
        print("  Interpretation: MODERATE discrimination — system shows meaningful preference.")
    else:
        print("  Interpretation: WEAK discrimination — top results barely outscore random pairs.")

    return sep_df


# ──────────────────────────────────────────────────────────────────────────────
# STRATEGY 3 — WORD COUNT SENSITIVITY
# ──────────────────────────────────────────────────────────────────────────────

def strategy4_word_count(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Group resumes into word-count buckets and report P@10 per bucket.
    Bucket boundaries match real resume length distribution:
        <100 words  : very short (likely incomplete)
        100–200     : short
        200–350     : medium (most common range)
        350–500     : long
        500+        : very long (may be truncated at 5,000 words by pipeline)
    """

    print("\n" + "=" * 60)
    print("  STRATEGY 3 — Word Count Sensitivity")
    print("=" * 60)

    df      = results_df.copy()
    df["wc_bucket"] = pd.cut(
        df["word_count"],
        bins   = [0, 100, 200, 350, 500, 99999],
        labels = ["<100", "100–200", "200–350", "350–500", "500+"],
    )

    agg_cols = [f"P@{k}" for k in K_VALUES] + ["MRR"]
    agg_cols = [c for c in agg_cols if c in df.columns]

    wc_df = (
        df.groupby("wc_bucket", observed=True)[agg_cols]
        .agg(["mean", "count"])
    )
    wc_df.columns = ["_".join(col) for col in wc_df.columns]
    mean_cols = [c for c in wc_df.columns if c.endswith("_mean")]
    n_col     = [c for c in wc_df.columns if c.endswith("_count")][:1]
    wc_df     = wc_df[n_col + mean_cols].copy()
    wc_df     = wc_df.rename(columns={n_col[0]: "n"} if n_col else {})
    wc_df.columns = [c.replace("_mean", "") for c in wc_df.columns]

    print(wc_df.round(4).to_string())

    path = os.path.join(OUTPUT_DIR, "word_count_metrics.csv")
    wc_df.to_csv(path)
    print(f"\n  Saved: {path}")

    return wc_df


# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────────────────────────────────────

def plot_precision_at_k(results_df: pd.DataFrame):
    """
    Plot 1: Precision@K curve — how precision changes as K increases.
    Also shows per-category lines to reveal spread.
    """

    k_vals  = K_VALUES
    mean_p  = [results_df[f"P@{k}"].mean() for k in k_vals if f"P@{k}" in results_df.columns]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Strategy 1 — Precision@K and MRR", fontsize=13, fontweight="bold")

    # Left: overall P@K curve
    axes[0].plot(k_vals, mean_p, "o-", color="#4C72B0", linewidth=2.5, markersize=7)
    for k, p in zip(k_vals, mean_p):
        axes[0].annotate(f"{p:.3f}", (k, p), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=9)
    axes[0].set_title("Overall Precision@K")
    axes[0].set_xlabel("K (number of results retrieved)")
    axes[0].set_ylabel("Precision@K")
    axes[0].set_xticks(k_vals)
    axes[0].set_ylim(0, 1)

    # Right: MRR distribution as histogram
    if "MRR" in results_df.columns:
        axes[1].hist(results_df["MRR"].dropna(), bins=20, color="#DD8452", edgecolor="white")
        axes[1].axvline(results_df["MRR"].mean(), color="#C44E52", linestyle="--",
                        linewidth=2, label=f"Mean MRR = {results_df['MRR'].mean():.3f}")
        axes[1].set_title("MRR Distribution Across Resumes")
        axes[1].set_xlabel("MRR (1 = top-1 always correct)")
        axes[1].set_ylabel("Count")
        axes[1].legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "01_precision_at_k.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_score_separation(sep_df: pd.DataFrame):
    """
    Plot 2: Score separation — top-10 vs random cosine score distributions.
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Strategy 2 — Score Separation (Top-10 vs Random)", fontsize=13, fontweight="bold")

    # Left: overlapping histograms
    axes[0].hist(sep_df["top10_mean"], bins=40, alpha=0.65, color="#55A868",
                 label="Mean top-10 score", edgecolor="white")
    axes[0].hist(sep_df["random_mean"], bins=40, alpha=0.65, color="#C44E52",
                 label="Mean random score", edgecolor="white")
    axes[0].axvline(sep_df["top10_mean"].mean(), color="#1a6e38", linestyle="--",
                    linewidth=1.5, label=f"Top-10 mean = {sep_df['top10_mean'].mean():.3f}")
    axes[0].axvline(sep_df["random_mean"].mean(), color="#7d1c1c", linestyle="--",
                    linewidth=1.5, label=f"Random mean = {sep_df['random_mean'].mean():.3f}")
    axes[0].set_title("Cosine Score Distributions")
    axes[0].set_xlabel("Mean Cosine Similarity")
    axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=8)

    # Right: gap distribution
    axes[1].hist(sep_df["gap"], bins=40, color="#4C72B0", edgecolor="white")
    axes[1].axvline(sep_df["gap"].mean(), color="#1c3a7d", linestyle="--",
                    linewidth=2, label=f"Mean gap = {sep_df['gap'].mean():.3f}")
    axes[1].axvline(0, color="black", linestyle="-", linewidth=1, alpha=0.4)
    axes[1].set_title("Score Separation Gap per Resume\n(top-10 mean − random mean)")
    axes[1].set_xlabel("Score Gap")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_score_separation.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_word_count_sensitivity(wc_df: pd.DataFrame):
    """
    Plot 3: P@10 by word-count bucket — bar chart.
    """

    if "P@10" not in wc_df.columns or wc_df.empty:
        print("  Skipping word count plot — P@10 column not available.")
        return

    plot_data = wc_df["P@10"].dropna()
    n_data    = wc_df["n"].reindex(plot_data.index) if "n" in wc_df.columns else None

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        range(len(plot_data)),
        plot_data.values,
        color="#8172B3", edgecolor="white",
    )
    ax.set_xticks(range(len(plot_data)))
    ax.set_xticklabels(plot_data.index, fontsize=10)
    ax.set_title("Strategy 3 — P@10 by Resume Word Count Range", fontsize=13, fontweight="bold")
    ax.set_xlabel("Word Count Range")
    ax.set_ylabel("Precision@10")
    ax.set_ylim(0, 1)

    # Annotate bars with value and n
    for i, (bar, val) in enumerate(zip(bars, plot_data.values)):
        label = f"{val:.3f}"
        if n_data is not None and not pd.isna(n_data.iloc[i]):
            label += f"\n(n={int(n_data.iloc[i])})"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            label, ha="center", va="bottom", fontsize=9,
        )

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_word_count_sensitivity.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():

    t_start = time.time()

    print("=" * 65)
    print("  PHASE 4 — MATCHING SYSTEM EVALUATION")
    print("  3 strategies: P@K+MRR | Score Separation | Word Count")
    print("=" * 65)

    # ── Load ──────────────────────────────────────────────────────────────────
    labeled, all_resumes, index, job_meta, model = load_data()

    # Pre-warm the relevance cache for all categories present in the data.
    # This avoids repeated full scans of job_meta during the evaluation loop.
    print("\nPre-warming relevance cache...")
    unique_cats = labeled["category"].unique().tolist()
    for cat in unique_cats:
        get_relevant_indices(cat, job_meta)
    print(f"  Cached relevance sets for {len(unique_cats)} categories")

    # ── Shared encode + evaluate pass (used by Strategies 1 and 3) ───────────
    results_df = evaluate_labeled_resumes(labeled, model, index, job_meta)

    if results_df.empty:
        print("\nERROR: No resumes were successfully evaluated.")
        print("Check that CATEGORY_KEYWORDS matches the category values in your CSV.")
        return

    # ── Strategy 1 ────────────────────────────────────────────────────────────
    summary_df = strategy1_overall_metrics(results_df)

    # ── Strategy 2 ────────────────────────────────────────────────────────────
    sep_df = strategy2_score_separation(all_resumes, model, index)

    # ── Strategy 3 ────────────────────────────────────────────────────────────
    wc_df = strategy4_word_count(results_df)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_precision_at_k(results_df)
    plot_score_separation(sep_df)
    plot_word_count_sensitivity(wc_df)

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'=' * 65}")
    print(f"  EVALUATION COMPLETE  ({elapsed / 60:.1f} min)")
    print(f"{'=' * 65}")
    print(f"  Resumes evaluated  : {len(results_df):,}")
    print(f"  Categories covered : {results_df['category'].nunique()}")

    for k in K_VALUES:
        col = f"P@{k}"
        if col in results_df.columns:
            print(f"  {col:<8}         : {results_df[col].mean():.4f}")
    if "MRR" in results_df.columns:
        print(f"  MRR              : {results_df['MRR'].mean():.4f}")
    print(f"  Score gap (mean) : {sep_df['gap'].mean():.4f}")
    print(f"\n  Output directory : {OUTPUT_DIR}/")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
