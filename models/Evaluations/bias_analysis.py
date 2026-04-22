"""
bias_analysis.py
================
Phase 4 — Bias Check & Error Analysis

Investigates whether the matching system performs differently across:
  1. Resume source        (dataset1 / dataset2 / dataset3)
  2. Resume category      (Java Developer / Data Science / etc.)
  3. Job experience level (Entry / Mid / Senior)
  4. Resume word count    (short / medium / long)

For each subgroup, reports P@10, F1@10 and MRR. Flags groups where
performance drops significantly below the overall mean (potential bias).

NOTE: Only Dataset 2 resumes (2,481 rows) have category labels.
      Analyses that depend on category labels use only those rows.
      Word-count and source analyses are scoped to the same labeled set
      to keep the comparison fair.

Outputs:
  - outputs/evaluation/bias_by_source.csv
  - outputs/evaluation/bias_by_category.csv
  - outputs/evaluation/bias_by_experience_level.csv
  - outputs/evaluation/bias_by_word_count.csv
  - outputs/evaluation/bias_summary.png
"""

import os
import numpy as np
import pandas as pd
import faiss
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================

RESUME_FILE       = "data_processed/resumes_cleaned.csv"
FAISS_INDEX_FILE  = "models/faiss_index.bin"
JOB_METADATA_FILE = "models/job_metadata.csv"

OUTPUT_DIR        = "outputs/evaluation"

MODEL_NAME        = "all-MiniLM-L6-v2"
MAX_SEQ_LENGTH    = 384

TOP_K             = 10
N_SAMPLE_PER_GROUP = 100     # Max resumes to evaluate per subgroup (for speed)

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

# Same keyword mapping used in evaluate_matching.py
CATEGORY_KEYWORDS = {
    "Data Science"             : ["data scientist", "data science", "machine learning", "ml engineer", "ai engineer"],
    "Java Developer"           : ["java", "j2ee", "spring", "backend developer", "software engineer"],
    "Python Developer"         : ["python", "django", "flask", "backend developer", "software engineer"],
    "Web Designing"            : ["web designer", "ui designer", "ux designer", "frontend", "web developer"],
    "DevOps Engineer"          : ["devops", "site reliability", "cloud engineer", "infrastructure", "sre"],
    "Database"                 : ["database", "dba", "sql developer", "data engineer"],
    "HR"                       : ["hr", "human resources", "recruiter", "talent acquisition", "people operations"],
    "Advocate"                 : ["lawyer", "attorney", "legal", "counsel", "paralegal"],
    "Arts"                     : ["artist", "creative", "designer", "art director", "illustrator"],
    "Mechanical Engineer"      : ["mechanical", "manufacturing", "process engineer", "product engineer"],
    "Sales"                    : ["sales", "account executive", "business development", "account manager"],
    "Health and Fitness"       : ["fitness", "personal trainer", "health coach", "wellness", "nutritionist"],
    "Civil Engineer"           : ["civil", "structural", "construction", "site engineer"],
    "Finance"                  : ["finance", "financial analyst", "accountant", "cpa", "controller"],
    "Hadoop"                   : ["hadoop", "big data", "spark", "data engineer", "etl"],
    "Blockchain"               : ["blockchain", "smart contract", "solidity", "web3", "crypto"],
    "ETL Developer"            : ["etl", "data pipeline", "data engineer", "informatica", "talend"],
    "Operations Manager"       : ["operations", "operations manager", "supply chain", "logistics"],
    "PMO"                      : ["project manager", "pmo", "scrum master", "agile coach", "program manager"],
    "Business Analyst"         : ["business analyst", "product analyst", "requirements", "systems analyst"],
    "DotNet Developer"         : [".net", "dotnet", "c#", "asp.net", "microsoft developer"],
    "Automation Testing"       : ["qa", "quality assurance", "test engineer", "automation", "selenium"],
    "Electrical Engineering"   : ["electrical", "electronics", "power systems", "embedded"],
    "SAP Developer"            : ["sap", "abap", "sap consultant", "erp"],
    "Testing"                  : ["tester", "qa engineer", "quality engineer", "manual testing"],
    "Network Security Engineer": ["network", "security", "cybersecurity", "infosec", "firewall"],
}


# ==============================
# LOAD
# ==============================

def load_data():
    print("Loading data...")

    resumes  = pd.read_csv(RESUME_FILE)
    index    = faiss.read_index(FAISS_INDEX_FILE)
    job_meta = pd.read_csv(JOB_METADATA_FILE, index_col="embed_idx")
    job_meta["title_lower"] = job_meta["title"].fillna("").str.lower()
    job_meta["exp_level"]   = job_meta["experience_level"].fillna("Unknown").str.strip()

    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = MAX_SEQ_LENGTH

    # Only keep labeled resumes that map to a known category
    labeled = resumes[
        resumes["category"].notna() &
        (resumes["category"].str.strip() != "") &
        (resumes["category"] != "NaN") &
        resumes["category"].isin(CATEGORY_KEYWORDS)
    ].copy().reset_index(drop=True)

    print(f"  Resumes (total)  : {len(resumes):,}")
    print(f"  Labeled + mapped : {len(labeled):,}")
    print(f"  FAISS index      : {index.ntotal:,} vectors")
    return labeled, index, job_meta, model


# ==============================
# RELEVANCE
# ==============================

def get_relevant_indices(category: str, job_meta: pd.DataFrame) -> set:
    keywords = CATEGORY_KEYWORDS.get(category, [])
    if not keywords:
        return set()
    mask = job_meta["title_lower"].apply(lambda t: any(kw in t for kw in keywords))
    return set(job_meta[mask].index.tolist())


# ==============================
# SINGLE-RESUME EVALUATION
# ==============================

def evaluate_one(row: pd.Series, model, index, job_meta, top_k: int = TOP_K) -> dict | None:
    """Evaluate one resume row. Returns dict with P@K, F1@K, MRR, or None to skip."""

    category = row.get("category", "")
    text     = str(row.get("cleaned_text", ""))

    if not text.strip() or category not in CATEGORY_KEYWORDS:
        return None

    vector = model.encode([text], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(vector)

    scores, indices  = index.search(vector, top_k)
    retrieved        = indices[0].tolist()
    relevant_set     = get_relevant_indices(category, job_meta)

    if not relevant_set:
        return None

    hits    = sum(1 for idx in retrieved if idx in relevant_set)
    p_at_k  = hits / top_k
    # Recall denominator can be very large; this value will be small — expected
    r_at_k  = hits / len(relevant_set)
    f1_at_k = 2 * p_at_k * r_at_k / (p_at_k + r_at_k) if (p_at_k + r_at_k) > 0 else 0.0

    mrr = 0.0
    for rank, idx in enumerate(retrieved, 1):
        if idx in relevant_set:
            mrr = 1.0 / rank
            break

    return {
        "p_at_k"    : p_at_k,
        "r_at_k"    : r_at_k,
        "f1_at_k"   : f1_at_k,
        "mrr"       : mrr,
        "top1_score": float(scores[0][0]) if len(scores[0]) > 0 else 0.0,
        "n_relevant": len(relevant_set),
    }


# ==============================
# GROUP EVALUATION
# ==============================

def evaluate_group(
    df_group : pd.DataFrame,
    model, index, job_meta,
    label    : str = "group",
    max_n    : int = N_SAMPLE_PER_GROUP,
) -> dict:
    """Evaluate a subgroup of resumes. Returns aggregated metrics dict."""

    sample  = df_group.sample(min(max_n, len(df_group)), random_state=42)
    results = []

    for _, row in tqdm(sample.iterrows(), total=len(sample), desc=f"  {label}", leave=False):
        r = evaluate_one(row, model, index, job_meta)
        if r:
            results.append(r)

    if not results:
        return {"n": 0, "P@10": None, "R@10": None, "F1@10": None, "MRR": None}

    df = pd.DataFrame(results)
    return {
        "n"             : len(df),
        "P@10"          : round(df["p_at_k"].mean(), 4),
        "R@10"          : round(df["r_at_k"].mean(), 4),
        "F1@10"         : round(df["f1_at_k"].mean(), 4),
        "MRR"           : round(df["mrr"].mean(), 4),
        "top1_score_mean": round(df["top1_score"].mean(), 4),
    }


# ==============================
# FOUR BIAS DIMENSIONS
# ==============================

def bias_by_source(labeled, model, index, job_meta):
    """Performance breakdown by resume source (dataset1/2/3)."""
    print("\n[1/4] Bias by Resume Source...")
    rows = []
    for source, group in labeled.groupby("source"):
        m = evaluate_group(group, model, index, job_meta, label=source)
        m["source"] = source
        rows.append(m)

    df = pd.DataFrame(rows).set_index("source")
    print(df[["n", "P@10", "R@10", "F1@10", "MRR"]].to_string())
    return df


def bias_by_category(labeled, model, index, job_meta):
    """Performance breakdown by resume category."""
    print("\n[2/4] Bias by Resume Category...")
    rows = []
    for category, group in labeled.groupby("category"):
        m = evaluate_group(group, model, index, job_meta, label=category)
        m["category"] = category
        rows.append(m)

    df = pd.DataFrame(rows).set_index("category").sort_values("F1@10", ascending=False)
    print(df[["n", "P@10", "R@10", "F1@10", "MRR"]].to_string())
    return df


def bias_by_experience_level(labeled, model, index, job_meta):
    """
    For each resume, retrieve the top-1 matched job and record its experience level.
    Reports mean top-1 cosine score grouped by that experience level.
    Answers: does the system consistently send candidates to a particular seniority tier?
    """
    print("\n[3/4] Bias by Job Experience Level (top-1 match)...")

    sample = labeled.sample(min(300, len(labeled)), random_state=42)
    rows   = []

    for _, row in tqdm(sample.iterrows(), total=len(sample), leave=False):
        text = str(row.get("cleaned_text", ""))
        if not text.strip():
            continue

        vector = model.encode([text], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(vector)
        scores, indices = index.search(vector, 1)

        top_idx   = int(indices[0][0])
        top_score = float(scores[0][0])

        try:
            exp_level = job_meta.iloc[top_idx]["exp_level"]
        except (IndexError, KeyError):
            exp_level = "Unknown"

        rows.append({
            "category" : row.get("category", ""),
            "exp_level": exp_level if exp_level and exp_level != "nan" else "Unknown",
            "top_score": top_score,
        })

    df      = pd.DataFrame(rows)
    summary = (
        df.groupby("exp_level")["top_score"]
        .agg(["mean", "std", "count"])
        .round(4)
        .rename(columns={"mean": "mean_score", "std": "std_score", "count": "count"})
        .sort_values("mean_score", ascending=False)
    )
    print(summary.to_string())
    return summary


def bias_by_word_count(labeled, model, index, job_meta):
    """Performance by resume word count bucket."""
    print("\n[4/4] Bias by Resume Word Count...")

    df = labeled.copy()
    df["wc_bucket"] = pd.cut(
        df["word_count"],
        bins=[0, 100, 200, 350, 500, 5000],
        labels=["<100", "100–200", "200–350", "350–500", "500+"]
    )

    rows = []
    for bucket, group in df.groupby("wc_bucket", observed=True):
        m = evaluate_group(group, model, index, job_meta, label=str(bucket))
        m["word_count_range"] = str(bucket)
        rows.append(m)

    result = pd.DataFrame(rows).set_index("word_count_range")
    print(result[["n", "P@10", "R@10", "F1@10", "MRR"]].to_string())
    return result


# ==============================
# PLOTTING
# ==============================

def plot_bias_summary(source_df, category_df, exp_df, wc_df, output_dir):
    """4-panel bias summary chart."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Bias Analysis — System Performance Across Subgroups",
        fontsize=14, fontweight="bold"
    )

    # Panel 1: By source
    ax = axes[0, 0]
    if not source_df.empty and "F1@10" in source_df.columns:
        source_df["F1@10"].dropna().plot(kind="bar", ax=ax, color="#4C72B0", edgecolor="white")
    ax.set_title("F1@10 by Resume Source")
    ax.set_ylabel("F1@10")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
    ax.set_ylim(0, 1)

    # Panel 2: By category (top 15 by F1)
    ax = axes[0, 1]
    if not category_df.empty and "F1@10" in category_df.columns:
        top15 = category_df["F1@10"].dropna().sort_values(ascending=True).tail(15)
        top15.plot(kind="barh", ax=ax, color="#DD8452", edgecolor="white")
    ax.set_title("F1@10 by Resume Category (Top 15)")
    ax.set_xlabel("F1@10")
    ax.set_xlim(0, 1)

    # Panel 3: By experience level (mean top-1 score)
    ax = axes[1, 0]
    if not exp_df.empty and "mean_score" in exp_df.columns:
        exp_df["mean_score"].dropna().sort_values(ascending=False).plot(
            kind="bar", ax=ax, color="#55A868", edgecolor="white"
        )
    ax.set_title("Mean Top-1 Cosine Score\nby Matched Job Experience Level")
    ax.set_ylabel("Mean Cosine Score")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
    ax.set_ylim(0, 1)

    # Panel 4: By word count
    ax = axes[1, 1]
    if not wc_df.empty and "F1@10" in wc_df.columns:
        wc_df["F1@10"].dropna().plot(kind="bar", ax=ax, color="#8172B3", edgecolor="white")
    ax.set_title("F1@10 by Resume Word Count Range")
    ax.set_ylabel("F1@10")
    ax.set_xlabel("Word Count Range")
    ax.tick_params(axis="x", rotation=15)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(output_dir, "bias_summary.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {path}")


# ==============================
# MAIN
# ==============================

def main():

    print("=" * 60)
    print("  PHASE 4 — BIAS ANALYSIS & ERROR CHECK")
    print("=" * 60)

    labeled, index, job_meta, model = load_data()

    source_df   = bias_by_source(labeled, model, index, job_meta)
    category_df = bias_by_category(labeled, model, index, job_meta)
    exp_df      = bias_by_experience_level(labeled, model, index, job_meta)
    wc_df       = bias_by_word_count(labeled, model, index, job_meta)

    # Save CSVs
    source_df.to_csv(os.path.join(OUTPUT_DIR, "bias_by_source.csv"))
    category_df.to_csv(os.path.join(OUTPUT_DIR, "bias_by_category.csv"))
    exp_df.to_csv(os.path.join(OUTPUT_DIR, "bias_by_experience_level.csv"))
    wc_df.to_csv(os.path.join(OUTPUT_DIR, "bias_by_word_count.csv"))
    print(f"\n  CSVs saved to {OUTPUT_DIR}/")

    # Plot
    print("\nGenerating bias summary chart...")
    plot_bias_summary(source_df, category_df, exp_df, wc_df, OUTPUT_DIR)

    # Flag underperforming categories
    print("\n--- Bias Flag Report ---")
    f1_vals = category_df["F1@10"].dropna()
    if f1_vals.empty:
        print("  No F1 data available.")
        return

    overall_f1 = f1_vals.mean()
    threshold  = overall_f1 * 0.7    # groups below 70% of mean are flagged

    print(f"  Overall mean F1@10 : {overall_f1:.4f}")
    print(f"  Flag threshold     : {threshold:.4f}  (70% of mean)")

    flagged = f1_vals[f1_vals < threshold]
    if not flagged.empty:
        print(f"\n  Underperforming categories (potential bias):")
        for cat, score in flagged.items():
            print(f"    {cat:40s} F1@10 = {score:.4f}")
    else:
        print("  No significantly underperforming categories detected.")

    print(f"\n{'='*60}")
    print(f"  Bias analysis complete!  Outputs in {OUTPUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
