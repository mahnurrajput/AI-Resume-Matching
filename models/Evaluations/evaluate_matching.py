"""
evaluate_matching.py
====================
Phase 4 — Model Evaluation: Precision@K, Recall@K, F1@K, MRR, AUC-ROC

Evaluation Strategy:
--------------------
This system uses Sentence-BERT for semantic matching — not a supervised
classifier — so traditional train/test splits don't apply directly.

Instead, we use CATEGORY-BASED EVALUATION:
  - Dataset 2 resumes have labeled categories (e.g. "Data Science", "Java Developer")
  - LinkedIn jobs have title text we can map to those categories
  - For each resume, we run matching → check if top-K results contain
    jobs from the correct category (the "relevant" jobs)
  - This gives us Precision@K, Recall@K, F1@K, and MRR

  NOTE on Recall@K: The denominator is ALL relevant jobs in the index
  (can be thousands). So Recall@K at small K values will always be very low.
  This is expected and correct — it just reflects that no top-10 list can
  cover thousands of matching jobs. F1@K and P@K are the more meaningful metrics.

K-Fold Cross-Validation:
  - We split the labeled resumes into 5 folds
  - Each fold is evaluated independently
  - We report mean ± std across folds

Metrics Reported:
  - Precision@K  : What fraction of top-K results are in the correct category
  - Recall@K     : What fraction of all correct-category jobs appear in top-K
  - F1@K         : Harmonic mean of Precision@K and Recall@K
  - MRR          : Mean Reciprocal Rank — position of first correct result
  - AUC-ROC      : Computed from score distributions (relevant vs random pairs)

Output:
  - outputs/evaluation/metrics_summary.csv
  - outputs/evaluation/per_category_metrics.csv
  - outputs/evaluation/kfold_results.csv
  - outputs/evaluation/score_distributions.png
  - outputs/evaluation/precision_recall_at_k.png
  - outputs/evaluation/kfold_f1_variance.png
"""

import os
import numpy as np
import pandas as pd
import faiss
import matplotlib.pyplot as plt
import seaborn as sns

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================

RESUME_FILE         = "data_processed/resumes_cleaned.csv"
FAISS_INDEX_FILE    = "models/faiss_index.bin"
JOB_METADATA_FILE   = "models/job_metadata.csv"

OUTPUT_DIR          = "outputs/evaluation"

MODEL_NAME          = "all-MiniLM-L6-v2"
MAX_SEQ_LENGTH      = 384

# K values to evaluate at
K_VALUES            = [1, 3, 5, 10, 20]

# Number of K-Fold splits
N_FOLDS             = 5

# How many jobs to retrieve per query during evaluation
# Needs to be >= max(K_VALUES) to compute all metrics correctly
RETRIEVAL_TOP_K     = max(K_VALUES) * 2   # 40

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120


# ==============================
# CATEGORY → JOB TITLE MAPPING
# ==============================

# Maps resume categories (from Dataset 2) to keywords found in job titles.
# A job is "relevant" for a resume if its title contains any of these keywords.
# Keys must match category values in the resumes CSV exactly (case-sensitive).

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
# DATA LOADING
# ==============================

def load_data():
    """Load resumes, FAISS index, job metadata, and SBERT model."""

    print("Loading data...")

    resumes = pd.read_csv(RESUME_FILE)

    # Only Dataset 2 has category labels — filter to labeled rows
    labeled = resumes[
        resumes["category"].notna() &
        (resumes["category"].str.strip() != "") &
        (resumes["category"] != "NaN") &
        resumes["category"].isin(CATEGORY_KEYWORDS)
    ].copy().reset_index(drop=True)

    print(f"  Total resumes      : {len(resumes):,}")
    print(f"  Labeled (in map)   : {len(labeled):,}")
    print(f"  Unique categories  : {labeled['category'].nunique()}")

    # Load FAISS index
    index = faiss.read_index(FAISS_INDEX_FILE)
    print(f"  FAISS index loaded : {index.ntotal:,} vectors")

    # Load job metadata — embed_idx is the row position in the FAISS index
    job_meta = pd.read_csv(JOB_METADATA_FILE, index_col="embed_idx")
    job_meta["title_lower"] = job_meta["title"].fillna("").str.lower()

    # Load SBERT model — same model used to build the index
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = MAX_SEQ_LENGTH
    print(f"  SBERT model loaded : {MODEL_NAME}")

    return labeled, index, job_meta, model


# ==============================
# RELEVANCE JUDGMENT
# ==============================

def get_relevant_job_indices(category: str, job_meta: pd.DataFrame) -> set:
    """
    Return the set of FAISS embed_idx values for jobs relevant to a category.
    A job is relevant if its title contains any keyword for that category.
    """
    keywords = CATEGORY_KEYWORDS.get(category, [])
    if not keywords:
        return set()

    mask = job_meta["title_lower"].apply(
        lambda t: any(kw in t for kw in keywords)
    )
    return set(job_meta[mask].index.tolist())


# ==============================
# METRIC COMPUTATION
# ==============================

def precision_at_k(retrieved: list, relevant: set, k: int) -> float:
    """Fraction of top-K retrieved items that are relevant."""
    hits = sum(1 for idx in retrieved[:k] if idx in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved: list, relevant: set, k: int) -> float:
    """Fraction of all relevant items that appear in top-K.
    NOTE: denominator can be large (thousands of jobs) so this value
    will naturally be very small — that is correct behaviour."""
    if not relevant:
        return 0.0
    hits = sum(1 for idx in retrieved[:k] if idx in relevant)
    return hits / len(relevant)


def f1_at_k(p: float, r: float) -> float:
    """Harmonic mean of precision and recall."""
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def reciprocal_rank(retrieved: list, relevant: set) -> float:
    """1 / rank of first relevant result. 0 if none found."""
    for rank, idx in enumerate(retrieved, start=1):
        if idx in relevant:
            return 1.0 / rank
    return 0.0


# ==============================
# EVALUATE ONE RESUME
# ==============================

def evaluate_resume(
    resume_text : str,
    category    : str,
    model       : SentenceTransformer,
    index       : faiss.Index,
    job_meta    : pd.DataFrame,
    k_values    : list,
    top_k       : int = RETRIEVAL_TOP_K,
) -> dict:
    """
    Encode a resume, search the FAISS index, compute all metrics.
    Returns a dict with keys like 'P@1', 'R@5', 'F1@10', 'MRR', 'top1_score'.
    """
    vector = model.encode([resume_text], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(vector)

    scores, indices = index.search(vector, top_k)
    retrieved    = indices[0].tolist()
    relevant_set = get_relevant_job_indices(category, job_meta)

    result = {
        "category"   : category,
        "n_relevant" : len(relevant_set),
        "MRR"        : reciprocal_rank(retrieved, relevant_set),
        "top1_score" : float(scores[0][0]) if len(scores[0]) > 0 else 0.0,
    }

    for k in k_values:
        p = precision_at_k(retrieved, relevant_set, k)
        r = recall_at_k(retrieved, relevant_set, k)
        result[f"P@{k}"]  = p
        result[f"R@{k}"]  = r
        result[f"F1@{k}"] = f1_at_k(p, r)

    return result


# ==============================
# K-FOLD EVALUATION
# ==============================

def run_kfold_evaluation(
    labeled  : pd.DataFrame,
    model    : SentenceTransformer,
    index    : faiss.Index,
    job_meta : pd.DataFrame,
    n_folds  : int = N_FOLDS,
    k_values : list = K_VALUES,
) -> pd.DataFrame:
    """
    K-Fold evaluation across all labeled resumes.
    Each fold evaluates a held-out subset; we aggregate across all folds.
    Returns a DataFrame with one row per resume and all metric columns.
    """

    print(f"\nRunning {n_folds}-Fold Evaluation ({len(labeled):,} resumes)...")

    kf          = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_results = []

    for fold_idx, (_, test_idx) in enumerate(kf.split(labeled)):
        fold_df = labeled.iloc[test_idx]
        print(f"  Fold {fold_idx + 1}/{n_folds}  ({len(fold_df)} resumes)...")

        for _, row in tqdm(fold_df.iterrows(), total=len(fold_df), leave=False):
            category = row["category"]
            text     = str(row.get("cleaned_text", ""))

            if not text.strip() or category not in CATEGORY_KEYWORDS:
                continue

            result         = evaluate_resume(text, category, model, index, job_meta, k_values)
            result["fold"] = fold_idx + 1
            all_results.append(result)

    return pd.DataFrame(all_results)


# ==============================
# AUC-ROC FROM SCORE DISTRIBUTIONS
# ==============================

def compute_auc_roc(
    labeled   : pd.DataFrame,
    model     : SentenceTransformer,
    index     : faiss.Index,
    job_meta  : pd.DataFrame,
    n_samples : int = 500,
) -> tuple:
    """
    Estimate AUC-ROC from score distributions.

    For each sampled resume:
      Positive pairs: top-20 retrieved job scores (likely relevant)
      Negative pairs: scores of 20 randomly sampled jobs from the full index
                      (queried with the SAME resume vector — correct approach)

    The binary label is 1 if the job index is in the relevant set for the
    resume's category, 0 otherwise.

    Returns:
        (auc_score, y_true_list, y_score_list)
    """

    print(f"\nComputing AUC-ROC (n_samples={n_samples})...")

    sample = labeled[labeled["category"].isin(CATEGORY_KEYWORDS)].sample(
        min(n_samples, len(labeled)), random_state=42
    )

    n_jobs  = index.ntotal
    rng     = np.random.default_rng(42)
    y_true  = []
    y_score = []

    for _, row in tqdm(sample.iterrows(), total=len(sample)):
        category = row["category"]
        text     = str(row.get("cleaned_text", ""))
        if not text.strip():
            continue

        # Encode resume once
        vector = model.encode([text], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(vector)

        relevant_set = get_relevant_job_indices(category, job_meta)

        # --- Positive evidence: top-20 retrieved ---
        scores, indices = index.search(vector, 20)
        for idx, score in zip(indices[0].tolist(), scores[0].tolist()):
            y_true.append(1 if idx in relevant_set else 0)
            y_score.append(float(score))

        # --- Negative evidence: 20 random job indices, scored via reconstruct ---
        # We reconstruct the stored vectors and compute dot product directly.
        # This correctly gives us the cosine similarity between THIS resume
        # and randomly chosen jobs — a true negative sample.
        random_idxs = rng.integers(0, n_jobs, size=20).tolist()
        for r_idx in random_idxs:
            try:
                # Reconstruct the job vector from the FAISS index
                job_vec = np.zeros((1, index.d), dtype="float32")
                index.reconstruct(int(r_idx), job_vec[0])
                # Dot product of L2-normalised vectors = cosine similarity
                sim = float(np.dot(vector[0], job_vec[0]))
            except Exception:
                sim = 0.0
            y_true.append(1 if r_idx in relevant_set else 0)
            y_score.append(sim)

    if len(set(y_true)) < 2:
        print("  WARNING: AUC-ROC requires both positive and negative samples. Skipping.")
        return 0.5, y_true, y_score

    auc_score = roc_auc_score(y_true, y_score)
    print(f"  AUC-ROC : {auc_score:.4f}")
    return auc_score, y_true, y_score


# ==============================
# PLOTTING
# ==============================

def plot_score_distributions(results_df: pd.DataFrame, y_true, y_score, output_dir: str):
    """Plot cosine score distributions for relevant vs random pairs."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Score Distributions — Relevant vs Random Job Pairs", fontsize=13, fontweight="bold")

    # Left: mean top-1 score by category
    cat_scores = (
        results_df.groupby("category")["top1_score"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
    )
    axes[0].barh(cat_scores.index, cat_scores.values, color="#4C72B0")
    axes[0].set_title("Mean Top-1 Cosine Score by Category")
    axes[0].set_xlabel("Mean Cosine Similarity Score")
    axes[0].invert_yaxis()

    # Right: relevant vs random score histogram
    y_true_arr  = np.array(y_true)
    y_score_arr = np.array(y_score)

    relevant_scores = y_score_arr[y_true_arr == 1]
    random_scores   = y_score_arr[y_true_arr == 0]

    axes[1].hist(relevant_scores, bins=40, alpha=0.6, color="#55A868", label="Relevant (top-K)")
    axes[1].hist(random_scores,   bins=40, alpha=0.6, color="#C44E52", label="Random (negative)")
    axes[1].set_title("Cosine Score: Relevant vs Random Pairs")
    axes[1].set_xlabel("Cosine Similarity Score")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "score_distributions.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_precision_recall_at_k(results_df: pd.DataFrame, k_values: list, output_dir: str):
    """Plot Precision@K, Recall@K, F1@K as K increases."""

    mean_p  = [results_df[f"P@{k}"].mean()  for k in k_values]
    mean_r  = [results_df[f"R@{k}"].mean()  for k in k_values]
    mean_f1 = [results_df[f"F1@{k}"].mean() for k in k_values]

    plt.figure(figsize=(9, 5))
    plt.plot(k_values, mean_p,  "o-", label="Precision@K", color="#4C72B0", linewidth=2)
    plt.plot(k_values, mean_r,  "s-", label="Recall@K",    color="#DD8452", linewidth=2)
    plt.plot(k_values, mean_f1, "^-", label="F1@K",        color="#55A868", linewidth=2)

    plt.title("Precision, Recall, and F1 at K", fontsize=13, fontweight="bold")
    plt.xlabel("K (number of results retrieved)")
    plt.ylabel("Score")
    plt.xticks(k_values)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(output_dir, "precision_recall_at_k.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_kfold_variance(results_df: pd.DataFrame, k_values: list, output_dir: str):
    """Plot per-fold F1@K to show stability across folds."""

    f1_cols  = [f"F1@{k}" for k in k_values if f"F1@{k}" in results_df.columns]
    fold_f1  = results_df.groupby("fold")[f1_cols].mean()

    plt.figure(figsize=(9, 5))
    for col in fold_f1.columns:
        plt.plot(fold_f1.index, fold_f1[col], "o-", label=col)

    plt.title(f"F1@K Across {results_df['fold'].nunique()} Folds", fontsize=13, fontweight="bold")
    plt.xlabel("Fold")
    plt.ylabel("F1 Score")
    plt.xticks(fold_f1.index)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(output_dir, "kfold_f1_variance.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_per_category_heatmap(cat_metrics: pd.DataFrame, output_dir: str):
    """Heatmap of P@K, F1@K per category — makes it easy to spot weak areas."""

    cols = [c for c in cat_metrics.columns if c.startswith("P@") or c.startswith("F1@")]
    if not cols:
        return

    plot_df = cat_metrics[cols].dropna(how="all")
    if plot_df.empty:
        return

    plt.figure(figsize=(12, max(6, len(plot_df) * 0.4)))
    sns.heatmap(
        plot_df,
        annot=True, fmt=".2f",
        cmap="YlGnBu", linewidths=0.5,
        vmin=0, vmax=1,
    )
    plt.title("Per-Category Precision & F1 at K", fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, "per_category_heatmap.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ==============================
# MAIN
# ==============================

def main():

    print("=" * 60)
    print("  PHASE 4 — MATCHING EVALUATION")
    print("  Strategy: Category-based, K-Fold (5 folds)")
    print(f"  K values : {K_VALUES}")
    print("=" * 60)

    # 1. Load
    labeled, index, job_meta, model = load_data()

    # 2. K-Fold evaluation
    results_df = run_kfold_evaluation(labeled, model, index, job_meta)
    print(f"\n  Total evaluated resumes: {len(results_df):,}")

    # 3. Overall summary
    print("\n--- Overall Metrics Summary ---")
    summary_rows = []
    metric_cols  = ["MRR"] + [f"{m}@{k}" for m in ["P", "R", "F1"] for k in K_VALUES]

    for col in metric_cols:
        if col in results_df.columns:
            mean_val = results_df[col].mean()
            std_val  = results_df[col].std()
            print(f"  {col:12s} : {mean_val:.4f} ± {std_val:.4f}")
            summary_rows.append({"metric": col, "mean": round(mean_val, 4), "std": round(std_val, 4)})

    summary_df   = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Saved: {summary_path}")

    # 4. Per-category metrics
    group_cols  = [f"P@{k}" for k in K_VALUES] + [f"R@{k}" for k in K_VALUES] + \
                  [f"F1@{k}" for k in K_VALUES] + ["MRR"]
    cat_metrics = results_df.groupby("category")[group_cols].mean().round(4)

    cat_path = os.path.join(OUTPUT_DIR, "per_category_metrics.csv")
    cat_metrics.to_csv(cat_path)
    print(f"  Saved: {cat_path}")

    # 5. K-Fold raw results
    kfold_path = os.path.join(OUTPUT_DIR, "kfold_results.csv")
    results_df.to_csv(kfold_path, index=False)
    print(f"  Saved: {kfold_path}")

    # 6. AUC-ROC
    auc_score, y_true, y_score = compute_auc_roc(labeled, model, index, job_meta)
    summary_df = pd.concat(
        [summary_df, pd.DataFrame([{"metric": "AUC-ROC", "mean": round(auc_score, 4), "std": 0.0}])],
        ignore_index=True,
    )
    summary_df.to_csv(summary_path, index=False)

    # 7. Plots
    print("\nGenerating plots...")
    plot_score_distributions(results_df, y_true, y_score, OUTPUT_DIR)
    plot_precision_recall_at_k(results_df, K_VALUES, OUTPUT_DIR)
    plot_kfold_variance(results_df, K_VALUES, OUTPUT_DIR)
    plot_per_category_heatmap(cat_metrics, OUTPUT_DIR)

    print(f"\n{'='*60}")
    print(f"  Evaluation complete!")
    print(f"  MRR          : {results_df['MRR'].mean():.4f}")
    print(f"  P@10         : {results_df['P@10'].mean():.4f}")
    print(f"  R@10         : {results_df['R@10'].mean():.4f}")
    print(f"  F1@10        : {results_df['F1@10'].mean():.4f}")
    print(f"  AUC-ROC      : {auc_score:.4f}")
    print(f"  Output dir   : {OUTPUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
