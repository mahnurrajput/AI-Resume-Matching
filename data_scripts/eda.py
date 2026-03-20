# This script performs Exploratory Data Analysis (EDA) on the cleaned resume
# and job datasets — generating charts and stats to understand the data better.

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter

# ==============================
# CONFIG
# ==============================

RESUME_FILE  = "data_processed/resumes_cleaned.csv"
JOBS_FILE    = "data_processed/jobs_cleaned.csv"

# All EDA plots will be saved here
OUTPUT_DIR   = "outputs/eda"

# How many top items to show in bar charts
TOP_N = 15


# ==============================
# SETUP
# ==============================

# Create output folder if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use a clean visual style for all plots
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120


# ==============================
# HELPER FUNCTIONS
# ==============================

def save_plot(filename):
    """Save the current matplotlib figure to the output folder."""
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def extract_top_keywords(text_series, top_n=30):
    """
    Extract the most common meaningful words from a text column.
    Removes very short words and common stopwords that add no insight.
    """

    # Basic stopwords — just enough to clean noise without hurting meaning
    STOPWORDS = {
        "the", "and", "to", "of", "in", "a", "for", "with", "is", "on",
        "are", "be", "as", "an", "at", "by", "or", "this", "that", "have",
        "will", "from", "we", "our", "you", "your", "their", "they", "it",
        "its", "not", "but", "all", "was", "has", "can", "which", "more",
        "about", "up", "also", "been", "other", "new", "into", "than",
        "s", "e", "c", "i"
    }

    all_words = []

    for text in text_series.dropna():
        words = str(text).lower().split()
        for word in words:
            # Keep words longer than 2 characters that are not stopwords
            if len(word) > 2 and word not in STOPWORDS:
                all_words.append(word)

    counter = Counter(all_words)
    return counter.most_common(top_n)


# ==============================
# LOAD DATA
# ==============================

print("\nLoading data...")
resumes = pd.read_csv(RESUME_FILE)
jobs    = pd.read_csv(JOBS_FILE)

print(f"  Resumes : {len(resumes):,} rows")
print(f"  Jobs    : {len(jobs):,} rows")


# ==============================
# SECTION 1 — BASIC STATS
# ==============================

print("\n--- Resume Dataset Info ---")
print(resumes.info())
print("\nMissing values in resumes:")
print(resumes.isnull().sum())

print("\n--- Jobs Dataset Info ---")
print(jobs.info())
print("\nMissing values in jobs:")
print(jobs.isnull().sum())


# ==============================
# SECTION 2 — WORD COUNT DISTRIBUTIONS
# ==============================

print("\nGenerating word count distribution plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Word Count Distributions", fontsize=14, fontweight="bold")

# Resume word count
axes[0].hist(resumes["word_count"].dropna(), bins=40, color="#4C72B0", edgecolor="white")
axes[0].set_title("Resumes — Word Count")
axes[0].set_xlabel("Word Count")
axes[0].set_ylabel("Number of Resumes")

# Job word count
axes[1].hist(jobs["word_count"].dropna(), bins=40, color="#DD8452", edgecolor="white")
axes[1].set_title("Jobs — Word Count")
axes[1].set_xlabel("Word Count")
axes[1].set_ylabel("Number of Jobs")

plt.tight_layout()
save_plot("01_word_count_distributions.png")


# ==============================
# SECTION 3 — RESUME SOURCE DISTRIBUTION
# ==============================

print("Generating resume source distribution...")

source_counts = resumes["source"].value_counts()

plt.figure(figsize=(7, 4))
bars = plt.bar(source_counts.index, source_counts.values, color=["#4C72B0", "#DD8452", "#55A868"])
plt.title("Resume Source Distribution", fontsize=13, fontweight="bold")
plt.xlabel("Source")
plt.ylabel("Count")

# Add count labels on top of each bar
for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 5,
        str(int(bar.get_height())),
        ha="center", fontsize=10
    )

plt.tight_layout()
save_plot("02_resume_source_distribution.png")


# ==============================
# SECTION 4 — RESUME CATEGORY DISTRIBUTION
# ==============================

print("Generating resume category distribution...")

# Only dataset2 has meaningful category labels
cat_df = resumes[resumes["category"].notna() & (resumes["category"] != "NaN")]

if not cat_df.empty:
    cat_counts = cat_df["category"].value_counts().head(TOP_N)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=cat_counts.values, y=cat_counts.index, palette="Blues_r")
    plt.title(f"Top {TOP_N} Resume Categories (Dataset 2)", fontsize=13, fontweight="bold")
    plt.xlabel("Count")
    plt.ylabel("Category")
    plt.tight_layout()
    save_plot("03_resume_category_distribution.png")
else:
    print("  No category data found — skipping category chart.")


# ==============================
# SECTION 5 — TOP JOB TITLES
# ==============================

print("Generating top job titles chart...")

title_counts = jobs["title"].dropna().value_counts().head(TOP_N)

plt.figure(figsize=(10, 6))
sns.barplot(x=title_counts.values, y=title_counts.index, palette="Oranges_r")
plt.title(f"Top {TOP_N} Job Titles", fontsize=13, fontweight="bold")
plt.xlabel("Count")
plt.ylabel("Job Title")
plt.tight_layout()
save_plot("04_top_job_titles.png")


# ==============================
# SECTION 6 — JOB WORK TYPE DISTRIBUTION
# ==============================

print("Generating work type distribution...")

work_counts = jobs["work_type"].dropna()
work_counts = work_counts[work_counts != "nan"].value_counts()

if not work_counts.empty:
    plt.figure(figsize=(7, 4))
    bars = plt.bar(work_counts.index, work_counts.values, color="#4C72B0", edgecolor="white")
    plt.title("Job Work Type Distribution", fontsize=13, fontweight="bold")
    plt.xlabel("Work Type")
    plt.ylabel("Count")
    plt.xticks(rotation=20)

    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            str(int(bar.get_height())),
            ha="center", fontsize=9
        )

    plt.tight_layout()
    save_plot("05_work_type_distribution.png")


# ==============================
# SECTION 7 — JOB EXPERIENCE LEVEL DISTRIBUTION
# ==============================

print("Generating experience level distribution...")

exp_counts = jobs["experience_level"].dropna()
exp_counts = exp_counts[exp_counts != "nan"].value_counts()

if not exp_counts.empty:
    plt.figure(figsize=(8, 4))
    sns.barplot(x=exp_counts.index, y=exp_counts.values, palette="Set2")
    plt.title("Job Experience Level Distribution", fontsize=13, fontweight="bold")
    plt.xlabel("Experience Level")
    plt.ylabel("Count")
    plt.xticks(rotation=20)
    plt.tight_layout()
    save_plot("06_experience_level_distribution.png")


# ==============================
# SECTION 8 — SALARY DISTRIBUTION
# ==============================

print("Generating salary distribution...")

# Only rows where salary data actually exists
salary_df = jobs[jobs["salary_min"].notna() & jobs["salary_max"].notna()].copy()

if len(salary_df) > 100:
    # Add a mid-salary column for easier analysis
    salary_df["salary_mid"] = (salary_df["salary_min"] + salary_df["salary_max"]) / 2

    # Remove extreme outliers (keep salaries between 1k and 500k)
    salary_df = salary_df[
        (salary_df["salary_mid"] > 1000) &
        (salary_df["salary_mid"] < 500000)
    ]

    plt.figure(figsize=(9, 4))
    plt.hist(salary_df["salary_mid"], bins=50, color="#55A868", edgecolor="white")
    plt.title("Job Salary Distribution (Mid-point)", fontsize=13, fontweight="bold")
    plt.xlabel("Salary (USD)")
    plt.ylabel("Count")
    plt.gca().xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${int(x):,}")
    )
    plt.tight_layout()
    save_plot("07_salary_distribution.png")
else:
    print("  Not enough salary data — skipping salary chart.")


# ==============================
# SECTION 9 — TOP JOB LOCATIONS
# ==============================

print("Generating top job locations chart...")

loc_counts = jobs["location"].dropna()
loc_counts = loc_counts[loc_counts != "nan"].value_counts().head(TOP_N)

if not loc_counts.empty:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=loc_counts.values, y=loc_counts.index, palette="Purples_r")
    plt.title(f"Top {TOP_N} Job Locations", fontsize=13, fontweight="bold")
    plt.xlabel("Count")
    plt.ylabel("Location")
    plt.tight_layout()
    save_plot("08_top_job_locations.png")


# ==============================
# SECTION 10 — TOP KEYWORDS IN RESUMES
# ==============================

print("Generating top resume keywords chart...")

resume_keywords = extract_top_keywords(resumes["cleaned_text"], top_n=TOP_N)
kw_words  = [item[0] for item in resume_keywords]
kw_counts = [item[1] for item in resume_keywords]

plt.figure(figsize=(10, 6))
sns.barplot(x=kw_counts, y=kw_words, palette="Blues_r")
plt.title(f"Top {TOP_N} Keywords in Resumes", fontsize=13, fontweight="bold")
plt.xlabel("Frequency")
plt.ylabel("Keyword")
plt.tight_layout()
save_plot("09_top_resume_keywords.png")


# ==============================
# SECTION 11 — TOP KEYWORDS IN JOB DESCRIPTIONS
# ==============================

print("Generating top job keywords chart...")

job_keywords = extract_top_keywords(jobs["job_text"], top_n=TOP_N)
jk_words  = [item[0] for item in job_keywords]
jk_counts = [item[1] for item in job_keywords]

plt.figure(figsize=(10, 6))
sns.barplot(x=jk_counts, y=jk_words, palette="Oranges_r")
plt.title(f"Top {TOP_N} Keywords in Job Descriptions", fontsize=13, fontweight="bold")
plt.xlabel("Frequency")
plt.ylabel("Keyword")
plt.tight_layout()
save_plot("10_top_job_keywords.png")


# ==============================
# SECTION 12 — REMOTE JOBS
# ==============================

print("Generating remote job distribution...")

remote_counts = jobs["remote_allowed"].dropna().value_counts()

if not remote_counts.empty:
    labels = {1.0: "Remote Allowed", 0.0: "Not Remote"}
    remote_counts.index = [labels.get(i, str(i)) for i in remote_counts.index]

    plt.figure(figsize=(5, 5))
    plt.pie(
        remote_counts.values,
        labels=remote_counts.index,
        autopct="%1.1f%%",
        colors=["#55A868", "#C44E52"],
        startangle=90
    )
    plt.title("Remote vs Non-Remote Jobs", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_plot("11_remote_job_distribution.png")


# ==============================
# DONE
# ==============================

print(f"\n{'='*50}")
print(f"  EDA complete!")
print(f"  All charts saved to: {OUTPUT_DIR}/")
print(f"  Total charts generated: 11")
print(f"{'='*50}")