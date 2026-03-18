import os
import re
import pandas as pd
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================

# LinkedIn dataset
LINKEDIN_DIR          = "data/linkedin_dataset"
LINKEDIN_JOBS_FILE    = os.path.join(LINKEDIN_DIR, "postings.csv")
LINKEDIN_COMPANY_FILE = os.path.join(LINKEDIN_DIR, "companies/companies.csv")

# Output
OUTPUT_FILE = "data_processed/jobs_cleaned.csv"

MIN_WORDS = 30
MAX_WORDS = 5000


# ==============================
# BOILERPLATE PATTERNS
# ==============================

BOILERPLATE_PATTERNS = [
    r'\bequal opportunity employer\b',
    r'\beo[e]?\b',
    r'\bwe are an equal\b',
    r'\bdiversity and inclusion\b',
    r'\bclick (here )?to apply\b',
    r'\bapply now\b',
    r'\bapply (at|on|via|through)\b',
    r'\bjob (id|ref|reference|code)\s*[:\-]?\s*\w*',
    r'\bposted (on|by|at)\b',
    r'\bjob posting\b',
    r'\bfull[- ]time\b',
    r'\bpart[- ]time\b',
    r'\bremote\b',
    r'\bhybrid\b',
    r'\bon[- ]site\b',
    r'\bsalary\s*[:\-]?\s*[\$\d,k\.]+',
    r'\b\$[\d,]+(\.\d+)?\s*(per\s+\w+)?\b',
    r'\bconfidential\b',
]


# ==============================
# CLEANING
# ==============================

def clean_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""

    text = text.lower()
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\+?\d[\d\s\-]{8,}', ' ', text)

    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, ' ', text)

    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def combine_and_clean(parts):
    combined = " ".join([str(p) for p in parts if pd.notna(p) and str(p).strip() != ""])
    return clean_text(combined)


# ==============================
# LINKEDIN DATASET PROCESSING
# ==============================

def process_linkedin():
    print("\nLoading LinkedIn dataset...")

    jobs = pd.read_csv(LINKEDIN_JOBS_FILE, low_memory=False)

    try:
        companies = pd.read_csv(LINKEDIN_COMPANY_FILE, low_memory=False)
        companies = companies[["company_id", "name"]].rename(columns={"name": "company"})
        jobs = jobs.merge(companies, on="company_id", how="left")
    except Exception as e:
        print(f"  Warning: Could not load companies.csv — company name will be empty. ({e})")
        jobs["company"] = None

    records = []

    for _, row in tqdm(jobs.iterrows(), total=len(jobs), desc="LinkedIn Jobs"):

        job_text = combine_and_clean([
            row.get("title"),
            row.get("description"),
            row.get("skills_desc"),
        ])

        if not job_text:
            continue

        wc = len(job_text.split())

        if wc < MIN_WORDS:
            continue

        if wc > MAX_WORDS:
            job_text = " ".join(job_text.split()[:MAX_WORDS])
            wc = MAX_WORDS

        records.append({
            "source":           "linkedin",
            "job_id":           row.get("job_id"),
            "title":            str(row.get("title", "")).strip(),
            "company":          str(row.get("company", "")).strip(),
            "location":         str(row.get("location", "")).strip(),
            "experience_level": str(row.get("formatted_experience_level", "")).strip(),
            "work_type":        str(row.get("formatted_work_type", "")).strip(),
            "remote_allowed":   row.get("remote_allowed"),
            "salary_min":       row.get("min_salary"),
            "salary_max":       row.get("max_salary"),
            "job_text":         job_text,
            "word_count":       wc,
        })

    return records


# ==============================
# MAIN
# ==============================

def main():
    records = process_linkedin()

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["job_text"])

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{'='*45}")
    print(f"  LinkedIn jobs processed  : {len(records)}")
    print(f"  Total after deduplication: {len(df)}")
    print(f"  Output saved to          : {OUTPUT_FILE}")
    print(f"{'='*45}")


if __name__ == "__main__":
    main()