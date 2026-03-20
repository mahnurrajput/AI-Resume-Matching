import os
import re
import ast
import pandas as pd
import pdfplumber
from docx import Document
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================

RAW_DIR        = "data/resumes_raw"
CSV_PATH       = "data/resumes_csv/resumes.csv"
SECONDARY_PATH = "data/resumes_secondary/resume_data.csv"

OUTPUT_FILE    = "data_processed/resumes_cleaned.csv"

MIN_WORDS = 50
MAX_WORDS = 5000


# ==============================
# RESUME BOILERPLATE PATTERNS
# ==============================
# Structural resume phrases with zero semantic value.
# Safe to remove regardless of which AI model is used.

BOILERPLATE_PATTERNS = [
    r'\bcurriculum\s+vitae\b',
    r'\breferences?\s+(available\s+)?(upon\s+request)?\b',
    r'\bpage\s+\d+\s+(of\s+\d+)?\b',
    r'\bconfidential\b',
    r'\bpersonal\s+information\b',
    r'\bdate\s+of\s+birth\b',
    r'\bnationality\b',
    r'\bmarital\s+status\b',
    r'\bgender\b',
    r'\bdob\b',
    r'\bfax\b',
]


# ==============================
# TEXT EXTRACTION — DATASET 1
# ==============================

def extract_pdf(path):
    text = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text.append(t)
    except Exception:
        return ""
    return "\n".join(text)


def extract_docx(path):
    text = []
    try:
        doc = Document(path)
        for para in doc.paragraphs:
            if para.text:
                text.append(para.text)
    except Exception:
        return ""
    return "\n".join(text)


# ==============================
# CLEANING
# ==============================

def clean_text(text):
    """
    Cleans resume text for use with Sentence-BERT and semantic matching.

    Only removes actual noise — keeps natural language intact
    because BERT relies on context and sentence structure.

    Steps:
      1. Type guard
      2. Lowercase
      3. Remove emails
      4. Remove URLs
      5. Remove phone numbers
      6. Remove boilerplate resume phrases
      7. Remove non-ASCII characters
      8. Remove punctuation
      9. Normalize whitespace
    """

    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove emails
    text = re.sub(r'\S+@\S+', ' ', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)

    # Remove phone numbers
    text = re.sub(r'\+?\d[\d\s\-]{8,}', ' ', text)

    # Remove boilerplate resume phrases
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, ' ', text)

    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Remove punctuation (keep letters, digits, spaces)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ==============================
# DATASET 3 HELPER — LIST PARSER
# ==============================

def parse_list_field(value):
    """
    Dataset 3 stores many fields as string representations of Python lists,
    e.g. "['Big Data', 'Hadoop', 'Python']" or "[None]" or "['N/A']".

    This function:
      - Safely parses the string into an actual list using ast.literal_eval
      - Filters out None, 'N/A', empty strings
      - Joins remaining items into a single plain text string

    If the value is not a list-like string, returns it as plain text directly.
    """

    if not isinstance(value, str) or value.strip() == "":
        return ""

    # Try to parse as a Python literal (list, nested list, etc.)
    try:
        parsed = ast.literal_eval(value)

        # Flatten nested lists (e.g. [['Big Data'], ['Python']])
        if isinstance(parsed, list):
            flat = []
            for item in parsed:
                if isinstance(item, list):
                    flat.extend(item)
                else:
                    flat.append(item)

            # Filter out None, 'N/A', empty strings, keep only real text
            cleaned_items = [
                str(item).strip()
                for item in flat
                if item is not None
                and str(item).strip() not in ("", "N/A", "None", "nan")
            ]
            return " ".join(cleaned_items)

    except (ValueError, SyntaxError):
        # Not a list string — treat as plain text
        pass

    # Plain text fallback — strip N/A type values
    stripped = value.strip()
    if stripped in ("N/A", "None", "nan", "[]", "[None]", "['N/A']"):
        return ""

    return stripped


def build_resume_text_from_row(row):
    """
    Reconstructs a single resume text string from Dataset 3's structured columns.

    Only uses the resume/candidate side columns — ignores the job requirement
    columns (job_position_name, skills_required, educational_requirements,
    experience_requirement, responsibilities.1) and matched_score.

    Columns used (in order of semantic importance):
      1. career_objective    — candidate's own summary (most important)
      2. skills              — technical and soft skills list
      3. positions           — job titles held
      4. responsibilities    — work experience descriptions
      5. related_skils_in_job — skills used in each job role
      6. certification_skills — skills from certifications
      7. major_field_of_studies — education field
      8. degree_names        — degrees obtained
    """

    resume_columns = [
        "career_objective",
        "skills",
        "positions",
        "responsibilities",
        "related_skils_in_job",
        "certification_skills",
        "major_field_of_studies",
        "degree_names",
    ]

    parts = []
    for col in resume_columns:
        raw_value = row.get(col, "")
        parsed = parse_list_field(str(raw_value) if pd.notna(raw_value) else "")
        if parsed:
            parts.append(parsed)

    return " ".join(parts)


# ==============================
# DATASET 1 PROCESSING (Raw Files)
# ==============================

def process_raw_files():
    records = []
    files = os.listdir(RAW_DIR)

    for file in tqdm(files, desc="Dataset 1 (Raw Files)"):
        path = os.path.join(RAW_DIR, file)
        text = ""

        if file.lower().endswith(".pdf"):
            text = extract_pdf(path)
        elif file.lower().endswith(".docx"):
            text = extract_docx(path)

        if not text:
            continue

        cleaned = clean_text(text)
        wc = len(cleaned.split())

        if wc < MIN_WORDS:
            continue

        if wc > MAX_WORDS:
            cleaned = " ".join(cleaned.split()[:MAX_WORDS])
            wc = MAX_WORDS

        records.append({
            "source":       "dataset1",
            "file_name":    file,
            "category":     None,
            "cleaned_text": cleaned,
            "word_count":   wc,
        })

    return records


# ==============================
# DATASET 2 PROCESSING (CSV)
# ==============================

def process_csv_dataset():
    df = pd.read_csv(CSV_PATH)
    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Dataset 2 (CSV)"):
        text = row.get("Resume_str", "")

        if not isinstance(text, str) or len(text.strip()) == 0:
            continue

        cleaned = clean_text(text)
        wc = len(cleaned.split())

        if wc < MIN_WORDS:
            continue

        if wc > MAX_WORDS:
            cleaned = " ".join(cleaned.split()[:MAX_WORDS])
            wc = MAX_WORDS

        records.append({
            "source":       "dataset2",
            "file_name":    row.get("ID"),
            "category":     row.get("Category"),
            "cleaned_text": cleaned,
            "word_count":   wc,
        })

    return records


# ==============================
# DATASET 3 PROCESSING (Secondary CSV)
# ==============================

def process_secondary_dataset():
    # encoding="utf-8-sig" handles the BOM character (ï»¿) at start of file
    # which corrupts the first column name if not handled
    df = pd.read_csv(SECONDARY_PATH, encoding="utf-8-sig", low_memory=False)

    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Dataset 3 (Secondary CSV)"):

        # Reconstruct resume text from structured columns
        raw_text = build_resume_text_from_row(row)

        if not raw_text:
            continue

        cleaned = clean_text(raw_text)
        wc = len(cleaned.split())

        if wc < MIN_WORDS:
            continue

        if wc > MAX_WORDS:
            cleaned = " ".join(cleaned.split()[:MAX_WORDS])
            wc = MAX_WORDS

        records.append({
            "source":       "dataset3",
            "file_name":    None,
            "category":     None,
            "cleaned_text": cleaned,
            "word_count":   wc,
        })

    return records


# ==============================
# MAIN
# ==============================

def main():
    data1 = process_raw_files()
    data2 = process_csv_dataset()
    data3 = process_secondary_dataset()

    all_data = data1 + data2 + data3
    df = pd.DataFrame(all_data)

    # Remove exact duplicates across all three datasets
    df = df.drop_duplicates(subset=["cleaned_text"])

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{'='*50}")
    print(f"  Dataset 1 (Raw files) : {len(data1)}")
    print(f"  Dataset 2 (CSV)       : {len(data2)}")
    print(f"  Dataset 3 (Secondary) : {len(data3)}")
    print(f"  Total before dedup    : {len(all_data)}")
    print(f"  Total after dedup     : {len(df)}")
    print(f"  Output saved to       : {OUTPUT_FILE}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()


