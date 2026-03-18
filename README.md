# AI Resume–Job Matching System

An AI-powered system that matches a user's resume with relevant job postings using semantic similarity. Built with Sentence-BERT and FAISS for fast vector search across 124,000+ real LinkedIn job descriptions.

---

## Project Overview

This system:
- Accepts resume uploads (PDF or DOCX)
- Extracts and cleans resume text
- Converts resumes and job descriptions into semantic vector embeddings
- Computes cosine similarity between resume and job vectors
- Returns top-ranked job matches with similarity scores
- Identifies missing skills (skill gap analysis)

---

## Tech Stack

| Layer | Tools |
|---|---|
| NLP / AI | Sentence-BERT (MiniLM), FAISS, spaCy |
| Data | pandas, numpy, scikit-learn |
| Visualization | matplotlib, seaborn |
| Interface | Streamlit |
| Dev | Python 3.10+, VS Code, Git |

---

## Project Structure

```
AI-RESUME-MATCHING/
│
├── data/                    # Raw datasets (gitignored — not uploaded)
│   ├── resumes_raw/         # 228 raw PDF/DOCX resumes
│   ├── resumes_csv/         # resumes.csv (2484 entries)
│   ├── resumes_secondary/   # secondary structured resume dataset
│   └── linkedin_dataset/    # LinkedIn job postings (~124k)
│
├── data_processed/          # Cleaned output files (gitignored)
│   ├── resumes_cleaned.csv
│   └── jobs_cleaned.csv
│
├── data_scripts/            # All Phase 2 data pipeline scripts
│   ├── resume_pipeline.py   # Resume extraction and cleaning
│   ├── job_pipeline.py      # Job description cleaning
│   └── eda.py               # Exploratory data analysis
│
├── models/                  # Saved embeddings and FAISS index (gitignored)
├── outputs/                 # Generated charts and results (gitignored)
├── app/                     # Streamlit application
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/mahnurrajput/AI-Resume-Matching.git
cd AI-Resume-Matching
```

### 2. Create and activate virtual environment

```bash
python -m venv venv

# Windows (PowerShell)
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download spaCy language model

```bash
python -m spacy download en_core_web_sm
```

### 5. Add datasets

Place datasets in the correct folders under `data/` (see structure above). Datasets are not included in this repository due to size.

**Resume datasets:**
- Dataset 1: Raw PDF/DOCX resumes → `data/resumes_raw/`
- Dataset 2: resumes.csv → `data/resumes_csv/`
- Dataset 3: resume_data.csv → `data/resumes_secondary/`

**Job dataset:**
- LinkedIn job postings → `data/linkedin_dataset/`

---

## Running the Pipelines

```bash
# Step 1 — Process and clean resumes
python data_scripts/resume_pipeline.py

# Step 2 — Process and clean job descriptions
python data_scripts/job_pipeline.py

# Step 3 — Generate EDA visualizations
python data_scripts/eda.py
```

---

## Dataset Summary

| Dataset | Size | Purpose |
|---|---|---|
| Resume Dataset 1 | 228 files | Real-world PDF/DOCX parsing |
| Resume Dataset 2 | 2,484 entries | Scalable CSV text data |
| Resume Dataset 3 | 9,500+ entries | Validation and skill analysis |
| LinkedIn Jobs | ~124,000 postings | Primary job matching dataset |

---

## Current Status

- [x] Phase 1 — Problem definition and system design
- [x] Phase 2 — Data acquisition and cleaning pipelines
- [ ] Phase 3 — Embedding generation (Sentence-BERT)
- [ ] Phase 4 — FAISS index and similarity engine
- [ ] Phase 5 — Skill gap analysis
- [ ] Phase 6 — Streamlit UI

---

## Team

Built as an AI course project.
