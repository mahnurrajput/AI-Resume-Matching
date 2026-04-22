# 🤖 AI Resume–Job Matching System

> An AI-powered career assistant that matches your resume with the most relevant job postings
> and identifies your skill gaps — built with Sentence-BERT and FAISS over 124,000+ real LinkedIn jobs.


---

## 🎯 What This System Does

This system works like a **personal career assistant**. You upload your resume, and the system:

1. Reads and cleans your resume text
2. Compares it semantically against 124,000+ real job descriptions
3. Returns the **top matching jobs** with similarity scores
4. Shows you exactly which **skills you are missing** for those jobs

**Example output:**

```
Top Job Matches
───────────────────────────────────
1.  Data Analyst              — 84%
2.  Business Intelligence     — 79%
3.  Financial Data Analyst    — 74%

```

---

## ⚙️ How It Works

The system follows this pipeline:

```
Resume Upload (PDF / DOCX)
         │
         ▼
  Text Extraction
  (pdfplumber / python-docx)
         │
         ▼
  Text Cleaning
  (lowercase, remove noise, normalize)
         │
         ▼
  Sentence-BERT Embedding
  (converts text into a 384-dim vector)
         │
         ▼
  FAISS Index Search
  (fast nearest-neighbor search over job vectors)
         │
         ▼
  Cosine Similarity Ranking
         │
         ▼
  Top Job Matches  +  Skill Gap Analysis
         │
         ▼
  Streamlit UI Output
```

**Why Sentence-BERT instead of TF-IDF?**

TF-IDF only matches exact keywords. Sentence-BERT understands *meaning* — so a resume
saying "built data pipelines" will still match a job asking for "ETL experience", even
though those words are completely different.

---

## 🛠 Tech Stack

| Layer             | Tools                                      |
|-------------------|--------------------------------------------|
| Core AI Model     | Sentence-BERT (MiniLM-L6-v2)              |
| Vector Search     | FAISS (Facebook AI Similarity Search)      |
| Similarity Metric | Cosine Similarity                          |
| NLP Processing    | spaCy, Python                              |
| Data Handling     | pandas, numpy, scikit-learn                |
| Visualization     | matplotlib, seaborn                        |
| User Interface    | Streamlit                                  |
| Backend (optional)| FastAPI + uvicorn                         |
| File Parsing      | pdfplumber, python-docx                    |
| Dev Tools         | Python 3.10+, VS Code, Git + GitHub        |
| Storage           | Local CSV / JSON (no database needed)      |

---

## 📁 Project Structure

```
AI-Resume-Matching/
│
├── data/                          # Raw datasets — NOT uploaded to GitHub (too large)
│   ├── resumes_raw/               # 228 real PDF/DOCX resume files
│   ├── resumes_csv/               # resumes.csv — 2,484 resume entries
│   ├── resumes_secondary/         # resume_data.csv — 9,500+ structured resume entries
│   └── linkedin_dataset/          # LinkedIn job postings (~124k jobs)
│       ├── postings.csv           # Main job listings file
│       └── companies/
│           └── companies.csv      # Company name lookup table
│
├── data_processed/                # Cleaned output files — NOT uploaded to GitHub
│   ├── resumes_cleaned.csv        # 2,709 cleaned resumes (output of resume_pipeline.py)
│   └── jobs_cleaned.csv           # Cleaned job postings (output of job_pipeline.py)
│
├── src/                           # All source code lives here
│   ├── resume_pipeline.py         # Extracts and cleans resume text from all 3 datasets
│   ├── job_pipeline.py            # Cleans and prepares LinkedIn job descriptions
│   ├── eda.py                     # Generates EDA charts and statistics
│   ├── embedder.py                # (Upcoming) Generates Sentence-BERT embeddings
│   ├── faiss_index.py             # (Upcoming) Builds and saves the FAISS index
│   ├── similarity.py              # (Upcoming) Matches resume vector against job vectors
│   └── skill_gap.py               # (Upcoming) Extracts and compares skills
│
├── app/                           # Streamlit web interface
│   └── app.py                     # (Upcoming) Main app file
│
├── models/                        # Saved embeddings and FAISS index — NOT uploaded to GitHub
│   ├── job_embeddings.npy         # (Upcoming) NumPy array of job vectors
│   └── faiss_index.bin            # (Upcoming) FAISS index file
│
├── outputs/                       # Generated outputs — NOT uploaded to GitHub
│   └── eda/                       # EDA charts saved here (11 PNG files)
│
├── .gitignore                     # Prevents large files from being pushed
├── requirements.txt               # All Python dependencies with versions
└── README.md                      # This file
```

Link to folders and files > 100MB:
https://drive.google.com/drive/folders/1ZxS0OCNjpIsh6QPBzP_LSdf_rqNh6lZn?usp=sharing

---

## 📊 Dataset Summary

### Resume Datasets

| Dataset   | Format      | Size           | Role in System                      |
|-----------|-------------|----------------|-------------------------------------|
| Dataset 1 | PDF / DOCX  | 228 files      | Real-world resume parsing tests     |
| Dataset 2 | CSV         | 2,484 entries  | Main scalable resume text source    |
| Dataset 3 | CSV         | 9,500+ entries | Validation and skill analysis       |

**Combined output:** 2,709 cleaned resumes after deduplication

### Job Datasets

| Dataset       | Format | Size           | Role in System                      |
|---------------|--------|----------------|-------------------------------------|
| LinkedIn Jobs | CSV    | ~124,000 jobs  | PRIMARY — main matching database    |
| Dataset 6     | CSV    | ~358 MB        | Secondary — reserved for future use |
| Analyst Jobs  | CSV    | ~2,000 entries | Small experiments only              |

**Why LinkedIn dataset?** It contains real recruiter-written job descriptions with rich
fields: `title`, `description`, `skills_desc`, `experience_level`, `salary`, `location`.
This gives the BERT model realistic language to work with.

---

## 🚀 Setup Instructions

### Step 1 — Clone the repository

```bash
git clone https://github.com/mahnurrajput/AI-Resume-Matching.git
cd AI-Resume-Matching
```

### Step 2 — Create and activate virtual environment

```bash
python -m venv venv
```

```bash
# Windows (PowerShell)
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

> ⚠️ **Windows PowerShell fix** — if activation is blocked, run this once:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### Step 3 — Install all dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Download the spaCy language model

```bash
python -m spacy download en_core_web_sm
```

### Step 5 — Add the datasets

Datasets are not included in this repository due to their large size.
Place them in the correct folders manually:

```
data/resumes_raw/          ← paste raw PDF/DOCX resume files here
data/resumes_csv/          ← paste resumes.csv here
data/resumes_secondary/    ← paste resume_data.csv here
data/linkedin_dataset/     ← paste postings.csv and companies/companies.csv here
```

---

## ▶️ Running the Project

Always activate your virtual environment first:

```bash
venv\Scripts\activate
```

Then run each step in order:

```bash
# Step 1 — Clean and process all resume datasets
python src/resume_pipeline.py

# Step 2 — Clean and process LinkedIn job descriptions
python src/job_pipeline.py

# Step 3 — Generate EDA charts and statistics
python src/eda.py

# Step 4 — Generate Sentence-BERT embeddings  [upcoming]
python src/embedder.py

# Step 5 — Build the FAISS search index  [upcoming]
python src/faiss_index.py

# Step 6 — Launch the Streamlit app  [upcoming]
streamlit run app/app.py
```

---

## 🧠 Key Design Decisions

### Why light text cleaning only?

The pipeline does NOT remove stopwords or lemmatize text — even though many NLP tutorials
recommend this. Here is why:

Sentence-BERT was trained on full natural English sentences. It understands meaning through
the relationships between all words in a sentence, including small words like "with", "for",
"in". Removing these breaks the sentence structure that BERT relies on.

Stopword removal and lemmatization are only appropriate for **TF-IDF** based systems.
Since this project uses **Sentence-BERT**, preserving natural language gives better results.

### Why FAISS instead of brute-force search?

With 124,000 job vectors, comparing a resume against every single job one by one would be
slow. FAISS builds an optimized index that finds the closest matches in milliseconds, even
at this scale.

### Why combine title + description + skills_desc?

Each field alone is incomplete. The title is too short. The description has context but no
skill list. The skills_desc has keywords but no context. Combining all three gives the
embedding model the richest possible representation of each job.

---

## ✅ Project Progress

| Phase | Task                                          | Status      |
|-------|-----------------------------------------------|-------------|
| 1     | Problem definition and system design          | ✅ Done      |
| 2     | Resume data pipeline (extraction + cleaning)  | ✅ Done      |
| 2     | Job data pipeline (LinkedIn cleaning)         | ✅ Done      |
| 2     | EDA — charts and statistics                   | ✅ Done      |
| 3     | Sentence-BERT embedding generation            | ⏳ Upcoming  |
| 4     | FAISS index building                          | ⏳ Upcoming  |
| 4     | Cosine similarity matching engine             | ⏳ Upcoming  |
| 5     | Skill gap analysis                            | ⏳ Upcoming  |
| 6     | Streamlit UI                                  | ⏳ Upcoming  |

---

## 👥 Team

Built as an AI course project.

| Members        | 
|--------------- |
| Fatima Riaz    |
| Mahnoor Naveed |

---

