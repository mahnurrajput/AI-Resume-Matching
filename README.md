# 🤖 AI Resume–Job Matching System

> An AI-powered career assistant that matches your resume with the most relevant job postings
> and identifies your skill gaps — built with Sentence-BERT and FAISS over 108,702 real LinkedIn jobs.

🔗 **Live App:** [resumeiq-rjm.streamlit.app](https://resumeiq-rjm.streamlit.app)
📁 **Repo:** [github.com/mahnurrajput/AI-Resume-Matching](https://github.com/mahnurrajput/AI-Resume-Matching)
📦 **Hugging Face Dataset:** [huggingface.co/datasets/mahnoor-24r/ai-resume-matching-models](https://huggingface.co/datasets/mahnoor-24r/ai-resume-matching-models)


---

## 🎯 What This System Does

Upload your resume and the system:

1. Reads and cleans your resume text (PDF, DOCX, or plain text)
2. Compares it semantically against 108,702 real LinkedIn job descriptions
3. Returns the **top matching jobs** with similarity scores
4. Performs a **two-stage skill gap analysis** — structured extraction + AI reasoning

---

## ⚙️ How It Works

```
Resume Upload (PDF / DOCX / TXT)
         │
         ▼
  Text Extraction + Cleaning
         │
         ▼
  Sentence-BERT Embedding  (384-dim vector)
         │
         ▼
  FAISS Index Search  (~17ms over 108,702 jobs)
         │
         ▼
  Cosine Similarity Ranking
         │
         ▼
  Skill Gap Analysis (Stage 1: spaCy + Taxonomy + SBERT)
         │
         ▼
  AI Reasoning (Stage 2: Gemini API)
         │
         ▼
  Streamlit UI — ranked matches + skill gap panels
```

---

## 🛠 Tech Stack

| Layer             | Tools                                               |
|-------------------|-----------------------------------------------------|
| Core AI Model     | Sentence-BERT (all-MiniLM-L6-v2, 384-dim)          |
| Vector Search     | FAISS IndexFlatIP (exact cosine similarity)         |
| AI Reasoning      | Google Gemini API (gemini-2.5-flash)                |
| NLP / NER         | spaCy (en_core_web_sm)                              |
| Skill Taxonomy    | Custom 400+ skills across 9 categories              |
| Data Handling     | pandas, numpy, scikit-learn                         |
| Visualization     | matplotlib, seaborn                                 |
| File Parsing      | pdfplumber, python-docx                             |
| User Interface    | Streamlit                                           |
| Deployment        | Streamlit Cloud (free tier)                         |
| Model Storage     | Hugging Face Hub (Dataset repo)                     |
| Language          | Python 3.11                                         |

---

## 📁 Project Structure

```
AI-Resume-Matching/
├── app.py                      ← Phase 5: Streamlit web UI (main entry point)
├── download_models.py          ← Phase 5: HF Hub model download script (runs at startup)
├── requirements.txt            ← Phase 5: Production dependencies for Streamlit Cloud
├── requirements-dev.txt        ← Dev-only: pinned exact versions for local reproducibility
├── upload_to_hf.py             ← Phase 5: One-time script to upload large files to HF Hub
├── .streamlit/
│   └── config.toml             ← Phase 5: Streamlit server config (port 8501, dark theme)
├── models/
│   ├── matching_engine.py      ← Phase 3 Step 3: Core matching + skill gap integration
│   ├── skill_analyzer.py       ← Phase 3 Step 4: Two-stage skill gap analysis
│   ├── faiss_index.bin         ← gitignored — hosted on HF Hub
│   ├── job_metadata.csv        ← gitignored — hosted on HF Hub
│   ├── resume_metadata.csv     ← gitignored — hosted on HF Hub
│   ├── job_embeddings.npy      ← gitignored — hosted on HF Hub
│   ├── resume_embeddings.npy   ← gitignored — hosted on HF Hub
│   └── index_config.json
├── data/                       ← gitignored (raw datasets)
│   ├── dataset_secondary
│   ├── linkedin_dataset
│   ├── resumes_csv
│   ├── resumes_raw
│   └── resumes_secondary
├── data_processed/             ← gitignored (cleaned CSVs — jobs_cleaned.csv on HF Hub)
├── data_scripts/
│   ├── eda.py
│   ├── job_pipeline.py
│   └── resume_pipeline.py
├── evaluation/
│   ├── evaluate_matching.py
├── outputs/                    ← gitignored (EDA charts, evaluation plots)
│   ├── eda
│   └── evaluation
├── venv
├── .env
└── .gitignore
```

---

## 📊 Dataset Summary

### Resumes — Three Sources

| Dataset   | Format      | Size           | Notes                               |
|-----------|-------------|----------------|-------------------------------------|
| Dataset 1 | PDF / DOCX  | 228 files      | Real resume files                   |
| Dataset 2 | CSV         | 2,484 entries  | 25 labeled categories — used for evaluation |
| Dataset 3 | CSV         | 9,500+ entries | Structured fields reconstructed into text |
| **Output**| CSV         | **11,654 rows**| After deduplication and filtering   |

### Jobs — LinkedIn Only

108,702 cleaned job postings from the LinkedIn Job Postings dataset (~124k raw).
Fields: `title`, `description`, `skills_desc`, `company`, `location`, `experience_level`, `salary`, `work_type`, `remote_allowed`.

---

## 🧠 Skill Gap Analysis — Two Stages

**Stage 1 — Structured Extraction** (always runs, fast):
- spaCy NER extracts skill candidates from resume and job text
- Matched against a 400+ skill taxonomy across 9 categories
- Alias resolution (e.g. `k8s → kubernetes`) and SBERT semantic similarity for synonyms
- Outputs: matched skills, missing skills, overlap score, gap severity

**Stage 2 — AI Reasoning** (requires `GEMINI_API_KEY`):
- Sends resume, job description, and Stage 1 results to Gemini
- Returns: candidacy verdict, dealbreaker skills, compensatable gaps, learning path, time-to-ready, hiring risks

---

## 🚀 Setup Instructions

```bash
# 1. Clone
git clone https://github.com/mahnurrajput/AI-Resume-Matching.git
cd AI-Resume-Matching

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Gemini API key (optional — enables AI reasoning)
# Create a .env file with: GEMINI_API_KEY=your_key_here

# 5. Run the app (model files download automatically from HF Hub on first run)
streamlit run app.py
```

> ⚠️ **First boot** downloads ~340 MB of model files from Hugging Face Hub. Subsequent runs skip this if files exist locally.

---

## ▶️ Running the Pipeline (local reproduction)

```bash
# Process resumes
python resume_pipeline.py

# Process jobs
python job_pipeline.py

# EDA charts
python eda.py

# Generate embeddings (one-time, ~4 hours CPU)
python embedding_generator.py

# Build FAISS index
python faiss_index_builder.py

# Launch app
streamlit run app.py
```

---

## 🧪 Key Design Decisions

**Why light text cleaning only?**
SBERT relies on natural sentence structure — stopwords and word relationships carry meaning. Removing them (appropriate for TF-IDF) degrades BERT-based matching.

**Why IndexFlatIP (exact search)?**
108,702 vectors fit in RAM. Exact search has zero approximation error. Approximate methods (IVFFlat, HNSW) are only needed at 1M+ vectors.

**Why cosine similarity?**
SBERT embeddings are directional. Cosine measures angle, not magnitude — unaffected by text length. Achieved by L2-normalizing vectors before adding to the FAISS index.

**Why Hugging Face Hub for model storage?**
GitHub's 100 MB file limit excludes `faiss_index.bin` (159.2 MB) and `job_embeddings.npy` (159.2 MB). HF Hub is free, reliable, and has a clean Python download API.

---

## ✅ Project Progress

| Phase | Task                                        | Status      |
|-------|---------------------------------------------|-------------|
| 1     | Problem definition and system design        | ✅ Complete  |
| 2     | Resume + job data pipelines + EDA           | ✅ Complete  |
| 3     | SBERT embeddings + FAISS index              | ✅ Complete  |
| 3     | Matching engine + skill gap analysis        | ✅ Complete  |
| 4     | Evaluation (P@K, MRR, score separation)     | ✅ Complete  |
| 5     | Streamlit UI + cloud deployment             | ✅ Complete  |

---

## 👥 Team

| Members        |
|----------------|
| Fatima Riaz    |
| Mahnoor Naveed |
