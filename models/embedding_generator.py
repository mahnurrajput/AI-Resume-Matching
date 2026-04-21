"""
embedding_generator.py
======================
Phase 3 — Step 1: Sentence-BERT Embedding Generation

Generates dense vector embeddings for all cleaned resumes and job descriptions
using the all-MiniLM-L6-v2 Sentence-BERT model. Embeddings are saved as .npy
files and will be used to build the FAISS index in the next step.

Model: sentence-transformers/all-MiniLM-L6-v2
  - 384-dimensional output vectors
  - Optimized for semantic similarity tasks
  - Fast inference, small footprint (~80MB)
  - Trained on 1B+ sentence pairs

Output files:
  - models/resume_embeddings.npy     : shape (N_resumes, 384)
  - models/job_embeddings.npy        : shape (N_jobs, 384)
  - models/job_metadata.csv          : job_id, title, company, location, experience_level, work_type
  - models/resume_metadata.csv       : source, file_name, category, word_count
"""

import os
import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ==============================
# CONFIG
# ==============================

RESUME_FILE         = "data_processed/resumes_cleaned.csv"
JOBS_FILE           = "data_processed/jobs_cleaned.csv"

OUTPUT_DIR          = "models"

RESUME_EMBEDDINGS   = os.path.join(OUTPUT_DIR, "resume_embeddings.npy")
JOB_EMBEDDINGS      = os.path.join(OUTPUT_DIR, "job_embeddings.npy")
JOB_METADATA        = os.path.join(OUTPUT_DIR, "job_metadata.csv")
RESUME_METADATA     = os.path.join(OUTPUT_DIR, "resume_metadata.csv")

# Sentence-BERT model — MiniLM variant (fast, accurate, 384-dim output)
MODEL_NAME          = "all-MiniLM-L6-v2"

# Batch size for encoding — increase if you have more RAM/GPU
# 64 is safe for 8GB RAM; use 128–256 on machines with 16GB+
BATCH_SIZE          = 64

# Max sequence length — MiniLM supports up to 512 tokens
# 384 tokens captures ~250–300 words which covers most resume/job text
MAX_SEQ_LENGTH      = 384


# ==============================
# SETUP
# ==============================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================
# LOAD DATA
# ==============================

def load_data():
    """Load cleaned resume and job CSVs. Validate required columns exist."""

    print("\nLoading data...")

    resumes = pd.read_csv(RESUME_FILE)
    jobs    = pd.read_csv(JOBS_FILE)

    print(f"  Resumes loaded : {len(resumes):,} rows")
    print(f"  Jobs loaded    : {len(jobs):,} rows")

    # Validate required columns
    assert "cleaned_text" in resumes.columns, "resumes_cleaned.csv must have 'cleaned_text' column"
    assert "job_text"     in jobs.columns,    "jobs_cleaned.csv must have 'job_text' column"

    # Drop rows where text is missing or empty
    resumes = resumes[resumes["cleaned_text"].notna() & (resumes["cleaned_text"].str.strip() != "")]
    jobs    = jobs[jobs["job_text"].notna()            & (jobs["job_text"].str.strip() != "")]

    print(f"  Resumes after null drop : {len(resumes):,}")
    print(f"  Jobs after null drop    : {len(jobs):,}")

    return resumes, jobs


# ==============================
# LOAD MODEL
# ==============================

def load_model():
    """Load the Sentence-BERT model. Downloads on first run (~80MB)."""

    print(f"\nLoading Sentence-BERT model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = MAX_SEQ_LENGTH

    print(f"  Model loaded successfully")
    print(f"  Embedding dimension : {model.get_sentence_embedding_dimension()}")
    print(f"  Max sequence length : {model.max_seq_length} tokens")

    return model


# ==============================
# ENCODE
# ==============================

def encode_texts(model, texts, label="texts"):
    """
    Encode a list of texts into dense embedding vectors.

    Args:
        model  : SentenceTransformer model
        texts  : list of strings
        label  : name shown in progress output

    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """

    print(f"\nEncoding {len(texts):,} {label}...")
    print(f"  Batch size      : {BATCH_SIZE}")
    print(f"  Estimated time  : ~{max(1, len(texts) // 1000)} min (CPU) / faster on GPU")

    start = time.time()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,   # We normalize in FAISS builder — keep raw here
    )

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s  |  Shape: {embeddings.shape}  |  dtype: {embeddings.dtype}")

    return embeddings


# ==============================
# SAVE
# ==============================

def save_embeddings(embeddings, path, label="embeddings"):
    """Save numpy array to disk."""
    np.save(path, embeddings)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Saved {label} → {path}  ({size_mb:.1f} MB)")


def save_metadata(df, path, cols, label="metadata"):
    """Save a subset of columns from a DataFrame as metadata CSV."""

    # Only keep columns that actually exist in the dataframe
    available_cols = [c for c in cols if c in df.columns]
    df[available_cols].reset_index(drop=True).to_csv(path, index=True, index_label="embed_idx")
    print(f"  Saved {label} → {path}  ({len(df):,} rows, cols: {available_cols})")


# ==============================
# MAIN
# ==============================

def main():

    print("=" * 55)
    print("  PHASE 3 — EMBEDDING GENERATOR")
    print("  Model: Sentence-BERT (all-MiniLM-L6-v2)")
    print("=" * 55)

    # 1. Load data
    resumes, jobs = load_data()

    # 2. Load model
    model = load_model()

    # 3. Encode resumes
    resume_texts      = resumes["cleaned_text"].tolist()
    resume_embeddings = encode_texts(model, resume_texts, label="resumes")

    # 4. Encode jobs
    job_texts      = jobs["job_text"].tolist()
    job_embeddings = encode_texts(model, job_texts, label="jobs")

    # 5. Save embeddings
    print("\nSaving embeddings...")
    save_embeddings(resume_embeddings, RESUME_EMBEDDINGS, label="resume embeddings")
    save_embeddings(job_embeddings,    JOB_EMBEDDINGS,    label="job embeddings")

    # 6. Save metadata — used later for displaying match results
    print("\nSaving metadata...")
    save_metadata(
        resumes, RESUME_METADATA,
        cols=["source", "file_name", "category", "word_count"],
        label="resume metadata"
    )
    save_metadata(
        jobs, JOB_METADATA,
        cols=["job_id", "title", "company", "location", "experience_level", "work_type", "remote_allowed", "salary_min", "salary_max"],
        label="job metadata"
    )

    # 7. Sanity check — print a few shapes and sample values
    print("\n--- Sanity Check ---")
    print(f"  Resume embeddings shape : {resume_embeddings.shape}")
    print(f"  Job embeddings shape    : {job_embeddings.shape}")
    print(f"  Resume embed[0] min/max : {resume_embeddings[0].min():.4f} / {resume_embeddings[0].max():.4f}")
    print(f"  Job embed[0] min/max    : {job_embeddings[0].min():.4f} / {job_embeddings[0].max():.4f}")
    print(f"  Embedding dim           : {resume_embeddings.shape[1]}")

    print(f"\n{'='*55}")
    print(f"  Embedding generation complete!")
    print(f"  Resume embeddings : {RESUME_EMBEDDINGS}")
    print(f"  Job embeddings    : {JOB_EMBEDDINGS}")
    print(f"  Resume metadata   : {RESUME_METADATA}")
    print(f"  Job metadata      : {JOB_METADATA}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
