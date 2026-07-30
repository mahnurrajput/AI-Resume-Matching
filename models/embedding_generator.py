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
  - models/_shards/                  : intermediate per-chunk checkpoints. Safe to delete once
                                        the final .npy files above exist. ALSO delete this folder
                                        manually if you re-run resume_pipeline.py / job_pipeline.py
                                        with different data before re-running this script — the
                                        shard cache does not detect that on its own.
  - models/truncation_report.json    : how much text SBERT silently dropped per dataset

─────────────────────────────────────────────────────────────────────────────
CHANGELOG — fixes applied to the previous version of this script
─────────────────────────────────────────────────────────────────────────────
1. CHECKPOINTING       : Encoding now happens in chunks, each written to disk
                         immediately. A killed/crashed run resumes from the
                         last completed chunk instead of starting over. (Simple
                         by design: a shard is reused if it exists and has the
                         right row count. If your source data changes, delete
                         models/_shards/ before re-running — see note above.)
2. ALIGNMENT ASSERTS   : reset_index(drop=True) now happens once, immediately
                         after loading, on the exact DataFrame used to build
                         both the text list and the metadata CSV. An explicit
                         assert checks embeddings/DataFrame row-count equality
                         right where the arrays are created — not left to
                         convention or to a downstream file to catch.
3. TRUNCATION LOGGING  : Before encoding, every text is tokenized (no
                         truncation) to measure how many tokens it actually
                         has. Anything over max_seq_length is counted,
                         reported to the console, and written to
                         truncation_report.json. MAX_SEQ_LENGTH is also raised
                         to 512 (MiniLM's real max) to reduce how much is lost.
4. ERROR HANDLING      : Model loading and encoding are wrapped so failures
                         fail fast with a clear message instead of a raw
                         traceback hours into a run — and so a failure mid-way
                         still leaves completed shards on disk for resume.
5. DTYPE CONTROL       : Embeddings are explicitly cast and asserted to be
                         float32 before saving. An optional SAVE_DTYPE="float16"
                         mode is supported to roughly halve file size on disk;
                         faiss_index_builder.py already casts back to float32
                         on load, so this is safe for the downstream pipeline.
6. REAL TIME ESTIMATES : The old "len(texts)//1000 min" guess is replaced with
                         a measured rows/sec rate from the first completed
                         chunk, used to project remaining time for that run.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers is not installed. Run: pip install sentence-transformers")
    sys.exit(1)


# ==============================
# CONFIG
# ==============================

RESUME_FILE         = "data_processed/resumes_cleaned.csv"
JOBS_FILE            = "data_processed/jobs_cleaned.csv"

OUTPUT_DIR           = "models"
SHARD_DIR            = os.path.join(OUTPUT_DIR, "_shards")

RESUME_EMBEDDINGS    = os.path.join(OUTPUT_DIR, "resume_embeddings.npy")
JOB_EMBEDDINGS       = os.path.join(OUTPUT_DIR, "job_embeddings.npy")
JOB_METADATA         = os.path.join(OUTPUT_DIR, "job_metadata.csv")
RESUME_METADATA      = os.path.join(OUTPUT_DIR, "resume_metadata.csv")
TRUNCATION_REPORT    = os.path.join(OUTPUT_DIR, "truncation_report.json")

# Sentence-BERT model — MiniLM variant (fast, accurate, 384-dim output)
MODEL_NAME           = "all-MiniLM-L6-v2"

# Batch size for encoding — increase if you have more RAM/GPU
# 64 is safe for 8GB RAM; use 128–256 on machines with 16GB+
BATCH_SIZE           = 64

# Max sequence length — MiniLM supports up to 512 tokens.
# Raised from 384 → 512 (the model's real ceiling) to reduce silent truncation.
# This does NOT eliminate truncation for long resumes/jobs — see the
# truncation report generated below for exactly how much text is still cut.
MAX_SEQ_LENGTH       = 512

# Rows encoded per checkpoint shard. Smaller = more resumable but more disk
# I/O overhead; larger = fewer checkpoints but more re-work lost on a crash.
CHUNK_SIZE           = 5000

# Storage dtype for the final .npy files. "float32" (default, exact) or
# "float16" (~50% smaller on disk). faiss_index_builder.py casts to float32
# on load either way, so float16 storage is safe for this pipeline.
SAVE_DTYPE           = "float32"


# ==============================
# SETUP
# ==============================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SHARD_DIR, exist_ok=True)


# ==============================
# LOAD DATA
# ==============================

def load_data():
    """
    Load cleaned resume and job CSVs, validate required columns, drop empty
    text rows, and reset the index exactly once here.

    Issue 2 fix: this is the single place the DataFrames are filtered and
    reindexed. Everything downstream (text list for encoding, metadata CSV)
    reads from these same reset DataFrames, so there is no second filter/sort
    step anywhere that could silently break positional alignment.
    """

    print("\nLoading data...")

    if not os.path.exists(RESUME_FILE):
        raise FileNotFoundError(f"{RESUME_FILE} not found. Run resume_pipeline.py first.")
    if not os.path.exists(JOBS_FILE):
        raise FileNotFoundError(f"{JOBS_FILE} not found. Run job_pipeline.py first.")

    resumes = pd.read_csv(RESUME_FILE)
    jobs    = pd.read_csv(JOBS_FILE)

    print(f"  Resumes loaded : {len(resumes):,} rows")
    print(f"  Jobs loaded    : {len(jobs):,} rows")

    assert "cleaned_text" in resumes.columns, "resumes_cleaned.csv must have 'cleaned_text' column"
    assert "job_text"     in jobs.columns,    "jobs_cleaned.csv must have 'job_text' column"

    resumes = resumes[resumes["cleaned_text"].notna() & (resumes["cleaned_text"].str.strip() != "")]
    jobs    = jobs[jobs["job_text"].notna()            & (jobs["job_text"].str.strip() != "")]

    # Issue 2 fix: reset once, here, on the exact objects used everywhere else.
    resumes = resumes.reset_index(drop=True)
    jobs    = jobs.reset_index(drop=True)

    print(f"  Resumes after null drop : {len(resumes):,}")
    print(f"  Jobs after null drop    : {len(jobs):,}")

    return resumes, jobs


# ==============================
# LOAD MODEL
# ==============================

def load_model():
    """Load the Sentence-BERT model. Downloads on first run (~80MB)."""

    print(f"\nLoading Sentence-BERT model: {MODEL_NAME}")

    # Issue 4 fix: fail fast with a clear message instead of a raw traceback
    # (e.g. HF Hub unreachable, corrupted cache, missing torch backend).
    try:
        model = SentenceTransformer(MODEL_NAME)
        model.max_seq_length = MAX_SEQ_LENGTH
    except Exception as e:
        print(f"\nERROR: Failed to load Sentence-BERT model '{MODEL_NAME}'.")
        print(f"  {type(e).__name__}: {e}")
        print("  Common causes: no internet access to Hugging Face Hub on first run,")
        print("  a corrupted local cache (~/.cache/huggingface), or a missing/broken")
        print("  torch install. Fix the underlying issue and re-run — completed")
        print("  encoding shards from a previous run (if any) are preserved on disk.")
        raise

    print(f"  Model loaded successfully")
    print(f"  Embedding dimension : {model.get_sentence_embedding_dimension()}")
    print(f"  Max sequence length : {model.max_seq_length} tokens")

    return model


# ==============================
# TRUNCATION INSTRUMENTATION  (Issue 3)
# ==============================

def measure_truncation(model, texts, label, max_len, tokenize_batch=1000):
    """
    Tokenize every text WITHOUT truncation to find its real token length,
    then report how many texts exceed max_len and would therefore be
    silently truncated inside model.encode().

    This makes an otherwise invisible quality ceiling visible: retrieval
    ranking is built only on the first max_len tokens of anything longer.

    Returns a dict summary (also written to TRUNCATION_REPORT by the caller).
    """
    print(f"\nMeasuring real token lengths for {len(texts):,} {label} (no truncation)...")

    tokenizer = model.tokenizer
    lengths = []

    t0 = time.time()
    for i in range(0, len(texts), tokenize_batch):
        batch = [str(t) for t in texts[i:i + tokenize_batch]]
        enc = tokenizer(batch, truncation=False, padding=False)
        lengths.extend(len(ids) for ids in enc["input_ids"])
    elapsed = time.time() - t0

    lengths = np.array(lengths)
    n_truncated = int((lengths > max_len).sum())
    pct_truncated = 100.0 * n_truncated / len(lengths) if len(lengths) else 0.0

    summary = {
        "label"            : label,
        "n_texts"          : int(len(lengths)),
        "max_seq_length"   : int(max_len),
        "n_truncated"      : n_truncated,
        "pct_truncated"    : round(pct_truncated, 2),
        "mean_tokens"      : round(float(lengths.mean()), 1) if len(lengths) else 0.0,
        "median_tokens"    : float(np.median(lengths)) if len(lengths) else 0.0,
        "p95_tokens"       : float(np.percentile(lengths, 95)) if len(lengths) else 0.0,
        "max_tokens"       : int(lengths.max()) if len(lengths) else 0,
    }

    print(f"  Tokenized in {elapsed:.1f}s")
    print(f"  Truncated (> {max_len} tokens) : {n_truncated:,} / {len(lengths):,}  ({pct_truncated:.1f}%)")
    print(f"  Token length — mean: {summary['mean_tokens']:.1f}  "
          f"median: {summary['median_tokens']:.0f}  "
          f"p95: {summary['p95_tokens']:.0f}  "
          f"max: {summary['max_tokens']}")
    if pct_truncated > 15:
        print(f"  WARNING: {pct_truncated:.1f}% of {label} exceed the model's sequence window.")
        print(f"           Retrieval ranking for these is based on a partial reading of the text —")
        print(f"           the tail (often recent roles / skills sections) is dropped by SBERT.")

    return summary


# ==============================
# ENCODE  (Issue 1: checkpointed / resumable)
# ==============================

def encode_texts(model, texts, label, shard_dir=None, chunk_size=None):
    """
    Encode texts in checkpointed chunks.

    Each chunk is encoded, saved to shard_dir immediately, and skipped on
    future runs if already present — so a crash/interruption partway through
    a multi-hour encode only costs the current chunk, not the whole run.

    IMPORTANT: if you change the source CSV (re-run resume_pipeline.py /
    job_pipeline.py with different data) after a partial or full encode,
    delete the models/_shards/ folder before re-running this script. The
    shard cache does not detect data changes on its own — it only checks
    that a shard has the expected number of rows, not that its content
    still matches the current input.
    """
    shard_dir  = SHARD_DIR   if shard_dir  is None else shard_dir
    chunk_size = CHUNK_SIZE  if chunk_size is None else chunk_size

    label_dir = os.path.join(shard_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    n_chunks = max(1, (len(texts) + chunk_size - 1) // chunk_size)
    print(f"\nEncoding {len(texts):,} {label} in {n_chunks} chunk(s) of ≤{chunk_size:,} rows...")
    print(f"  Batch size : {BATCH_SIZE}")

    chunk_arrays = []
    first_chunk_time = None

    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx   = min(start_idx + chunk_size, len(texts))
        shard_path = os.path.join(label_dir, f"shard_{i:05d}.npy")

        if os.path.exists(shard_path):
            arr = np.load(shard_path)
            expected_rows = end_idx - start_idx
            if arr.shape[0] == expected_rows:
                print(f"  [{i+1}/{n_chunks}] Shard already complete — skipping ({arr.shape[0]} rows)")
                chunk_arrays.append(arr)
                continue
            else:
                print(f"  [{i+1}/{n_chunks}] Shard row-count mismatch — re-encoding.")

        chunk_texts = [str(t) for t in texts[start_idx:end_idx]]
        t0 = time.time()

        # Issue 4 fix: encoding failures fail fast with context; completed
        # shards from earlier chunks remain on disk for the next attempt.
        try:
            chunk_emb = model.encode(
                chunk_texts,
                batch_size=BATCH_SIZE,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=False,   # normalization happens in the FAISS builder
            )
        except Exception as e:
            print(f"\nERROR: Encoding failed on chunk {i+1}/{n_chunks} of {label} "
                  f"(rows {start_idx}:{end_idx}).")
            print(f"  {type(e).__name__}: {e}")
            print(f"  {i} chunk(s) already completed and saved under {label_dir} — "
                  f"re-run this script to resume from chunk {i+1}.")
            raise

        chunk_elapsed = time.time() - t0
        np.save(shard_path, chunk_emb.astype("float32"))
        chunk_arrays.append(chunk_emb)

        # Issue 6 fix: real measured throughput instead of a decorative guess.
        if first_chunk_time is None:
            first_chunk_time = chunk_elapsed
            rows_per_sec = len(chunk_texts) / max(chunk_elapsed, 1e-6)
            remaining_rows = len(texts) - end_idx
            eta_sec = remaining_rows / rows_per_sec if rows_per_sec > 0 else 0
            print(f"  [{i+1}/{n_chunks}] Done in {chunk_elapsed:.1f}s "
                  f"({rows_per_sec:.1f} rows/sec) — "
                  f"estimated remaining time: {eta_sec/60:.1f} min")
        else:
            print(f"  [{i+1}/{n_chunks}] Done in {chunk_elapsed:.1f}s")

    embeddings = np.concatenate(chunk_arrays, axis=0)
    print(f"  {label} encoding complete — shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    return embeddings


# ==============================
# SAVE
# ==============================

def save_embeddings(embeddings, path, label="embeddings"):
    """
    Save numpy array to disk with explicit dtype control (Issue 5).

    Casts to float32 (exact, default) or float16 (~50% smaller) per
    SAVE_DTYPE. faiss_index_builder.py already does
    `embeddings.astype("float32")` before building the index, so either
    storage dtype is safe for the downstream pipeline.
    """
    if SAVE_DTYPE == "float16":
        to_save = embeddings.astype("float16")
    elif SAVE_DTYPE == "float32":
        to_save = embeddings.astype("float32")
    else:
        raise ValueError(f"Unsupported SAVE_DTYPE: {SAVE_DTYPE!r}. Use 'float32' or 'float16'.")

    np.save(path, to_save)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Saved {label} → {path}  ({size_mb:.1f} MB, dtype={to_save.dtype})")


def save_metadata(df, path, cols, label="metadata"):
    """
    Save a subset of columns from a DataFrame as metadata CSV.

    df is expected to already be reset_index(drop=True) by load_data() —
    reset_index() here is kept as a defensive no-op, not the primary
    guarantee (see Issue 2 fix in load_data()).
    """
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

    # 1. Load data (reset_index happens once, here — Issue 2)
    resumes, jobs = load_data()

    # 2. Load model (fails fast on error — Issue 4)
    model = load_model()

    resume_texts = resumes["cleaned_text"].tolist()
    job_texts    = jobs["job_text"].tolist()

    # 3. Truncation instrumentation — Issue 3
    truncation_summary = {}
    truncation_summary["resumes"] = measure_truncation(model, resume_texts, "resumes", MAX_SEQ_LENGTH)
    truncation_summary["jobs"]    = measure_truncation(model, job_texts,    "jobs",    MAX_SEQ_LENGTH)

    with open(TRUNCATION_REPORT, "w") as f:
        json.dump(truncation_summary, f, indent=2)
    print(f"\n  Truncation report saved → {TRUNCATION_REPORT}")

    # 4. Encode resumes and jobs — chunked + checkpointed (Issue 1)
    resume_embeddings = encode_texts(model, resume_texts, label="resumes")
    job_embeddings    = encode_texts(model, job_texts,    label="jobs")

    # Issue 2 fix: explicit alignment assert right where the arrays are
    # created, rather than relying on convention or a downstream check.
    assert len(resume_embeddings) == len(resumes), (
        f"Positional misalignment: {len(resume_embeddings)} resume embeddings "
        f"vs {len(resumes)} resume rows."
    )
    assert len(job_embeddings) == len(jobs), (
        f"Positional misalignment: {len(job_embeddings)} job embeddings "
        f"vs {len(jobs)} job rows."
    )
    print("\n  Alignment check (embeddings vs. DataFrame rows): OK")

    # Issue 5 fix: dtype is explicit and asserted before saving.
    resume_embeddings = resume_embeddings.astype("float32")
    job_embeddings    = job_embeddings.astype("float32")
    assert resume_embeddings.dtype == np.float32
    assert job_embeddings.dtype == np.float32

    # 5. Save embeddings
    print("\nSaving embeddings...")
    save_embeddings(resume_embeddings, RESUME_EMBEDDINGS, label="resume embeddings")
    save_embeddings(job_embeddings,    JOB_EMBEDDINGS,    label="job embeddings")

    # 6. Save metadata
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

    # 7. Sanity check
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
    print(f"  Truncation report : {TRUNCATION_REPORT}")
    print(f"{'='*55}")
    print(f"\n  Note: intermediate shards in {SHARD_DIR}/ can be deleted once")
    print(f"  the .npy files above exist — they're only needed for resuming.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Completed chunks are saved under "
              f"{SHARD_DIR}/ — re-run this script to resume.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL: {type(e).__name__}: {e}")
        print(f"Completed chunks (if any) are saved under {SHARD_DIR}/ — "
              f"re-run this script to resume once the issue above is fixed.")
        sys.exit(1)
