"""
faiss_index_builder.py
======================
Phase 3 — Step 2: FAISS Vector Index Construction

Loads pre-generated job embeddings, L2-normalizes them so that inner
product equals cosine similarity, then builds a FAISS IndexFlatIP index
for exact nearest-neighbor search.

Why IndexFlatIP (exact search)?
  - The job dataset (~124k vectors of 384 dims) fits comfortably in RAM
  - Exact search guarantees no approximation error in results
  - Approximate methods (IVFFlat, HNSW) are only needed for 1M+ vectors
  - For this project scope, exact search is the correct and final choice

Why cosine similarity (not L2 distance)?
  - Sentence-BERT embeddings are directional — two semantically similar
    texts point in the same direction regardless of text length
  - L2 distance is affected by vector magnitude, which introduces bias
    toward shorter or longer texts
  - Cosine similarity is the standard metric for SBERT-based matching

Input:
  - models/job_embeddings.npy       : raw job embeddings (N, 384)
  - data_processed/jobs_cleaned.csv : optional — only used for an informational
                                       row-count check, not required to run

Output:
  - models/faiss_index.bin          : FAISS index (normalized, inner product)
  - models/index_config.json        : metadata about the index build

─────────────────────────────────────────────────────────────────────────────
CHANGELOG — fixes applied to the previous version of this script
─────────────────────────────────────────────────────────────────────────────
1. NO IN-PLACE MUTATION OF CALLER'S ARRAY : build_faiss_index() now makes an
   explicit copy before normalizing, so the `embeddings` array passed in by
   main() (and handed to verify_index() afterward) is never silently changed
   out from under the caller.
2. VERIFY_INDEX DOCUMENTED AS A SANITY CHECK, NOT A QUALITY CHECK : clarified
   in the docstring and console output that this is a self-match integrity
   check (did the index build correctly), not a measure of retrieval quality
   (that's evaluate_matching.py's job).
3. SHAPE/EMPTINESS GUARD : a clear assert now catches a corrupted or empty
   embeddings file immediately, instead of an unhelpful IndexError deep in
   FAISS.
4. MODEL NAME NO LONGER HARDCODED : reads MODEL_NAME from embedding_generator.py
   when that file is importable (single source of truth), falling back to a
   documented default only if it isn't.
5. SOURCE FRESHNESS VISIBILITY : index_config.json now records the embeddings
   file's size/modified-time, and — if jobs_cleaned.csv is present — its row
   count and modified-time too, plus a console note if the row counts don't
   match the number of indexed vectors. This is informational only (some
   row-count drift is expected — embedding_generator.py drops empty-text
   rows), not a hard failure, since asserting equality here would produce
   false alarms on legitimate runs.
"""

import os
import sys
import json
import time
import datetime
import numpy as np
import faiss

# ==============================
# CONFIG
# ==============================

JOB_EMBEDDINGS_FILE = "models/job_embeddings.npy"
INDEX_OUTPUT_FILE   = "models/faiss_index.bin"
CONFIG_OUTPUT_FILE  = "models/index_config.json"

# Optional — only used for the informational row-count note in Fix 5.
# Not required for the script to run.
JOBS_CSV_FILE        = "data_processed/jobs_cleaned.csv"

# Issue 4 fix: read the model name from embedding_generator.py so this file
# doesn't carry its own independent, driftable copy of the constant. Falls
# back to a documented default if embedding_generator.py isn't on the path
# (e.g. this script is run from a different working directory).
try:
    from embedding_generator import MODEL_NAME as EMBEDDING_MODEL_NAME
except ImportError:
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # keep in sync with embedding_generator.py MODEL_NAME
    print("  NOTE: could not import MODEL_NAME from embedding_generator.py — "
          f"falling back to hardcoded default '{EMBEDDING_MODEL_NAME}' for index_config.json. "
          "Run this script from the same directory as embedding_generator.py to avoid this.")


# ==============================
# BUILD INDEX
# ==============================

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS IndexFlatIP index from L2-normalized embeddings.

    Steps:
      1. Validate shape (Issue 3 fix)
      2. Copy + cast to float32  (Issue 1 fix — never mutate the caller's array)
      3. L2-normalize the copy   (so inner product == cosine similarity)
      4. Build IndexFlatIP (exact inner product search)
      5. Add all vectors

    Args:
        embeddings : numpy array of shape (N, D), raw float embeddings.
                     This array is NOT modified — the function works on an
                     internal copy.

    Returns:
        faiss.IndexFlatIP  with all N vectors added
    """

    print(f"\nBuilding FAISS index...")
    print(f"  Input shape  : {embeddings.shape}")
    print(f"  Input dtype  : {embeddings.dtype}")

    # Issue 3 fix: fail with a clear message instead of a confusing IndexError
    # or undefined FAISS behavior if the file is corrupted/empty/malformed.
    assert embeddings.ndim == 2 and embeddings.shape[0] > 0 and embeddings.shape[1] > 0, (
        f"Bad embeddings array — expected 2D shape (N, D) with N > 0 and D > 0, "
        f"got shape {embeddings.shape}. The .npy file may be corrupted or empty — "
        f"re-run embedding_generator.py."
    )

    # Issue 1 fix: astype(..., copy=True) always returns a new array, even
    # when the dtype already matches — so the caller's original array (which
    # main() later passes to verify_index()) is never touched by the
    # in-place faiss.normalize_L2() call below.
    embeddings = embeddings.astype("float32", copy=True)

    # L2 normalize the copy — makes dot product equal cosine similarity
    faiss.normalize_L2(embeddings)
    print(f"  Normalized   : L2 norm of first vector = {np.linalg.norm(embeddings[0]):.6f} (should be 1.0)")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    print(f"  Index type   : IndexFlatIP (exact cosine similarity)")
    print(f"  Dimension    : {dim}")

    start = time.time()
    index.add(embeddings)
    elapsed = time.time() - start

    print(f"  Vectors added: {index.ntotal:,}")
    print(f"  Build time   : {elapsed:.2f}s")

    return index


# ==============================
# SAVE
# ==============================

def save_index(index: faiss.Index, path: str):
    """Write FAISS index to disk."""
    faiss.write_index(index, path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"\n  Index saved → {path}  ({size_mb:.1f} MB)")


def _file_info(path: str) -> dict:
    """Cheap, dependency-free size + modified-time lookup for provenance notes."""
    if not os.path.exists(path):
        return {"exists": False}
    stat = os.stat(path)
    return {
        "exists"        : True,
        "size_mb"       : round(stat.st_size / (1024 * 1024), 2),
        "modified_time" : datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
    }


def _count_csv_rows(path: str) -> int:
    """
    Cheap row count (line count minus header). Not exact for CSVs with
    embedded newlines inside quoted fields, but good enough as an
    informational freshness signal — this is not used for any hard check.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f) - 1


def save_config(embeddings: np.ndarray, index: faiss.Index, path: str):
    """
    Save index build configuration for reproducibility and documentation.

    Issue 5 fix: adds source-file size/mtime for the embeddings, and — if
    jobs_cleaned.csv is present — its row count and mtime too, so anyone
    inspecting this file later can visually check whether the index looks
    like it was built from stale data. This is informational only; it is
    not enforced anywhere, since some row-count drift between the CSV and
    the index is expected (embedding_generator.py drops empty-text rows).
    """
    embeddings_info = _file_info(JOB_EMBEDDINGS_FILE)

    jobs_csv_info = {"exists": False}
    if os.path.exists(JOBS_CSV_FILE):
        jobs_csv_info = _file_info(JOBS_CSV_FILE)
        try:
            jobs_csv_info["row_count"] = _count_csv_rows(JOBS_CSV_FILE)
        except Exception as e:
            jobs_csv_info["row_count"] = None
            jobs_csv_info["row_count_error"] = str(e)

    config = {
        "index_type":        "IndexFlatIP",
        "similarity_metric": "cosine (L2-normalized inner product)",
        "embedding_model":   EMBEDDING_MODEL_NAME,
        "embedding_dim":     int(embeddings.shape[1]),
        "num_vectors":       int(index.ntotal),
        "normalization":     "L2 (faiss.normalize_L2)",
        "input_dtype":       "float32",
        "index_file":        path.replace("index_config.json", "faiss_index.bin"),
        "source_embeddings_file": {
            "path": JOB_EMBEDDINGS_FILE,
            **embeddings_info,
        },
        "source_jobs_csv": {
            "path": JOBS_CSV_FILE,
            **jobs_csv_info,
        },
        "notes": (
            "IndexFlatIP chosen over IVFFlat/HNSW because the dataset (~124k vectors) "
            "fits in RAM and exact search is preferable for a research/demo system. "
            "L2 normalization ensures dot product equals cosine similarity. "
            "'source_jobs_csv' is informational only — used to eyeball whether this "
            "index might be stale relative to the current jobs_cleaned.csv, not an "
            "enforced check."
        )
    }

    with open(path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"  Config saved → {path}")


# ==============================
# VERIFY INDEX
# ==============================

def verify_index(index: faiss.Index, embeddings: np.ndarray, k: int = 5):
    """
    Quick sanity check: query the index with the first embedding.
    The top result should be itself (score = 1.0).

    Issue 2 note: this confirms the INDEX was built and saved correctly
    (self-match integrity) — it does NOT measure retrieval quality (whether
    semantically similar-but-different texts actually rank well). That
    evaluation lives in evaluate_matching.py. Don't read "Verification
    passed" as "matching works well" — it only means the index isn't corrupt.
    """

    print("\n--- Index Verification (self-match integrity check, not a quality check) ---")

    query = embeddings[0:1].astype("float32").copy()
    faiss.normalize_L2(query)

    scores, indices = index.search(query, k)

    print(f"  Query: embedding[0] (self-match test)")
    print(f"  Top-{k} results:")

    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        marker = " ← should be 1.0000 (self)" if rank == 0 else ""
        print(f"    Rank {rank+1}: idx={idx:6d}  score={score:.4f}{marker}")

    assert indices[0][0] == 0,        "ERROR: Top result is not embedding[0] — index may be corrupt"
    assert abs(scores[0][0] - 1.0) < 1e-4, "ERROR: Self-similarity score is not 1.0 — normalization failed"

    print("  Verification passed ✓ (index integrity confirmed — see note above re: retrieval quality)")


# ==============================
# MAIN
# ==============================

def main():

    print("=" * 55)
    print("  PHASE 3 — FAISS INDEX BUILDER")
    print("  Type: IndexFlatIP (exact cosine similarity)")
    print("=" * 55)

    # 1. Load job embeddings
    print(f"\nLoading job embeddings from: {JOB_EMBEDDINGS_FILE}")
    if not os.path.exists(JOB_EMBEDDINGS_FILE):
        raise FileNotFoundError(
            f"Job embeddings not found at {JOB_EMBEDDINGS_FILE}. "
            f"Run embedding_generator.py first."
        )

    embeddings = np.load(JOB_EMBEDDINGS_FILE)
    print(f"  Shape  : {embeddings.shape}")
    print(f"  dtype  : {embeddings.dtype}")
    print(f"  Memory : {embeddings.nbytes / (1024*1024):.1f} MB")

    # 2. Build index — embeddings is NOT mutated by this call (Issue 1 fix)
    index = build_faiss_index(embeddings)

    # 3. Save
    save_index(index, INDEX_OUTPUT_FILE)
    save_config(embeddings, index, CONFIG_OUTPUT_FILE)

    # Issue 5 fix: informational-only freshness note printed to console too,
    # not just buried in the JSON config.
    if os.path.exists(JOBS_CSV_FILE):
        try:
            csv_rows = _count_csv_rows(JOBS_CSV_FILE)
            print(f"\n  jobs_cleaned.csv rows : {csv_rows:,}   (indexed vectors: {index.ntotal:,})")
            if csv_rows != index.ntotal:
                print("  NOTE: these counts differ. Some difference is expected if rows with")
                print("        empty text were dropped during embedding — but if this looks")
                print("        unexpected, confirm job_embeddings.npy was regenerated from the")
                print("        CURRENT jobs_cleaned.csv before trusting this index.")
        except Exception as e:
            print(f"  (Could not read {JOBS_CSV_FILE} for the row-count note: {e})")

    # 4. Verify — embeddings here is still the original array loaded from disk,
    #    unmodified by build_faiss_index() (Issue 1 fix)
    verify_index(index, embeddings)

    print(f"\n{'='*55}")
    print(f"  FAISS index build complete!")
    print(f"  Index file : {INDEX_OUTPUT_FILE}")
    print(f"  Vectors    : {index.ntotal:,}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
