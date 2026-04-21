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

Output:
  - models/faiss_index.bin          : FAISS index (normalized, inner product)
  - models/index_config.json        : metadata about the index build
"""

import os
import json
import time
import numpy as np
import faiss

# ==============================
# CONFIG
# ==============================

JOB_EMBEDDINGS_FILE = "models/job_embeddings.npy"
INDEX_OUTPUT_FILE   = "models/faiss_index.bin"
CONFIG_OUTPUT_FILE  = "models/index_config.json"


# ==============================
# BUILD INDEX
# ==============================

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS IndexFlatIP index from L2-normalized embeddings.

    Steps:
      1. Cast to float32  (FAISS requires float32)
      2. L2-normalize     (so inner product == cosine similarity)
      3. Build IndexFlatIP (exact inner product search)
      4. Add all vectors

    Args:
        embeddings : numpy array of shape (N, D), raw float embeddings

    Returns:
        faiss.IndexFlatIP  with all N vectors added
    """

    print(f"\nBuilding FAISS index...")
    print(f"  Input shape  : {embeddings.shape}")
    print(f"  Input dtype  : {embeddings.dtype}")

    # Step 1: Ensure float32
    embeddings = embeddings.astype("float32")

    # Step 2: L2 normalize — makes dot product equal cosine similarity
    # faiss.normalize_L2 modifies in-place
    faiss.normalize_L2(embeddings)
    print(f"  Normalized   : L2 norm of first vector = {np.linalg.norm(embeddings[0]):.6f} (should be 1.0)")

    # Step 3: Build index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    print(f"  Index type   : IndexFlatIP (exact cosine similarity)")
    print(f"  Dimension    : {dim}")

    # Step 4: Add all vectors
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


def save_config(embeddings: np.ndarray, index: faiss.Index, path: str):
    """Save index build configuration for reproducibility and documentation."""
    config = {
        "index_type":        "IndexFlatIP",
        "similarity_metric": "cosine (L2-normalized inner product)",
        "embedding_model":   "all-MiniLM-L6-v2",
        "embedding_dim":     int(embeddings.shape[1]),
        "num_vectors":       int(index.ntotal),
        "normalization":     "L2 (faiss.normalize_L2)",
        "input_dtype":       "float32",
        "index_file":        path.replace("index_config.json", "faiss_index.bin"),
        "notes": (
            "IndexFlatIP chosen over IVFFlat/HNSW because the dataset (~124k vectors) "
            "fits in RAM and exact search is preferable for a research/demo system. "
            "L2 normalization ensures dot product equals cosine similarity."
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
    """

    print("\n--- Index Verification ---")

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

    print("  Verification passed ✓")


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

    # 2. Build index
    index = build_faiss_index(embeddings)

    # 3. Save
    save_index(index, INDEX_OUTPUT_FILE)
    save_config(embeddings, index, CONFIG_OUTPUT_FILE)

    # 4. Verify
    verify_index(index, embeddings)

    print(f"\n{'='*55}")
    print(f"  FAISS index build complete!")
    print(f"  Index file : {INDEX_OUTPUT_FILE}")
    print(f"  Vectors    : {index.ntotal:,}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
