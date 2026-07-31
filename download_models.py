"""
download_models.py
==================
Downloads large model artifacts from Hugging Face Hub at startup.
Called automatically by app.py before the matching engine loads.

This file is the single source of truth for which files live in the
Hugging Face dataset repo and where they belong locally. upload_to_hf.py
imports MODEL_FILES from here so the upload and download file lists can
never drift out of sync with each other.
"""

import os
import shutil

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    raise ImportError(
        "huggingface_hub is not installed. Run: pip install huggingface_hub"
    )

REPO_ID   = "mahnoor-24r/ai-resume-matching-models"
REPO_TYPE = "dataset"

# ──────────────────────────────────────────────────────────────────────────
# FILE MANIFEST — single source of truth for both this script and
# upload_to_hf.py. Format: (repo_filename, local_path, required_for_app)
#
# required_for_app=True  : needed by the live Streamlit app (MatchingEngine).
#                           Always downloaded.
# required_for_app=False : only needed for offline scripts (e.g.
#                           evaluate_matching.py). NOT downloaded automatically
#                           at app startup — keeps cold-start fast and reduces
#                           what can fail on every launch. Call
#                           download_offline_extras() explicitly if you need
#                           these for a local/offline run.
# ──────────────────────────────────────────────────────────────────────────
MODEL_FILES = [
    ("faiss_index.bin",       "models/faiss_index.bin",         True),
    ("job_metadata.csv",      "models/job_metadata.csv",        True),
    ("jobs_cleaned.csv",      "data_processed/jobs_cleaned.csv", True),
    ("resume_metadata.csv",   "models/resume_metadata.csv",     False),
    ("resume_embeddings.npy","models/resume_embeddings.npy",   False),
    ("job_embeddings.npy",    "models/job_embeddings.npy",      False),
]


def _download_one(repo_filename: str, local_path: str) -> None:
    """
    Download a single file, with three layers of defense:
      1. Skip only if the file exists AND is non-empty (a 0-byte leftover
         from an interrupted download is treated as "not really there").
      2. Wrap the network call so failures raise a clear, actionable message
         instead of a raw huggingface_hub traceback.
      3. Verify the file actually landed at local_path afterward, copying it
         there if hf_hub_download's internal behavior ever changes.
    """
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        print(f"  Already exists, skipping: {local_path}")
        return

    print(f"  Downloading {repo_filename}...")
    local_dir = os.path.dirname(local_path) or "."
    os.makedirs(local_dir, exist_ok=True)

    try:
        downloaded_path = hf_hub_download(
            repo_id   = REPO_ID,
            repo_type = REPO_TYPE,
            filename  = repo_filename,
            local_dir = local_dir,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to download '{repo_filename}' from Hugging Face Hub "
            f"(repo: {REPO_ID}). Common causes: no internet access, the "
            f"repo is private/renamed, or an HF rate limit was hit. "
            f"Original error: {type(e).__name__}: {e}"
        ) from e

    if os.path.abspath(downloaded_path) != os.path.abspath(local_path):
        shutil.copy2(downloaded_path, local_path)

    if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
        raise RuntimeError(
            f"Download of '{repo_filename}' completed but '{local_path}' "
            f"is missing or empty — treating this as a failed download."
        )

    print(f"  Saved to: {local_path}")


def download_all(include_offline_extras: bool = False) -> None:
    """
    Download the files the live app needs.

    Args:
        include_offline_extras : Also fetch files only needed for offline
                                  scripts. Off by default so the Streamlit
                                  cold start stays fast and only downloads
                                  what MatchingEngine actually uses.
    """
    os.makedirs("models",         exist_ok=True)
    os.makedirs("data_processed", exist_ok=True)

    for repo_filename, local_path, required_for_app in MODEL_FILES:
        if required_for_app or include_offline_extras:
            _download_one(repo_filename, local_path)


def download_offline_extras() -> None:
    """Fetch the extra artifacts only needed for local/offline scripts."""
    download_all(include_offline_extras=True)


if __name__ == "__main__":
    download_all(include_offline_extras=True)