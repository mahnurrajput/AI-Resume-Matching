"""
download_models.py
==================
Downloads large model files from Hugging Face Hub at startup.
Called automatically by app.py before the engine loads.
"""

import os
from huggingface_hub import hf_hub_download

REPO_ID   = "mahnoor-24r/ai-resume-matching-models" 
REPO_TYPE = "dataset"

FILES = [
    # (filename in HF repo,          local destination path)
    ("faiss_index.bin",       "models/faiss_index.bin"),
    ("job_metadata.csv",      "models/job_metadata.csv"),
    ("resume_metadata.csv",   "models/resume_metadata.csv"),
    ("job_embeddings.npy",    "models/job_embeddings.npy"),
    ("resume_embeddings.npy", "models/resume_embeddings.npy"),
    ("jobs_cleaned.csv",      "data_processed/jobs_cleaned.csv"),
]

def download_all():
    os.makedirs("models",          exist_ok=True)
    os.makedirs("data_processed",  exist_ok=True)

    for repo_filename, local_path in FILES:
        if os.path.exists(local_path):
            print(f"  Already exists, skipping: {local_path}")
            continue

        print(f"  Downloading {repo_filename}...")
        hf_hub_download(
            repo_id   = REPO_ID,
            repo_type = REPO_TYPE,
            filename  = repo_filename,
            local_dir = os.path.dirname(local_path),
        )
        print(f"  Saved to: {local_path}")

if __name__ == "__main__":
    download_all()