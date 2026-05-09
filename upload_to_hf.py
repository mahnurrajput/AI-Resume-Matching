from huggingface_hub import HfApi

api = HfApi()
repo_id = "mahnoor-24r/ai-resume-matching-models"  # ← change this

files_to_upload = [
    ("models/faiss_index.bin",        "faiss_index.bin"),
    ("models/job_metadata.csv",       "job_metadata.csv"),
    ("models/resume_metadata.csv",    "resume_metadata.csv"),
    ("models/job_embeddings.npy",     "job_embeddings.npy"),
    ("models/resume_embeddings.npy",  "resume_embeddings.npy"),
    ("data_processed/jobs_cleaned.csv", "jobs_cleaned.csv"),
]

for local_path, repo_filename in files_to_upload:
    print(f"Uploading {local_path}...")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_filename,
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"  Done: {repo_filename}")