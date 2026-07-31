"""
upload_to_hf.py
===============
Pushes large model artifacts to the Hugging Face Hub dataset repo.

Imports MODEL_FILES from download_models.py so the upload list and the
download list can never drift apart — add or remove a file in exactly
one place (download_models.py's MODEL_FILES) and both scripts stay in sync.
"""

from huggingface_hub import HfApi
from download_models import REPO_ID, REPO_TYPE, MODEL_FILES

api = HfApi()
failures = []

for repo_filename, local_path, _required_for_app in MODEL_FILES:
    print(f"Uploading {local_path} -> {repo_filename}...")
    try:
        api.upload_file(
            path_or_fileobj = local_path,
            path_in_repo    = repo_filename,
            repo_id         = REPO_ID,
            repo_type       = REPO_TYPE,
        )
        print(f"  Done: {repo_filename}")
    except FileNotFoundError:
        print(f"  SKIPPED: local file not found at '{local_path}'.")
        failures.append(repo_filename)
    except Exception as e:
        print(f"  FAILED: {repo_filename} — {type(e).__name__}: {e}")
        failures.append(repo_filename)

print("\n" + "=" * 50)
if failures:
    print(f"Upload finished with {len(failures)} failure(s): {failures}")
else:
    print("All files uploaded successfully.")
print("=" * 50)