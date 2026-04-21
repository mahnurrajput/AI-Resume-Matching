"""
matching_engine.py
==================
Phase 3 — Step 3: Core Resume–Job Matching Engine

The central module of the AI Resume Matching system. Loads the FAISS index
and job metadata, then provides a clean API for:
  - Matching a single resume text → top-K job results
  - Matching a batch of resumes   → all results
  - Retrieving results with full job metadata

Usage (standalone test):
    python models/matching_engine.py

Usage (imported):
    from models.matching_engine import MatchingEngine
    engine = MatchingEngine()
    results = engine.match(resume_text, top_k=10)
"""

import os
import sys
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from typing import List, Optional

# ==============================
# CONFIG
# ==============================

FAISS_INDEX_FILE    = "models/faiss_index.bin"
JOB_METADATA_FILE   = "models/job_metadata.csv"
RESUME_METADATA_FILE= "models/resume_metadata.csv"

MODEL_NAME          = "all-MiniLM-L6-v2"
MAX_SEQ_LENGTH      = 384
DEFAULT_TOP_K       = 10


# ==============================
# RESULT DATACLASS
# ==============================

@dataclass
class MatchResult:
    """
    A single job match result returned by the engine.

    Attributes:
        rank            : 1-based rank in the result list
        score           : cosine similarity score (0.0 – 1.0)
        job_idx         : index in the FAISS index / job_metadata
        job_id          : original LinkedIn job ID
        title           : job title
        company         : company name
        location        : job location
        experience_level: entry / mid / senior / etc.
        work_type       : full-time / part-time / contract / etc.
        remote_allowed  : True / False / NaN
        salary_min      : minimum salary where available
        salary_max      : maximum salary where available
    """
    rank            : int
    score           : float
    job_idx         : int
    job_id          : str
    title           : str
    company         : str
    location        : str
    experience_level: str
    work_type       : str
    remote_allowed  : Optional[bool]
    salary_min      : Optional[float]
    salary_max      : Optional[float]

    def to_dict(self) -> dict:
        return {
            "rank"            : self.rank,
            "score"           : round(self.score, 4),
            "job_id"          : self.job_id,
            "title"           : self.title,
            "company"         : self.company,
            "location"        : self.location,
            "experience_level": self.experience_level,
            "work_type"       : self.work_type,
            "remote_allowed"  : self.remote_allowed,
            "salary_min"      : self.salary_min,
            "salary_max"      : self.salary_max,
        }


# ==============================
# MATCHING ENGINE
# ==============================

class MatchingEngine:
    """
    Sentence-BERT + FAISS resume–job matching engine.

    Initialization loads:
      - The FAISS job index from disk
      - The job metadata CSV for display
      - The Sentence-BERT model for encoding new resume text

    All heavy loading happens once at init — subsequent match() calls
    are fast (milliseconds per query).
    """

    def __init__(
        self,
        faiss_index_path : str = FAISS_INDEX_FILE,
        job_metadata_path: str = JOB_METADATA_FILE,
        model_name       : str = MODEL_NAME,
        max_seq_length   : int = MAX_SEQ_LENGTH,
    ):
        print("Initializing MatchingEngine...")

        # Load FAISS index
        self.index = self._load_index(faiss_index_path)

        # Load job metadata for enriching results
        self.job_meta = self._load_metadata(job_metadata_path)

        # Load Sentence-BERT model for encoding resume queries
        self.model = self._load_model(model_name, max_seq_length)

        print(f"  Engine ready — {self.index.ntotal:,} jobs indexed\n")


    # ---- Loaders --------------------------------------------------------

    def _load_index(self, path: str) -> faiss.Index:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"FAISS index not found at '{path}'. "
                "Run faiss_index_builder.py first."
            )
        index = faiss.read_index(path)
        print(f"  FAISS index loaded  : {path}  ({index.ntotal:,} vectors)")
        return index

    def _load_metadata(self, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Job metadata not found at '{path}'. "
                "Run embedding_generator.py first."
            )
        df = pd.read_csv(path, index_col="embed_idx")
        print(f"  Job metadata loaded : {path}  ({len(df):,} rows)")
        return df

    def _load_model(self, model_name: str, max_seq_length: int) -> SentenceTransformer:
        model = SentenceTransformer(model_name)
        model.max_seq_length = max_seq_length
        print(f"  SBERT model loaded  : {model_name}")
        return model


    # ---- Core Matching --------------------------------------------------

    def _embed(self, text: str) -> np.ndarray:
        """
        Embed a single text string into a normalized 384-dim vector.

        The vector is L2-normalized so that FAISS inner product search
        returns cosine similarity scores directly.
        """
        vector = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        ).astype("float32")

        faiss.normalize_L2(vector)
        return vector


    def match(self, resume_text: str, top_k: int = DEFAULT_TOP_K) -> List[MatchResult]:
        """
        Match a resume text against the job index.

        Args:
            resume_text : raw or cleaned resume text string
            top_k       : number of top job results to return

        Returns:
            List of MatchResult objects sorted by cosine similarity (desc)
        """

        if not resume_text or not resume_text.strip():
            raise ValueError("resume_text cannot be empty")

        # Encode the resume into an embedding vector
        query_vector = self._embed(resume_text)

        # Search FAISS index
        scores, indices = self.index.search(query_vector, top_k)

        # Build result objects
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx == -1:
                # FAISS returns -1 for padded results (shouldn't happen with FlatIP)
                continue

            meta = self._get_job_meta(idx)

            results.append(MatchResult(
                rank             = rank + 1,
                score            = float(score),
                job_idx          = int(idx),
                job_id           = meta.get("job_id",           ""),
                title            = meta.get("title",            ""),
                company          = meta.get("company",          ""),
                location         = meta.get("location",         ""),
                experience_level = meta.get("experience_level", ""),
                work_type        = meta.get("work_type",        ""),
                remote_allowed   = meta.get("remote_allowed",   None),
                salary_min       = meta.get("salary_min",       None),
                salary_max       = meta.get("salary_max",       None),
            ))

        return results


    def match_batch(
        self,
        resume_texts : List[str],
        top_k        : int = DEFAULT_TOP_K,
        show_progress: bool = True,
    ) -> List[List[MatchResult]]:
        """
        Match multiple resumes in a single batch encoding call.
        More efficient than calling match() in a loop.

        Args:
            resume_texts : list of resume text strings
            top_k        : number of top results per resume
            show_progress: show tqdm progress bar during encoding

        Returns:
            List of result lists — one per resume
        """

        if not resume_texts:
            return []

        # Batch encode all resumes
        vectors = self.model.encode(
            resume_texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=show_progress,
        ).astype("float32")

        faiss.normalize_L2(vectors)

        # Batch search
        all_scores, all_indices = self.index.search(vectors, top_k)

        # Build results for each resume
        all_results = []
        for r_idx in range(len(resume_texts)):
            results = []
            for rank, (idx, score) in enumerate(zip(all_indices[r_idx], all_scores[r_idx])):
                if idx == -1:
                    continue
                meta = self._get_job_meta(idx)
                results.append(MatchResult(
                    rank             = rank + 1,
                    score            = float(score),
                    job_idx          = int(idx),
                    job_id           = meta.get("job_id",           ""),
                    title            = meta.get("title",            ""),
                    company          = meta.get("company",          ""),
                    location         = meta.get("location",         ""),
                    experience_level = meta.get("experience_level", ""),
                    work_type        = meta.get("work_type",        ""),
                    remote_allowed   = meta.get("remote_allowed",   None),
                    salary_min       = meta.get("salary_min",       None),
                    salary_max       = meta.get("salary_max",       None),
                ))
            all_results.append(results)

        return all_results


    # ---- Metadata Helper ------------------------------------------------

    def _get_job_meta(self, idx: int) -> dict:
        """Return metadata dict for a job by its FAISS index position."""
        try:
            row = self.job_meta.iloc[idx]
            return row.to_dict()
        except IndexError:
            return {}


    # ---- Display Helpers ------------------------------------------------

    @staticmethod
    def print_results(results: List[MatchResult], resume_label: str = "Resume"):
        """Pretty-print a list of match results to stdout."""

        print(f"\n{'─'*65}")
        print(f"  Top {len(results)} Matches for: {resume_label}")
        print(f"{'─'*65}")

        for r in results:
            salary_str = ""
            if r.salary_min and r.salary_max:
                salary_str = f"  |  ${r.salary_min:,.0f}–${r.salary_max:,.0f}"

            remote_str = ""
            if r.remote_allowed == 1 or r.remote_allowed is True:
                remote_str = "  [Remote OK]"

            print(f"\n  #{r.rank}  Score: {r.score:.4f}")
            print(f"      Title   : {r.title}")
            print(f"      Company : {r.company}")
            print(f"      Location: {r.location}{remote_str}")
            print(f"      Level   : {r.experience_level}  |  {r.work_type}{salary_str}")

        print(f"\n{'─'*65}")


# ==============================
# STANDALONE TEST
# ==============================

if __name__ == "__main__":

    # --- Sample resume text (short excerpt for quick testing) ---
    SAMPLE_RESUME = """
    experienced software engineer with 6 years in java j2ee spring boot microservices
    restful web services angular javascript html css oracle postgresql mysql aws cloud
    agile methodology ci cd jenkins maven git docker kubernetes junit testing
    designed and developed enterprise web applications using spring mvc hibernate jpa
    worked with angular for frontend react for ui components
    strong understanding of design patterns mvc singleton factory dao
    experience with nosql databases mongodb redis
    """

    print("=" * 65)
    print("  MATCHING ENGINE — STANDALONE TEST")
    print("=" * 65)

    engine = MatchingEngine()

    results = engine.match(SAMPLE_RESUME, top_k=10)
    engine.print_results(results, resume_label="Sample Java Developer Resume")

    # Print as dict for inspection
    print("\nResults as dicts:")
    for r in results[:3]:
        print(f"  {r.to_dict()}")
