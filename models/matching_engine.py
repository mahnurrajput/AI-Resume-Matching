"""
matching_engine.py
==================
Phase 3 — Core Resume-Job Matching Engine

Integrates AI-powered skill gap analysis as a first-class feature.

Usage (basic — no skill gap):
    engine  = MatchingEngine()
    results = engine.match(resume_text, top_k=10)

Usage (with skill gap analysis):
    results = engine.match(
        resume_text,
        top_k            = 10,
        enable_skill_gap = True,   # Adds AI-powered gap analysis to each result
        enable_ai        = True,   # Use Gemini API for deep reasoning
    )

Each MatchResult has an optional skill_gap field (SkillGapResult) containing
both structured skill sets and AI-generated insights.

File placement:
    This file lives at the project root (same level as the models/ folder).
    skill_analyzer.py lives at models/skill_analyzer.py.

    Project layout expected:
        AI-Resume-Matching/
        ├── matching_engine.py       ← this file
        ├── models/
        │   ├── faiss_index.bin
        │   ├── job_metadata.csv
        │   └── skill_analyzer.py
        └── data_processed/
            └── jobs_cleaned.csv
"""

import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, field
from typing import List, Optional, Dict

# ── Import SkillAnalyzer from models/ sub-package ─────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.skill_analyzer import SkillGapResult, SkillAnalyzer, get_analyzer  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

FAISS_INDEX_FILE  = "models/faiss_index.bin"
JOB_METADATA_FILE = "models/job_metadata.csv"
JOBS_CSV_FILE     = "data_processed/jobs_cleaned.csv"

MODEL_NAME     = "all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 384
DEFAULT_TOP_K  = 10


# ══════════════════════════════════════════════════════════════════════════════
# RESULT DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MatchResult:
    """
    A single job match result returned by MatchingEngine.match().

    Core fields (always populated):
        rank, score, job_idx, job_id, title, company,
        location, experience_level, work_type,
        remote_allowed, salary_min, salary_max

    Optional extension:
        skill_gap : SkillGapResult or None
            Populated when enable_skill_gap=True is passed to match().
            Contains structured skill sets (Stage 1) and optional
            AI-generated insights (Stage 2 / Gemini).
    """
    rank             : int
    score            : float
    job_idx          : int
    job_id           : str
    title            : str
    company          : str
    location         : str
    experience_level : str
    work_type        : str
    remote_allowed   : Optional[float]
    salary_min       : Optional[float]
    salary_max       : Optional[float]
    skill_gap        : Optional[SkillGapResult] = None

    def to_dict(self) -> dict:
        d: Dict = {
            "rank"             : self.rank,
            "score"            : round(self.score, 4),
            "job_id"           : self.job_id,
            "title"            : self.title,
            "company"          : self.company,
            "location"         : self.location,
            "experience_level" : self.experience_level,
            "work_type"        : self.work_type,
            "remote_allowed"   : self.remote_allowed,
            "salary_min"       : self.salary_min,
            "salary_max"       : self.salary_max,
        }
        if self.skill_gap is not None:
            d["skill_gap"] = self.skill_gap.to_dict()
        return d


# ══════════════════════════════════════════════════════════════════════════════
# MATCHING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class MatchingEngine:
    """
    Sentence-BERT + FAISS resume-job matching engine.

    Core components (loaded eagerly at __init__):
      - FAISS index         : pre-built job vector index
      - Job metadata CSV    : display info (title, company, location, etc.)
      - Sentence-BERT model : encodes resume query text

    Lazy-loaded (first time enable_skill_gap=True):
      - SkillAnalyzer       : spaCy + SBERT + optional Gemini
      - jobs_cleaned.csv    : full job text needed for skill gap analysis
    """

    def __init__(
        self,
        faiss_index_path  : str = FAISS_INDEX_FILE,
        job_metadata_path : str = JOB_METADATA_FILE,
        jobs_csv_path     : str = JOBS_CSV_FILE,
        model_name        : str = MODEL_NAME,
        max_seq_length    : int = MAX_SEQ_LENGTH,
    ):
        print("Initializing MatchingEngine...")
        self.index    = self._load_index(faiss_index_path)
        self.job_meta = self._load_metadata(job_metadata_path)
        self.model    = self._load_model(model_name, max_seq_length)

        self._jobs_csv_path  : str = jobs_csv_path
        self._jobs_df        : Optional[pd.DataFrame] = None
        self._skill_analyzer : Optional[SkillAnalyzer] = None

        print(f"  Engine ready — {self.index.ntotal:,} jobs indexed\n")

    # ── Loaders ──────────────────────────────────────────────────────────────

    def _load_index(self, path: str) -> faiss.Index:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"FAISS index not found at '{path}'. "
                "Run faiss_index_builder.py first."
            )
        idx = faiss.read_index(path)
        print(f"  FAISS index loaded  : {path}  ({idx.ntotal:,} vectors)")
        return idx

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
        m = SentenceTransformer(model_name)
        m.max_seq_length = max_seq_length
        print(f"  SBERT model loaded  : {model_name}")
        return m

    def _load_jobs_df(self) -> pd.DataFrame:
        if self._jobs_df is not None:
            return self._jobs_df

        if not os.path.exists(self._jobs_csv_path):
            raise FileNotFoundError(
                f"jobs_cleaned.csv not found at '{self._jobs_csv_path}'. "
                "Disable enable_skill_gap or ensure the file exists."
            )

        print("  Loading jobs_cleaned.csv for skill gap analysis...")
        self._jobs_df = pd.read_csv(self._jobs_csv_path, low_memory=False)

        print(f"  Jobs CSV loaded: {len(self._jobs_df):,} rows")
        return self._jobs_df

    def _get_skill_analyzer(self, enable_ai: bool) -> SkillAnalyzer:
        if self._skill_analyzer is None:
            self._skill_analyzer = get_analyzer(
                enable_semantic=True,
                enable_ai=enable_ai,
            )
            return self._skill_analyzer

        existing_ai = getattr(self._skill_analyzer, "_ai", None)
        if enable_ai and existing_ai is None:
            self._skill_analyzer = get_analyzer(
                enable_semantic=True,
                enable_ai=enable_ai,
            )

        return self._skill_analyzer

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        """Encode a single text and return an L2-normalised (1, D) float32 vector."""
        vec = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        ).astype("float32")
        faiss.normalize_L2(vec)
        return vec

    # ── Job helpers ───────────────────────────────────────────────────────────

    def _get_job_meta(self, embed_idx: int) -> dict:
        try:
            return self.job_meta.iloc[embed_idx].to_dict()
        except IndexError:
            return {}

    def _get_job_text(self, embed_idx: int, jobs_df: Optional[pd.DataFrame]) -> str:
        if jobs_df is None:
            return ""
        try:
            row = jobs_df.iloc[embed_idx]
            return str(row["job_text"]) if "job_text" in row.index else ""
        except (IndexError, KeyError):
            return ""

    # ── Core match ────────────────────────────────────────────────────────────

    def match(
        self,
        resume_text      : str,
        top_k            : int  = DEFAULT_TOP_K,
        enable_skill_gap : bool = True,
        enable_ai        : bool = True,
    ) -> List[MatchResult]:
        """
        Match a resume against all indexed jobs and return the top-k results.

        Args:
            resume_text      : Resume text (raw or cleaned).
            top_k            : Number of results to return.
            enable_skill_gap : Run skill gap analysis for each result.
                               On first call this lazy-loads spaCy + SBERT +
                               jobs_cleaned.csv — subsequent calls are fast.
            enable_ai        : Use Gemini API for Stage 2 reasoning inside
                               skill gap analysis.  Ignored when
                               enable_skill_gap=False.

        Returns:
            List[MatchResult] sorted by cosine similarity (descending).
        """
        if not resume_text or not resume_text.strip():
            raise ValueError("resume_text cannot be empty or whitespace.")

        # Encode and search
        query_vec       = self._embed(resume_text)
        scores, indices = self.index.search(query_vec, top_k)

        jobs_df = self._load_jobs_df() if enable_skill_gap else None

        results: List[MatchResult] = []
        for rank_0, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx == -1:
                # FAISS returns -1 when fewer than top_k results exist
                continue

            embed_idx = int(idx)
            meta      = self._get_job_meta(embed_idx)
            job_text  = self._get_job_text(embed_idx, jobs_df)

            skill_gap: Optional[SkillGapResult] = None
            if enable_skill_gap and job_text:
                try:
                    skill_gap = self._get_skill_analyzer(enable_ai).analyze(
                        resume_text, job_text, enable_ai=enable_ai
                    )
                except Exception as e:
                    print(f"  [SkillGap] Error for result #{rank_0 + 1}: {e}")

            results.append(MatchResult(
                rank             = rank_0 + 1,
                score            = float(score),
                job_idx          = embed_idx,
                job_id           = str(meta.get("job_id",           "") or ""),
                title            = str(meta.get("title",            "") or ""),
                company          = str(meta.get("company",          "") or ""),
                location         = str(meta.get("location",         "") or ""),
                experience_level = str(meta.get("experience_level", "") or ""),
                work_type        = str(meta.get("work_type",        "") or ""),
                remote_allowed   = meta.get("remote_allowed",  None),
                salary_min       = meta.get("salary_min",      None),
                salary_max       = meta.get("salary_max",      None),
                skill_gap        = skill_gap,
            ))

        return results

    # ── Batch match ───────────────────────────────────────────────────────────

    def match_batch(
        self,
        resume_texts     : List[str],
        top_k            : int  = DEFAULT_TOP_K,
        show_progress    : bool = True,
        enable_skill_gap : bool = False,
        enable_ai        : bool = False,
    ) -> List[List[MatchResult]]:
        """
        Batch-encode and match multiple resumes efficiently.

        Skill gap is disabled by default — running it per (resume x job) pair
        is expensive (one Gemini call each).  Enable only when needed.

        Args:
            resume_texts     : List of resume text strings.
            top_k            : Number of results per resume.
            show_progress    : Show SBERT encoding progress bar.
            enable_skill_gap : Run skill gap for every result (slow).
            enable_ai        : Use Gemini Stage 2 inside skill gap.

        Returns:
            List of lists — one inner list of MatchResult per resume.
        """
        if not resume_texts:
            return []

        # Batch encode all resumes in one SBERT call (more efficient than looping)
        vecs = self.model.encode(
            resume_texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=show_progress,
        ).astype("float32")
        faiss.normalize_L2(vecs)

        all_scores, all_indices = self.index.search(vecs, top_k)

        jobs_df = self._load_jobs_df() if enable_skill_gap else None

        all_results: List[List[MatchResult]] = []

        for r_idx, resume_text in enumerate(resume_texts):
            results: List[MatchResult] = []
            for rank_0, (idx, score) in enumerate(
                zip(all_indices[r_idx], all_scores[r_idx])
            ):
                if idx == -1:
                    continue

                embed_idx = int(idx)
                meta      = self._get_job_meta(embed_idx)
                job_text  = self._get_job_text(embed_idx, jobs_df)

                skill_gap: Optional[SkillGapResult] = None
                if enable_skill_gap and job_text:
                    try:
                        skill_gap = self._get_skill_analyzer(enable_ai).analyze(
                            resume_text, job_text, enable_ai=enable_ai
                        )
                    except Exception:
                        pass

                results.append(MatchResult(
                    rank             = rank_0 + 1,
                    score            = float(score),
                    job_idx          = embed_idx,
                    job_id           = str(meta.get("job_id",           "") or ""),
                    title            = str(meta.get("title",            "") or ""),
                    company          = str(meta.get("company",          "") or ""),
                    location         = str(meta.get("location",         "") or ""),
                    experience_level = str(meta.get("experience_level", "") or ""),
                    work_type        = str(meta.get("work_type",        "") or ""),
                    remote_allowed   = meta.get("remote_allowed",  None),
                    salary_min       = meta.get("salary_min",      None),
                    salary_max       = meta.get("salary_max",      None),
                    skill_gap        = skill_gap,
                ))

            all_results.append(results)

        return all_results

    # ── Display ───────────────────────────────────────────────────────────────

    @staticmethod
    def print_results(results: List[MatchResult], resume_label: str = "Resume") -> None:
        print(f"\n{'─'*70}")
        print(f"  Top {len(results)} Matches for: {resume_label}")
        print(f"{'─'*70}")

        for r in results:
            # Salary string (only when both min and max are present and valid)
            salary_str = ""
            try:
                if r.salary_min and str(r.salary_min) not in ("nan", "None", ""):
                    salary_str = (
                        f"  |  ${float(r.salary_min):,.0f}"
                        f"–${float(r.salary_max):,.0f}"
                    )
            except Exception:
                pass

            remote_str = "  [Remote OK]" if str(r.remote_allowed) in ("1.0", "1", "True") else ""

            print(f"\n  #{r.rank}  Score: {r.score:.4f}")
            print(f"      Title    : {r.title}")
            print(f"      Company  : {r.company}")
            print(f"      Location : {r.location}{remote_str}")
            print(f"      Level    : {r.experience_level}  |  {r.work_type}{salary_str}")

            if r.skill_gap:
                sg = r.skill_gap
                print(f"      Overlap  : {sg.overlap_score:.0%}  ({sg.gap_severity} gap)")
                matched = sg.flat_matched()
                missing = sg.flat_missing()
                if matched:
                    suffix = "..." if len(matched) > 6 else ""
                    print(f"      Matched  : {', '.join(matched[:6])}{suffix}")
                if missing:
                    suffix = "..." if len(missing) > 6 else ""
                    print(f"      Missing  : {', '.join(missing[:6])}{suffix}")

                if sg.ai_available:
                    print(f"      Verdict  : {sg.candidacy_verdict} ({sg.verdict_confidence} confidence)")
                    if sg.executive_summary:
                        print(f"      AI Note  : {sg.executive_summary}")
                    if sg.dealbreaker_skills:
                        print(f"      Blockers : {', '.join(sg.dealbreaker_skills)}")
                    if sg.time_to_ready:
                        print(f"      Ready in : {sg.time_to_ready}")

        print(f"\n{'─'*70}")


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    SAMPLE = """
    Python Developer with 4+ years of experience building scalable backend systems and data-driven applications.
    Proficient in Python, Django, Flask, and REST API development.

    Experienced in working with relational and NoSQL databases including PostgreSQL and MongoDB.
    Strong understanding of object-oriented programming, data structures, and software design patterns.

    Built and deployed production-level applications with Docker and AWS (EC2, S3, RDS).
    Worked in agile teams, collaborating with frontend developers and product managers.

    Optimized application performance, reduced API response time by 35%, and implemented secure authentication systems (JWT, OAuth).
    """

    print("=" * 70)
    print("  MATCHING ENGINE — STANDALONE TEST")
    print("=" * 70)

    engine = MatchingEngine()

    print("\n[Test 1] Basic match (no skill gap, fast)...")
    r1 = engine.match(SAMPLE, top_k=3, enable_skill_gap=False)
    engine.print_results(r1, "Chef — Basic")

    print("\n[Test 2] With skill gap (structured only, no AI)...")
    r2 = engine.match(SAMPLE, top_k=3, enable_skill_gap=True, enable_ai=False)
    engine.print_results(r2, "Chef — Structured Gap")

    print("\n[Test 3] Full pipeline (structured + AI reasoning)...")
    r3 = engine.match(SAMPLE, top_k=3, enable_skill_gap=True, enable_ai=True)
    engine.print_results(r3, "Chef — Full AI Analysis")
