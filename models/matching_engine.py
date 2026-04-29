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
    This file lives inside the models/ folder, alongside skill_analyzer.py
    and the FAISS / embedding artefacts.

    Project layout expected:
        AI-Resume-Matching/
        ├── data_processed/
        │   └── jobs_cleaned.csv
        └── models/
            ├── matching_engine.py   ← this file
            ├── skill_analyzer.py
            ├── faiss_index.bin
            ├── job_metadata.csv
            ├── resume_embeddings.npy
            └── job_embeddings.npy

IMPORTANT — Streamlit usage:
    MatchingEngine loads ~240 MB of data at startup (FAISS index + SBERT model
    + jobs CSV).  In Streamlit, always cache it with @st.cache_resource so it
    is only initialised once per server process:

        @st.cache_resource
        def get_engine():
            return MatchingEngine()

        engine = get_engine()
"""

import os
import sys
import math
import traceback

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

# ── Import skill_analyzer — both files live in models/, so a direct import
# works when this file is run from inside models/ or when models/ is on
# sys.path (e.g. via Streamlit or the test suite).  The insert below makes
# the import work regardless of the current working directory.
_MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
if _MODELS_DIR not in sys.path:
    sys.path.insert(0, _MODELS_DIR)

from skill_analyzer import SkillGapResult, SkillAnalyzer, get_analyzer  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  —  paths resolved relative to this file's location
# ══════════════════════════════════════════════════════════════════════════════

# Project root = parent of the models/ directory
_PROJECT_ROOT = os.path.dirname(_MODELS_DIR)

# Artefacts that live in models/ (same folder as this file)
FAISS_INDEX_FILE  = os.path.join(_MODELS_DIR,   "faiss_index.bin")
JOB_METADATA_FILE = os.path.join(_MODELS_DIR,   "job_metadata.csv")

# jobs_cleaned.csv lives one level up in data_processed/
JOBS_CSV_FILE     = os.path.join(_PROJECT_ROOT, "data_processed", "jobs_cleaned.csv")

MODEL_NAME     = "all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 384
DEFAULT_TOP_K  = 10


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _safe_float(value) -> Optional[float]:
    """
    Convert a value to float, returning None for NaN, None, or unparseable input.

    Pandas reads missing CSV cells as float('nan'), not Python None.
    json.dumps() raises ValueError on NaN, so we normalise here at MatchResult
    construction time rather than patching every downstream consumer.
    """
    if value is None:
        return None
    try:
        f = float(value)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


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

    Note: remote_allowed, salary_min, salary_max are always Python float or
    None — never float('nan') — so to_dict() output is always JSON-safe.
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

    Index alignment guarantee:
      _validate_alignment() asserts len(job_metadata) == index.ntotal at startup.
      If they differ a clear RuntimeError is raised immediately rather than
      silently returning wrong job metadata for FAISS results.
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

        # Validate positional alignment between FAISS index and metadata CSV.
        self._validate_alignment()

        self._jobs_csv_path  : str                     = jobs_csv_path
        self._jobs_df        : Optional[pd.DataFrame]  = None
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
        # Read without index_col so the DataFrame keeps a clean 0-based RangeIndex.
        # iloc[embed_idx] is then unambiguously positional — no gap ambiguity from
        # any embed_idx column that may exist in the CSV.
        df = pd.read_csv(path)
        print(f"  Job metadata loaded : {path}  ({len(df):,} rows)")
        return df

    def _load_model(self, model_name: str, max_seq_length: int) -> SentenceTransformer:
        m = SentenceTransformer(model_name)
        m.max_seq_length = max_seq_length
        print(f"  SBERT model loaded  : {model_name}")
        return m

    def _validate_alignment(self) -> None:
        """
        Assert FAISS index size == metadata row count.

        FAISS returns positional indices (0 … N-1).  _get_job_meta() maps those
        positions to metadata rows via iloc.  A mismatch means every result past
        the shorter source would either crash or return the wrong job's metadata.
        Better to fail loudly at startup with a clear message.
        """
        n_index = self.index.ntotal
        n_meta  = len(self.job_meta)
        if n_index != n_meta:
            raise RuntimeError(
                f"Index/metadata misalignment: FAISS index has {n_index:,} vectors "
                f"but job_metadata.csv has {n_meta:,} rows. "
                "Regenerate embeddings and metadata from the same jobs_cleaned.csv "
                "using embedding_generator.py, then rebuild the FAISS index."
            )
        print(f"  Alignment check     : OK ({n_index:,} vectors == {n_meta:,} metadata rows)")

    def _load_jobs_df(self) -> pd.DataFrame:
        """
        Lazy-load jobs_cleaned.csv for skill-gap text retrieval.

        The CSV is only read once per MatchingEngine instance (guarded by
        self._jobs_df is not None).  In Streamlit wrap MatchingEngine in
        @st.cache_resource so the same instance is reused across all reruns.
        """
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
        """
        Return (and lazily initialise) a SkillAnalyzer that shares this engine's
        already-loaded SBERT model.

        Why we construct SkillAnalyzer directly instead of calling get_analyzer():
            get_analyzer() is the module-level singleton in skill_analyzer.py.
            Its signature is get_analyzer(enable_semantic, enable_ai) — it does
            NOT accept a sbert_model argument.  sbert_model is only accepted by
            SkillAnalyzer.__init__() and StructuredSkillExtractor.__init__().
            Constructing directly lets us pass sbert_model=self.model to avoid
            loading a second ~80 MB copy of all-MiniLM-L6-v2 into memory.

        AI-upgrade logic mirrors get_analyzer() in skill_analyzer.py:
            AISkillReasoningEngine is always instantiated but its internal
            _client is None when no API key is present.  We check _client
            (not the engine object) so the analyzer is properly recreated
            when a user provides a key and retries — not just on first call.
        """
        if self._skill_analyzer is None:
            print("  Loading SkillAnalyzer (shared SBERT model)...")
            self._skill_analyzer = SkillAnalyzer(
                enable_semantic=True,
                enable_ai=enable_ai,
                sbert_model=self.model,
            )
            return self._skill_analyzer

        # AI-upgrade check: does the cached instance have a live Gemini client?
        existing_ai_client = getattr(
            getattr(self._skill_analyzer, "_ai", None),
            "_client",
            None,
        )
        if enable_ai and existing_ai_client is None:
            print("  Reinitializing SkillAnalyzer with AI enabled...")
            self._skill_analyzer = SkillAnalyzer(
                enable_semantic=True,
                enable_ai=True,
                sbert_model=self.model,
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
        """
        Return metadata dict for FAISS position embed_idx.

        Safe because _validate_alignment() guarantees len(job_meta) == index.ntotal,
        so every valid FAISS result index maps to an existing metadata row.
        """
        try:
            return self.job_meta.iloc[embed_idx].to_dict()
        except IndexError:
            return {}

    def _get_job_text(self, embed_idx: int, jobs_df: Optional[pd.DataFrame]) -> str:
        """
        Return job_text for FAISS position embed_idx from jobs_cleaned.csv.

        Positional iloc is correct: embedding_generator.py encodes jobs in the
        same row order it writes jobs_cleaned.csv, so FAISS position N == row N
        in both the metadata CSV and the full jobs CSV.
        """
        if jobs_df is None:
            return ""
        try:
            row = jobs_df.iloc[embed_idx]
            return str(row["job_text"]) if "job_text" in row.index else ""
        except (IndexError, KeyError):
            return ""

    def _build_match_result(
        self,
        rank_0    : int,
        embed_idx : int,
        score     : float,
        meta      : dict,
        skill_gap : Optional[SkillGapResult],
    ) -> "MatchResult":
        """
        Construct a MatchResult from raw FAISS output + metadata dict.

        _safe_float() normalises float('nan') → None so all optional numeric
        fields are always JSON-serialisable (json.dumps raises on NaN).
        """
        return MatchResult(
            rank             = rank_0 + 1,
            score            = float(score),
            job_idx          = embed_idx,
            job_id           = str(meta.get("job_id",           "") or ""),
            title            = str(meta.get("title",            "") or ""),
            company          = str(meta.get("company",          "") or ""),
            location         = str(meta.get("location",         "") or ""),
            experience_level = str(meta.get("experience_level", "") or ""),
            work_type        = str(meta.get("work_type",        "") or ""),
            remote_allowed   = _safe_float(meta.get("remote_allowed")),
            salary_min       = _safe_float(meta.get("salary_min")),
            salary_max       = _safe_float(meta.get("salary_max")),
            skill_gap        = skill_gap,
        )

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

        query_vec       = self._embed(resume_text)
        scores, indices = self.index.search(query_vec, top_k)

        jobs_df = self._load_jobs_df() if enable_skill_gap else None

        results: List[MatchResult] = []
        for rank_0, (idx, score) in enumerate(zip(indices[0], scores[0])):
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
                except Exception as e:
                    print(f"  [SkillGap] Error for result #{rank_0 + 1}: {type(e).__name__}: {e}")
                    traceback.print_exc()

            results.append(self._build_match_result(rank_0, embed_idx, score, meta, skill_gap))

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

        Skill gap is disabled by default — running it per (resume × job) pair
        is expensive (one Gemini call each).  Enable only when needed.

        Empty or whitespace-only strings in resume_texts are validated before
        encoding; invalid entries produce an empty inner list rather than being
        silently passed to SBERT and returning garbage matches.

        Args:
            resume_texts     : List of resume text strings.
            top_k            : Number of results per resume.
            show_progress    : Show SBERT encoding progress bar.
            enable_skill_gap : Run skill gap for every result (slow).
            enable_ai        : Use Gemini Stage 2 inside skill gap.

        Returns:
            List of lists — one inner list of MatchResult per input resume.
            Invalid (empty/whitespace) resume entries produce an empty inner list.
        """
        if not resume_texts:
            return []

        # Validate each entry before touching SBERT.
        valid_mask  : List[bool] = [bool(t and t.strip()) for t in resume_texts]
        valid_texts : List[str]  = [t for t, ok in zip(resume_texts, valid_mask) if ok]

        if not valid_texts:
            return [[] for _ in resume_texts]

        # Batch-encode all valid resumes in one SBERT call (more efficient than looping).
        vecs = self.model.encode(
            valid_texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=show_progress,
        ).astype("float32")
        faiss.normalize_L2(vecs)

        all_scores, all_indices = self.index.search(vecs, top_k)

        jobs_df = self._load_jobs_df() if enable_skill_gap else None

        valid_results: List[List[MatchResult]] = []

        for v_idx, resume_text in enumerate(valid_texts):
            results: List[MatchResult] = []
            for rank_0, (idx, score) in enumerate(
                zip(all_indices[v_idx], all_scores[v_idx])
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
                    except Exception as e:
                        print(
                            f"  [SkillGap Batch] Error for resume #{v_idx + 1}, "
                            f"result #{rank_0 + 1}: {type(e).__name__}: {e}"
                        )
                        traceback.print_exc()

                results.append(self._build_match_result(rank_0, embed_idx, score, meta, skill_gap))

            valid_results.append(results)

        # Reconstruct the full output list, inserting [] at invalid positions.
        all_results: List[List[MatchResult]] = []
        valid_iter = iter(valid_results)
        for ok in valid_mask:
            all_results.append(next(valid_iter) if ok else [])

        return all_results

    # ── Display ───────────────────────────────────────────────────────────────

    @staticmethod
    def print_results(results: List[MatchResult], resume_label: str = "Resume") -> None:
        print(f"\n{'─'*70}")
        print(f"  Top {len(results)} Matches for: {resume_label}")
        print(f"{'─'*70}")

        for r in results:
            # Salary string — always safe because salary fields are float or None
            salary_str = ""
            if r.salary_min is not None and r.salary_max is not None:
                salary_str = f"  |  ${r.salary_min:,.0f}–${r.salary_max:,.0f}"

            remote_str = "  [Remote OK]" if r.remote_allowed == 1.0 else ""

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

    Optimized application performance, reduced API response time by 35%, and implemented secure
    authentication systems (JWT, OAuth).
    """

    print("=" * 70)
    print("  MATCHING ENGINE — STANDALONE TEST")
    print("=" * 70)

    engine = MatchingEngine()

    print("\n[Test 1] Basic match (no skill gap, fast)...")
    r1 = engine.match(SAMPLE, top_k=3, enable_skill_gap=False)
    engine.print_results(r1, "Python Developer — Basic")

    print("\n[Test 2] With skill gap (structured only, no AI)...")
    r2 = engine.match(SAMPLE, top_k=3, enable_skill_gap=True, enable_ai=False)
    engine.print_results(r2, "Python Developer — Structured Gap")

    print("\n[Test 3] Full pipeline (structured + AI reasoning)...")
    r3 = engine.match(SAMPLE, top_k=3, enable_skill_gap=True, enable_ai=True)
    engine.print_results(r3, "Python Developer — Full AI Analysis")
