"""
skill_analyzer.py
=================
Phase 3 — AI-Powered Skill Gap Analysis

Industry-standard, two-stage skill gap analysis pipeline:

STAGE 1 — STRUCTURED EXTRACTION (spaCy + Taxonomy + SBERT)
  Extracts skills from both resume and job text using:
  - spaCy NER for candidate skill phrase detection
  - 400+ skill taxonomy with exact + substring + semantic matching
  - SBERT cosine similarity for synonym resolution (k8s → kubernetes, etc.)
  Produces deterministic, structured skill sets — fast and reproducible.

STAGE 2 — LLM REASONING (Gemini via Google Generative AI API)
  Sends the resume, job description, AND the structured extraction to Gemini.
  Gemini does what no keyword algorithm can:
  - Determines which missing skills are dealbreakers vs nice-to-have
  - Assesses whether existing skills partially compensate for gaps
  - Gives a hiring manager's perspective on the candidacy
  - Provides a prioritized, actionable learning path
  - Identifies transferable skills the taxonomy may have missed
  - Returns structured JSON — not free text — so the UI can render properly

Why this architecture?
  Structured extraction gives us speed, reproducibility, and structured data.
  LLM reasoning gives us contextual intelligence that hardcoded rules cannot.
  Together they match what LinkedIn's AI Recruiter, Workday Skills Cloud,
  and enterprise ATS systems do under the hood.

Output: SkillGapResult dataclass — fully structured, JSON-serializable,
  UI-ready with both raw skill data and AI-generated insights.

Dependencies:
    pip install spacy sentence-transformers google-genai python-dotenv
    python -m spacy download en_core_web_sm
"""

import re
import os
import json
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

# ── spaCy ─────────────────────────────────────────────────────────────────────
try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False

# ── sentence-transformers ─────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False

# ── Google Gemini ──────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

try:
    from google import genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False
    print("WARNING: google-genai package not installed. Run: pip install google-genai")


# ══════════════════════════════════════════════════════════════════════════════
# SKILL TAXONOMY  (400+ skills across 9 categories)
# ══════════════════════════════════════════════════════════════════════════════

SKILL_TAXONOMY: Dict[str, List[str]] = {

    "programming_languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "golang", "go",
        "rust", "kotlin", "swift", "scala", "r", "matlab", "perl", "ruby", "php",
        "dart", "julia", "lua", "haskell", "erlang", "elixir", "clojure", "groovy",
        "objective-c", "assembly", "cobol", "fortran", "vba", "bash", "shell script",
        "powershell", "sql", "pl/sql", "t-sql",
    ],

    "web_frontend": [
        "react", "react.js", "angular", "angularjs", "vue", "vue.js", "next.js",
        "nuxt.js", "svelte", "html", "css", "sass", "scss", "less", "tailwind",
        "bootstrap", "material ui", "jquery", "webpack", "vite", "babel",
        "redux", "mobx", "graphql", "rest", "restful", "soap", "websockets",
        "responsive design", "web components", "pwa", "d3.js", "three.js",
    ],

    "web_backend": [
        "node.js", "express", "fastapi", "flask", "django", "spring", "spring boot",
        "spring mvc", "hibernate", "jpa", "j2ee", "asp.net", ".net", "laravel",
        "rails", "ruby on rails", "fastify", "nestjs", "grpc", "microservices",
        "rest api", "graphql api", "oauth", "jwt", "api gateway",
        "message queue", "rabbitmq", "kafka", "celery", "nginx", "apache",
    ],

    "databases": [
        "postgresql", "mysql", "sqlite", "oracle", "sql server", "mariadb",
        "mongodb", "redis", "elasticsearch", "cassandra", "dynamodb", "couchdb",
        "neo4j", "influxdb", "clickhouse", "snowflake", "bigquery", "redshift",
        "hbase", "firebase", "supabase", "prisma", "sqlalchemy",
        "database design", "query optimization", "stored procedures", "indexing",
        "nosql", "relational database", "data modeling", "etl",
    ],

    "cloud_devops": [
        "aws", "amazon web services", "gcp", "google cloud", "azure",
        "docker", "kubernetes", "terraform", "ansible", "chef devops", "puppet",
        "jenkins", "github actions", "gitlab ci", "circleci", "travis ci",
        "ci/cd", "devops", "infrastructure as code", "helm", "istio",
        "lambda", "serverless", "ec2", "s3", "rds", "ecs", "eks",
        "load balancer", "cdn", "cloudflare", "heroku", "digitalocean", "vercel",
        "prometheus", "grafana", "datadog", "splunk", "elk stack",
        "linux", "unix", "git", "github", "gitlab", "version control",
        "agile", "scrum", "kanban", "jira", "confluence",
    ],

    "data_ai_ml": [
        "machine learning", "deep learning", "neural networks",
        "natural language processing", "nlp", "computer vision",
        "reinforcement learning", "supervised learning", "unsupervised learning",
        "classification", "regression", "clustering",
        "tensorflow", "pytorch", "keras", "scikit-learn", "xgboost",
        "lightgbm", "catboost", "hugging face", "transformers", "bert", "gpt",
        "llm", "large language model", "fine-tuning", "rag", "langchain",
        "pandas", "numpy", "matplotlib", "seaborn", "plotly", "jupyter",
        "data science", "data analysis", "data engineering", "feature engineering",
        "model deployment", "mlops", "data pipeline", "apache spark", "hadoop",
        "pyspark", "airflow", "dbt", "data warehouse", "data lake",
        "tableau", "power bi", "looker", "statistics", "a/b testing",
        "faiss", "vector database", "embedding", "sentence transformers",
    ],

    "mobile": [
        "android", "ios", "react native", "flutter", "dart", "swift", "kotlin",
        "objective-c", "xamarin", "ionic", "mobile development",
        "firebase", "push notifications",
    ],

    "security_networking": [
        "cybersecurity", "network security", "penetration testing", "ethical hacking",
        "vulnerability assessment", "siem", "splunk", "wireshark", "nmap",
        "firewall", "vpn", "ssl", "tls", "encryption", "oauth",
        "soc", "incident response", "threat hunting", "malware analysis",
        "cissp", "ceh", "security+", "tcp/ip", "dns",
        "zero trust", "identity management", "iam", "ldap", "active directory",
    ],

    "culinary_hospitality": [
        "chef", "head chef", "sous chef", "menu planning", "menu design",
        "food preparation", "kitchen management", "kitchen operations", "inventory control",
        "food costing", "portion sizing", "supplier coordination", "food safety", "haccp",
        "italian cuisine", "continental cuisine", "baking", "pastry", "fine dining",
        "banquet management", "catering", "restaurant operations", "staff training",
    ],
}

# Canonical alias map — resolves abbreviations and common synonyms
SKILL_ALIASES: Dict[str, str] = {
    "py"              : "python",
    "js"              : "javascript",
    "ts"              : "typescript",
    "k8s"             : "kubernetes",
    "node"            : "node.js",
    "react js"        : "react",
    "angular js"      : "angular",
    "ml"              : "machine learning",
    "dl"              : "deep learning",
    "cv"              : "computer vision",
    "nlp"             : "natural language processing",
    "postgres"        : "postgresql",
    "mongo"           : "mongodb",
    "elastic"         : "elasticsearch",
    "tf"              : "tensorflow",
    "sklearn"         : "scikit-learn",
    "hf"              : "hugging face",
    "ci cd"           : "ci/cd",
    "ci-cd"           : "ci/cd",
    "aws cloud"       : "aws",
    "dotnet"          : ".net",
    "dot net"         : ".net",
    "asp net"         : "asp.net",
    "spring framework": "spring",
    "spring-boot"     : "spring boot",
    "jee"             : "j2ee",
    "iac"             : "infrastructure as code",
    "gh actions"      : "github actions",
    "llms"            : "large language model",
    "gpt-4"           : "large language model",
    "openai api"      : "large language model",
    "sql db"          : "sql",
    "rdbms"           : "relational database",
    "haccp compliance" : "haccp",
    "food safety standards": "food safety",
    "menu design"     : "menu planning",
}

# Build flat index and reverse map
ALL_SKILLS_FLAT: List[str] = []
SKILL_TO_CATEGORY: Dict[str, str] = {}

for _cat, _skills in SKILL_TAXONOMY.items():
    for _sk in _skills:
        ALL_SKILLS_FLAT.append(_sk)
        SKILL_TO_CATEGORY[_sk] = _cat

# ══════════════════════════════════════════════════════════════════════════════
# RESULT DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StructuredSkillSets:
    """
    Raw structured skill sets extracted from resume and job.
    Output of Stage 1 (deterministic extraction).
    """
    resume_skills       : Dict[str, List[str]]   # {category: [skills]}
    job_skills          : Dict[str, List[str]]   # {category: [skills]}
    matched             : Dict[str, List[str]]   # skills in both
    missing             : Dict[str, List[str]]   # job requires, resume lacks
    extra               : Dict[str, List[str]]   # resume has, job doesn't need
    unrecognized_resume : List[str]              # NER found, not in taxonomy
    unrecognized_job    : List[str]
    overlap_score       : float                  # Jaccard overlap 0-1

    def flat_matched(self) -> List[str]:
        return [s for v in self.matched.values() for s in v]

    def flat_missing(self) -> List[str]:
        return [s for v in self.missing.values() for s in v]

    def flat_extra(self) -> List[str]:
        return [s for v in self.extra.values() for s in v]


@dataclass
class AISkillInsight:
    """
    AI-generated insight for a single skill or skill group.
    Generated by Gemini in Stage 2.
    """
    skill            : str
    importance       : str    # "critical" / "important" / "nice_to_have"
    is_dealbreaker   : bool
    compensation     : str    # How resume skills partially compensate (if any)
    learning_priority: int    # 1 = learn first, higher = lower priority


@dataclass
class SkillGapResult:
    """
    Complete skill gap analysis result — combines structured extraction
    with AI-generated reasoning.

    This is the final output of the full pipeline and what gets attached
    to each MatchResult in the matching engine.
    """
    # Stage 1: Structured extraction (always present)
    structured          : StructuredSkillSets

    # Stage 2: AI analysis (None if API call failed or was skipped)
    ai_available        : bool              = False

    # AI-generated fields (populated if ai_available=True)
    candidacy_verdict   : str               = ""
    verdict_confidence  : str               = ""
    executive_summary   : str               = ""
    dealbreaker_skills  : List[str]         = field(default_factory=list)
    compensatable_gaps  : List[str]         = field(default_factory=list)
    transferable_skills : List[str]         = field(default_factory=list)
    skill_insights      : List[AISkillInsight] = field(default_factory=list)
    learning_path       : List[str]         = field(default_factory=list)
    time_to_ready       : str               = ""
    strengths           : List[str]         = field(default_factory=list)
    hiring_risks        : List[str]         = field(default_factory=list)

    # ── Convenience properties (delegate to structured) ──────────────────────

    @property
    def overlap_score(self) -> float:
        return self.structured.overlap_score

    @property
    def gap_severity(self) -> str:
        if self.ai_available and self.candidacy_verdict:
            return {
                "strong_fit"  : "low",
                "moderate_fit": "medium",
                "weak_fit"    : "high",
            }.get(self.candidacy_verdict, "medium")
        # Fallback: derive from Jaccard overlap
        s = self.structured.overlap_score
        if s >= 0.60:
            return "low"
        if s >= 0.35:
            return "medium"
        return "high"

    def flat_matched(self) -> List[str]:
        return self.structured.flat_matched()

    def flat_missing(self) -> List[str]:
        return self.structured.flat_missing()

    def flat_extra(self) -> List[str]:
        return self.structured.flat_extra()

    def to_dict(self) -> dict:
        return {
            # ── Structured (always present) ──────────────────────────────────
            "resume_skills"       : self.structured.resume_skills,
            "job_skills"          : self.structured.job_skills,
            "matched_skills"      : self.structured.matched,
            "missing_skills"      : self.structured.missing,
            "extra_skills"        : self.structured.extra,
            "unrecognized_resume" : self.structured.unrecognized_resume,
            "unrecognized_job"    : self.structured.unrecognized_job,
            "overlap_score"       : round(self.overlap_score, 4),
            "gap_severity"        : self.gap_severity,
            # ── AI analysis ──────────────────────────────────────────────────
            "ai_available"        : self.ai_available,
            "candidacy_verdict"   : self.candidacy_verdict,
            "verdict_confidence"  : self.verdict_confidence,
            "executive_summary"   : self.executive_summary,
            "dealbreaker_skills"  : self.dealbreaker_skills,
            "compensatable_gaps"  : self.compensatable_gaps,
            "transferable_skills" : self.transferable_skills,
            "learning_path"       : self.learning_path,
            "time_to_ready"       : self.time_to_ready,
            "strengths"           : self.strengths,
            "hiring_risks"        : self.hiring_risks,
            "skill_insights"      : [
                {
                    "skill"            : s.skill,
                    "importance"       : s.importance,
                    "is_dealbreaker"   : s.is_dealbreaker,
                    "compensation"     : s.compensation,
                    "learning_priority": s.learning_priority,
                }
                for s in self.skill_insights
            ],
        }


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — STRUCTURED SKILL EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

class StructuredSkillExtractor:
    """
    Deterministic, fast skill extractor using spaCy NER + taxonomy matching.

        Three-stage matching pipeline per candidate:
            1. Exact match against taxonomy
            2. Substring match (handles "postgresql database" → "postgresql")
            3. SBERT semantic similarity ≥ 0.72 (handles synonyms/abbreviations)

    Plus a direct keyword scan as a safety net for common skills that
    NER might miss (git, sql, etc.).
    """

    SEMANTIC_THRESHOLD = 0.78

    def __init__(self, enable_semantic: bool = True):
        self.enable_semantic = enable_semantic and _SBERT_AVAILABLE

        # Load spaCy — required for NER candidate extraction
        if _SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                raise OSError(
                    "spaCy model not found. Run: python -m spacy download en_core_web_sm"
                )
        else:
            self.nlp = None

        # Pre-encode the full taxonomy once for fast semantic similarity lookup
        if self.enable_semantic:
            self._sbert = SentenceTransformer("all-MiniLM-L6-v2")
            self._skill_embeddings = self._sbert.encode(
                ALL_SKILLS_FLAT,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            self._sbert = None
            self._skill_embeddings = None

    # ── Public interface ──────────────────────────────────────────────────────

    def extract(self, text: str) -> Tuple[Dict[str, List[str]], List[str]]:
        """
        Extract skills from text.

        Returns:
            (skills_by_category, unrecognized_candidates)
            where skills_by_category is {category_name: [skill, ...]}
        """
        if not text or not text.strip():
            return {}, []

        candidates   = self._get_candidates(text)
        matched      : Dict[str, List[str]] = {}
        unrecognized : List[str] = []

        for cand in candidates:
            cat, skill = self._match(cand)
            if cat and skill:
                matched.setdefault(cat, [])
                if skill not in matched[cat]:
                    matched[cat].append(skill)
            elif len(cand) > 2:
                unrecognized.append(cand)

        # Safety-net: direct keyword scan catches skills NER misses (e.g. "git", "sql")
        for cat, skills in self._keyword_scan(text).items():
            matched.setdefault(cat, [])
            for s in skills:
                if s not in matched[cat]:
                    matched[cat].append(s)

        # Clean unrecognized: remove anything that's actually in taxonomy
        unrecognized = list(set(
            u for u in unrecognized
            if not any(u in sk or sk in u for sk in ALL_SKILLS_FLAT if len(sk) >= 3)
        ))[:15]

        return matched, unrecognized

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_candidates(self, text: str) -> List[str]:
        """Use spaCy NER + noun chunks + tokens to build candidate skill phrases."""
        if not self.nlp:
            # Fallback: simple word split when spaCy is unavailable
            return list(set(text.lower().split()))

        doc   = self.nlp(text[:12000])
        cands = set()

        # Named entities (ORG/PRODUCT are the most useful for tech skills)
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PRODUCT", "GPE", "WORK_OF_ART"):
                c = ent.text.lower().strip()
                if 2 < len(c) < 50:
                    cands.add(SKILL_ALIASES.get(c, c))

        # Noun chunks (e.g. "machine learning engineer" → "machine learning")
        for chunk in doc.noun_chunks:
            for candidate in (chunk.root.text.lower().strip(), chunk.text.lower().strip()):
                if 2 < len(candidate) < 60:
                    cands.add(SKILL_ALIASES.get(candidate, candidate))

        # Individual tokens — proper nouns, uppercase acronyms, and plain nouns
        for tok in doc:
            t  = tok.text.strip()
            tl = t.lower()
            if (
                2 < len(t) < 30
                and not tok.is_stop
                and not tok.is_punct
                and (tok.pos_ in ("PROPN", "NOUN") or t.isupper())
            ):
                cands.add(SKILL_ALIASES.get(tl, tl))

        return list(cands)

    def _match(self, candidate: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Attempt to match a candidate phrase to a known taxonomy skill.

        Matching order (fast-to-slow):
          1. Exact match
          2. Substring match (handles "postgresql database" → "postgresql")
          3. Semantic similarity for synonym resolution
        """
        if not candidate or len(candidate) < 2:
            return None, None

        cl = candidate.lower().strip()

        # 1. Exact match
        if cl in SKILL_TO_CATEGORY:
            return SKILL_TO_CATEGORY[cl], cl

        # Substring
        # Only allow taxonomy-skill-in-candidate direction to avoid noisy
        # matches like "design" -> "responsive design" and "restaurant" -> "rest".
        for sk in ALL_SKILLS_FLAT:
            if len(sk) >= 5 and sk in cl:
                return SKILL_TO_CATEGORY[sk], sk

        # Semantic
        if self.enable_semantic and self._skill_embeddings is not None:
            try:
                vec = self._sbert.encode(
                    [cl],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                sims     = vec @ self._skill_embeddings.T
                best_idx = int(np.argmax(sims))
                best_score = float(sims[0, best_idx])
                # Require a clear winner over the second-best skill to reduce
                # noisy cross-domain matches on non-technical text.
                if sims.shape[1] > 1:
                    second_best = float(np.partition(sims[0], -2)[-2])
                else:
                    second_best = 0.0

                if best_score >= self.SEMANTIC_THRESHOLD and (best_score - second_best) >= 0.08:
                    sk = ALL_SKILLS_FLAT[best_idx]
                    return SKILL_TO_CATEGORY[sk], sk
            except Exception:
                pass

        return None, None

    def _keyword_scan(self, text: str) -> Dict[str, List[str]]:
        tl = text.lower()
        found: Dict[str, List[str]] = {}
        for sk in ALL_SKILLS_FLAT:
            escaped = re.escape(sk)
            # For symbol-heavy skills (c++, c#, .net, ci/cd), word boundaries are too strict.
            if re.search(rf'(?<!\w){escaped}(?!\w)', tl):
                cat = SKILL_TO_CATEGORY[sk]
                found.setdefault(cat, [])
                if sk not in found[cat]:
                    found[cat].append(sk)
        return found


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — AI REASONING ENGINE (Gemini)
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """You are a senior technical recruiter and career advisor with 15 years of \
experience evaluating software engineering candidates. You have deep expertise in:
- Reading between the lines of resumes to find hidden strengths
- Distinguishing dealbreaker gaps from easily bridgeable ones
- Understanding which skills are truly required vs. listed as aspirational
- Providing actionable, honest career guidance

You always respond with valid JSON only — no markdown, no explanation outside the JSON.
Your analysis is honest, specific, and actionable — never generic."""

_ANALYSIS_PROMPT = """Analyze the skill fit between this candidate and job. You have been given:
1. The full resume text
2. The full job description
3. Pre-extracted structured skill sets (from a taxonomy system)

Your task: produce a deep, honest assessment that a senior hiring manager would find useful.

=== RESUME TEXT ===
{resume_text}

=== JOB DESCRIPTION ===
{job_text}

=== PRE-EXTRACTED SKILLS (structured, use as reference) ===
Resume skills: {resume_skills}
Job requirements: {job_skills}
Matched skills: {matched}
Missing skills: {missing}
Extra skills resume brings: {extra}
Unrecognized terms in resume: {unrecognized_resume}
Unrecognized terms in job: {unrecognized_job}
Raw overlap score: {overlap_score:.2%}

=== YOUR ANALYSIS TASK ===
Return a JSON object with EXACTLY these fields:

{{
  "candidacy_verdict": "<string: exactly one of: strong_fit, moderate_fit, weak_fit>",
  "verdict_confidence": "<string: exactly one of: high, medium, low>",

  "executive_summary": "<string: 2-3 sentences from a hiring manager's perspective. \
Be direct and specific. Mention the strongest qualification and the biggest risk.>",

  "dealbreaker_skills": [
    "<list of strings: skills in the job requirements that are TRULY required and the candidate lacks. \
Only include real dealbreakers — not nice-to-haves. Can be empty list.>"
  ],

  "compensatable_gaps": [
    "<list of strings: skills the candidate lacks but could compensate for with existing experience. \
Explain HOW they compensate, e.g. 'Lacks Kafka but has RabbitMQ — same messaging paradigm'>"
  ],

  "transferable_skills": [
    "<list of strings: skills the candidate has that weren't in the job description but add value. \
Include non-obvious transfers, e.g. 'Spring Boot to FastAPI: OOP patterns transfer directly'>"
  ],

  "strengths": [
    "<list of 3-5 strings: the candidate's strongest qualifications FOR THIS SPECIFIC JOB. \
Be specific — don't just list skills, explain why they matter for this role.>"
  ],

  "hiring_risks": [
    "<list of 2-4 strings: concrete risks a hiring manager would note. Be honest. \
E.g., 'No production ML deployment experience despite strong modeling skills'>"
  ],

  "skill_insights": [
    {{
      "skill": "<skill name>",
      "importance": "<critical | important | nice_to_have>",
      "is_dealbreaker": <true | false>,
      "compensation": "<how existing skills compensate, or empty string if no compensation>",
      "learning_priority": <integer 1-10, where 1 = learn immediately>
    }}
  ],

  "learning_path": [
    "<ordered list of 3-5 strings: specific, actionable steps to close the most important gaps. \
Include resource types (course, project, certification) not just skill names. \
E.g., 'Build a Kafka consumer/producer project using existing Java skills (1-2 weeks)'>"
  ],

  "time_to_ready": "<string: realistic estimate for candidate to be ready if gaps exist. \
E.g., '2-3 months with focused study' or 'Ready now' or 'Not viable without career change'>"
}}

Rules:
- skill_insights should cover the TOP 5 missing skills only (most important ones)
- Be specific to THIS resume and THIS job — no generic statements
- If the candidate is actually a strong fit, say so clearly and explain why
- If there are serious gaps, be honest about whether they are bridgeable
- The unrecognized terms may contain real skills not in our taxonomy — consider them"""


class AISkillReasoningEngine:
    """
    Sends structured skill extraction to Gemini for deep reasoning.

    Tries multiple Gemini model names for forward compatibility.
    Returns an empty dict (graceful degradation) if the API is unavailable.
    """

    # Ordered list of model candidates — first valid one wins
    MODEL_CANDIDATES = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-flash-latest",
    ]

    def __init__(self):
        self._client = None

        if not _GEMINI_AVAILABLE:
            return

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("  [SkillAnalyzer] No GEMINI_API_KEY / GOOGLE_API_KEY found — AI stage disabled.")
            return

        try:
            self._client = genai.Client(api_key=api_key)
        except Exception as e:
            print(f"  [SkillAnalyzer] Gemini client init error: {e}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _extract_json(self, raw: str) -> Dict[str, Any]:
        """Strip optional markdown fences and parse JSON."""
        text = (raw or "").strip()
        if not text:
            return {}
        # Remove ```json ... ``` fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$",          "", text)
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _generate_with_fallback(self, prompt: str) -> Dict[str, Any]:
        """Try each model in MODEL_CANDIDATES and return the first valid JSON response."""
        saw_not_found = False
        saw_quota_or_rate = False
        saw_service_unavailable = False
        saw_other_error = False

        for model_name in self.MODEL_CANDIDATES:
            try:
                response = self._client.models.generate_content(
                    model=model_name,
                    contents=f"{_SYSTEM_PROMPT}\n\n{prompt}",
                )
                raw    = getattr(response, "text", "") or ""
                parsed = self._extract_json(raw)
                if parsed:
                    return parsed
            except Exception as e:
                msg = str(e).lower()
                # Keep trying on model-availability errors
                if any(k in msg for k in ("not_found", "not found", "not supported", "404")):
                    saw_not_found = True
                    continue
                # Keep trying on transient service/rate-limit issues.
                if any(k in msg for k in (
                    "503", "502", "500", "504", "429", "unavailable",
                    "resource exhausted", "rate", "deadline", "timeout", "temporar"
                )):
                    if any(k in msg for k in ("429", "resource exhausted", "quota", "rate")):
                        saw_quota_or_rate = True
                    else:
                        saw_service_unavailable = True
                    continue

                # Non-recoverable errors: log and bail
                saw_other_error = True
                continue

        if saw_quota_or_rate:
            print("  [SkillAnalyzer] AI skipped for this job: Gemini quota or rate limit reached.")
        elif saw_service_unavailable:
            print("  [SkillAnalyzer] AI skipped for this job: Gemini service is temporarily unavailable.")
        elif saw_not_found:
            print("  [SkillAnalyzer] AI skipped for this job: no compatible Gemini model is available for this key.")
        elif saw_other_error:
            print("  [SkillAnalyzer] AI skipped for this job: Gemini request failed this run.")
        else:
            print("  [SkillAnalyzer] AI skipped for this job: no valid response returned by Gemini.")
        return {}

    # ── Public interface ──────────────────────────────────────────────────────

    def analyze(
        self,
        resume_text : str,
        job_text    : str,
        structured  : StructuredSkillSets,
    ) -> Dict[str, Any]:
        """
        Call Gemini to produce AI-powered skill gap reasoning.

        Returns a dict with all AI fields, or {} if unavailable.
        """
        if self._client is None:
            return {}

        # Cap inputs at 4 000 chars each — most signal is in the first portion
        prompt = _ANALYSIS_PROMPT.format(
            resume_text         = resume_text.strip()[:4000],
            job_text            = job_text.strip()[:4000],
            resume_skills       = json.dumps(structured.resume_skills),
            job_skills          = json.dumps(structured.job_skills),
            matched             = json.dumps(structured.flat_matched()),
            missing             = json.dumps(structured.flat_missing()),
            extra               = json.dumps(structured.flat_extra()),
            unrecognized_resume = json.dumps(structured.unrecognized_resume),
            unrecognized_job    = json.dumps(structured.unrecognized_job),
            overlap_score       = structured.overlap_score,
        )

        try:
            return self._generate_with_fallback(prompt)
        except Exception as e:
            print(f"  [SkillAnalyzer] Unexpected error during AI analysis: {type(e).__name__}: {e}")
            return {}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE — SkillAnalyzer
# ══════════════════════════════════════════════════════════════════════════════

class SkillAnalyzer:
    """
    Full two-stage skill gap analysis pipeline.

    Stage 1: StructuredSkillExtractor  — fast, deterministic, always runs.
    Stage 2: AISkillReasoningEngine    — deep Gemini reasoning, runs when API available.

    Usage:
        analyzer = SkillAnalyzer()
        result   = analyzer.analyze(resume_text, job_text)

        # Always available:
        result.structured.flat_matched()   → ["python", "aws", "docker"]
        result.structured.flat_missing()   → ["kafka", "spark", "kubernetes"]
        result.overlap_score               → 0.42

        # Available when AI ran (result.ai_available == True):
        result.executive_summary
        result.dealbreaker_skills
        result.learning_path
        result.candidacy_verdict
    """

    def __init__(self, enable_semantic: bool = True, enable_ai: bool = True):
        print("  Loading SkillAnalyzer (Stage 1: spaCy + SBERT)...")
        self._extractor = StructuredSkillExtractor(enable_semantic=enable_semantic)

        if enable_ai:
            print("  Loading AISkillReasoningEngine (Stage 2: Gemini API)...")
            self._ai: Optional[AISkillReasoningEngine] = AISkillReasoningEngine()
        else:
            self._ai = None

        print("  SkillAnalyzer ready.")

    def analyze(
        self,
        resume_text : str,
        job_text    : str,
        enable_ai   : bool = True,
    ) -> SkillGapResult:
        """
        Run the full pipeline: extract → compute gap → AI reasoning.

        Args:
            resume_text : resume text (raw or cleaned)
            job_text    : job description text (raw or cleaned)
            enable_ai   : set False to skip Stage 2 (faster, no API cost)

        Returns:
            SkillGapResult — always contains structured data.
                             AI fields populated only when API call succeeds.
        """
        # ── Stage 1: Structured Extraction ───────────────────────────────────
        resume_skills, unrec_resume = self._extractor.extract(resume_text)
        job_skills,    unrec_job    = self._extractor.extract(job_text)

        # Set-algebra per category to compute matched / missing / extra
        all_cats = set(list(resume_skills.keys()) + list(job_skills.keys()))
        matched : Dict[str, List[str]] = {}
        missing : Dict[str, List[str]] = {}
        extra   : Dict[str, List[str]] = {}

        for cat in all_cats:
            r = set(resume_skills.get(cat, []))
            j = set(job_skills.get(cat, []))
            if r & j:
                matched[cat] = sorted(r & j)
            if j - r:
                missing[cat] = sorted(j - r)
            if r - j:
                extra[cat]   = sorted(r - j)

        # Jaccard overlap score
        n_res  = sum(len(v) for v in resume_skills.values())
        n_job  = sum(len(v) for v in job_skills.values())
        n_both = sum(len(v) for v in matched.values())
        union  = n_res + n_job - n_both
        overlap = n_both / union if union > 0 else 0.0

        structured = StructuredSkillSets(
            resume_skills       = resume_skills,
            job_skills          = job_skills,
            matched             = matched,
            missing             = missing,
            extra               = extra,
            unrecognized_resume = unrec_resume,
            unrecognized_job    = unrec_job,
            overlap_score       = overlap,
        )

        # ── Stage 2: AI Reasoning ─────────────────────────────────────────────
        ai_data     : Dict[str, Any] = {}
        ai_available: bool           = False

        if enable_ai and self._ai is not None:
            ai_data      = self._ai.analyze(resume_text, job_text, structured)
            ai_available = bool(ai_data)

        # Parse AI output into typed fields
        skill_insights: List[AISkillInsight] = []
        for si in ai_data.get("skill_insights", []):
            try:
                skill_insights.append(AISkillInsight(
                    skill             = str(si.get("skill", "")),
                    importance        = str(si.get("importance", "important")),
                    is_dealbreaker    = bool(si.get("is_dealbreaker", False)),
                    compensation      = str(si.get("compensation", "")),
                    learning_priority = int(si.get("learning_priority", 5)),
                ))
            except Exception:
                pass

        return SkillGapResult(
            structured          = structured,
            ai_available        = ai_available,
            candidacy_verdict   = str(ai_data.get("candidacy_verdict",  "")),
            verdict_confidence  = str(ai_data.get("verdict_confidence", "")),
            executive_summary   = str(ai_data.get("executive_summary",  "")),
            dealbreaker_skills  = list(ai_data.get("dealbreaker_skills",  [])),
            compensatable_gaps  = list(ai_data.get("compensatable_gaps",  [])),
            transferable_skills = list(ai_data.get("transferable_skills", [])),
            skill_insights      = skill_insights,
            learning_path       = list(ai_data.get("learning_path",      [])),
            time_to_ready       = str(ai_data.get("time_to_ready",       "")),
            strengths           = list(ai_data.get("strengths",           [])),
            hiring_risks        = list(ai_data.get("hiring_risks",        [])),
        )


# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON  (avoids reloading heavy models on every call)
# ══════════════════════════════════════════════════════════════════════════════

_default_analyzer: Optional[SkillAnalyzer] = None


def get_analyzer(enable_semantic: bool = True, enable_ai: bool = True) -> SkillAnalyzer:
    """
    Return a module-level singleton SkillAnalyzer.

    If the cached instance was created with lower capabilities than requested
    (e.g. AI was disabled, now requested), the instance is recreated.
    """
    global _default_analyzer

    if _default_analyzer is None:
        _default_analyzer = SkillAnalyzer(
            enable_semantic=enable_semantic,
            enable_ai=enable_ai,
        )
        return _default_analyzer

    needs_ai_upgrade = enable_ai and (_default_analyzer._ai is None)
    needs_sem_upgrade = (
        enable_semantic and not _default_analyzer._extractor.enable_semantic
    )
    if needs_ai_upgrade or needs_sem_upgrade:
        _default_analyzer = SkillAnalyzer(
            enable_semantic=enable_semantic,
            enable_ai=enable_ai,
        )

    return _default_analyzer


def analyze_skill_gap(
    resume_text: str,
    job_text   : str,
    enable_ai  : bool = True,
) -> SkillGapResult:
    """Convenience function — uses module-level singleton."""
    return get_analyzer(enable_ai=enable_ai).analyze(resume_text, job_text, enable_ai=enable_ai)


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    RESUME = """
    Professional chef with 6 years of experience in high-end restaurants and hotel kitchens.
    Skilled in menu planning, food preparation, kitchen management, and inventory control.
    Expert in Italian and Continental cuisine with strong knowledge of baking and pastry.

    Worked in fast-paced environments handling 200+ covers per service.
    Managed kitchen staff of 8-12 people and ensured food safety standards (HACCP compliance).
    Familiar with cost control, portion sizing, and supplier coordination.
    """

    JOB = """
    Head Chef at Gourmet Fusion Restaurant

    We are looking for a Head Chef to lead our kitchen operations.

    Required:
    - 5+ years experience in professional kitchens
    - Expertise in multi-cuisine (Italian, French, Asian fusion preferred)
    - Strong leadership and kitchen team management
    - Menu design and seasonal planning
    - Knowledge of food costing and inventory management
    - HACCP and food safety compliance

    Nice to have:
    - Experience in Michelin-star or fine dining restaurants
    - Event catering experience
    - Experience training junior chefs
    """

    print("=" * 70)
    print("  SKILL ANALYZER — FULL PIPELINE TEST")
    print("=" * 70)

    analyzer = SkillAnalyzer(enable_semantic=True, enable_ai=True)

    print("\nRunning analysis...")
    t0      = time.time()
    result  = analyzer.analyze(RESUME, JOB)
    elapsed = time.time() - t0

    print(f"\n  Total analysis time: {elapsed:.1f}s")
    print(f"\n{'─'*60}")
    print("  STAGE 1 — Structured Extraction")
    print(f"{'─'*60}")
    n_res_skills = sum(len(v) for v in result.structured.resume_skills.values())
    n_job_skills = sum(len(v) for v in result.structured.job_skills.values())
    print(f"  Resume skills  : {n_res_skills} skills across {len(result.structured.resume_skills)} categories")
    print(f"  Job skills     : {n_job_skills} skills across {len(result.structured.job_skills)} categories")
    print(f"  Matched        : {result.flat_matched()}")
    print(f"  Missing        : {result.flat_missing()}")
    print(f"  Overlap score  : {result.overlap_score:.2%}")

    if result.ai_available:
        print(f"\n{'─'*60}")
        print("  STAGE 2 — AI Analysis (Gemini)")
        print(f"{'─'*60}")
        print(f"  Verdict        : {result.candidacy_verdict} (confidence: {result.verdict_confidence})")
        print(f"  Gap severity   : {result.gap_severity}")
        print(f"\n  Executive Summary:")
        print(f"  {result.executive_summary}")
        print(f"\n  Dealbreakers   : {result.dealbreaker_skills}")
        print(f"  Compensatable  : {result.compensatable_gaps}")
        print(f"  Strengths      : {result.strengths}")
        print(f"  Hiring Risks   : {result.hiring_risks}")
        print(f"  Time to Ready  : {result.time_to_ready}")
        if result.learning_path:
            print(f"\n  Learning Path:")
            for i, step in enumerate(result.learning_path, 1):
                print(f"    {i}. {step}")
        if result.skill_insights:
            print(f"\n  Skill Insights (top missing skills):")
            for si in result.skill_insights:
                db = " [DEALBREAKER]" if si.is_dealbreaker else ""
                print(f"    • {si.skill} ({si.importance}){db}")
                if si.compensation:
                    print(f"      → {si.compensation}")
    else:
        print("\n  [Stage 2 unavailable — API key missing, quota exceeded, or service/model unavailable]")
        print("  If key is set, check Gemini quota/billing and retry after cooldown.")
