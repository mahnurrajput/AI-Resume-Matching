"""
skill_taxonomy.py
==================
Data module for the skill gap analysis system: the skill taxonomy, alias
map, and domain-guard configuration. Split out from skill_analyzer.py
because the taxonomy is large enough that mixing it with extraction/AI
logic in one file made both harder to read and maintain.

─────────────────────────────────────────────────────────────────────────────
WHY THIS FILE WAS EXPANDED
─────────────────────────────────────────────────────────────────────────────
The previous taxonomy covered 8 tech categories plus a single culinary
category — 9 total. But evaluate_matching.py's own CATEGORY_KEYWORDS lists
25 real resume categories in the labeled dataset (HR, Advocate, Arts,
Mechanical Engineer, Sales, Health-and-Fitness, Civil Engineer, Finance,
Operations Manager, PMO, Business Analyst, Electrical Engineering, etc.).
Most of those had zero taxonomy coverage, meaning skill gap analysis
couldn't produce anything meaningful for roughly two-thirds of the labeled
categories (and, by extension, a large share of the full 11,654-resume
corpus).

This file adds 14 new categories so all 25 CATEGORY_KEYWORDS domains have
at least one matching taxonomy category, without removing or shrinking any
existing tech category.

This revision further widens every category's keyword list. Tech
categories were broadened the most (they cover the largest number of
distinct sub-fields, tools, and frameworks in practice — e.g. "cloud
devops" alone spans a dozen tool ecosystems), while the non-tech
categories were also given a meaningfully larger and more specific
vocabulary so guard checks and matching have more to work with.

─────────────────────────────────────────────────────────────────────────────
GENERALIZED DOMAIN GUARD (replaces the culinary-only patch)
─────────────────────────────────────────────────────────────────────────────
The previous version had one hardcoded guard (_CULINARY_INDICATORS /
_CULINARY_MIN_INDICATORS) to stop generic business phrases like "inventory
control" or "staff training" from falsely tagging a non-culinary resume as
having culinary skills. That guard only protected one category.

Many of the new categories added here use the same kind of everyday
business phrasing (HR, Sales, Finance, Operations, Business Analysis,
Legal, Arts, and the engineering disciplines all have this risk to varying
degrees) — so instead of hand-writing a one-off guard per category, this
file defines a single DOMAIN_GUARDS table: any category listed there
requires a minimum number of category-specific "anchor" terms (things like
"hr", "attorney", "gym", "cpa", "solidworks") to appear in the source text
before its skills are allowed to match at all. Categories whose vocabulary
is inherently unambiguous (programming languages, cloud/devops tooling,
ML frameworks, etc.) are not listed and need no guard.

This is a config table, not a mechanism you need to extend per category —
adding a new guarded category later just means adding one entry here.
"""

import re
from typing import Dict, List


# ══════════════════════════════════════════════════════════════════════════════
# SKILL TAXONOMY
# ══════════════════════════════════════════════════════════════════════════════

SKILL_TAXONOMY: Dict[str, List[str]] = {

    # ── Original technical categories (broadened, kept intact in spirit) ──────

    "programming_languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "golang", "go",
        "rust", "kotlin", "swift", "scala", "r", "matlab", "perl", "ruby", "php",
        "dart", "julia", "lua", "haskell", "erlang", "elixir", "clojure", "groovy",
        "objective-c", "assembly", "cobol", "fortran", "vba", "bash", "shell script",
        "powershell", "sql", "pl/sql", "t-sql",
        "zig", "nim", "f#", "vb.net", "visual basic", "delphi", "pascal", "ada",
        "scheme", "common lisp", "prolog", "racket", "ocaml", "abap", "apex",
        "awk", "sed", "crystal", "elm", "purescript", "solidity", "coffeescript",
        "webassembly", "wasm", "smalltalk", "tcl", "verilog", "vhdl", "labview",
    ],

    "web_frontend": [
        "react", "react.js", "angular", "angularjs", "vue", "vue.js", "next.js",
        "nuxt.js", "svelte", "sveltekit", "html", "css", "sass", "scss", "less",
        "tailwind", "bootstrap", "material ui", "jquery", "webpack", "vite",
        "babel", "redux", "mobx", "graphql", "rest", "restful", "soap",
        "websockets", "responsive design", "web components", "pwa", "d3.js",
        "three.js", "ember.js", "backbone.js", "storybook", "styled-components",
        "css-in-js", "emotion", "remix", "astro", "solid.js", "alpine.js",
        "gatsby", "chakra ui", "ant design", "postcss", "eslint", "prettier",
        "rollup", "esbuild", "parcel", "web workers", "service workers",
        "figma", "sketch", "zeplin", "bem methodology", "css grid", "flexbox",
        "ssr", "server-side rendering", "ssg", "static site generation",
        "hydration", "micro-frontends", "accessibility", "a11y", "web vitals",
        "cypress", "playwright", "jest", "react testing library", "storybook.js",
    ],

    "web_backend": [
        "node.js", "express", "fastapi", "flask", "django", "spring", "spring boot",
        "spring mvc", "hibernate", "jpa", "j2ee", "asp.net", ".net", "laravel",
        "rails", "ruby on rails", "fastify", "nestjs", "grpc", "microservices",
        "rest api", "graphql api", "oauth", "jwt", "api gateway",
        "message queue", "rabbitmq", "kafka", "celery", "nginx", "apache",
        "koa", "hapi", "phoenix framework", "gin framework", "echo framework",
        "fiber framework", "symfony", "codeigniter", "cakephp", "play framework",
        "micronaut", "quarkus", "vert.x", "actix", "rocket framework", "trpc",
        "socket.io", "protocol buffers", "openapi", "swagger", "api design",
        "rate limiting", "circuit breaker pattern", "event-driven architecture",
        "cqrs", "domain-driven design", "service mesh", "monorepo tooling",
        "nx monorepo", "turborepo", "server-side architecture", "webhooks",
    ],

    "databases": [
        "postgresql", "mysql", "sqlite", "oracle", "sql server", "mariadb",
        "mongodb", "redis", "elasticsearch", "cassandra", "dynamodb", "couchdb",
        "neo4j", "influxdb", "clickhouse", "snowflake", "bigquery", "redshift",
        "hbase", "firebase", "supabase", "prisma", "sqlalchemy",
        "database design", "query optimization", "stored procedures", "indexing",
        "nosql", "relational database", "data modeling", "etl",
        "db2", "teradata", "vertica", "timescaledb", "cockroachdb", "scylladb",
        "arangodb", "couchbase", "rethinkdb", "memcached", "amazon aurora",
        "google cloud spanner", "database sharding", "database replication",
        "database migration", "flyway", "liquibase", "database normalization",
        "acid transactions", "cap theorem", "olap", "oltp", "data warehousing",
        "star schema", "snowflake schema", "database partitioning",
        "connection pooling", "database backup and recovery",
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
        "vagrant", "packer", "hashicorp vault", "consul", "nomad", "openshift",
        "rancher", "argocd", "flux cd", "spinnaker", "cloud foundry",
        "alibaba cloud", "oracle cloud infrastructure", "ibm cloud", "vmware",
        "vsphere", "esxi", "nagios", "zabbix", "new relic", "pagerduty",
        "site reliability engineering", "sre", "chaos engineering",
        "blue-green deployment", "canary deployment", "gitops", "linkerd",
        "docker compose", "docker swarm", "podman", "containerd", "cri-o",
        "cloudformation", "azure devops", "bitbucket pipelines",
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
        "apache flink", "apache beam", "apache nifi", "kafka streams",
        "presto", "trino", "apache hive", "apache pig", "databricks",
        "mlflow", "kubeflow", "weights and biases", "dvc", "feature store",
        "model monitoring", "model registry", "onnx", "tensorrt", "opencv",
        "yolo object detection", "gans", "generative adversarial networks",
        "diffusion models", "transfer learning", "time series forecasting",
        "arima", "prophet forecasting", "anomaly detection",
        "recommender systems", "collaborative filtering", "bayesian statistics",
        "hypothesis testing", "causal inference", "prompt engineering",
        "vector search", "pinecone", "weaviate", "milvus", "chroma db",
        "semantic search", "agentic ai", "multi-agent systems",
    ],

    "mobile": [
        "android", "ios", "react native", "flutter", "dart", "swift", "kotlin",
        "objective-c", "xamarin", "ionic", "mobile development",
        "firebase", "push notifications",
        "jetpack compose", "swiftui", "uikit", "android studio", "xcode",
        "cocoapods", "gradle", "app store deployment", "google play deployment",
        "arkit", "arcore", "capacitor", "cordova", "phonegap",
        "fastlane", "mvvm architecture", "mobile app architecture",
        "mobile ci/cd", "app performance optimization", "mobile ui/ux",
    ],

    "security_networking": [
        "cybersecurity", "network security", "penetration testing", "ethical hacking",
        "vulnerability assessment", "siem", "splunk", "wireshark", "nmap",
        "firewall", "vpn", "ssl", "tls", "encryption", "oauth",
        "soc", "incident response", "threat hunting", "malware analysis",
        "cissp", "ceh", "security+", "tcp/ip", "dns",
        "zero trust", "identity management", "iam", "ldap", "active directory",
        "owasp", "soc 2 compliance", "iso 27001", "pci dss compliance",
        "gdpr compliance", "hipaa compliance", "red teaming", "blue teaming",
        "soar", "edr", "xdr", "dlp", "pki", "digital forensics",
        "reverse engineering", "burp suite", "metasploit", "kali linux",
        "oscp", "cism", "security awareness training", "risk assessment",
        "network segmentation", "ids/ips", "vlan", "bgp", "ospf",
        "subnetting", "routing protocols", "sase", "sd-wan",
    ],

    "culinary_hospitality": [
        "chef", "head chef", "sous chef", "menu planning",
        "food preparation", "kitchen management", "kitchen operations", "inventory control",
        "food costing", "portion sizing", "supplier coordination", "food safety", "haccp",
        "italian cuisine", "continental cuisine", "baking", "pastry", "fine dining",
        "banquet management", "catering", "restaurant operations", "staff training",
        "menu costing", "wine pairing", "sommelier", "mixology", "bartending",
        "front of house", "back of house", "hotel management", "guest services",
        "housekeeping management", "event planning", "room service",
        "revenue management", "front desk operations", "concierge services",
        "culinary arts", "molecular gastronomy", "food plating",
        "sanitation standards", "food handler certification",
    ],

    # ── New categories — added to cover the ~16 previously-uncovered labeled
    #    resume categories from evaluate_matching.py's CATEGORY_KEYWORDS ────────

    "human_resources": [
        "human resources", "recruitment", "talent acquisition", "onboarding", "employee relations",
        "performance management", "compensation and benefits", "hr policies",
        "hris", "workforce planning", "employee engagement", "payroll processing",
        "labor relations", "diversity and inclusion programs", "exit interviews",
        "succession planning", "hr compliance", "applicant tracking system",
        "employee training programs", "conflict resolution", "hr generalist",
        "hr business partner", "benefits administration",
        "compensation benchmarking", "job analysis", "workforce diversity",
        "hr metrics", "hr analytics", "employee handbook", "disciplinary actions",
        "unemployment claims", "fmla administration", "ada compliance",
        "eeoc compliance", "organizational development", "talent management",
        "employer branding", "staffing agencies", "background checks",
        "i-9 verification", "workday hris", "sap successfactors", "adp payroll",
        "ceridian", "bamboohr",
    ],

    "legal": [
        "contract law", "litigation", "legal research", "legal writing",
        "corporate law", "intellectual property law", "regulatory compliance",
        "due diligence", "case management", "legal drafting", "negotiation",
        "arbitration", "mediation", "paralegal support", "court filings",
        "legal documentation", "employment law", "criminal law",
        "civil litigation", "contract negotiation", "legal counsel", "compliance review",
        "compliance officer", "contract management", "e-discovery", "legal ethics",
        "bar admission", "trial preparation", "deposition", "discovery process",
        "tort law", "family law", "real estate law", "immigration law",
        "bankruptcy law", "trademark law", "patent law", "copyright law",
        "notary public", "legal billing", "westlaw", "lexisnexis",
        "regulatory filings", "securities law",
    ],

    "arts": [
        "illustration", "graphic design", "fine arts", "sculpture", "painting",
        "portfolio development", "art direction", "creative concept development",
        "gallery exhibitions", "mixed media art", "digital illustration",
        "storyboarding", "visual arts", "art curation", "printmaking",
        "art critique", "adobe illustrator", "adobe photoshop", "concept art",
        "typography", "color theory", "adobe indesign", "adobe after effects",
        "motion graphics", "animation", "3d modeling", "blender", "autodesk maya",
        "ceramics", "textile art", "photography", "art history",
        "museum studies", "exhibition design", "public art", "mural painting",
        "figure drawing", "watercolor painting", "printmaking techniques",
    ],

    "mechanical_engineering": [
        "mechanical design", "solidworks", "autocad", "cad modeling",
        "finite element analysis", "manufacturing processes", "product design",
        "thermodynamics", "fluid mechanics", "mechanical drafting",
        "tolerance analysis", "prototyping", "cnc machining", "hvac systems",
        "mechanical engineering", "gd&t", "materials science",
        "quality control engineering", "six sigma", "lean manufacturing",
        "ansys", "catia", "creo parametric", "siemens nx", "pro/engineer",
        "sheet metal design", "welding", "machining", "injection molding",
        "robotics", "mechatronics", "kinematics", "dynamics analysis",
        "vibration analysis", "heat transfer", "pneumatics", "hydraulics",
        "product lifecycle management", "geometric dimensioning and tolerancing",
        "3d printing", "additive manufacturing",
    ],

    "sales": [
        "sales strategy", "business development", "account management",
        "lead generation", "cold calling", "sales pipeline management",
        "crm software", "salesforce crm", "negotiation skills", "quota attainment",
        "client relationship management", "upselling", "cross-selling",
        "sales forecasting", "territory management", "b2b sales", "b2c sales",
        "sales presentations", "closing deals", "customer acquisition",
        "hubspot crm", "zoho crm", "sales enablement", "key account management",
        "channel sales", "inside sales", "outside sales", "sales operations",
        "sales analytics", "value selling", "solution selling", "spin selling",
        "consultative selling", "sales training", "revenue growth",
        "market penetration", "competitive analysis", "proposal writing",
    ],

    "health_and_fitness": [
        "personal training", "fitness coaching", "nutrition counseling",
        "exercise programming", "group fitness instruction",
        "strength and conditioning", "wellness coaching",
        "weight management programs", "certified personal trainer",
        "yoga instruction", "physical therapy support", "health assessments",
        "fitness assessments", "corrective exercise",
        "sports performance training", "client fitness plans",
        "crossfit coaching", "pilates instruction", "sports nutrition",
        "injury prevention", "rehabilitation exercises", "athletic training",
        "biomechanics", "body composition analysis", "functional training",
        "hiit training", "aquatic fitness instruction", "senior fitness programs",
        "prenatal fitness", "nasm certification", "ace certification",
    ],

    "civil_engineering": [
        "structural design", "civil engineering", "site engineering",
        "construction management", "autocad civil 3d", "surveying",
        "geotechnical engineering", "structural analysis", "construction drawings",
        "project estimation", "building codes compliance", "land development",
        "highway design", "water resources engineering", "concrete design",
        "steel structure design", "environmental engineering",
        "revit", "building information modeling", "primavera p6",
        "cost estimation", "quantity surveying", "transportation engineering",
        "urban planning", "land surveying", "topographic surveys",
        "drainage design", "retaining wall design", "bridge design",
        "dam engineering", "seismic design", "leed certification",
        "construction scheduling",
    ],

    "finance": [
        "financial analysis", "financial modeling", "budgeting and forecasting",
        "accounting", "accounts payable", "accounts receivable",
        "financial reporting", "auditing", "cpa", "gaap", "cash flow management",
        "risk management", "investment analysis", "portfolio management",
        "tax preparation", "reconciliation", "variance analysis",
        "financial planning", "cost accounting", "internal controls",
        "bloomberg terminal", "quickbooks", "sap fico", "oracle financials",
        "hyperion", "private equity", "mergers and acquisitions",
        "treasury management", "derivatives trading", "equity research",
        "credit analysis", "fp&a", "sox compliance", "ifrs", "capital budgeting",
        "valuation modeling", "dcf analysis", "wealth management",
        "corporate finance", "regulatory reporting",
    ],

    "blockchain": [
        "blockchain", "smart contract development", "solidity", "web3",
        "ethereum", "cryptocurrency", "decentralized applications",
        "dapp development", "nft development", "consensus algorithms",
        "distributed ledger technology", "hyperledger", "cryptography",
        "tokenomics", "defi protocols", "truffle framework", "hardhat",
        "chainlink", "polkadot", "cosmos sdk", "layer 2 scaling",
        "zero-knowledge proofs", "zk-snarks", "dao governance", "ipfs",
        "metamask integration", "web3.js", "ethers.js", "erc-20", "erc-721",
        "cross-chain bridges", "rollups",
    ],

    "operations_management": [
        "operations management", "supply chain management", "logistics management",
        "inventory management", "procurement", "vendor management",
        "process improvement", "warehouse management", "production planning",
        "quality assurance processes", "lean operations", "continuous improvement",
        "operational efficiency", "distribution management", "demand planning",
        "fleet management",
        "kaizen", "5s methodology", "erp systems", "sap erp", "oracle scm",
        "just-in-time inventory", "bottleneck analysis", "capacity planning",
        "workforce scheduling", "osha safety compliance", "cost reduction strategies",
        "operations analytics", "materials management",
    ],

    "project_management": [
        "project management", "pmo", "scrum master", "agile project management",
        "program management", "project planning", "risk management planning",
        "stakeholder management", "project scheduling", "resource allocation",
        "pmp certification", "gantt charts", "waterfall methodology",
        "sprint planning", "project budgeting", "change management",
        "ms project", "monday.com", "asana", "prince2", "critical path method",
        "earned value management", "risk register", "raci matrix",
        "lean project management", "project charter", "milestone tracking",
        "capstone project delivery", "portfolio governance",
    ],

    "business_analysis": [
        "business analysis", "requirements gathering", "stakeholder management",
        "business process modeling", "use case development", "user stories",
        "gap analysis", "swot analysis", "data-driven decision making",
        "business process reengineering", "functional specifications",
        "uat testing", "business requirements document",
        "process mapping", "cost-benefit analysis",
        "visio diagramming", "wireframing", "process automation",
        "robotic process automation", "agile business analysis",
        "product ownership", "backlog grooming", "kpi development",
        "competitive benchmarking", "requirements traceability matrix",
    ],

    "automation_testing": [
        "test automation", "selenium", "manual testing", "quality assurance",
        "test case design", "regression testing", "junit", "testng", "cypress",
        "appium", "performance testing", "load testing", "api testing", "postman",
        "test plans", "bug tracking", "defect management", "sdlc", "stlc",
        "continuous testing",
        "playwright", "robot framework", "katalon studio", "cucumber",
        "behavior-driven development", "test-driven development", "jmeter",
        "loadrunner", "soapui", "mobile testing", "cross-browser testing",
        "accessibility testing", "security testing", "exploratory testing",
    ],

    "electrical_engineering": [
        "electrical design", "circuit design", "power systems", "embedded systems",
        "plc programming", "electrical schematics", "control systems",
        "power electronics", "signal processing", "pcb design", "microcontrollers",
        "electrical engineering", "renewable energy systems", "instrumentation",
        "scada systems", "motor control",
        "matlab simulink", "altium designer", "eagle pcb", "vhdl", "verilog",
        "fpga design", "asic design", "analog circuit design",
        "digital circuit design", "rf engineering", "telecommunications",
        "5g networks", "iot device design", "sensors and actuators",
    ],
}


# Canonical alias map — resolves abbreviations and common synonyms.
# Applied during NER candidate resolution AND during keyword scanning.
SKILL_ALIASES: Dict[str, str] = {
    "py"                   : "python",
    "js"                   : "javascript",
    "ts"                   : "typescript",
    "k8s"                  : "kubernetes",
    "node"                 : "node.js",
    "react js"             : "react",
    "angular js"           : "angular",
    "ml"                   : "machine learning",
    "dl"                   : "deep learning",
    "cv"                   : "computer vision",
    "nlp"                  : "natural language processing",
    "postgres"             : "postgresql",
    "mongo"                : "mongodb",
    "elastic"              : "elasticsearch",
    "tf"                   : "tensorflow",
    "sklearn"              : "scikit-learn",
    "hf"                   : "hugging face",
    "ci cd"                : "ci/cd",
    "ci-cd"                : "ci/cd",
    "aws cloud"            : "aws",
    "dotnet"               : ".net",
    "dot net"              : ".net",
    "asp net"              : "asp.net",
    "spring framework"     : "spring",
    "spring-boot"          : "spring boot",
    "jee"                  : "j2ee",
    "iac"                  : "infrastructure as code",
    "gh actions"           : "github actions",
    "llms"                 : "large language model",
    "gpt-4"                : "large language model",
    "openai api"           : "large language model",
    "sql db"               : "sql",
    "rdbms"                : "relational database",
    "haccp compliance"     : "haccp",
    "food safety standards": "food safety",
    "menu design"          : "menu planning",
    # New-category aliases
    "hr"                   : "human resources",
    "human resource management": "human resources",
    "ats"                  : "applicant tracking system",
    "attorney"             : "legal counsel",
    "lawyer"               : "legal counsel",
    "fea"                  : "finite element analysis",
    "crm"                  : "crm software",
    "salesforce"           : "salesforce crm",
    "pt"                   : "personal training",
    "cad"                  : "cad modeling",
    "pm"                   : "project management",
    "pmp"                  : "pmp certification",
    "ba"                   : "business analysis",
    "brd"                  : "business requirements document",
    "qa"                   : "quality assurance",
    "sdlc"                 : "sdlc",
    "plc"                  : "plc programming",
    "web 3"                : "web3",
    "defi"                 : "defi protocols",
    # Additional aliases for newly added terms
    "bim"                  : "building information modeling",
    "rpa"                  : "robotic process automation",
    "tdd"                  : "test-driven development",
    "bdd"                  : "behavior-driven development",
    "fpga"                 : "fpga design",
    "sox"                  : "sox compliance",
    "m&a"                  : "mergers and acquisitions",
    "dcf"                  : "dcf analysis",
    "plm"                  : "product lifecycle management",
    "gdpr"                 : "gdpr compliance",
    "hipaa"                : "hipaa compliance",
    "pci dss"              : "pci dss compliance",
    "ada compliance"       : "ada compliance",
    "eeoc"                 : "eeoc compliance",
    "fmla"                 : "fmla administration",
    "sre"                  : "site reliability engineering",
    "iam"                  : "identity management",
    "soc2"                 : "soc 2 compliance",
    "iso27001"             : "iso 27001",
    "gd and t"             : "geometric dimensioning and tolerancing",
    "3d print"             : "3d printing",
}


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN GUARDS  (generalized version of the old culinary-only guard)
# ══════════════════════════════════════════════════════════════════════════════
#
# A category listed here only contributes matches when the source text
# contains at least `min_indicators` distinct anchor terms from its
# `indicators` list. This stops everyday business phrases that happen to be
# taxonomy entries in one of these categories (e.g. "inventory management",
# "staff training", "process improvement") from firing on resumes that are
# actually in a different field but happen to use similar generic wording.
#
# Categories NOT listed here (programming languages, cloud/devops tooling,
# ML frameworks, mobile, security, blockchain, test-automation tooling) use
# specific-enough jargon that this kind of guard isn't needed.

DOMAIN_GUARDS: Dict[str, Dict] = {
    "culinary_hospitality": {
        "indicators": [
            "chef", "kitchen", "cuisine", "culinary", "restaurant", "menu",
            "pastry", "baking", "haccp", "catering", "sous", "banquet",
            "food", "dining", "sommelier", "mixology", "bartending",
            "concierge", "housekeeping",
        ],
        "min_indicators": 2,
    },
    "human_resources": {
        "indicators": [
            "hr", "human resources", "recruiter", "recruitment",
            "talent acquisition", "onboarding", "payroll", "employee relations",
            "hris", "workday", "successfactors", "fmla", "eeoc",
        ],
        "min_indicators": 1,
    },
    "legal": {
        "indicators": [
            "lawyer", "attorney", "legal", "law", "litigation", "court",
            "paralegal", "counsel", "arbitration", "westlaw", "lexisnexis",
            "notary",
        ],
        "min_indicators": 2,
    },
    "arts": {
        "indicators": [
            "artist", "gallery", "illustration", "portfolio", "sculpture",
            "painting", "art director", "fine arts", "curator", "typography",
            "printmaking", "ceramics",
        ],
        "min_indicators": 2,
    },
    "sales": {
        "indicators": [
            "sales", "quota", "crm", "cold calling", "account executive",
            "pipeline", "business development", "closing deals", "hubspot",
            "zoho",
        ],
        "min_indicators": 2,
    },
    "health_and_fitness": {
        "indicators": [
            "fitness", "gym", "personal trainer", "nutrition", "wellness",
            "yoga", "health coach", "workout", "crossfit", "pilates",
            "nasm", "ace certification",
        ],
        "min_indicators": 1,
    },
    "mechanical_engineering": {
        "indicators": [
            "mechanical", "solidworks", "autocad", "manufacturing",
            "thermodynamics", "cnc", "mechanical engineering", "ansys",
            "catia", "creo",
        ],
        "min_indicators": 1,
    },
    "civil_engineering": {
        "indicators": [
            "civil engineering", "construction", "structural", "surveying",
            "site engineer", "building codes", "geotechnical", "revit",
            "primavera",
        ],
        "min_indicators": 1,
    },
    "electrical_engineering": {
        "indicators": [
            "electrical", "circuit", "plc", "power systems",
            "embedded systems", "pcb", "electronics", "fpga", "vhdl",
            "verilog",
        ],
        "min_indicators": 1,
    },
    "finance": {
        "indicators": [
            "finance", "accounting", "audit", "cpa", "gaap", "budgeting",
            "financial analyst", "investment", "bloomberg terminal",
            "treasury", "equity research",
        ],
        "min_indicators": 2,
    },
    "operations_management": {
        "indicators": [
            "operations", "supply chain", "logistics", "warehouse",
            "procurement", "vendor management", "fleet", "kaizen", "erp",
        ],
        "min_indicators": 2,
    },
    "project_management": {
        "indicators": [
            "project manager", "pmo", "scrum master", "agile",
            "program manager", "project management", "prince2",
        ],
        "min_indicators": 1,
    },
    "business_analysis": {
        "indicators": [
            "business analyst", "requirements gathering", "stakeholder",
            "business analysis", "use case", "gap analysis", "wireframing",
        ],
        "min_indicators": 1,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# DERIVED LOOKUPS  (built once at import time)
# ══════════════════════════════════════════════════════════════════════════════

ALL_SKILLS_FLAT: List[str] = []
SKILL_TO_CATEGORY: Dict[str, str] = {}

for _cat, _skills in SKILL_TAXONOMY.items():
    for _sk in _skills:
        ALL_SKILLS_FLAT.append(_sk)
        SKILL_TO_CATEGORY[_sk] = _cat

# Pre-compiled word-boundary regex per taxonomy skill — shared by the
# extractor's substring-match stage and its keyword-scan safety net.
SKILL_PATTERNS: Dict[str, re.Pattern] = {}
for _sk in ALL_SKILLS_FLAT:
    _escaped = re.escape(_sk)
    SKILL_PATTERNS[_sk] = re.compile(rf'(?<!\w){_escaped}(?!\w)', re.IGNORECASE)

# Pre-compiled word-boundary regex per guard indicator term, so counting
# indicator hits in a text is just as cheap as the skill-matching itself.
_GUARD_INDICATOR_PATTERNS: Dict[str, re.Pattern] = {}
for _cat, _cfg in DOMAIN_GUARDS.items():
    for _ind in _cfg["indicators"]:
        if _ind not in _GUARD_INDICATOR_PATTERNS:
            _GUARD_INDICATOR_PATTERNS[_ind] = re.compile(
                rf'(?<!\w){re.escape(_ind)}(?!\w)', re.IGNORECASE
            )


def count_domain_indicators(text: str, category: str) -> int:
    """
    Count how many distinct guard indicator terms for `category` appear in
    `text`. Returns 0 for categories with no guard configured (i.e. they are
    always allowed — see DOMAIN_GUARDS docstring above).
    """
    cfg = DOMAIN_GUARDS.get(category)
    if not cfg:
        return 0
    tl = text.lower()
    return sum(
        1 for ind in cfg["indicators"]
        if _GUARD_INDICATOR_PATTERNS[ind].search(tl)
    )


def is_domain_allowed(text_indicator_counts: Dict[str, int], category: str) -> bool:
    """
    Given a precomputed {category: indicator_count} dict for a piece of text
    (see StructuredSkillExtractor._compute_guard_counts in skill_analyzer.py),
    return whether `category` is allowed to contribute matches for that text.

    Categories with no guard entry are always allowed.
    """
    cfg = DOMAIN_GUARDS.get(category)
    if not cfg:
        return True
    return text_indicator_counts.get(category, 0) >= cfg["min_indicators"]