from __future__ import annotations

from typing import Final

# Internal runtime settings.
# Keep all non-secret knobs here during testing.

# Pipeline
MAX_HARD_CONSTRAINTS: Final[int] = 3  # LLM extraction cap (keywords per JD)
MAX_KEYWORD_SET: Final[int] = 8  # Max scoring keywords after user reconciliation
MIN_GOOD_MATCH_SCORE: Final[float] = 6.0
MIN_POOL_SIZE: Final[int] = 100
MAX_FILTER_POOL_SIZE: Final[int] = 1000  # Fetch multiple Manatal pages (100/page) up to this cap
FILTER_PROGRESS_EVERY_PAGES: Final[int] = 5  # Log page progress every N pages (and always page 1)
SHORTLIST_SIZE: Final[int] = 20
PHASE2_GOOD_MATCH_TARGET: Final[int] = 20  # Trigger broader search when good matches are below this target
DEFAULT_TOP_X: Final[int] = 5

# Embedding cache
EMBEDDING_CACHE_TTL_DAYS: Final[int] = 7
EMBEDDING_CACHE_PATH: Final[str] = "cache/embeddings.db"

# Output
RESULTS_DIR: Final[str] = "results"

# Telemetry
TELEMETRY_LOG_PATH: Final[str] = "logs/telemetry.jsonl"
FEEDBACK_LOG_PATH: Final[str] = "logs/feedback.jsonl"

# Model/provider settings
EMBEDDING_PROVIDER: Final[str] = "openai"
EMBEDDING_MODEL: Final[str] = "text-embedding-3-small"
EMBEDDING_PRICE_PER_MILLION_TOKENS: Final[float] = 0.02

LLM_PROVIDER: Final[str] = "openai"
LLM_MODEL: Final[str] = "gpt-4.1-mini"
LLM_INPUT_PRICE_PER_MILLION_TOKENS: Final[float] = 0.40
LLM_OUTPUT_PRICE_PER_MILLION_TOKENS: Final[float] = 1.60

# Resilience
MCP_TIMEOUT_SECONDS: Final[int] = 10
EMBEDDING_TIMEOUT_SECONDS: Final[int] = 15
LLM_TIMEOUT_SECONDS: Final[int] = 60
MAX_RETRIES: Final[int] = 2
RETRY_BACKOFF: Final[str] = "exponential"
