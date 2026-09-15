"""Environment-driven configuration.

Every external provider is configurable here and read from the environment only -
no keys are ever hardcoded. Stage 0 validation is deliberately lenient: missing AI
provider keys produce a startup warning, not a failure, so the skeleton boots with
zero paid credentials.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    env: str = "development"
    log_level: str = "info"

    # --- service wiring ---
    rag_core_port: int = 8000
    rag_core_shared_secret: str = ""

    # --- qdrant ---
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "knowledge_base"
    qdrant_collection_naive: str = "knowledge_base_naive"  # Stage 3 baseline, kept separate on purpose

    # --- AI providers (configurable) ---
    embedding_provider: str = "voyage"
    embedding_model: str = "voyage-4-lite"
    embedding_dimensions: int = 1024
    voyage_api_key: str = ""
    # Voyage's free tier caps a single call at 10K TPM - a single large document's
    # chunks embedded in one call can legitimately exceed that on its own (not a
    # transient rate-limit; waiting doesn't help). VoyageEmbeddingProvider splits
    # into multiple calls under this budget, paced by voyage_batch_delay_seconds
    # to also respect the free tier's 3 RPM. Live-calibrated, not just derived
    # from the documented 10K TPM figure: a 20-chunk (~7.8K token) batch
    # consistently failed even after 30s/60s/90s backoff retries, while a
    # 16-chunk (~5.9K token) batch consistently succeeded - the free tier's
    # real effective ceiling per call is well below the documented 10K TPM.
    # Conservative defaults for the free tier; raise both once a paid plan is
    # in place.
    voyage_max_batch_tokens: int = 4000
    voyage_batch_delay_seconds: float = 21.0
    openai_api_key: str = ""
    rerank_provider: str = "cohere"
    rerank_model: str = "rerank-v3.5"
    cohere_api_key: str = ""
    anthropic_api_key: str = ""
    generation_model: str = "claude-sonnet-5"
    query_rewrite_model: str = "claude-haiku-4-5"
    generation_max_tokens: int = 1500
    generation_effort: str = "low"
    query_rewrite_max_tokens: int = 200  # a rewritten question, not an essay (Stage 7)

    # --- retrieval / cost-control knobs ---
    retrieval_score_floor: float = 0.35
    retrieve_top_k: int = 20  # hybrid (dense+sparse RRF) candidate pool size, pre-rerank (Stage 5)
    rerank_top_n: int = 5  # final chunk count after Cohere reranks the candidate pool (Stage 5)

    # --- multi-turn (Stage 7) ---
    conversation_history_turns: int = 5  # prior turns fed into query rewriting, most-recent-N

    # --- hosted-demo ingestion gate ---
    ingest_enabled: bool = True
    ingest_api_key: str = ""

    # --- ingestion (Stage 1) ---
    sqlite_path: str = "data/registry.db"
    storage_dir: str = "data/storage"
    chunk_target_tokens: int = 500
    chunk_overlap_ratio: float = 0.15

    # --- PDF / OCR (Stage 4) ---
    ocr_enabled: bool = True  # local Tesseract only - free, no paid API
    pdf_scanned_threshold_chars_per_page: float = 20.0
    pdf_ocr_dpi: int = 200

    @property
    def is_development(self) -> bool:
        return self.env.lower() in {"development", "dev", "local"}

    def missing_ai_keys(self) -> list[str]:
        """AI provider keys that are absent. None of these are required in Stage 0."""
        missing: list[str] = []
        if not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")
        if self.embedding_provider == "voyage" and not self.voyage_api_key:
            missing.append("VOYAGE_API_KEY")
        if self.embedding_provider == "openai" and not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if self.rerank_provider == "cohere" and not self.cohere_api_key:
            missing.append("COHERE_API_KEY")
        return missing


@lru_cache
def get_settings() -> Settings:
    return Settings()
