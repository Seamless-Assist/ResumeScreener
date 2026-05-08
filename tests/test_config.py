from __future__ import annotations

import pytest

import sa_candidate_finder.secrets as secrets
import sa_candidate_finder.settings as settings
from sa_candidate_finder.config import Config, load_config


def test_default_config_values():
    cfg = Config(openai_api_key="k", manatal_base_url="https://secret-manatal-url")
    assert cfg.max_hard_constraints == settings.MAX_HARD_CONSTRAINTS
    assert cfg.shortlist_size == settings.SHORTLIST_SIZE
    assert cfg.top_x == settings.DEFAULT_TOP_X
    assert cfg.embedding_cache_ttl_days == settings.EMBEDDING_CACHE_TTL_DAYS
    assert cfg.embedding_model == settings.EMBEDDING_MODEL
    assert cfg.llm_model == settings.LLM_MODEL
    assert cfg.max_retries == settings.MAX_RETRIES
    assert cfg.retry_backoff == settings.RETRY_BACKOFF


def test_load_config_from_internal_settings(monkeypatch):
    monkeypatch.setattr(secrets, "OPENAI_API_KEY", "my-openai-key")
    monkeypatch.setattr(secrets, "MANATAL_BASE_URL", "https://secret-manatal-url")
    cfg = load_config()

    assert cfg.top_x == settings.DEFAULT_TOP_X
    assert cfg.shortlist_size == settings.SHORTLIST_SIZE
    assert cfg.openai_api_key == "my-openai-key"
    assert cfg.manatal_base_url == "https://secret-manatal-url"


def test_load_config_missing_openai_key(monkeypatch):
    monkeypatch.setattr(secrets, "OPENAI_API_KEY", "")
    monkeypatch.setattr(secrets, "MANATAL_BASE_URL", "https://secret-manatal-url")

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        load_config()


def test_load_config_missing_manatal_base_url(monkeypatch):
    monkeypatch.setattr(secrets, "OPENAI_API_KEY", "my-openai-key")
    monkeypatch.setattr(secrets, "MANATAL_BASE_URL", "")

    with pytest.raises(ValueError, match="MANATAL_BASE_URL"):
        load_config()


def test_invalid_retry_backoff():
    with pytest.raises(ValueError, match="retry_backoff"):
        Config(openai_api_key="k", manatal_base_url="https://secret-manatal-url", retry_backoff="invalid")
