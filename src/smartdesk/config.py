"""Application configuration using Pydantic Settings.

Enterprise pattern: all config flows through ONE typed object. No scattered os.getenv() calls.
Environment variables override YAML defaults, which override class defaults.
"""
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PreprocessingConfig(BaseSettings):
    """Phase 1: Text preprocessing knobs."""

    model_config = SettingsConfigDict(env_prefix="PREPROC_")

    # Toggle each step independently (Strategy pattern)
    remove_urls: bool = True
    remove_emails: bool = True
    remove_emojis: bool = False  # keep for sentiment signal
    lowercase: bool = True
    remove_stopwords: bool = False  # task-dependent
    lemmatize: bool = True
    min_token_length: int = 2
    max_token_length: int = 50


class AppConfig(BaseSettings):
    """Root application config."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "SmartDesk"
    environment: str = Field(default="dev", description="dev | staging | prod")
    log_level: str = "INFO"

    # Paths
    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"

    # Nested configs
    preprocessing: PreprocessingConfig = PreprocessingConfig()


@lru_cache
def get_config() -> AppConfig:
    """Cached singleton. Import this everywhere — never instantiate AppConfig directly."""
    return AppConfig()
