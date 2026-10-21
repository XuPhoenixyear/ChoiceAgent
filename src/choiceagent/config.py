"""Application configuration with project-local defaults and explicit providers."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CHOICEAGENT_", env_file=PROJECT_ROOT / ".env", extra="ignore"
    )

    project_root: Path = PROJECT_ROOT
    catalog_path: Path = PROJECT_ROOT / "data/processed/catalog.jsonl"
    catalog_db: Path = PROJECT_ROOT / "runtime/catalog.sqlite3"
    state_db: Path = PROJECT_ROOT / "runtime/state.sqlite3"
    index_dir: Path = PROJECT_ROOT / "artifacts/indexes"
    log_dir: Path = PROJECT_ROOT / "logs"
    embedding_model_path: Path = Path("/datas/wuxi/Projects/datas/models/Qwen3-Embedding-4B")
    model_base_url: str = "http://127.0.0.1:8001"
    embedding_device: str = "cuda:0"
    gpu_ids: str = "5"
    model_timeout_seconds: float = Field(default=30, gt=0, le=120)
    turn_timeout_seconds: float = Field(default=60, gt=0, le=180)
    max_llm_calls: int = Field(default=4, ge=1, le=4)
    max_tool_calls: int = Field(default=6, ge=1, le=6)
    max_output_tokens: int = Field(default=384, ge=32, le=768)
    context_tokens: int = Field(default=8192, ge=1024, le=8192)
    api_key: str | None = None
    max_sessions: int = Field(default=10000, ge=1)
