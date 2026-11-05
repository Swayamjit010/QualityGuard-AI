"""Environment-based configuration (prefix ``QG_``). See .env.example."""
from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QG_", env_file=str(ROOT / ".env"), extra="ignore")

    app_name: str = "QualityGuard AI"
    environment: str = "development"
    database_url: str = f"sqlite:///{(ROOT / 'storage' / 'qualityguard.db').as_posix()}"
    schema_mode: str = "create_all"           # create_all (development) | migrations (Alembic, production)
    storage_dir: Path = ROOT / "storage"
    secret_key: str = ""                      # MUST be set in production
    access_token_minutes: int = 480
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    max_upload_mb: int = 25
    max_video_mb: int = 300
    max_batch_files: int = 200
    workers: int = 2
    rate_limit_per_minute: int = 240
    login_rate_limit_per_minute: int = 20
    allow_registration: bool = True           # self-registration creates 'viewer' accounts only
    default_admin_email: str = "admin@qualityguard.local"
    default_admin_password: str = ""          # if empty, a random one is generated and logged once
    seed_demo_users: bool = True
    data_retention_days: int = 0              # 0 = keep forever
    webhook_url: str = ""
    webhook_events: str = "alert"             # comma list of event types to POST: alert, inspection, review, job
    mqtt_host: str = ""                       # optional MQTT adapter (pip install paho-mqtt)
    mqtt_port: int = 1883
    mqtt_topic_prefix: str = "qualityguard"
    mqtt_username: str = ""
    mqtt_password: str = ""
    mqtt_tls: bool = False
    anthropic_api_key: str = ""
    assistant_model: str = "claude-opus-5-5"
    torch_threads: int = 0                    # 0 = torch default

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.storage_dir.mkdir(parents=True, exist_ok=True)
    if not s.secret_key:
        key_file = s.storage_dir / ".secret_key"
        if key_file.exists():
            s.secret_key = key_file.read_text().strip()
        else:
            s.secret_key = secrets.token_urlsafe(48)
            key_file.write_text(s.secret_key)
    return s
