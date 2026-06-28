"""Configuration for the PET-CT visualization project."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


@dataclass
class RateLimitConfig:
    rpm: int = 60
    concurrency: int = 5
    max_retries: int = 1
    retry_backoff: float = 2.0
    timeout: float = 300.0


@dataclass
class CsvConfig:
    text_column_candidates: List[str] = field(
        default_factory=lambda: ["检查所见", "检查结论", "诊断", "报告内容", "report", "content"]
    )
    user_text_cols: List[str] = field(default_factory=list)
    id_column_candidates: List[str] = field(
        default_factory=lambda: [
            "门诊号/住院号",
            "门诊号",
            "住院号",
            "ID",
            "id",
            "patient_id",
            "accession_number",
        ]
    )
    default_encoding: str = "utf-8-sig"
    max_chars: int = 5000


@dataclass
class AppConfig:
    api_key: str = os.getenv("NANOBANANA_API_KEY", "")
    api_mode: str = os.getenv("NANOBANANA_API_MODE", "chat")
    api_url: str = os.getenv(
        "NANOBANANA_API_URL",
        "https://api.sydney-ai.com/v1/chat/completions",
    )
    model: str = os.getenv("NANOBANANA_MODEL", "gemini-2.5-flash-image")
    output_dir: str = "outputs"
    width: int = 1024
    height: int = 1024
    size: str = "1024x1024"
    num_images: int = 1
    reference_image_url: Optional[str] = os.getenv("REFERENCE_IMAGE_URL")
    extra_params: Dict[str, Any] = field(default_factory=dict)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    csv: CsvConfig = field(default_factory=CsvConfig)

    @staticmethod
    def from_env() -> "AppConfig":
        config = AppConfig()
        config.api_key = os.getenv("NANOBANANA_API_KEY", config.api_key)
        config.api_mode = os.getenv("NANOBANANA_API_MODE", config.api_mode)
        config.api_url = os.getenv("NANOBANANA_API_URL", config.api_url)
        config.model = os.getenv("NANOBANANA_MODEL", config.model)
        config.output_dir = os.getenv("OUTPUT_DIR", config.output_dir)
        config.size = os.getenv("NANOBANANA_SIZE", config.size)
        config.rate_limit.rpm = int(os.getenv("RPM", config.rate_limit.rpm))
        config.rate_limit.concurrency = int(os.getenv("CONCURRENCY", config.rate_limit.concurrency))
        config.rate_limit.max_retries = int(
            os.getenv("MAX_RETRIES", config.rate_limit.max_retries)
        )
        config.rate_limit.timeout = float(os.getenv("TIMEOUT", config.rate_limit.timeout))
        return config
