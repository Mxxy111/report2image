"""Configuration for NanoBanana PET-CT Project."""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    rpm: int = 60  # Requests per minute
    concurrency: int = 5  # Max concurrent requests
    max_retries: int = 1  # 默认重试 1 次 (共尝试 2 次)
    retry_backoff: float = 2.0
    timeout: float = 300.0  # 增加默认超时到 5 分钟

@dataclass
class CsvConfig:
    """CSV input configuration."""
    # 自动检测的候选列名（仅当用户未指定 --text-cols 时使用）
    text_column_candidates: List[str] = field(
        default_factory=lambda: ["检查所见", "报告内容", "report", "content"]
    )
    # 用户明确指定的列名列表
    user_text_cols: List[str] = field(default_factory=list)
    
    id_column_candidates: List[str] = field(
        default_factory=lambda: ["门诊号/住院号", "ID", "id", "patient_id", "accession_number"]
    )
    default_encoding: str = "utf-8-sig"
    max_chars: int = 5000

@dataclass
class AppConfig:
    """Application configuration."""
    # ================= API 配置 (请根据文档修改) =================
    
    # API 密钥
    api_key: str = os.getenv("NANOBANANA_API_KEY", "")
    
    # API 模式: 'chat' (v1/chat/completions) 或 'image' (v1/images/generations)
    api_mode: str = os.getenv("NANOBANANA_API_MODE", "chat")
    
    # API 完整 URL
    # Chat 示例: https://api.com/v1/chat/completions
    # Image 示例: https://api.com/v1/images/generations
    api_url: str = os.getenv("NANOBANANA_API_URL", "")
    
    # 模型名称
    model: str = os.getenv("NANOBANANA_MODEL", "")
    
    # ===========================================================
    
    # Output Configuration
    output_dir: str = "outputs"
    
    # Generation Parameters
    width: int = 1024
    height: int = 1024
    size: str = "1024x1024" # 仅用于 image 模式, 如 1024x1024 或 1x1
    num_images: int = 1
    
    # 如果需要参考图，可在此填入 URL
    reference_image_url: Optional[str] = os.getenv("REFERENCE_IMAGE_URL", None)
    
    # 额外参数 (如果 API 需要特定参数，可在此添加)
    extra_params: Dict[str, Any] = field(default_factory=lambda: {})

    # Sub-configs
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    csv: CsvConfig = field(default_factory=CsvConfig)

    @staticmethod
    def from_env() -> "AppConfig":
        """Create config from environment variables."""
        config = AppConfig()
        
        # Load simple env vars
        config.api_key = os.getenv("NANOBANANA_API_KEY", config.api_key)
        config.api_mode = os.getenv("NANOBANANA_API_MODE", config.api_mode)
        config.api_url = os.getenv("NANOBANANA_API_URL", config.api_url)
        config.model = os.getenv("NANOBANANA_MODEL", config.model)
        config.output_dir = os.getenv("OUTPUT_DIR", config.output_dir)
        config.size = os.getenv("NANOBANANA_SIZE", config.size)
        
        # Load rate limit config
        config.rate_limit.rpm = int(os.getenv("RPM", config.rate_limit.rpm))
        config.rate_limit.concurrency = int(os.getenv("CONCURRENCY", config.rate_limit.concurrency))
        config.rate_limit.timeout = float(os.getenv("TIMEOUT", config.rate_limit.timeout))
        
        return config

