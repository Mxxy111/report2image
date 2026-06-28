"""Fetch and normalize provider model catalogs."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import aiohttp


def normalize_base_url(base_url: str | None) -> str:
    value = (base_url or "https://api.openai.com/v1").strip().rstrip("/")
    endpoint_suffixes = (
        "/chat/completions",
        "/images/generations",
        "/images/edits",
        "/images/variations",
        "/responses",
        "/embeddings",
        "/models",
    )
    changed = True
    while changed:
        changed = False
        lower = value.lower()
        for suffix in endpoint_suffixes:
            if lower.endswith(suffix):
                value = value[: -len(suffix)].rstrip("/")
                changed = True
                break
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc and not parsed.path:
        return f"{value}/v1"
    return value.rstrip("/")


def parse_model_ids(payload: Any) -> list[str]:
    if isinstance(payload, dict):
        candidates = payload.get("data", payload.get("models", []))
    else:
        candidates = payload

    model_ids: list[str] = []
    if not isinstance(candidates, list):
        return model_ids

    for item in candidates:
        model_id = None
        if isinstance(item, str):
            model_id = item
        elif isinstance(item, dict):
            for key in ("id", "name", "model"):
                value = item.get(key)
                if isinstance(value, str):
                    model_id = value
                    break
        if model_id and model_id not in model_ids:
            model_ids.append(model_id)
    return model_ids


async def fetch_openai_compatible_models(
    *,
    api_key: str,
    base_url: str | None,
    timeout_seconds: float = 60,
) -> tuple[list[str], str]:
    if not api_key.strip():
        raise ValueError("API key is required to fetch model list.")

    normalized = normalize_base_url(base_url)
    urls = [f"{normalized}/models"]
    if not normalized.lower().endswith("/v1"):
        urls.append(f"{normalized}/v1/models")

    headers = {"Authorization": f"Bearer {api_key.strip()}"}
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    last_error = ""
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for url in urls:
            try:
                async with session.get(url, headers=headers) as response:
                    text = await response.text()
                    if response.status >= 400:
                        last_error = f"{response.status}: {text[:300]}"
                        continue
                    payload = await response.json(content_type=None)
                    models = parse_model_ids(payload)
                    if models:
                        return models, url
                    last_error = "Provider returned no model IDs."
            except Exception as exc:
                last_error = str(exc)
    raise RuntimeError(f"Unable to fetch model list from provider: {last_error}")
