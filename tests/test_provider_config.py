import json

import pytest

from petct.provider_config import BindingOverride, ProviderRegistry


def test_registry_resolves_independent_task_bindings(tmp_path, monkeypatch):
    config_path = tmp_path / "providers.json"
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "defaultPipelineId": "mixed",
                "providers": [
                    {
                        "id": "openai",
                        "label": "OpenAI",
                        "apiStyle": "openai",
                        "apiKeyEnv": "OPENAI_API_KEY",
                    },
                    {
                        "id": "gateway",
                        "label": "Gateway",
                        "apiStyle": "openai_compatible",
                        "apiKeyEnv": "GATEWAY_API_KEY",
                        "baseUrlEnv": "GATEWAY_BASE_URL",
                    },
                ],
                "pipelines": [
                    {
                        "id": "mixed",
                        "label": "Mixed pipeline",
                        "image": {
                            "providerId": "openai",
                            "model": "gpt-image-2",
                            "options": {"size": "1024x1536", "quality": "high"},
                        },
                        "review": {"providerId": "gateway", "model": "vision-model"},
                        "questions": {"providerId": "gateway", "model": "text-model"},
                        "maxRevisions": 2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "image-key")
    monkeypatch.setenv("GATEWAY_API_KEY", "gateway-key")
    monkeypatch.setenv("GATEWAY_BASE_URL", "https://gateway.example/v1")

    resolved = ProviderRegistry.load(config_path).resolve_pipeline("mixed")

    assert resolved.image.api_key == "image-key"
    assert resolved.review.base_url == "https://gateway.example/v1"
    assert resolved.questions.model == "text-model"
    assert resolved.max_revisions == 2


def test_registry_rejects_pipeline_with_unknown_provider(tmp_path):
    config_path = tmp_path / "providers.json"
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "defaultPipelineId": "invalid",
                "providers": [],
                "pipelines": [
                    {
                        "id": "invalid",
                        "label": "Invalid",
                        "image": {"providerId": "missing", "model": "image"},
                        "review": {"providerId": "missing", "model": "vision"},
                        "questions": {"providerId": "missing", "model": "text"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown provider"):
        ProviderRegistry.load(config_path)


def test_public_config_reports_key_state_without_exposing_key(tmp_path, monkeypatch):
    config_path = tmp_path / "providers.json"
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "defaultPipelineId": "default",
                "providers": [
                    {
                        "id": "openai",
                        "label": "OpenAI",
                        "apiStyle": "openai",
                        "apiKeyEnv": "OPENAI_API_KEY",
                    }
                ],
                "pipelines": [
                    {
                        "id": "default",
                        "label": "Default",
                        "image": {"providerId": "openai", "model": "gpt-image-2"},
                        "review": {"providerId": "openai", "model": "gpt-5.4"},
                        "questions": {"providerId": "openai", "model": "gpt-5.4"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "super-secret")

    public = ProviderRegistry.load(config_path).public_config()

    assert public["providers"][0]["isConfigured"] is True
    assert "super-secret" not in json.dumps(public)


def test_pipeline_allows_per_task_model_and_option_overrides(tmp_path, monkeypatch):
    config_path = tmp_path / "providers.json"
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "defaultPipelineId": "default",
                "providers": [
                    {
                        "id": "openai",
                        "label": "OpenAI",
                        "apiStyle": "openai",
                        "apiKeyEnv": "OPENAI_API_KEY",
                    }
                ],
                "pipelines": [
                    {
                        "id": "default",
                        "label": "Default",
                        "image": {
                            "providerId": "openai",
                            "model": "old-image",
                            "options": {"quality": "medium"},
                        },
                        "review": {"providerId": "openai", "model": "vision"},
                        "questions": {"providerId": "openai", "model": "text"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "key")

    resolved = ProviderRegistry.load(config_path).resolve_pipeline(
        image_override=BindingOverride(
            model="new-image",
            options={"quality": "high", "size": "1024x1536"},
        )
    )

    assert resolved.image.model == "new-image"
    assert resolved.image.options == {"quality": "high", "size": "1024x1536"}


def test_disabled_optional_tasks_do_not_require_their_keys(tmp_path, monkeypatch):
    config_path = tmp_path / "providers.json"
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "defaultPipelineId": "default",
                "providers": [
                    {
                        "id": "image",
                        "label": "Image",
                        "apiStyle": "openai",
                        "apiKeyEnv": "IMAGE_KEY",
                    },
                    {
                        "id": "optional",
                        "label": "Optional",
                        "apiStyle": "openai",
                        "apiKeyEnv": "OPTIONAL_KEY",
                    },
                ],
                "pipelines": [
                    {
                        "id": "default",
                        "label": "Default",
                        "image": {"providerId": "image", "model": "image-model"},
                        "review": {"providerId": "optional", "model": "vision-model"},
                        "questions": {"providerId": "optional", "model": "text-model"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("IMAGE_KEY", "image-key")
    monkeypatch.delenv("OPTIONAL_KEY", raising=False)

    resolved = ProviderRegistry.load(config_path).resolve_pipeline(
        require_review=False,
        require_questions=False,
    )

    assert resolved.image.api_key == "image-key"
    assert resolved.review.api_key == ""


def test_required_compatible_provider_requires_base_url(tmp_path):
    config_path = tmp_path / "providers.json"
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "defaultPipelineId": "default",
                "providers": [
                    {
                        "id": "gateway",
                        "label": "Gateway",
                        "apiStyle": "openai_compatible",
                        "apiKey": "stored-key",
                    }
                ],
                "pipelines": [
                    {
                        "id": "default",
                        "label": "Default",
                        "image": {"providerId": "gateway", "model": "image"},
                        "review": {"providerId": "gateway", "model": "vision"},
                        "questions": {"providerId": "gateway", "model": "text"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires a Base URL"):
        ProviderRegistry.load(config_path).resolve_pipeline(
            require_review=False,
            require_questions=False,
        )


def test_registry_merges_local_provider_with_stored_key(tmp_path, monkeypatch):
    config_path = tmp_path / "providers.json"
    local_path = tmp_path / "local_providers.json"
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "defaultPipelineId": "default",
                "providers": [
                    {
                        "id": "openai",
                        "label": "OpenAI",
                        "apiStyle": "openai",
                        "apiKeyEnv": "OPENAI_API_KEY",
                    }
                ],
                "pipelines": [
                    {
                        "id": "default",
                        "label": "Default",
                        "image": {"providerId": "openai", "model": "image"},
                        "review": {"providerId": "openai", "model": "vision"},
                        "questions": {"providerId": "openai", "model": "text"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    local_path.write_text(
        json.dumps(
            {
                "version": 1,
                "providers": [
                    {
                        "id": "siliconflow",
                        "label": "SiliconFlow",
                        "apiStyle": "openai_compatible",
                        "apiKey": "stored-secret",
                        "baseUrl": "https://api.siliconflow.cn/v1",
                        "cachedModels": ["deepseek-ai/DeepSeek-V3.2"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")

    registry = ProviderRegistry.load(config_path, local_path=local_path)
    public = registry.public_config()
    resolved = registry.resolve_pipeline(
        image_override=BindingOverride(
            providerId="siliconflow",
            model="deepseek-ai/DeepSeek-V3.2",
        )
    )

    local_public = next(
        provider for provider in public["providers"] if provider["id"] == "siliconflow"
    )
    assert resolved.image.api_key == "stored-secret"
    assert resolved.image.base_url == "https://api.siliconflow.cn/v1"
    assert local_public["source"] == "local"
    assert local_public["isConfigured"] is True
    assert local_public["cachedModels"] == ["deepseek-ai/DeepSeek-V3.2"]
    assert "stored-secret" not in json.dumps(public)


def test_registry_normalizes_endpoint_urls_and_exposes_task_models(tmp_path, monkeypatch):
    config_path = tmp_path / "providers.json"
    local_path = tmp_path / "local_providers.json"
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "defaultPipelineId": "default",
                "providers": [],
                "pipelines": [
                    {
                        "id": "default",
                        "label": "Default",
                        "image": {"providerId": "gateway", "model": "fallback-image"},
                        "review": {"providerId": "gateway", "model": "fallback-review"},
                        "questions": {"providerId": "gateway", "model": "fallback-question"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    local_path.write_text(
        json.dumps(
            {
                "version": 1,
                "providers": [
                    {
                        "id": "gateway",
                        "label": "Gateway",
                        "apiStyle": "openai_compatible",
                        "apiKey": "stored-secret",
                        "baseUrl": "https://api.vectorengine.ai/v1/images/generations",
                        "taskModels": {
                            "image": "gpt-image-2",
                            "review": "moonshotai/Kimi-K2",
                            "questions": "deepseek-ai/DeepSeek-V3.2",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    registry = ProviderRegistry.load(config_path, local_path=local_path)
    resolved = registry.resolve_pipeline()
    public_provider = registry.public_config()["providers"][0]

    assert resolved.image.base_url == "https://api.vectorengine.ai/v1"
    assert resolved.image.model == "gpt-image-2"
    assert resolved.review.model == "moonshotai/Kimi-K2"
    assert resolved.questions.model == "deepseek-ai/DeepSeek-V3.2"
    assert public_provider["taskModels"]["image"] == "gpt-image-2"
    assert "stored-secret" not in json.dumps(public_provider)
