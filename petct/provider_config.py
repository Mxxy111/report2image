"""Validated provider profiles and task-specific model bindings."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from petct.model_catalog import normalize_base_url


class ProviderProfile(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    api_style: Literal["openai", "openai_compatible"] = Field(alias="apiStyle")
    api_key_env: str | None = Field(default=None, alias="apiKeyEnv")
    api_key: str | None = Field(default=None, alias="apiKey")
    base_url: str | None = Field(default=None, alias="baseUrl")
    base_url_env: str | None = Field(default=None, alias="baseUrlEnv")
    timeout_seconds: float = Field(default=300, alias="timeoutSeconds", gt=0)
    homepage: str | None = None
    notes: str | None = None
    cached_models: list[str] = Field(default_factory=list, alias="cachedModels")
    task_models: dict[str, str] = Field(default_factory=dict, alias="taskModels")

    def resolved_base_url(self) -> str | None:
        if self.base_url_env:
            env_value = os.getenv(self.base_url_env, "").strip()
            if env_value:
                return normalize_base_url(env_value)
        return normalize_base_url(self.base_url) if self.base_url else None

    def resolved_api_key(self) -> str:
        if self.api_key:
            return self.api_key.strip()
        if self.api_key_env:
            return os.getenv(self.api_key_env, "").strip()
        return ""


class ModelBinding(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    provider_id: str = Field(alias="providerId", min_length=1)
    model: str = Field(min_length=1)
    options: dict[str, Any] = Field(default_factory=dict)


class BindingOverride(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    provider_id: str | None = Field(default=None, alias="providerId")
    model: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class PipelineProfile(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    image: ModelBinding
    review: ModelBinding
    questions: ModelBinding
    max_revisions: int = Field(default=1, alias="maxRevisions", ge=0, le=5)


class RegistryDocument(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    version: Literal[1]
    default_pipeline_id: str = Field(alias="defaultPipelineId")
    providers: list[ProviderProfile]
    pipelines: list[PipelineProfile]

    @model_validator(mode="after")
    def validate_references(self):
        provider_ids = {provider.id for provider in self.providers}
        pipeline_ids = {pipeline.id for pipeline in self.pipelines}
        if len(provider_ids) != len(self.providers):
            raise ValueError("provider IDs must be unique")
        if len(pipeline_ids) != len(self.pipelines):
            raise ValueError("pipeline IDs must be unique")
        if self.default_pipeline_id not in pipeline_ids:
            raise ValueError("default pipeline does not exist")
        for pipeline in self.pipelines:
            for binding in (pipeline.image, pipeline.review, pipeline.questions):
                if binding.provider_id not in provider_ids:
                    raise ValueError(
                        f"pipeline '{pipeline.id}' references unknown provider "
                        f"'{binding.provider_id}'"
                    )
        return self


@dataclass(frozen=True)
class ResolvedBinding:
    provider_id: str
    provider_label: str
    api_style: str
    api_key: str
    base_url: str | None
    timeout_seconds: float
    model: str
    options: dict[str, Any]


@dataclass(frozen=True)
class ResolvedPipeline:
    id: str
    label: str
    image: ResolvedBinding
    review: ResolvedBinding
    questions: ResolvedBinding
    max_revisions: int


class ProviderRegistry:
    def __init__(
        self,
        document: RegistryDocument,
        source_path: Path,
        *,
        local_path: Path | None = None,
        local_provider_ids: set[str] | None = None,
    ):
        self.document = document
        self.source_path = source_path
        self.local_path = local_path
        self.local_provider_ids = local_provider_ids or set()
        self._providers = {provider.id: provider for provider in document.providers}
        self._pipelines = {pipeline.id: pipeline for pipeline in document.pipelines}

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        local_path: str | Path | None = None,
    ) -> "ProviderRegistry":
        source_path = Path(path)
        raw = json.loads(source_path.read_text(encoding="utf-8"))
        local_provider_ids: set[str] = set()
        resolved_local_path = Path(local_path) if local_path else None
        if resolved_local_path and resolved_local_path.exists():
            local_raw = json.loads(resolved_local_path.read_text(encoding="utf-8"))
            raw = _merge_provider_documents(raw, local_raw)
            local_provider_ids = {
                provider["id"]
                for provider in local_raw.get("providers", [])
                if provider.get("id")
            }
        return cls(
            RegistryDocument.model_validate(raw),
            source_path,
            local_path=resolved_local_path,
            local_provider_ids=local_provider_ids,
        )

    def resolve_pipeline(
        self,
        pipeline_id: str | None = None,
        *,
        image_override: BindingOverride | None = None,
        review_override: BindingOverride | None = None,
        question_override: BindingOverride | None = None,
        max_revisions: int | None = None,
        require_review: bool = True,
        require_questions: bool = True,
    ) -> ResolvedPipeline:
        selected_id = pipeline_id or self.document.default_pipeline_id
        pipeline = self._pipelines.get(selected_id)
        if pipeline is None:
            raise ValueError(f"unknown pipeline: {selected_id}")
        return ResolvedPipeline(
            id=pipeline.id,
            label=pipeline.label,
            image=self._resolve_binding(
                pipeline.image,
                image_override,
                require_key=True,
                task_model_key="image",
            ),
            review=self._resolve_binding(
                pipeline.review,
                review_override,
                require_key=require_review,
                task_model_key="review",
            ),
            questions=self._resolve_binding(
                pipeline.questions,
                question_override,
                require_key=require_questions,
                task_model_key="questions",
            ),
            max_revisions=(
                pipeline.max_revisions
                if max_revisions is None
                else max(0, min(max_revisions, 5))
            ),
        )

    def public_config(self) -> dict[str, Any]:
        providers = []
        for provider in self.document.providers:
            key_source = "stored" if provider.api_key else "environment"
            source = "local" if provider.id in self.local_provider_ids else "settings"
            providers.append(
                {
                    "id": provider.id,
                    "label": provider.label,
                    "apiStyle": provider.api_style,
                    "apiKeyEnv": provider.api_key_env,
                    "baseUrl": provider.resolved_base_url(),
                    "homepage": provider.homepage,
                    "notes": provider.notes,
                    "timeoutSeconds": provider.timeout_seconds,
                    "cachedModels": provider.cached_models,
                    "taskModels": provider.task_models,
                    "keySource": key_source,
                    "source": source,
                    "isConfigured": bool(provider.resolved_api_key()),
                }
            )
        pipelines = [
            pipeline.model_dump(by_alias=True)
            for pipeline in self.document.pipelines
        ]
        return {
            "version": self.document.version,
            "defaultPipelineId": self.document.default_pipeline_id,
            "providers": providers,
            "pipelines": pipelines,
        }

    def get_provider(self, provider_id: str) -> ProviderProfile | None:
        return self._providers.get(provider_id)

    def _resolve_binding(
        self,
        binding: ModelBinding,
        override: BindingOverride | None = None,
        require_key: bool = True,
        task_model_key: str | None = None,
    ) -> ResolvedBinding:
        provider_id = override.provider_id if override and override.provider_id else binding.provider_id
        provider = self._providers.get(provider_id)
        if provider is None:
            raise ValueError(f"unknown provider: {provider_id}")
        api_key = provider.resolved_api_key()
        if not api_key and require_key:
            raise ValueError(
                f"provider '{provider.id}' requires environment variable "
                f"{provider.api_key_env or 'or a stored API key'}"
            )
        base_url = provider.resolved_base_url()
        if require_key and provider.api_style == "openai_compatible" and not base_url:
            raise ValueError(
                f"provider '{provider.id}' uses openai_compatible API style "
                "and requires a Base URL"
            )
        provider_task_model = (
            provider.task_models.get(task_model_key or "")
            if task_model_key
            else None
        )
        return ResolvedBinding(
            provider_id=provider.id,
            provider_label=provider.label,
            api_style=provider.api_style,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=provider.timeout_seconds,
            model=(
                override.model
                if override and override.model
                else provider_task_model or binding.model
            ),
            options={
                **binding.options,
                **(override.options if override else {}),
            },
        )


def _merge_provider_documents(
    base_document: dict[str, Any],
    local_document: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(base_document)
    provider_by_id = {
        provider["id"]: dict(provider)
        for provider in base_document.get("providers", [])
        if provider.get("id")
    }
    for provider in local_document.get("providers", []):
        provider_id = provider.get("id")
        if provider_id:
            provider_by_id[provider_id] = dict(provider)
    merged["providers"] = list(provider_by_id.values())
    return merged
