from petct.model_catalog import normalize_base_url, parse_model_ids


def test_parse_model_ids_accepts_openai_compatible_response():
    payload = {
        "object": "list",
        "data": [
            {"id": "gpt-image-2"},
            {"id": "deepseek-ai/DeepSeek-V3.2"},
            {"id": "gpt-image-2"},
        ],
    }

    assert parse_model_ids(payload) == [
        "gpt-image-2",
        "deepseek-ai/DeepSeek-V3.2",
    ]


def test_parse_model_ids_accepts_simple_model_arrays():
    payload = {"models": ["model-a", {"name": "model-b"}]}

    assert parse_model_ids(payload) == ["model-a", "model-b"]


def test_normalize_base_url_strips_operation_suffixes():
    assert (
        normalize_base_url("https://example.com/v1/chat/completions/")
        == "https://example.com/v1"
    )
    assert (
        normalize_base_url("https://example.com/v1/images/generations/images/generations")
        == "https://example.com/v1"
    )


def test_normalize_base_url_adds_v1_for_provider_roots():
    assert normalize_base_url("https://api.siliconflow.cn") == "https://api.siliconflow.cn/v1"
    assert (
        normalize_base_url("https://example.com/v1/images/generations/images/generations")
        == "https://example.com/v1"
    )


def test_normalize_base_url_adds_v1_for_provider_roots():
    assert normalize_base_url("https://api.siliconflow.cn") == "https://api.siliconflow.cn/v1"
