from types import SimpleNamespace

from scripts.experiments.run_paired_experiment import _image_override


def test_image_override_exposes_publishable_generation_parameters():
    override = _image_override(
        SimpleNamespace(
            image_provider="image_api",
            image_model="gpt-image-2",
            image_size="1536x1024",
            image_quality="high",
            image_output_format="png",
            image_background="opaque",
            image_compression=None,
            image_input_fidelity="high",
            image_moderation="low",
        )
    )

    assert override.provider_id == "image_api"
    assert override.model == "gpt-image-2"
    assert override.options == {
        "size": "1536x1024",
        "quality": "high",
        "output_format": "png",
        "background": "opaque",
        "input_fidelity": "high",
        "moderation": "low",
    }
