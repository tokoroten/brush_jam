"""The validator's treatment of an explicitly-null optional field.

Node tests `!== undefined`, so `null` reaches the type check and is malformed.
Python used to test `is not None`, which silently accepted messages the
reference server rejected.
"""

from __future__ import annotations

from brushjam.validate import validate_client_message

def test_an_explicit_null_is_malformed_just_as_it_is_in_node() -> None:
    """Node tests `!== undefined`, so `null` reaches the type check and fails.

    Python used to test `is not None`, which treated an explicit null as an
    absent field - so a message the reference server rejected was accepted and
    applied here.
    """
    for field, message in (
        ("denoise", "set_ai_settings denoise must be a finite number"),
        ("negativePrompt", "set_ai_settings negativePrompt must be a string"),
        ("aiResolution", "set_ai_settings aiResolution must be a finite number"),
        ("aiProfile", "set_ai_settings aiProfile must be one of fast, quality"),
    ):
        result = validate_client_message({"t": "set_ai_settings", field: None})
        assert not result.ok, field
        assert result.error == message, field

    # Absence is still the only way to say "not supplied".
    ok = validate_client_message({"t": "set_ai_settings", "denoise": 0.5})
    assert ok.ok and "negativePrompt" not in ok.msg


def test_an_explicit_null_image_id_is_malformed() -> None:
    result = validate_client_message(
        {"t": "layer_create", "layer": {"kind": "reference", "imageId": None}}
    )
    assert not result.ok
    assert result.error == "imageId must be a string"

    # Omitting it is fine.
    assert validate_client_message({"t": "layer_create", "layer": {"kind": "reference"}}).ok
