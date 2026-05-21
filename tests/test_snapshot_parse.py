"""Tests for the snapshot-payload extractor that pulls the Verifier's
scene snapshot dict out of blender-mcp's `execute_blender_code` response."""

from __future__ import annotations

import json

from blendering.orchestrator import _parse_snapshot_payload


def test_extracts_clean_snapshot_after_blender_prefix() -> None:
    snapshot = {
        "objects": {
            "dog_body": {
                "primitive_guess": "cube",
                "vert_count": 8,
                "world_location": [0.0, 0.0, 0.0],
                "world_bbox_min": [-0.5, -0.5, -0.5],
                "world_bbox_max": [0.5, 0.5, 0.5],
                "rotation_euler_deg": [0.0, 0.0, 0.0],
            }
        }
    }
    text = f"Code executed successfully: {json.dumps(snapshot)}\n"
    assert _parse_snapshot_payload(text) == snapshot


def test_walks_to_matching_brace_not_last_brace() -> None:
    """Regression: the previous implementation used text.rfind('{') which
    landed on the deepest nested object, then JSON-parsed a fragment that
    didn't decode → fell back to {'objects': {}}."""
    snapshot = {
        "objects": {
            "a": {
                "primitive_guess": "cube",
                "vert_count": 8,
                "world_location": [0.0, 0.0, 0.0],
                "world_bbox_min": [-0.5, -0.5, -0.5],
                "world_bbox_max": [0.5, 0.5, 0.5],
                "rotation_euler_deg": [0.0, 0.0, 0.0],
            },
            "b": {
                "primitive_guess": "cube",
                "vert_count": 8,
                "world_location": [1.0, 0.0, 0.0],
                "world_bbox_min": [0.5, -0.5, -0.5],
                "world_bbox_max": [1.5, 0.5, 0.5],
                "rotation_euler_deg": [0.0, 0.0, 0.0],
            },
        }
    }
    text = f"Code executed successfully: {json.dumps(snapshot)}\n"
    result = _parse_snapshot_payload(text)
    assert set(result["objects"]) == {"a", "b"}, (
        "Both nested objects must round-trip — the bug skipped them by jumping "
        "to the last brace, which was inside the LAST nested object."
    )


def test_empty_objects_when_sentinel_missing() -> None:
    text = "Code executed successfully: (no output)"
    assert _parse_snapshot_payload(text) == {"objects": {}}


def test_empty_objects_when_text_blank() -> None:
    assert _parse_snapshot_payload("") == {"objects": {}}


def test_handles_braces_inside_string_values() -> None:
    """A material name or rotation field could conceivably contain a brace
    character. The walker must not be tricked into closing the JSON early."""
    snapshot = {
        "objects": {
            "obj_with_weird_string": {
                "primitive_guess": "cube",
                "vert_count": 8,
                "world_location": [0.0, 0.0, 0.0],
                "world_bbox_min": [-0.5, -0.5, -0.5],
                "world_bbox_max": [0.5, 0.5, 0.5],
                "rotation_euler_deg": [0.0, 0.0, 0.0],
            }
        }
    }
    text = (
        'Code executed successfully: Some prefix with {brace} chars '
        f'then: {json.dumps(snapshot)}'
    )
    assert _parse_snapshot_payload(text) == snapshot
