"""Tests for the auto-frame Blender script generator."""

from __future__ import annotations

from blendering.framing import reframe_script


def test_reframe_script_includes_padding_and_min_distance() -> None:
    script = reframe_script(padding=0.2, min_distance=3.0, exclude_tags=["_helper"])
    # Generated script must reference the configured numbers literally so they
    # land in Blender's runtime, not just in Python locals.
    assert "padding = 0.2" in script
    assert "min_distance = 3.0" in script
    assert "_helper" in script


def test_reframe_script_handles_empty_scene() -> None:
    script = reframe_script(padding=0.15, min_distance=2.0, exclude_tags=[])
    # The script must early-return when there are no mesh objects to frame.
    assert "if not bboxes" in script or "if not objs" in script


def test_reframe_script_creates_default_camera_when_missing() -> None:
    script = reframe_script(padding=0.15, min_distance=2.0, exclude_tags=[])
    # On first reframe the scene may have no camera; the script must create one.
    assert "bpy.data.cameras.new" in script
    assert "scene.camera" in script


def test_reframe_script_does_not_change_focal_length() -> None:
    script = reframe_script(padding=0.15, min_distance=2.0, exclude_tags=[])
    # We move the camera, not the lens — perspective stays consistent across steps.
    assert "cam.data.lens" not in script
    assert ".lens =" not in script


def test_reframe_script_excludes_lights_and_cameras() -> None:
    script = reframe_script(padding=0.15, min_distance=2.0, exclude_tags=[])
    # AABB must ignore non-content objects.
    assert "MESH" in script  # filtering by object.type
