"""Generates a Blender Python script that reframes the camera around scene content.

Sent verbatim through MCP's `execute_blender_code` before each screenshot so the
Critic always sees a tightly-framed view. No I/O happens in this module — it
just returns a string.
"""

from __future__ import annotations

import json


def reframe_script(
    padding: float,
    min_distance: float,
    exclude_tags: list[str],
) -> str:
    """Return Python source that reframes Blender's active camera around all mesh
    objects, with `padding` extra room on each side of the AABB and a hard
    `min_distance` floor. Objects whose names contain any string in `exclude_tags`
    are ignored. Adds a default 3/4-angle camera if none exists.
    """
    exclude_literal = json.dumps(exclude_tags)
    return f"""
import bpy
import math
from mathutils import Matrix, Vector

padding = {float(padding)}
min_distance = {float(min_distance)}
exclude_tags = {exclude_literal}
scene = bpy.context.scene

def _excluded(name):
    return any(tag in name for tag in exclude_tags)

objs = [o for o in scene.objects if o.type == "MESH" and not _excluded(o.name)]
bboxes = []
for o in objs:
    for corner in o.bound_box:
        bboxes.append(o.matrix_world @ Vector(corner))

if not bboxes:
    # Empty scene — leave the camera where it is.
    pass
else:
    mn = Vector((min(b.x for b in bboxes), min(b.y for b in bboxes), min(b.z for b in bboxes)))
    mx = Vector((max(b.x for b in bboxes), max(b.y for b in bboxes), max(b.z for b in bboxes)))
    centroid = (mn + mx) / 2.0
    diag = (mx - mn).length
    radius = (diag / 2.0) * (1.0 + padding)

    cam = scene.camera
    if cam is None:
        cam_data = bpy.data.cameras.new("AutoCam")
        cam = bpy.data.objects.new("AutoCam", cam_data)
        scene.collection.objects.link(cam)
        scene.camera = cam
        # Default 3/4 angle: from +X +Y +Z looking toward origin.
        cam.location = centroid + Vector((radius, -radius, radius))

    # Aim camera at centroid via track-to math.
    raw = cam.location - centroid
    if raw.length < 1e-4:
        direction = Vector((1.0, -1.0, 1.0)).normalized()
    else:
        direction = raw.normalized()

    # Solve dolly distance so the bounding sphere fits the camera frustum.
    fov = (
        min(cam.data.angle, cam.data.angle_y)
        if cam.data.type != "ORTHO"
        else math.radians(50.0)
    )
    fit_distance = radius / max(math.sin(fov / 2.0), 1e-4)
    distance = max(fit_distance, min_distance)
    cam.location = centroid + direction * distance

    # Point at centroid using rotation_euler from a tracking vector.
    look = (centroid - cam.location).normalized()
    # Convert look-direction into euler. Blender cameras look down -Z by default.
    up = Vector((0.0, 0.0, 1.0))
    right = look.cross(up)
    if right.length < 1e-4:
        up = Vector((0.0, 1.0, 0.0))
        right = look.cross(up)
    right = right.normalized()
    new_up = right.cross(look).normalized()
    mat = Matrix((
        (right.x, new_up.x, -look.x, cam.location.x),
        (right.y, new_up.y, -look.y, cam.location.y),
        (right.z, new_up.z, -look.z, cam.location.z),
        (0.0, 0.0, 0.0, 1.0),
    ))
    cam.matrix_world = mat
""".lstrip()
