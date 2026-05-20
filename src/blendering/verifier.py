"""Deterministic checker that compares the current Blender scene to the Plan.

Pure function: no I/O, no MCP. The orchestrator gathers the snapshot dict
and passes it in. Output is a structured VerifierDiff consumed by both
the Actor (next turn) and the Critic (this turn)."""

from __future__ import annotations

from typing import Any

from .config import VerifierConfig
from .schemas import PartDiff, PartSpec, Plan, PositionSpec, VerifierDiff


def verify(plan: Plan, snapshot: dict[str, Any], cfg: VerifierConfig) -> VerifierDiff:
    """Compare the Plan to the scene snapshot. Returns a structured diff."""
    objects: dict[str, dict[str, Any]] = snapshot.get("objects", {})
    part_diffs: list[PartDiff] = []
    seen_names: set[str] = set()
    off_count = 0

    # Index parts by id for relative-position lookups.
    by_id: dict[str, PartSpec] = {p.id: p for p in plan.parts}

    for part in plan.parts:
        obj = objects.get(part.id)
        # Fuzzy fallback by primitive guess if exact-name not found.
        if obj is None:
            for name, candidate in objects.items():
                if name in seen_names:
                    continue
                if _fuzzy_match(part, candidate, cfg):
                    obj = candidate
                    seen_names.add(name)
                    break

        if obj is None:
            part_diffs.append(
                PartDiff(part_id=part.id, status="missing", issues=["object not found"])
            )
            continue
        seen_names.add(part.id)

        issues: list[str] = []
        measured: dict[str, Any] = {
            "location": obj.get("world_location"),
            "vert_count": obj.get("vert_count"),
        }

        # Primitive sanity (lenient — used only as a soft hint).
        primitive_ok = _primitive_plausible(part.primitive, obj.get("vert_count"))
        if not primitive_ok:
            issues.append(
                f"vert_count {obj.get('vert_count')} unusual for primitive {part.primitive}"
            )

        # Dimensions — for relative-positioned parts only check the face-normal axis
        # (the perpendicular axes depend on how the part was modelled, not the face constraint).
        dim_primary_axis: int | None = None
        if part.position.mode == "relative" and part.position.anchor_face is not None:
            dim_primary_axis = _face_axis(part.position.anchor_face)
        dim_issues, dim_measured = _check_dimensions(part, obj, cfg, primary_axis=dim_primary_axis)
        issues.extend(dim_issues)
        measured.update(dim_measured)

        # Position (absolute or relative)
        pos_issues, expected = _check_position(part, obj, by_id, objects, cfg)
        issues.extend(pos_issues)
        measured["expected_location"] = expected

        # Orientation
        rot = obj.get("rotation_euler_deg") or [0.0, 0.0, 0.0]
        for axis_idx, axis in enumerate(("x", "y", "z")):
            delta = abs(rot[axis_idx] - part.orientation_deg[axis_idx])
            # Wrap delta to [0, 180]
            delta = delta % 360.0
            if delta > 180.0:
                delta = 360.0 - delta
            if delta > cfg.orientation_tolerance_deg:
                issues.append(
                    f"rotation_{axis} {rot[axis_idx]:.1f}° vs plan {part.orientation_deg[axis_idx]:.1f}° "
                    f"(delta {delta:.1f}°)"
                )

        status = "off" if issues else "ok"
        if status == "off":
            off_count += 1
        part_diffs.append(
            PartDiff(part_id=part.id, status=status, issues=issues, measured=measured)
        )

    extras = sorted(name for name in objects if name not in seen_names)

    missing_count = sum(1 for d in part_diffs if d.status == "missing")
    is_structural = (
        (cfg.missing_is_structural and missing_count > 0)
        or off_count >= cfg.off_threshold_for_structural
    )

    summary = _summary(part_diffs, extras)

    return VerifierDiff(
        plan_version=plan.version,
        parts=part_diffs,
        extras=extras,
        summary=summary,
        is_structural=is_structural,
    )


def _fuzzy_match(part: PartSpec, candidate: dict[str, Any], cfg: VerifierConfig) -> bool:
    """Used when exact-name lookup fails. Match on primitive guess + rough dims."""
    if candidate.get("primitive_guess") != part.primitive:
        return False
    extents = _extents(candidate)
    if extents is None:
        return False
    expected = _expected_extents(part)
    if expected is None:
        return False
    for axis_idx in range(3):
        if expected[axis_idx] == 0.0:
            continue
        if abs(extents[axis_idx] - expected[axis_idx]) / expected[axis_idx] > cfg.dimension_tolerance:
            return False
    return True


def _primitive_plausible(primitive: str, vert_count: Any) -> bool:
    if not isinstance(vert_count, int):
        return True  # don't penalize unknown
    table = {
        "cube": (8, 8),
        "plane": (4, 4),
        "cone": (3, 200),
        "cylinder": (6, 200),
        "sphere": (10, 5000),
    }
    if primitive not in table:
        return True
    lo, hi = table[primitive]
    return lo <= vert_count <= hi


def _extents(obj: dict[str, Any]) -> tuple[float, float, float] | None:
    mn = obj.get("world_bbox_min")
    mx = obj.get("world_bbox_max")
    if not mn or not mx:
        return None
    return (mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2])


def _expected_extents(part: PartSpec) -> tuple[float, float, float] | None:
    d = part.dimensions
    if part.primitive == "cube" or part.primitive == "plane":
        return (d.get("x", 0.0), d.get("y", 0.0), d.get("z", 0.0))
    if part.primitive in ("cylinder", "cone"):
        r = d.get("radius", 0.0)
        return (2 * r, 2 * r, d.get("height", 0.0))
    if part.primitive == "sphere":
        r = d.get("radius", 0.0)
        return (2 * r, 2 * r, 2 * r)
    return None


def _check_dimensions(
    part: PartSpec,
    obj: dict[str, Any],
    cfg: VerifierConfig,
    primary_axis: int | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Check dimensions against plan tolerances.

    When *primary_axis* is given (0=x, 1=y, 2=z) only that axis is checked.
    This is used for relative-positioned parts where only the face-normal axis
    is reliable — the other extents may differ due to how the part was modelled.
    """
    issues: list[str] = []
    extents = _extents(obj)
    expected = _expected_extents(part)
    if extents is None or expected is None:
        return issues, {"extents": extents}
    axes = [(primary_axis, ("x", "y", "z")[primary_axis])] if primary_axis is not None else list(enumerate(("x", "y", "z")))
    for axis_idx, axis in axes:
        exp = expected[axis_idx]
        if exp <= 0.0:
            continue
        actual = extents[axis_idx]
        frac = abs(actual - exp) / exp
        if frac > cfg.dimension_tolerance:
            issues.append(
                f"{axis} extent {actual:.3f} vs plan {exp:.3f} "
                f"({frac * 100:.0f}% off; tol {cfg.dimension_tolerance * 100:.0f}%)"
            )
    return issues, {"extents": list(extents)}


def _check_position(
    part: PartSpec,
    obj: dict[str, Any],
    by_id: dict[str, PartSpec],
    objects: dict[str, dict[str, Any]],
    cfg: VerifierConfig,
) -> tuple[list[str], list[float] | None]:
    actual = obj.get("world_location")
    if actual is None:
        return ["world_location missing in snapshot"], None
    expected = _expected_position(part, by_id, objects)
    if expected is None:
        return [], None
    delta = sum((actual[i] - expected[i]) ** 2 for i in range(3)) ** 0.5
    if delta > cfg.position_tolerance:
        return (
            [
                f"position {tuple(round(v, 3) for v in actual)} vs expected "
                f"{tuple(round(v, 3) for v in expected)} (delta {delta:.3f}m; "
                f"tol {cfg.position_tolerance}m)"
            ],
            list(expected),
        )
    return [], list(expected)


def _expected_position(
    part: PartSpec,
    by_id: dict[str, PartSpec],
    objects: dict[str, dict[str, Any]],
) -> tuple[float, float, float] | None:
    pos: PositionSpec = part.position
    if pos.mode == "absolute":
        return pos.xyz
    # Relative: find anchor object in the snapshot, compute face point + offset.
    if pos.anchor_part is None:
        return None
    anchor_obj = objects.get(pos.anchor_part)
    if anchor_obj is None:
        return None
    face_point = _face_point(anchor_obj, pos.anchor_face)
    if face_point is None:
        return None
    # Expected: place this part such that its corresponding face sits on the anchor's face.
    # We approximate by offsetting the face_point by half this part's extent along the face normal,
    # so the part's CENTER lands on (face_point + normal * half_extent + offset).
    own_extents = _expected_extents(part)
    half_along_normal = 0.0
    if own_extents is not None:
        face_axis = _face_axis(pos.anchor_face)
        if face_axis is not None:
            half_along_normal = own_extents[face_axis] / 2.0
    normal = _face_normal(pos.anchor_face)
    ox, oy, oz = pos.offset
    return (
        face_point[0] + normal[0] * half_along_normal + ox,
        face_point[1] + normal[1] * half_along_normal + oy,
        face_point[2] + normal[2] * half_along_normal + oz,
    )


def _face_point(obj: dict[str, Any], face: str | None) -> tuple[float, float, float] | None:
    mn = obj.get("world_bbox_min")
    mx = obj.get("world_bbox_max")
    if not mn or not mx or face is None:
        return None
    cx = (mn[0] + mx[0]) / 2.0
    cy = (mn[1] + mx[1]) / 2.0
    cz = (mn[2] + mx[2]) / 2.0
    if face == "top":
        return (cx, cy, mx[2])
    if face == "bottom":
        return (cx, cy, mn[2])
    if face == "front":
        return (cx, mn[1], cz)
    if face == "back":
        return (cx, mx[1], cz)
    if face == "left":
        return (mn[0], cy, cz)
    if face == "right":
        return (mx[0], cy, cz)
    if face == "center":
        return (cx, cy, cz)
    return None


def _face_normal(face: str | None) -> tuple[float, float, float]:
    return {
        "top": (0.0, 0.0, 1.0),
        "bottom": (0.0, 0.0, -1.0),
        "front": (0.0, -1.0, 0.0),
        "back": (0.0, 1.0, 0.0),
        "left": (-1.0, 0.0, 0.0),
        "right": (1.0, 0.0, 0.0),
        "center": (0.0, 0.0, 0.0),
    }.get(face or "", (0.0, 0.0, 0.0))


def _face_axis(face: str | None) -> int | None:
    return {
        "top": 2, "bottom": 2,
        "front": 1, "back": 1,
        "left": 0, "right": 0,
    }.get(face or "")


def _summary(parts: list[PartDiff], extras: list[str]) -> str:
    ok = sum(1 for p in parts if p.status == "ok")
    off = sum(1 for p in parts if p.status == "off")
    missing = sum(1 for p in parts if p.status == "missing")
    bits = [f"{ok} ok"]
    if off:
        bits.append(f"{off} off")
    if missing:
        bits.append(f"{missing} missing")
    if extras:
        bits.append(f"{len(extras)} extra")
    return ", ".join(bits)
